import numpy as np
import torch
from scipy.linalg import expm, logm
import warnings
import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace
from math import pi
import random
#from torchvectorized import vlinalg as tv
from torch_sym3eig import Sym3Eig as se
from util.tensors import batch_cholesky, make_pos_def
import SimpleITK as sitk
from data.io import WriteTensorNPArray, WriteScalarNPArray

'''
SplitEbinMetric.py stays the same from Atlas2D to Atlas3D
'''

def trKsquare(B, A):
    det_threshold=1e-11
    B[torch.det(B)<=det_threshold] = torch.eye((3)).double().to(device=B.device)
    #G = torch.linalg.cholesky(B)
    G = batch_cholesky(B)
    nonpsd_idx = torch.where(torch.isnan(G))
    if len(nonpsd_idx[0]) > 0:
      print(len(nonpsd_idx[0]), 'non psd entries found in trKsquare')
    for i in range(len(nonpsd_idx[0])):
      G[nonpsd_idx[0][i]] = torch.eye((3)).double().to(device=B.device)
    inv_G = torch.inverse(G)
    W = torch.einsum("...ij,...jk,...lk->...il", inv_G, A, inv_G)
    #W_sym = (W + torch.transpose(W,len(W.shape)-2,len(W.shape)-1))/2
    #print('W.shape:',W.shape)
    #print('W.shape[:-2]',W.shape[:-2])
    
    lamda , _ = se.apply(W.reshape((-1,3,3)))
    #print('lamda.shape:',lamda.shape)
    lamda = lamda.reshape((*W.shape[:-2],3))
    #lamda , _ = se.apply(W_sym.reshape((-1,3,3)))
    #lamda = lamda.reshape((*W_sym.shape[:-2],3))
    result = torch.sum(torch.log(lamda.clamp(min=1.0e-15)) ** 2, (-1))
    return result

def Squared_distance_Ebin_field(g0, g1, a, mask):
#     inputs: g0.shape, g1.shape = [h, w, d, 3, 3]
#     output: scalar
#     3.3.4 https://www.cs.utah.edu/~haocheng/notes/NoteonMatching.pdf
    print('g0.shape', g0.shape)
    print('g1.shape', g1.shape)
    det_threshold=1e-11
    g0[torch.det(g0)<=det_threshold] = torch.eye((3), dtype=g0.dtype)

    inv_g0_g1 = torch.einsum("...ik,...kj->...ij", torch.inverse(g0), g1)
    trK0square = trKsquare(g0, g1) - torch.log(torch.det(inv_g0_g1).clamp(min=1.0e-15)) ** 2 *a  
    theta = torch.min((trK0square.clamp(min=1.0e-15) / a).sqrt() / 4., torch.tensor(np.pi, dtype=torch.double))
    detg0 = torch.det(g0)
    detg1 = torch.det(g1)

    alpha, beta = detg0.clamp(min=1.0e-15).pow(1. / 4.), detg1.clamp(min=1.0e-15).pow(1. / 4.)
    E = 16 * a * (alpha ** 2 - 2 * alpha * beta * torch.cos(theta) + beta ** 2)
    return (E*mask)

def Squared_distance_Ebin(g0, g1, a, mask):
#     inputs: g0.shape, g1.shape = [hxwxd, 3, 3]
#     output: scalar
#     3.3.4 https://www.cs.utah.edu/~haocheng/notes/NoteonMatching.pdf
    #print('Squared_distance_Ebin g0 NaN?', g0.isnan().any(), 'g1 NaN?', g1.isnan().any())
    #print('Squared_distance_Ebin g0 Inf?', g0.isinf().any(), 'g1 Inf?', g1.isinf().any())
    #print('Squared_distance_Ebin g0.shape', g0.shape, 'g1.shape', g1.shape, 'a', a)
    det_threshold=1e-11
    g0[torch.det(g0)<=det_threshold] = torch.eye((3)).to(device=g0.device)

    inv_g0_g1 = torch.einsum("...ik,...kj->...ij", torch.inverse(g0), g1)
    trK0square = trKsquare(g0, g1) - torch.log(torch.det(inv_g0_g1).clamp(min=1.0e-15)) ** 2 *a  # torch.log(torch.det(inv_g0_g1) + 1e-25)
    #theta = torch.min((trK0square / a + 1e-40).sqrt() / 4., torch.tensor(np.pi, dtype=torch.double))  
    #theta = torch.min((trK0square / a + 1e-7).sqrt() / 4., torch.tensor(np.pi, dtype=torch.double))  # change 1e-40 to 1e-13, because there is only one negative error of 1e-15 in UKF brain experiment
    #print('len trK0square < 0:', len(torch.where(trK0square<0)[0]))
    theta = torch.min((trK0square.clamp(min=1.0e-15) / a).sqrt() / 4., torch.tensor(np.pi, dtype=torch.double).to(device=g0.device))  # change 1e-40 to 1e-13, because there is only one negative error of 1e-15 in UKF brain experiment
    #theta = torch.min((trK0square.clamp(min=1.0e-15)).sqrt() / 4., torch.tensor(np.pi, dtype=torch.double))  # change 1e-40 to 1e-13, because there is only one negative error of 1e-15 in UKF brain experiment
    detg0 = torch.det(g0)
    detg1 = torch.det(g1)
    #print('len det(g0) < 0:',len(torch.where(detg0<=0)[0]))
    #print('len det(g1) < 0:',len(torch.where(detg1<=0)[0]))

    alpha, beta = detg0.clamp(min=1.0e-15).pow(1. / 4.), detg1.clamp(min=1.0e-15).pow(1. / 4.)
    E = 16 * a * (alpha ** 2 - 2 * alpha * beta * torch.cos(theta) + beta ** 2)

    #TODO Add cases where geodesic goes through 0
    
    #print('Squared_distance_Ebin, trK0square NaN?', trK0square.isnan().any(), 'theta NaN?', theta.isnan().any(),
    #      'alpha NaN?', alpha.isnan().any(), 'beta NaN?', beta.isnan().any())
    #print('Squared_distance_Ebin, trK0square Inf?', trK0square.isinf().any(), 'theta Inf?', theta.isinf().any(),
    #      'alpha Inf?', alpha.isinf().any(), 'beta Inf?', beta.isinf().any())
    return torch.einsum("hwd,hwd->", E, mask)


def tensor_cleaning(g, det_threshold=1e-11):
    print("In First Definition of tensor_cleaning")
    g[torch.det(g)<=det_threshold] = torch.eye((3), device=g.device)
    g[torch.transpose(g,-1,-2)!=g] = torch.eye((3), device=g.device)
    # Sylvester's criterion https://en.wikipedia.org/wiki/Sylvester%27s_criterion
    psd_map = torch.where(g[...,0,0]>0, 1, 0) + torch.where(torch.det(g[...,:2,:2])>0, 1, 0) + torch.where(torch.det(g)>0, 1, 0)
    nonpsd_idx = torch.where(psd_map!=3)
    # nonpsd_idx = torch.where(torch.isnan(torch.sum(batch_cholesky(g), (3,4))))
    for i in range(len(nonpsd_idx[0])):
        g[nonpsd_idx[0][i], nonpsd_idx[1][i], nonpsd_idx[2][i]] = torch.eye((3), device=g.device)
    return g

def cpu_logm_invB_A(B, A):
    G = torch.linalg.cholesky(B)
    inv_G = torch.inverse(G)
    W = torch.einsum("...ij,...jk,...lk->...il", inv_G, A, inv_G)
    # torch.symeig is more accurate but much slower on GPU than CPU
    lamda, Q = torch.symeig(W.cpu(), eigenvectors=True)
    #log_lamda = torch.zeros((*lamda.shape, lamda.shape[-1]),dtype=torch.double)
    log_lamda = torch.diag_embed(torch.log(lamda.to(device=B.device)))
    V = torch.einsum('...ji,...jk->...ik', inv_G, Q.to(device=B.device))
    inv_V = torch.inverse(V)
    return torch.einsum('...ij,...jk,...kl->...il', V, log_lamda, inv_V)

def logm_invB_A(B, A):
#     inputs: A/B.shape = [hxwxd, 3, 3]
#     output: shape = [hxwxd, 3, 3]
    # To convert back
    # zDirection = i % d
    # yDirection = (i / d) % w
    # xDirection = i / (w * d)

    #G = torch.linalg.cholesky(B)
    G = batch_cholesky(B)
    nonpsd_idx = torch.where(torch.isnan(G))
    if len(nonpsd_idx[0]) > 0:
      print(len(nonpsd_idx[0]), 'non psd entries found in logm_invB_A', nonpsd_idx)
    for i in range(len(nonpsd_idx[0])):
      G[nonpsd_idx[0][i]] = torch.eye((3)).double().to(device=B.device)

    det_G = torch.det(G)
    inv_G = torch.zeros_like(G)
    inv_G[det_G>0.] = torch.pinverse(G[det_G>0.])
    inv_G[det_G<=0.] = torch.eye((3)).double().to(device=B.device)
    W = torch.einsum("...ij,...jk,...lk->...il", inv_G, A, inv_G)

    lamda, Q = se.apply(W)

    # Get consistent eigenvector sign following approach of:
    # https://www.osti.gov/servlets/purl/920802
    S = torch.ones_like(lamda)
    lQ0outer = torch.einsum('...i,...i,...j->...ij',lamda[...,0].reshape((-1,1)),Q[...,0],Q[...,0])
    lQ1outer = torch.einsum('...i,...i,...j->...ij',lamda[...,1].reshape((-1,1)),Q[...,1],Q[...,1])
    lQ2outer = torch.einsum('...i,...i,...j->...ij',lamda[...,2].reshape((-1,1)),Q[...,2],Q[...,2])
    Y0 = W - lQ1outer - lQ2outer
    Y1 = W - lQ0outer - lQ2outer
    Y2 = W - lQ0outer - lQ1outer
    q0y0 = torch.einsum('...i,...i->...',Q[...,0],Y0[...,0])
    q0y1 = torch.einsum('...i,...i->...',Q[...,0],Y0[...,1])
    q0y2 = torch.einsum('...i,...i->...',Q[...,0],Y0[...,2])
    S[...,0] = torch.sign(q0y0) * q0y0 * q0y0 + torch.sign(q0y1) * q0y1 * q0y1 + torch.sign(q0y2) * q0y2 * q0y2
    q1y0 = torch.einsum('...i,...i->...',Q[...,1],Y1[...,0])
    q1y1 = torch.einsum('...i,...i->...',Q[...,1],Y1[...,1])
    q1y2 = torch.einsum('...i,...i->...',Q[...,1],Y1[...,2])
    S[...,1] = torch.sign(q1y0) * q1y0 * q1y0 + torch.sign(q1y1) * q1y1 * q1y1 + torch.sign(q1y2) * q1y2 * q1y2
    q2y0 = torch.einsum('...i,...i->...',Q[...,2],Y2[...,0])
    q2y1 = torch.einsum('...i,...i->...',Q[...,2],Y2[...,1])
    q2y2 = torch.einsum('...i,...i->...',Q[...,2],Y2[...,2])
    S[...,2] = torch.sign(q2y0) * q2y0 * q2y0 + torch.sign(q2y1) * q2y1 * q2y1 + torch.sign(q2y2) * q2y2 * q2y2
    
    log_lamda = torch.diag_embed(torch.log(lamda.clamp(min=1.0e-15)))

    # include S here to use best signs for Q
    V = torch.einsum('...ji,...jk,...k->...ik', inv_G, Q, torch.sign(S))

    # KMC comment out following 4 lines and see if pseudo inverse sufficient instead
    det_V = torch.abs(torch.det(V))
    inv_V = torch.zeros_like(V)
    inv_V[det_V>1.e-8] = torch.pinverse(V[det_V>1.e-8])
    inv_V[det_V<=1.e-8] = torch.eye((3)).double().to(device=B.device)

    result = torch.einsum('...ij,...jk,...kl->...il', V, log_lamda, inv_V)
    ill_cond_idx = (inv_V > 1e20).nonzero().reshape(-1)
    num_ill_cond = len(ill_cond_idx)
    if num_ill_cond > 0:
      dbg_ill = ill_cond_idx[0]
      #print('Replacing', num_ill_cond, 'ill-conditioned results in logm_invB_A with cpu version. First index is', dbg_ill)
      # TODO this should be batchable
      #for ii in range(num_ill_cond):
      #  result[ill_cond_idx[0][ii]] = cpu_logm_invB_A(B[ill_cond_idx[0][ii]],A[ill_cond_idx[0][ii]])
      #result[ill_cond_idx] = cpu_logm_invB_A(B[ill_cond_idx],A[ill_cond_idx])
      print('Replacing', num_ill_cond, 'ill-conditioned results in logm_invB_A with identity. First index is', dbg_ill)
      result[ill_cond_idx] = torch.eye((3)).double().to(device=B.device)
    
    return result

# Vectorize scipy.linalg.logm function
vectorized_logm = np.vectorize(logm,signature='(m,m)->(m,m)')

def scipy_logm_invB_A(B, A):
#    import SimpleITK as sitk
#     inputs: A/B.shape = [hxwxd, 3, 3]
#     output: shape = [hxwxd, 3, 3]

    G = batch_cholesky(B)
    nonpsd_idx = torch.where(torch.isnan(G))
    if len(nonpsd_idx[0]) > 0:
      print(len(nonpsd_idx[0]), 'non psd entries found in logm_invB_A', nonpsd_idx)
    for i in range(len(nonpsd_idx[0])):
      G[nonpsd_idx[0][i]] = torch.eye((3)).double().to(device=B.device)

    # KMC comment out following 4 lines and see if pseudo inverse sufficient instead
    det_G = torch.det(G)
    inv_G = torch.zeros_like(G)
    inv_G[det_G>0.] = torch.pinverse(G[det_G>0.])
    inv_G[det_G<=0.] = torch.eye((3)).double().to(device=B.device)
        
    W = torch.einsum("...ij,...jk,...lk->...il", inv_G, A, inv_G)

    logm_W = torch.from_numpy(vectorized_logm(W.cpu().detach().numpy())).double().to(device=B.device)

    return(logm_W)

# 2 without for loops using Kyle's method
def inv_RieExp(g0, g1, a):  # g0,g1: two tensors of size (s,t,...,3,3), where g0\neq 0
    '''this function is to calculate the inverse Riemannian exponential of g1 in the image of the maximal domain of the Riemannian exponential at g0
    '''
    #print('entering inv_RieExp, max(g0)', torch.max(g0), 'max(g1)', torch.max(g1))
    n = g1.size(-1)
    #print('entering inv_RieExp, g0 NaN?', g0.isnan().any(), 'g1 NaN?', g1.isnan().any())
    #print('entering inv_RieExp, g0 Inf?', g0.isinf().any(), 'g1 Inf?', g1.isinf().any())
    #     matrix multiplication
    #inv_g0_g1 = torch.einsum("...ik,...kj->...ij", torch.inverse(g0), g1)  # (s,t,...,3,3)
    #inv_g0_g1 = make_pos_def(torch.einsum("...ik,...kj->...ij", torch.inverse(g0), g1),None, 1.0e-10)  # (s,t,...,3,3)
    logm_invg0_g1 = logm_invB_A(g0, g1)
    tr_g0_b = torch.einsum("...kii->...k", logm_invg0_g1)
    b = torch.einsum("...ik,...kj->...ij",g0, logm_invg0_g1)
    bT = b - torch.einsum("...,...ij->...ij", tr_g0_b, g0) * a
    #print('inv_RieExp, max(inv_g0_g1)', torch.max(inv_g0_g1), 'max(inverse(g0))', torch.max(torch.inverse(g0)))
    
    #print('inv_RieExp after make_pos_def, g0 NaN?', g0.isnan().any(), 'g1 NaN?', g1.isnan().any(),'inv_g0_g1 NaN?',
    #      inv_g0_g1.isnan().any())
    #print('inv_RieExp after make_pos_def, g0 Inf?', g0.isinf().any(), 'g1 Inf?', g1.isinf().any(),'inv_g0_g1 Inf?',
    #      inv_g0_g1.isinf().any())
    
    def get_u_g0direction(g0, logm_invg0_g1):  # (-1,3,3) first reshape g0,g1,inv_g..
        #         permute
        #inv_g0_g1 = torch.einsum("...ij->ij...", inv_g0_g1)  # (3,3,-1)
        #s = inv_g0_g1[0, 0].clamp(min=1.0e-15)  # (-1)
        #u = 4 / n * (s ** (n / 4) - 1) * torch.einsum("...ij->ij...", g0)  # (-1)@(3,3,-1) -> (3,3,-1)
        tr_g0_b = torch.einsum("...kii->...k", logm_invg0_g1)
        u = 4 / n * (torch.exp(tr_g0_b / 4.0) - 1) * torch.einsum("...ij->ij...", g0)
        # transpose/permute trick allows scalar multiplication, otherwise do following and skip permute
        #u = 4 / n * torch.einsum("...,...ij->...ij",(torch.exp(tr_g0_b / 4.0) - 1), g0)
        
        #print('inv_RieExp.get_u_g0direction, g0 NaN?', g0.isnan().any(), 'inv_g0_g1 NaN?', inv_g0_g1.isnan().any(),
        #      's NaN?', s.isnan().any(), 'u NaN?', u.isnan().any())
        #print('inv_RieExp.get_u_g0direction, g0 Inf?', g0.isinf().any(), 'inv_g0_g1 Inf?', inv_g0_g1.isinf().any(),
        #      's Inf?', s.isinf().any(), 'u Inf?', u.isinf().any())
        #print('inv_RieExp get_u_g0direction max(u)', torch.max(u), 'max(s)', torch.max(s))
        return u.permute(2, 0, 1)  # (-1,3,3)
        #return u  # (-1,3,3)

    def get_u_ng0direction(g0, g1, logm_invg0_g1, bT, a):  # (-1,3,3) first reshape g0,g1,inv_g..
        det_threshold=1e-11
        where_below = torch.where(torch.det(g0)<=det_threshold)
        num_below = len(where_below[0])
        if num_below > 0:
          print('inv_RieExp num det(g0) below thresh:', num_below)
        tr_g0_b = torch.einsum("...kii->...k", logm_invg0_g1)
        expTrg0invb = torch.exp(tr_g0_b / 4.0)
        #         AA^T
        g0inv = torch.inverse(g0)
        g0bT = torch.einsum("...ik,...kj->...ij", g0inv, bT)
        theta = ((1. / a * torch.einsum("...ik,...ki->...", g0bT, g0bT)).clamp(min=1.0e-15).sqrt() / 4.).clamp(min=1.0e-15)  # (-1)

        A = 4. / n * (expTrg0invb * torch.cos(theta) - 1)  # (-1)
        B = 1. / theta * expTrg0invb * torch.sin(theta)
        # Clarke
        u = A * torch.einsum("...ij->ij...", g0) + B * torch.einsum("...ij->ij...", bT)  # (-1)@(3,3,-1) -> (3,3,-1)
        # Kris
        #u = A * torch.einsum("...ij->ij...", g0) + B * torch.einsum("...ik,...kj->ij...", bT, g0)  # (-1)@(3,3,-1) -> (3,3,-1)
        #u = A * torch.einsum("...ij->ij...", g0) + B * torch.einsum("...ik,...kl,...lj->ij...", g0inv, bT, g0)  # (-1)@(3,3,-1) -> (3,3,-1)
        # Kris trying to use conversion between GMM and Clarke, BT = g0H0
        #u = A * torch.einsum("...ij->ij...", g0) + B * torch.einsum("...ik,...kj->ij...", g0inv, bT)  # (-1)@(3,3,-1) -> (3,3,-1)
  
        # # TODO if this works, move to get_karcher_mean  
        # #g0[torch.det(g0)<=det_threshold] = torch.eye((3))
        # # It moved the problem in unexpected ways
        # K = logm_invB_A(g0, g1)
        # g0invK = torch.einsum("...ik,...kj->...ij", torch.inverse(g0), K)
        # expTrg0invK = torch.exp(torch.einsum("...kii->...k", g0invK) / 4.0)
        # # KTrless = K - torch.einsum("...ii,kl->...kl", K, torch.eye(n, dtype=torch.double, device=g0.device)) / n  # (-1,3,3)
        # KTrless = K - torch.einsum("...ii,kl->...kl", g0invK, torch.eye(n, dtype=torch.double, device=g0.device)) / n  # (-1,3,3)
        # #         AA^T
        # theta = ((1. / a * torch.einsum("...ik,...ki->...", KTrless, KTrless)).clamp(min=1.0e-15).sqrt() / 4.).clamp(min=1.0e-15)  # (-1)
        # #gamma = torch.det(g1).pow(1 / 4) / (torch.det(g0).clamp(min=1.0e-15).pow(1 / 4))  # (-1)

        # #A = 4 / n * (gamma * torch.cos(theta) - 1)  # (-1)
        # #B = 1 / theta * gamma * torch.sin(theta)
        # A = 4. / n * (expTrg0invK * torch.cos(theta) - 1)  # (-1)
        # B = 1. / theta * expTrg0invK * torch.sin(theta)

        # #u = A * torch.einsum("...ij->ij...", g0) + B * torch.einsum("...ik,...kj->ij...", g0, KTrless)  # (-1)@(3,3,-1) -> (3,3,-1)
        # u = A * torch.einsum("...ij->ij...", g0) + B * torch.einsum("...ik,...kl,...lj->ij...", g0, KTrless, g0)  # (-1)@(3,3,-1) -> (3,3,-1)
        # #print('inv_RieExp.get_u_ng0direction, g0 NaN?', g0.isnan().any(), 'g1 NaN?', g1.isnan().any(),
        # #      'inv_g0_g1 NaN?', inv_g0_g1.isnan().any(), 'K NaN?', K.isnan().any(),'theta NaN?', theta.isnan().any(),
        # #      'gamma NaN?', gamma.isnan().any(), 'A NaN?', A.isnan().any(), 'B NaN?', B.isnan().any())
        # #print('inv_RieExp.get_u_ng0direction, g0 Inf?', g0.isinf().any(), 'g1 Inf?', g1.isinf().any(),
        # #      'inv_g0_g1 Inf?', inv_g0_g1.isinf().any(), 'K Inf?', K.isinf().any(),'theta Inf?', theta.isinf().any(),
        # #      'gamma Inf?', gamma.isinf().any(), 'A Inf?', A.isinf().any(), 'B Inf?', B.isinf().any())
        # #di=1398389
        # #di=1365089
        # #di=1147366 # inv_v > 1e22
        # #di=26131
        # #if g0.shape[0] > di:
        # #  print('\nget_u_ng0direction \ng0[',di,'] =\n', g0[di], '\ng1[',di,'] =\n', g1[di],
        # #        '\nK[',di,'] =\n', K[di], '\nKTrless[',di,'] =\n', KTrless[di],
        # #        '\ntheta[',di,'] =\n', theta[di],'\ngamma[',di,'] =\n', gamma[di],
        # #        '\nA[',di,'] =\n', A[di],'\nB[',di,'] =\n', B[di],
        # #        '\nA term[',di,'] =\n', (A * torch.einsum("...ij->ij...", g0)).permute(2,0,1)[di],
        # #        '\nB term[',di,'] =\n', (B * torch.einsum("...ik,...kj->ij...", g0, KTrless)).permute(2,0,1)[di])
        # where_huge = torch.where(K > 6e14)
        # if len(where_huge[0]) > 0:
        #   print('num K huge', len(where_huge[0]), 'first huge', where_huge[0][0], where_huge[1][0], where_huge[2][0])

        # #print(' where u > 1e15 ', torch.where(u > 1e15))
        # #print('inv_RieExp get_u_ng0direction max(u)', torch.max(u), 'max(K)', torch.max(K), 'max(KTrless)', torch.max(KTrless)
        # #      , 'max(theta)', torch.max(theta), 'max(gamma)', torch.max(gamma), 'max(A)', torch.max(A), 'max(B)', torch.max(B))
        return u.permute(2, 0, 1)  # (-1,3,3)

    #inv_g0_g1_trless = inv_g0_g1 - torch.einsum("...ii,kl->...kl", inv_g0_g1, torch.eye(n, dtype=torch.double, device='cuda:0')) / n  #
    #inv_g0_g1_trless = inv_g0_g1 - torch.einsum("...ii,kl->...kl", inv_g0_g1, torch.eye(n, dtype=torch.double, device=g0.device))  # (s,t,...,2,2)
    #norm0 = torch.einsum("...ij,...ij->...", inv_g0_g1_trless, inv_g0_g1_trless).reshape(-1)  # (-1)
    norm0 = torch.einsum("...ij,...ij->...", bT, bT).reshape(-1)  # (-1)


    # find the indices for which the entries are 0s and non0s
    Ind0 = (norm0 <= 1e-12).nonzero().reshape(-1)  # using squeeze results in [1,1]->[]
    Indn0 = (norm0 > 1e-12).nonzero().reshape(-1)

    u = torch.zeros(g0.reshape(-1, n, n).size(), dtype=torch.double, device=g0.device)  # (-1,3,3)
    if len(Indn0) == 0:
        #u = get_u_g0direction(g0.reshape(-1, n, n), inv_g0_g1.reshape(-1, n, n))
        u = get_u_g0direction(g0.reshape(-1, n, n), logm_invg0_g1.reshape(-1, n, n))
        #di=1398389
        #di=1365089
        #di=1147366 # inv_v > 1e22
        #di=26131
        #if u.shape[0] > di:
        #    print('\nget_u_g0direction u[',di,'] =\n', u[di])
    elif len(Ind0) == 0:
        #u = get_u_ng0direction(g0.reshape(-1, n, n), g1.reshape(-1, n, n), inv_g0_g1.reshape(-1, n, n), a)
        u = get_u_ng0direction(g0.reshape(-1, n, n), g1.reshape(-1, n, n), logm_invg0_g1.reshape(-1, n, n), bT.reshape(-1, n, n), a)
        #di=1398389
        #di=1365089
        #di=1147366 # inv_v > 1e22
        #di=26131
        #if u.shape[0] > di:
        #    print('\nget_u_ng0direction u[',di,'] =\n', u[di])
    else:
        #u[Ind0] = get_u_g0direction(g0.reshape(-1, n, n)[Ind0], inv_g0_g1.reshape(-1, n, n)[Ind0])
        #u[Indn0] = get_u_ng0direction(g0.reshape(-1, n, n)[Indn0], g1.reshape(-1, n, n)[Indn0], inv_g0_g1.reshape(-1, n, n)[Indn0], a)
        u[Ind0] = get_u_g0direction(g0.reshape(-1, n, n)[Ind0], logm_invg0_g1.reshape(-1, n, n)[Ind0])
        u[Indn0] = get_u_ng0direction(g0.reshape(-1, n, n)[Indn0], g1.reshape(-1, n, n)[Indn0], logm_invg0_g1.reshape(-1, n, n)[Indn0], bT.reshape(-1, n, n)[Indn0], a)
        #di=1398389
        #di=1365089
        #di=1147366 # inv_v > 1e22
        #di=26131
        #if u.shape[0] > di:
        #    print('\nget_u_g0direction get_u_ng0direction u[',di,'] =\n', u[di])
    
    #print('exiting inv_RieExp, g0 NaN?', g0.isnan().any(), 'g1 NaN?', g1.isnan().any(),'inv_g0_g1 NaN?',
    #      inv_g0_g1.isnan().any(),'u NaN?', u.isnan().any())
    #print('exiting inv_RieExp, g0 Inf?', g0.isinf().any(), 'g1 Inf?', g1.isinf().any(),'inv_g0_g1 Inf?',
    #      inv_g0_g1.isinf().any(),'u Inf?', u.isinf().any())
    return u.reshape(g1.size())

def inv_RieExp_scipy(g0, g1, a):  # g0,g1: two tensors of size (s,t,...,3,3), where g0\neq 0
    '''this function is to calculate the inverse Riemannian exponential of g1 in the image of the maximal domain of the Riemannian exponential at g0
    '''
    #print('entering inv_RieExp, max(g0)', torch.max(g0), 'max(g1)', torch.max(g1))
    n = g1.size(-1)
    #print('entering inv_RieExp, g0 NaN?', g0.isnan().any(), 'g1 NaN?', g1.isnan().any())
    #print('entering inv_RieExp, g0 Inf?', g0.isinf().any(), 'g1 Inf?', g1.isinf().any())
    #     matrix multiplication
    #inv_g0_g1 = torch.einsum("...ik,...kj->...ij", torch.inverse(g0), g1)  # (s,t,...,3,3)
    #inv_g0_g1 = make_pos_def(torch.einsum("...ik,...kj->...ij", torch.inverse(g0), g1),None, 1.0e-10)  # (s,t,...,3,3)
    logm_invg0_g1 = scipy_logm_invB_A(g0, g1)
    tr_g0_b = torch.einsum("...kii->...k", logm_invg0_g1)
    b = torch.einsum("...ik,...kj->...ij",g0, logm_invg0_g1)
    bT = b - torch.einsum("...,...ij->...ij", tr_g0_b, g0) * a

    #print('inv_RieExp, max(inv_g0_g1)', torch.max(inv_g0_g1), 'max(inverse(g0))', torch.max(torch.inverse(g0)))
    
    #print('inv_RieExp after make_pos_def, g0 NaN?', g0.isnan().any(), 'g1 NaN?', g1.isnan().any(),'inv_g0_g1 NaN?',
    #      inv_g0_g1.isnan().any())
    #print('inv_RieExp after make_pos_def, g0 Inf?', g0.isinf().any(), 'g1 Inf?', g1.isinf().any(),'inv_g0_g1 Inf?',
    #      inv_g0_g1.isinf().any())
    
    def get_u_g0direction(g0, logm_invg0_g1):  # (-1,3,3) first reshape g0,g1,inv_g..
        #         permute
        #inv_g0_g1 = torch.einsum("...ij->ij...", inv_g0_g1)  # (3,3,-1)
        #s = inv_g0_g1[0, 0].clamp(min=1.0e-15)  # (-1)
        #u = 4 / n * (s ** (n / 4) - 1) * torch.einsum("...ij->ij...", g0)  # (-1)@(3,3,-1) -> (3,3,-1)

        tr_g0_b = torch.einsum("...kii->...k", logm_invg0_g1)
        u = 4 / n * (torch.exp(tr_g0_b / 4.0) - 1) * torch.einsum("...ij->ij...", g0)
        #u = 4 / n * torch.einsum("...,...ij->...ij",(torch.exp(tr_g0_b / 4.0) - 1), g0)

        #print('inv_RieExp.get_u_g0direction, g0 NaN?', g0.isnan().any(), 'inv_g0_g1 NaN?', inv_g0_g1.isnan().any(),
        #      's NaN?', s.isnan().any(), 'u NaN?', u.isnan().any())
        #print('inv_RieExp.get_u_g0direction, g0 Inf?', g0.isinf().any(), 'inv_g0_g1 Inf?', inv_g0_g1.isinf().any(),
        #      's Inf?', s.isinf().any(), 'u Inf?', u.isinf().any())
        #print('inv_RieExp get_u_g0direction max(u)', torch.max(u), 'max(s)', torch.max(s))
        return u.permute(2, 0, 1)  # (-1,3,3)
        #return u  # (-1,3,3)

    def get_u_ng0direction(g0, g1, logm_invg0_g1, bT, a):  # (-1,3,3) first reshape g0,g1,inv_g..
        det_threshold=1e-11
        where_below = torch.where(torch.det(g0)<=det_threshold)
        num_below = len(where_below[0])
        if num_below > 0:
          print('inv_RieExp num det(g0) below thresh:', num_below)
        tr_g0_b = torch.einsum("...kii->...k", logm_invg0_g1)
        expTrg0invb = torch.exp(tr_g0_b / 4.0)
        #         AA^T
        g0inv = torch.inverse(g0)
        g0bT = torch.einsum("...ik,...kj->...ij", g0inv, bT)
        theta = ((1. / a * torch.einsum("...ik,...ki->...", g0bT, g0bT)).clamp(min=1.0e-15).sqrt() / 4.).clamp(min=1.0e-15)  # (-1)

        A = 4. / n * (expTrg0invb * torch.cos(theta) - 1)  # (-1)
        B = 1. / theta * expTrg0invb * torch.sin(theta)
        # Clarke
        u = A * torch.einsum("...ij->ij...", g0) + B * torch.einsum("...ij->ij...", bT)  # (-1)@(3,3,-1) -> (3,3,-1)
        # Kris
        #u = A * torch.einsum("...ij->ij...", g0) + B * torch.einsum("...ik,...kj->ij...", bT, g0)  # (-1)@(3,3,-1) -> (3,3,-1)
        #u = A * torch.einsum("...ij->ij...", g0) + B * torch.einsum("...ik,...kl,...lj->ij...", g0inv, bT, g0)  # (-1)@(3,3,-1) -> (3,3,-1)
        # Kris trying to use conversion between GMM and Clarke, BT = g0H0
        #u = A * torch.einsum("...ij->ij...", g0) + B * torch.einsum("...ik,...kj->ij...", g0inv, bT)  # (-1)@(3,3,-1) -> (3,3,-1)
  
        # # TODO if this works, move to get_karcher_mean  
        # #g0[torch.det(g0)<=det_threshold] = torch.eye((3))
        # # It moved the problem in unexpected ways
        # K = scipy_logm_invB_A(g0, g1)
        # g0invK = torch.einsum("...ik,...kj->...ij", torch.inverse(g0), K)
        # expTrg0invK = torch.exp(torch.einsum("...kii->...k", g0invK) / 4.0)
        # #KTrless = K - torch.einsum("...ii,kl->...kl", K, torch.eye(n, dtype=torch.double, device=g0.device)) / n  # (-1,3,3)
        # KTrless = K - torch.einsum("...ii,kl->...kl", g0invK, torch.eye(n, dtype=torch.double, device=g0.device)) / n  # (-1,3,3)
        # #         AA^T
        # theta = ((1. / a * torch.einsum("...ik,...ki->...", KTrless, KTrless)).clamp(min=1.0e-15).sqrt() / 4.).clamp(min=1.0e-15)  # (-1)
        # #gamma = torch.det(g1).pow(1 / 4) / (torch.det(g0).clamp(min=1.0e-15).pow(1 / 4))  # (-1)
        
        # #A = 4 / n * (gamma * torch.cos(theta) - 1)  # (-1)
        # #B = 1 / theta * gamma * torch.sin(theta)
        # A = 4. / n * (expTrg0invK * torch.cos(theta) - 1)  # (-1)
        # B = 1. / theta * expTrg0invK * torch.sin(theta)
        # #u = A * torch.einsum("...ij->ij...", g0) + B * torch.einsum("...ik,...kj->ij...", g0, KTrless)  # (-1)@(3,3,-1) -> (3,3,-1)
        # u = A * torch.einsum("...ij->ij...", g0) + B * torch.einsum("...ik,...kl,...lj->ij...", g0, KTrless, g0)  # (-1)@(3,3,-1) -> (3,3,-1)
        # #print('inv_RieExp.get_u_ng0direction, g0 NaN?', g0.isnan().any(), 'g1 NaN?', g1.isnan().any(),
        # #      'inv_g0_g1 NaN?', inv_g0_g1.isnan().any(), 'K NaN?', K.isnan().any(),'theta NaN?', theta.isnan().any(),
        # #      'gamma NaN?', gamma.isnan().any(), 'A NaN?', A.isnan().any(), 'B NaN?', B.isnan().any())
        # #print('inv_RieExp.get_u_ng0direction, g0 Inf?', g0.isinf().any(), 'g1 Inf?', g1.isinf().any(),
        # #      'inv_g0_g1 Inf?', inv_g0_g1.isinf().any(), 'K Inf?', K.isinf().any(),'theta Inf?', theta.isinf().any(),
        # #      'gamma Inf?', gamma.isinf().any(), 'A Inf?', A.isinf().any(), 'B Inf?', B.isinf().any())
        # #di=1398389
        # #di=1365089
        # #di=1147366 # inv_v > 1e22
        # #di=26131
        # #if g0.shape[0] > di:
        # #  print('\nget_u_ng0direction \ng0[',di,'] =\n', g0[di], '\ng1[',di,'] =\n', g1[di],
        # #        '\nK[',di,'] =\n', K[di], '\nKTrless[',di,'] =\n', KTrless[di],
        # #        '\ntheta[',di,'] =\n', theta[di],'\ngamma[',di,'] =\n', gamma[di],
        # #        '\nA[',di,'] =\n', A[di],'\nB[',di,'] =\n', B[di],
        # #        '\nA term[',di,'] =\n', (A * torch.einsum("...ij->ij...", g0)).permute(2,0,1)[di],
        # #        '\nB term[',di,'] =\n', (B * torch.einsum("...ik,...kj->ij...", g0, KTrless)).permute(2,0,1)[di])
        # where_huge = torch.where(K > 6e14)
        # if len(where_huge[0]) > 0:
        #   print('num K huge', len(where_huge[0]), 'first huge', where_huge[0][0], where_huge[1][0], where_huge[2][0])

        # #print(' where u > 1e15 ', torch.where(u > 1e15))
        # #print('inv_RieExp get_u_ng0direction max(u)', torch.max(u), 'max(K)', torch.max(K), 'max(KTrless)', torch.max(KTrless)
        # #      , 'max(theta)', torch.max(theta), 'max(gamma)', torch.max(gamma), 'max(A)', torch.max(A), 'max(B)', torch.max(B))
        return u.permute(2, 0, 1)  # (-1,3,3)

    #inv_g0_g1_trless = inv_g0_g1 - torch.einsum("...ii,kl->...kl", inv_g0_g1, torch.eye(n, dtype=torch.double, device='cuda:0')) / n  #
    #inv_g0_g1_trless = inv_g0_g1 - torch.einsum("...ii,kl->...kl", inv_g0_g1, torch.eye(n, dtype=torch.double, device=g0.device))  # (s,t,...,2,2)
    #norm0 = torch.einsum("...ij,...ij->...", inv_g0_g1_trless, inv_g0_g1_trless).reshape(-1)  # (-1)
    norm0 = torch.einsum("...ij,...ij->...", bT, bT).reshape(-1)  # (-1)

    # find the indices for which the entries are 0s and non0s
    Ind0 = (norm0 <= 1e-12).nonzero().reshape(-1)  # using squeeze results in [1,1]->[]
    Indn0 = (norm0 > 1e-12).nonzero().reshape(-1)

    u = torch.zeros(g0.reshape(-1, n, n).size(), dtype=torch.double, device=g0.device)  # (-1,3,3)
    if len(Indn0) == 0:
        #u = get_u_g0direction(g0.reshape(-1, n, n), inv_g0_g1.reshape(-1, n, n))
        u = get_u_g0direction(g0.reshape(-1, n, n), logm_invg0_g1.reshape(-1, n, n))
        #di=1398389
        #di=1365089
        #di=1147366 # inv_v > 1e22
        #di=26131
        #if u.shape[0] > di:
        #    print('\nget_u_g0direction u[',di,'] =\n', u[di])
    elif len(Ind0) == 0:
        #u = get_u_ng0direction(g0.reshape(-1, n, n), g1.reshape(-1, n, n), inv_g0_g1.reshape(-1, n, n), a)
        u = get_u_ng0direction(g0.reshape(-1, n, n), g1.reshape(-1, n, n), logm_invg0_g1.reshape(-1, n, n), bT.reshape(-1, n, n), a)
        #di=1398389
        #di=1365089
        #di=1147366 # inv_v > 1e22
        #di=26131
        #if u.shape[0] > di:
        #    print('\nget_u_ng0direction u[',di,'] =\n', u[di])
    else:
        #u[Ind0] = get_u_g0direction(g0.reshape(-1, n, n)[Ind0], inv_g0_g1.reshape(-1, n, n)[Ind0])
        #u[Indn0] = get_u_ng0direction(g0.reshape(-1, n, n)[Indn0], g1.reshape(-1, n, n)[Indn0], inv_g0_g1.reshape(-1, n, n)[Indn0], a)
        u[Ind0] = get_u_g0direction(g0.reshape(-1, n, n)[Ind0], logm_invg0_g1.reshape(-1, n, n)[Ind0])
        u[Indn0] = get_u_ng0direction(g0.reshape(-1, n, n)[Indn0], g1.reshape(-1, n, n)[Indn0], logm_invg0_g1.reshape(-1, n, n)[Indn0], bT.reshape(-1, n, n)[Indn0], a)
        #di=1398389
        #di=1365089
        #di=1147366 # inv_v > 1e22
        #di=26131
        #if u.shape[0] > di:
        #    print('\nget_u_g0direction get_u_ng0direction u[',di,'] =\n', u[di])
    
    #print('exiting inv_RieExp, g0 NaN?', g0.isnan().any(), 'g1 NaN?', g1.isnan().any(),'inv_g0_g1 NaN?',
    #      inv_g0_g1.isnan().any(),'u NaN?', u.isnan().any())
    #print('exiting inv_RieExp, g0 Inf?', g0.isinf().any(), 'g1 Inf?', g1.isinf().any(),'inv_g0_g1 Inf?',
    #      inv_g0_g1.isinf().any(),'u Inf?', u.isinf().any())
    return u.reshape(g1.size())


def Rie_Exp(g0, u, a, t=1.0):  # here g0 is of size (s,t,...,3,3) and u is of size (s,t,...,3,3), where g0\neq 0
    '''this function is to calculate the Riemannian exponential of u in the the maximal domain of the Riemannian exponential at g0
    '''
    n = g0.size(-1)
    #print('entering Rie_Exp, g0 Nan?', g0.isnan().any(), 'inverse(g0) Nan?', torch.inverse(g0).isnan().any(),
    #      'u Nan?', u.isnan().any())
    #print('entering Rie_Exp, g0 Inf?', g0.isinf().any(), 'inverse(g0) Inf?', torch.inverse(g0).isinf().any(),
    #      'u Inf?', u.isinf().any())
    #print('entering Rie_Exp, max(g0)', torch.max(g0), 'max(u)', torch.max(u))
    #U = torch.einsum("...ik,...kj->...ij", torch.inverse(g0), u)  # (s,t,...,3,3)
    #g0inv = torch.inverse(g0)
    #U = torch.einsum("...ik,...kj->...ij", g0inv, u)  # (s,t,...,3,3)
    #Ug0inv = torch.einsum("...ik,...kj->...ij", U, g0inv)
    #trU = torch.einsum("...kii->...k", Ug0inv)  # (s,t,...)
    #UTrless = U - torch.einsum("...,ij->...ij", trU, torch.eye(n, n, dtype=torch.double, device=g0.device)) / n  # (s,t,...,3,3)
    #print('Rie_Exp, max(U)', torch.max(U), 'max(trU)', torch.max(trU), 'max(UTrless)', torch.max(UTrless))

    g0inv = torch.inverse(g0)
    U = torch.einsum("...ik,...kj->...ij", g0inv, u)  # (s,t,...,3,3)
    #Ug0inv = torch.einsum("...ik,...kj->...ij", U, g0inv)
    trU = torch.einsum("...kii->...k", U)  # (s,t,...)
    # GMM UTrless = H0
    #UTrless = U - torch.einsum("...,ij->...ij", trU, torch.eye(n, n, dtype=torch.double, device=g0.device)) / n  # (s,t,...,3,3)
    # Clarke version of UTrless = BT   
    UTrless = u - torch.einsum("...,...ij->...ij", trU, g0) / n  # (s,t,...,3,3)


    #     in g0 direction:K_0=0
    def get_g1_g0direction(g0, trU, t):  # first reshape g0 (-1,3,3) and trU (-1)
        #g1 = (t * trU / 4. + 1).pow(4. / n) * torch.einsum("...ij->ij...", g0)  # (3,3,-1)
        g1 = (t * trU / 4. + 1).pow(4. / n) * torch.einsum("...ij->ij...", g0)  # (3,3,-1)
        #print('Rie_Exp, get_g1_g0direction, max(g1)', torch.max(g1))
        return g1.permute(2, 0, 1)  # (-1,3,3)

    #     not in g0 direction SplitEbinMetric.pdf Theorem 1 :K_0\not=0
    def get_g1_ng0direction(g0, trU, UTrless, a, t):  # first reshape g0,UTrless (-1,3,3) and trU (-1)
        if len((trU < -4).nonzero().reshape(-1)) != 0:
            warnings.warn('The tangent vector u is out of the maximal domain of the Riemannian exponential.', DeprecationWarning)

        q = t * trU / 4. + 1  # (-1)
        # GMM and Clarke q match    
        q = t * trU / 4. + 1  # (-1)
        g0UTrless = torch.einsum("...ik,...kj->...ij", torch.inverse(g0), UTrless)
        # GMM r
        #r = t * (1. / a * torch.einsum("...ik,...ki->...", UTrless, UTrless)).clamp(min=1.0e-15).sqrt() / 4.  # (-1)
        # Clarke r
        r = t * (1. / a * torch.einsum("...ik,...ki->...", g0UTrless, g0UTrless)).clamp(min=1.0e-15).sqrt() / 4.  # (-1)
        
        #ArctanUtrless = (torch.atan2(r, q) * torch.einsum("...ij->ij...", UTrless) / r.clamp(min=1.0e-15))  # use (2,2,-1) for computation
        # GMM Arctan
        #ArctanUtrless = t * (torch.atan2(r, q) * torch.einsum("...ij->ij...", UTrless) / r.clamp(min=1.0e-15))  # use (2,2,-1) for computation
        # Clarke Arctan, using conversion between GMM and Clarke, BT = g0H0
        ArctanUtrless = t * (torch.atan2(r, q) * torch.einsum("...ij->ij...", g0UTrless) / r.clamp(min=1.0e-15))  # use (2,2,-1) for computation
        # Kris Arctan
        #ArctanUtrless = t * (torch.atan2(r, q) * torch.einsum("...ik,...kj->ij...", g0UTrless, g0) / r.clamp(min=1.0e-15))  # use (2,2,-1) for computation
        #ArctanUtrless = t * (torch.atan2(r, q) * torch.einsum("...ik,...kj->ij...", g0inv, g0UTrless) / r.clamp(min=1.0e-15))  # use (2,2,-1) for computation

        ExpArctanUtrless = torch.nan_to_num(torch.matrix_exp(ArctanUtrless.permute(2, 0, 1)).permute(1, 2, 0))
        ExpArctanUtrless[torch.abs(ExpArctanUtrless) > 1e12] = 0

        # GMM, Clarke g1
        g1 = (q ** 2 + r ** 2).pow(2. / n) * torch.einsum("...ik,kj...->ij...", g0, ExpArctanUtrless)  # (2,2,-1)
        #di=26131
        #if g0.shape[0] > di:
        #    print('\nget_u_g0direction g0[',di,'] =\n', g0[di], '\ntrU[',di,'] =\n', trU[di],
        #          '\nUTrless[',di,'] =\n', UTrless[di], '\nr[',di,'] =\n', r[di],
        #          '\n ArctanUtrless[',di,'] =\n', ArctanUtrless.permute(2,0,1)[di], '\n ExpArctanUtrless[',di,'] =\n', ExpArctanUtrless.permute(2,0,1)[di],
        #          '\ng1[',di,'] =\n', g1.permute(2,0,1)[di])
        #print('Rie_exp where g0 > 1e15 ', torch.where(g0 > 1e15), 'where trU > 1e15 ', torch.where(trU > 1e15),
        #      'where UTrless > 1e15 ', torch.where(UTrless > 1e15), 'where r > 1e15 ', torch.where(r > 1e15),
        #      'where ArctanUtrless > 1e15 ', torch.where(ArctanUtrless > 1e15), 'where ExpArctanUtrless > 1e15 ', torch.where(ExpArctanUtrless > 1e15),
        #      'where g1 > 1e15 ', torch.where(g1.permute(2,0,1) > 1e15))

        #print('Rie_Exp get_g1_ng0direction, g0 NaN?', g0.isnan().any(), 'trU NaN?', trU.isnan().any(),
        #      'UTrless NaN?', UTrless.isnan().any(),'ArctanUtrless NaN?',ArctanUtrless.isnan().any(),
        #      'ExpArctanUtrless NaN?', ExpArctanUtrless.isnan().any(), 'g1 NaN?', g1.isnan().any())
        #print('Rie_Exp get_g1_ng0direction, g0 Inf?', g0.isinf().any(), 'trU Inf?', trU.isinf().any(),
        #      'UTrless Inf?', UTrless.isinf().any(),'ArctanUtrless Inf?',ArctanUtrless.isinf().any(),
        #      'ExpArctanUtrless Inf?', ExpArctanUtrless.isinf().any(), 'g1 Inf?', g1.isinf().any())
        #print('Rie_Exp, get_g1_ng0direction, max(g1)', torch.max(g1))
        return g1.permute(2, 0, 1)  # (-1,2,2)

    #     pointwise multiplication Tr(U^TU)
    #UMinusTrU = U - torch.einsum("...,ij->...ij", trU, torch.eye(n, n, dtype=torch.double, device=g0.device))  # (s,t,...,3,3)

    #norm0 = torch.einsum("...ij,...ij->...", UMinusTrU, UMinusTrU).reshape(-1)  # (-1)
    norm0 = torch.einsum("...ij,...ij->...", UTrless, UTrless).reshape(-1)  # (-1)

    # find the indices for which the entries are 0s and non0s
    #     k_0=0 or \not=0
    Ind0 = (norm0 <= 1e-12).nonzero().reshape(-1)
    Indn0 = (norm0 > 1e-12).nonzero().reshape(-1)

    g1 = torch.zeros(g0.reshape(-1, n, n).size(), dtype=torch.double, device=g0.device)  # (-1,2,2)
    if len(Indn0) == 0:
        g1 = get_g1_g0direction(g0.reshape(-1, n, n), trU.reshape(-1), t)
        #di=1398389
        #di=1365089
        #di=1147366 # inv_v > 1e22
        #di=26131
        #if g1.shape[0] > di:
        #    print('\nget_g1_g0 g1[',di,'] =\n', g1[di])
    elif len(Ind0) == 0:
        g1 = get_g1_ng0direction(g0.reshape(-1, n, n), trU.reshape(-1), UTrless.reshape(-1, n, n), a, t)
        #di=1398389
        #di=1365089
        #di=1147366 # inv_v > 1e22
        #di=26131
        #if g1.shape[0] > di:
        #    print('\nget_g1_ng0 g1[',di,'] =\n', g1[di])
    else:
        g1[Ind0] = get_g1_g0direction(g0.reshape(-1, n, n)[Ind0], trU.reshape(-1)[Ind0], t)
        g1[Indn0] = get_g1_ng0direction(g0.reshape(-1, n, n)[Indn0], trU.reshape(-1)[Indn0], UTrless.reshape(-1, n, n)[Indn0], a, t)
        #di=1398389
        #di=1365089
        #di=1147366 # inv_v > 1e22
        #di=26131
        #if g1.shape[0] > di:
        #    print('\nget_g1_g0 get_g1_ng0 g1[',di,'] =\n', g1[di])
    #print('exiting Rie_Exp, g0 Nan?', g0.isnan().any(), 'g1 Nan?', g1.isnan().any(), 'u Nan?', u.isnan().any(),
    #      'U Nan?', U.isnan().any())
    #print('exiting Rie_Exp, g0 Inf?', g0.isinf().any(), 'g1 Inf?', g1.isinf().any(), 'u Inf?', u.isinf().any(),
    #      'U Inf?', U.isinf().any())
    return g1.reshape(g0.size())



''' 
The following Riemannian exponential and inverse Riemannian exponential are extended to the case g0=0 
'''
def Rie_Exp_extended(g0, u, a, t=1):  # here g0 is of size (s,t,...,3,3) and u is of size (s,t,...,3,3)
    size = g0.size()
    g0, u = g0.reshape(-1, *size[-2:]), u.reshape(-1, *size[-2:])  # (-1,3,3)
    detg0 = torch.det(g0)

    Ind_g0_is0 = (detg0 == 0).nonzero().reshape(-1)
    Ind_g0_isnot0 = (detg0 != 0).nonzero().reshape(-1)

    if len(Ind_g0_isnot0) == 0:  # g0x are 0s for all x
        g1 = u * g0.size(-1) / 4
    elif len(Ind_g0_is0) == 0:  # g0x are PD for all x
        g1 = Rie_Exp(g0, u, a, t)
    else:
        g1 = torch.zeros(g0.size(), dtype=torch.double, device=g0.device)
        g1[Ind_g0_is0] = u[Ind_g0_is0] * g0.size(-1) / 4
        g1[Ind_g0_isnot0] = Rie_Exp(g0[Ind_g0_isnot0], u[Ind_g0_isnot0], a, t)
    #print('Rie_Exp_extended, g0 NaN?', g0.isnan().any(), 'g1 NaN?', g1.isnan().any(), 'u NaN?', u.isnan().any())
    #print('Rie_Exp_extended, g0 Inf?', g0.isinf().any(), 'g1 Inf?', g1.isinf().any(), 'u Inf?', u.isinf().any())
    
    return g1.reshape(size)


def inv_RieExp_extended(g0, g1, a):  # g0, g1: (s,t,...,3,3)
    #print('entering inv_RieExp_extended, g0 NaN?', g0.isnan().any(), 'g1 NaN?', g1.isnan().any())
    #print('entering inv_RieExp_extended, g0 Inf?', g0.isinf().any(), 'g1 Inf?', g1.isinf().any())

    size = g0.size()
    g0, g1 = g0.reshape(-1, *size[-2:]), g1.reshape(-1, *size[-2:])  # (-1,3,3)
    detg0 = torch.det(g0)

    Ind_g0_is0 = (detg0 == 0).nonzero().reshape(-1)
    Ind_g0_isnot0 = (detg0 != 0).nonzero().reshape(-1)

    if len(Ind_g0_isnot0) == 0:  # g0x are 0s for all x
        u = g1 * 4 / g0.size(-1)
    elif len(Ind_g0_is0) == 0:  # g0x are PD for all x
        u = inv_RieExp(g0, g1, a)
    else:
        u = torch.zeros(g0.size(), dtype=torch.double, device=g0.device)
        u[Ind_g0_is0] = g1[Ind_g0_is0] * 4 / g0.size(-1)
        u[Ind_g0_isnot0] = inv_RieExp(g0[Ind_g0_isnot0], g1[Ind_g0_isnot0], a)
    #print('exiting inv_RieExp_extended, g0 NaN?', g0.isnan().any(), 'g1 NaN?', g1.isnan().any(), 'u NaN?', u.isnan().any())
    #print('exiting inv_RieExp_extended, g0 Inf?', g0.isinf().any(), 'g1 Inf?', g1.isinf().any(), 'u Inf?', u.isinf().any())

    return u.reshape(size)

def inv_RieExp_extended_scipy(g0, g1, a):  # g0, g1: (s,t,...,3,3)
    #print('entering inv_RieExp_extended_scipy, g0 NaN?', g0.isnan().any(), 'g1 NaN?', g1.isnan().any())
    #print('entering inv_RieExp_extended_scipy, g0 Inf?', g0.isinf().any(), 'g1 Inf?', g1.isinf().any())

    size = g0.size()
    g0, g1 = g0.reshape(-1, *size[-2:]), g1.reshape(-1, *size[-2:])  # (-1,3,3)
    detg0 = torch.det(g0)

    Ind_g0_is0 = (detg0 == 0).nonzero().reshape(-1)
    Ind_g0_isnot0 = (detg0 != 0).nonzero().reshape(-1)

    if len(Ind_g0_isnot0) == 0:  # g0x are 0s for all x
        u = g1 * 4 / g0.size(-1)
    elif len(Ind_g0_is0) == 0:  # g0x are PD for all x
        u = inv_RieExp_scipy(g0, g1, a)
    else:
        u = torch.zeros(g0.size(), dtype=torch.double, device=g0.device)
        u[Ind_g0_is0] = g1[Ind_g0_is0] * 4 / g0.size(-1)
        u[Ind_g0_isnot0] = inv_RieExp_scipy(g0[Ind_g0_isnot0], g1[Ind_g0_isnot0], a)
    #print('exiting inv_RieExp_extended_scipy, g0 NaN?', g0.isnan().any(), 'g1 NaN?', g1.isnan().any(), 'u NaN?', u.isnan().any())
    #print('exiting inv_RieExp_extended_scipy, g0 Inf?', g0.isinf().any(), 'g1 Inf?', g1.isinf().any(), 'u Inf?', u.isinf().any())

    return u.reshape(size)



def get_geo(g0, g1, a, Tpts):  # (s,t,...,,3,3)
    '''
    use odd number Tpts of time points since the geodesic may go
    though the zero matrix which will give the middle point of the geodesic
    '''
    size = g0.size()

    g0, g1 = g0.reshape(-1, *size[-2:]), g1.reshape(-1, *size[-2:])  # (-1,3,3)

    Time = torch.arange(Tpts, out=torch.DoubleTensor()) / (Tpts - 1)  # (Tpts)

    U = logm_invB_A(g0, g1)
    UTrless = U - torch.einsum("...ii,kl->...kl", U, torch.eye(g1.size(-1), dtype=torch.double, device=g0.device)) / g1.size(-1)  # (...,3,3)
    theta = ((1 / a * torch.einsum("...ik,...ki->...", UTrless, UTrless)).sqrt() / 4 - np.pi)

    Ind_inRange = (theta < 0).nonzero().reshape(-1)
    Ind_notInRange = (theta >= 0).nonzero().reshape(-1)

    def geo_in_range(g0, g1, a, Tpts):
        u = inv_RieExp_extended(g0, g1, a)  # (-1,3,3)
        geo = torch.zeros(Tpts, *g0.size(), dtype=torch.double, device=g0.device)  # (Tpts,-1,3,3)
        geo[0], geo[-1] = g0, g1
        for i in range(1, Tpts - 1):
            #geo[i] = Rie_Exp_extended(g0, u * Time[i], a)
            geo[i] = Rie_Exp_extended(g0, u, a, Time[i])
        return geo  # (Tpts,-1,2,2)

    def geo_not_in_range(g0, g1, a, Tpts):  # (-1,3,3)
        m0 = torch.zeros(g0.size(), dtype=torch.double, device=g0.device)
        u0 = inv_RieExp_extended(g0, m0, a)
        u1 = inv_RieExp_extended(g1, m0, a)

        geo = torch.zeros(Tpts, *g0.size(), dtype=torch.double, device=g0.device)  # (Tpts,-1,3,3)
        geo[0], geo[-1] = g0, g1

        for i in range(1, int((Tpts - 1) / 2)):
            #geo[i] = Rie_Exp_extended(g0, u0 * Time[i], a)
            geo[i] = Rie_Exp_extended(g0, u0, a, Time[i])
        for j in range(-int((Tpts - 1) / 2), -1):
            #geo[j] = Rie_Exp_extended(g1, u1 * (1 - Time[j]), a)
            geo[j] = Rie_Exp_extended(g1, u1, a, (1 - Time[j]))
        return geo  # (Tpts,-1,2,2)

    # If g1 = 0, len(Ind_notInRange) and len(Ind_inRange) are both zero. In this case we say that g1 is in the range
    if (len(Ind_notInRange) == 0): # all in the range
        geo = geo_in_range(g0, g1, a, Tpts)
    elif (len(Ind_inRange) == 0):  # all not in range
        geo = geo_not_in_range(g0, g1, a, Tpts)
    else:
        geo = torch.zeros(Tpts, *g0.size(), dtype=torch.double, device=g0.device)  # (Tpts,-1,3,3)
        geo[:, Ind_inRange] = geo_in_range(g0[Ind_inRange], g1[Ind_inRange], a, Tpts)
        geo[:, Ind_notInRange] = geo_not_in_range(g0[Ind_notInRange], g1[Ind_notInRange], a, Tpts)
    return geo.reshape(Tpts, *size)

def get_geo_scipy(g0, g1, a, Tpts):  # (s,t,...,,3,3)
    '''
    use odd number Tpts of time points since the geodesic may go
    though the zero matrix which will give the middle point of the geodesic
    '''
    size = g0.size()

    g0, g1 = g0.reshape(-1, *size[-2:]), g1.reshape(-1, *size[-2:])  # (-1,3,3)

    Time = torch.arange(Tpts, out=torch.DoubleTensor()) / (Tpts - 1)  # (Tpts)

    U = scipy_logm_invB_A(g0, g1)
    UTrless = U - torch.einsum("...ii,kl->...kl", U, torch.eye(g1.size(-1), dtype=torch.double, device=g0.device)) / g1.size(-1)  # (...,3,3)
    theta = ((1 / a * torch.einsum("...ik,...ki->...", UTrless, UTrless)).sqrt() / 4 - np.pi)

    Ind_inRange = (theta < 0).nonzero().reshape(-1)
    Ind_notInRange = (theta >= 0).nonzero().reshape(-1)

    def geo_in_range(g0, g1, a, Tpts):
        u = inv_RieExp_extended_scipy(g0, g1, a)  # (-1,3,3)
        geo = torch.zeros(Tpts, *g0.size(), dtype=torch.double, device=g0.device)  # (Tpts,-1,3,3)
        geo[0], geo[-1] = g0, g1
        for i in range(1, Tpts - 1):
            #geo[i] = Rie_Exp_extended(g0, u * Time[i], a)
            geo[i] = Rie_Exp_extended(g0, u, a, Time[i])
        return geo  # (Tpts,-1,2,2)

    def geo_not_in_range(g0, g1, a, Tpts):  # (-1,3,3)
        print('geo not in range')
        m0 = torch.zeros(g0.size(), dtype=torch.double, device=g0.device)
        u0 = inv_RieExp_extended_scipy(g0, m0, a)
        u1 = inv_RieExp_extended_scipy(g1, m0, a)

        geo = torch.zeros(Tpts, *g0.size(), dtype=torch.double, device=g0.device)  # (Tpts,-1,3,3)
        geo[0], geo[-1] = g0, g1

        for i in range(1, int((Tpts - 1) / 2)):
            #geo[i] = Rie_Exp_extended(g0, u0 * Time[i], a)
            geo[i] = Rie_Exp_extended(g0, u0, a, Time[i])
        for j in range(-int((Tpts - 1) / 2), -1):
            #geo[j] = Rie_Exp_extended(g1, u1 * (1 - Time[j]), a)
            geo[j] = Rie_Exp_extended(g1, u1, a, (1 - Time[j]))
        return geo  # (Tpts,-1,2,2)

    # If g1 = 0, len(Ind_notInRange) and len(Ind_inRange) are both zero. In this case we say that g1 is in the range
    if (len(Ind_notInRange) == 0): # all in the range
        geo = geo_in_range(g0, g1, a, Tpts)
    elif (len(Ind_inRange) == 0):  # all not in range
        geo = geo_not_in_range(g0, g1, a, Tpts)
    else:
        geo = torch.zeros(Tpts, *g0.size(), dtype=torch.double, device=g0.device)  # (Tpts,-1,3,3)
        geo[:, Ind_inRange] = geo_in_range(g0[Ind_inRange], g1[Ind_inRange], a, Tpts)
        geo[:, Ind_notInRange] = geo_not_in_range(g0[Ind_notInRange], g1[Ind_notInRange], a, Tpts)
    return geo.reshape(Tpts, *size)

def ptPick_notInRange(g0, g1, logm_invg0_g1, i):  # (-1,3,3)
    #alpha = torch.det(g1).clamp(min=1.0e-15).pow(1 / 4) / torch.det(g0).clamp(min=1.0e-15).pow(1 / 4)  # (-1)
    tr_g0_b = torch.einsum("...kii->...k", logm_invg0_g1)
    alpha = torch.exp(tr_g0_b / 4.0)
    #print('ptPick_notInRange, g0 NaN?', g0.isnan().any(), 'g1 NaN?', g1.isnan().any(), 'alpha NaN?', alpha.isnan().any())
    #print('ptPick_notInRange, g0 Inf?', g0.isinf().any(), 'g1 Inf?', g1.isinf().any(), 'alpha Inf?', alpha.isinf().any())
    #Ind_close_to_g0 = (alpha <= i).nonzero().reshape(-1)
    #Ind_close_to_g1 = (alpha > i).nonzero().reshape(-1)
    Ind_close_to_g0 = (tr_g0_b <= i).nonzero().reshape(-1)
    Ind_close_to_g1 = (tr_g0_b > i).nonzero().reshape(-1)

    def get_gm_inLine_0g0(alpha, g0, i):
        kn_over4 = -(1 + alpha) / (i + 1)  # (-1)
        gm = (1 + kn_over4) ** (4 / g0.size(-1)) * torch.einsum("...ij->ij...", g0)  # (3,3,-1)
        return gm.permute(2, 0, 1)  # (-1,3,3)

    def get_gm_inLine_0g1(alpha, g1, i):
        kn_over4 = -i * (1 + 1 / alpha) / (i + 1)  # (-1)
        gm = (1 + kn_over4) ** (4 / g1.size(-1)) * torch.einsum("...ij->ij...", g1)  # (3,3,-1)
        return gm.permute(2, 0, 1)

    if len(Ind_close_to_g1) == 0:  # all are close to g0
        gm = get_gm_inLine_0g0(alpha, g0, i)
    elif len(Ind_close_to_g0) == 0:
        gm = get_gm_inLine_0g1(alpha, g1, i)
    else:
        gm = torch.zeros(g0.size(), dtype=torch.double, device=g0.device)
        gm[Ind_close_to_g0] = get_gm_inLine_0g0(alpha[Ind_close_to_g0], g0[Ind_close_to_g0], i)
        gm[Ind_close_to_g1] = get_gm_inLine_0g1(alpha[Ind_close_to_g1], g1[Ind_close_to_g1], i)
    return gm


def tensor_cleaning(g, det_threshold=1e-15):
    print("In Second Definition of tensor_cleaning")
    g[torch.det(g)<=det_threshold] = torch.eye((3), device=g.device)
    # # Sylvester's criterion https://en.wikipedia.org/wiki/Sylvester%27s_criterion
    # psd_map = torch.where(g[...,0,0]>0, 1, 0) + torch.where(torch.det(g[...,:2,:2])>0, 1, 0) + torch.where(torch.det(g)>0, 1, 0)
    # nonpsd_idx = torch.where(psd_map!=3)
    # # nonpsd_idx = torch.where(torch.isnan(torch.sum(batch_cholesky(g), (3,4))))
    # for i in range(len(nonpsd_idx[0])):
    #     g[nonpsd_idx[0][i], nonpsd_idx[1][i], nonpsd_idx[2][i]] = torch.eye((3))
    # torch.where()
    return g

# def get_karcher_mean(G, a, mask=None, scale_factor=1.0, filename='', device='cuda:0'):
#     size = G.size()
#     G = G.reshape(size[0], -1, *size[-2:])  # (T,-1,3,3)
#     gm = G[0].to(device)
#     #dbgi=1417992
#     #dbgi=[1384951,1378791,1398389,1398390,1404550] # problem with logm_invB_A
#     #dbgi=[1027390, 1365089, 1495564, 1499330, 1618888, 1635720, 1831681] # inv_V > 1e22
#     #dbgi=[1147366]
#     #dbgi = [26131]
#     #for di in dbgi:
#     #  if gm.shape[0] > di:
#     #    print('\nG[0,',di,'] =\n', gm[di])

#     for i in range(1, G.size(0)):
#         #print('logm_invB_A, i', i, 'max gm', torch.max(gm))
#         G_i = G[i].to(device)
#         #U = logm_invB_A(gm, G_i)
#         #U = logm_invB_A(make_pos_def(gm, mask.reshape(-1), 1.0e-10, skip_small_eval=True), G[i])
#         #UTrless = U - torch.einsum("...ii,kl->...kl", U, torch.eye(size[-1], dtype=torch.double, device=device)) / size[-1]  # (...,2,2)

#         #theta = ((torch.einsum("...ik,...ki->...", UTrless, UTrless) / a).sqrt() / 4 - np.pi)
#         #tr_gm_b = torch.einsum("...kii->...k", U)
#         #b = torch.einsum("...ik,...kj->...ij",gm, U)
#         #bT = b - torch.einsum("...,...ij->...ij", tr_gm_b, gm) * a
#         #theta = ((torch.einsum("...ik,...ki->...", bT, bT) / a).sqrt() / 4 - np.pi)

#         logm_invgm_gi = logm_invB_A(gm, G_i)
#         tr_gm_b = torch.einsum("...kii->...k", logm_invgm_gi)
#         b = torch.einsum("...ik,...kj->...ij",gm, logm_invgm_gi)
#         bT = b - torch.einsum("...,...ij->...ij", tr_gm_b, gm) * a
#         gminv = torch.inverse(gm)
#         gmbT = torch.einsum("...ik,...kj->...ij", gminv, bT)
#         theta = torch.einsum("...ik,...ki->...", gmbT, gmbT)
#         #print('tr(b):',tr_gm_b)
#         #print('theta(b):',theta)
    

#         #for di in dbgi:
#         #  if U.shape[0] > di:
#         #    print('\nG[',i,',',di,'] =\n', G[i,di],'\nU[',di,'] =\n', U[di],'\nUTrless[',di,'] =\n', UTrless[di],'\ntheta[',di,'] =\n', theta[di], '\n')
#         if filename:
#           sitk.WriteImage(sitk.GetImageFromArray(np.transpose(theta.reshape(*size[1:-2]).cpu(),(2,1,0))), filename+'_theta.nhdr')
#           U_lin = np.zeros((6,*size[1:4])).cpu()
#           #U_inv = torch.inverse(U) / tens_scale
#           U_lin[0] = U.reshape(*size[1:])[:,:,:,0,0].cpu()
#           U_lin[1] = U.reshape(*size[1:])[:,:,:,0,1].cpu()
#           U_lin[2] = U.reshape(*size[1:])[:,:,:,0,2].cpu()
#           U_lin[3] = U.reshape(*size[1:])[:,:,:,1,1].cpu()
#           U_lin[4] = U.reshape(*size[1:])[:,:,:,1,2].cpu()
#           U_lin[5] = U.reshape(*size[1:])[:,:,:,2,2].cpu()
#           WriteTensorNPArray(np.transpose(U_lin,(3,2,1,0)), filename+'_U.nhdr')

#         #thresh = 0
#         thresh = a * (4*np.pi)**2
  
#         Ind_inRange = (theta < thresh).nonzero().reshape(-1)  ## G[i] is in the range of the exponential map at gm
#         Ind_notInRange = (theta >= thresh).nonzero().reshape(-1)  ## G[i] is not in the range

#         # when g1 = 0, len(Ind_notInRange) and len(Ind_inRange) are both zero. So check len(Ind_notInRange) first
#         if len(Ind_notInRange) == 0:  # all in the range
#             #print('Before Rie_Exp_extended, i', i, 'max gm', torch.max(gm))
#             #gm = Rie_Exp_extended(gm, inv_RieExp_extended(gm, G_i, a) / (i + 1), a)
#             gm = Rie_Exp_extended(gm, inv_RieExp_extended(gm, G_i, a), a, 1.0 / (i + 1))
#             #print('after Rie_Exp_extended, i', i, 'max gm', torch.max(gm))
#         elif len(Ind_inRange) == 0:  # all not in range
#             #print('Before ptPick_notInRange, i', i, 'max gm', torch.max(gm))
#             gm = ptPick_notInRange(gm, G_i, i)
#             #print('after ptPick_notInRange, i', i, 'max gm', torch.max(gm))
#         else:
#             #print('Before Rie_Exp_extended, ptPick_notInRange, i', i, 'max gm', torch.max(gm))
#             #gm[Ind_inRange] = Rie_Exp_extended(gm[Ind_inRange],
#             #                                   inv_RieExp_extended(gm[Ind_inRange], G_i[Ind_inRange], a) / (i + 1),
#             #                                   a)  # stop here
#             gm[Ind_inRange] = Rie_Exp_extended(gm[Ind_inRange],
#                                                inv_RieExp_extended(gm[Ind_inRange], G_i[Ind_inRange], a),
#                                                                    a, 1.0 / (i + 1))  # stop here
#             #print('after Rie_Exp_extended, i', i, 'max gm', torch.max(gm))
#             gm[Ind_notInRange] = ptPick_notInRange(gm[Ind_notInRange], G_i[Ind_notInRange], i)
# #             print('end')
#             #print('after ptPick_notInRange, i', i, 'max gm', torch.max(gm))
#         #print('get_karcher_mean num zeros', len(torch.where(gm[:] == torch.zeros((size[-2],size[-2])))[0]))
#         #print("WARNING! Don't know why need to scale atlas by scale_factor")
#         #gm[:] = torch.where(gm[:] == torch.zeros((size[-2],size[-2]), device=device),
#         #               scale_factor * torch.eye(size[-2], dtype=G.dtype, device=device), scale_factor * gm[:])

#         #del G_i
#         #torch.cuda.empty_cache()


#     #return gm.reshape(*size[1:])
#     #for di in dbgi:
#     #  if gm.shape[0] > di:
#     #    print('\ngm[',di,'] =\n', gm[di])

#     #gm_cpu = gm.cpu()
#     #del gm
#     torch.cuda.empty_cache()

#     #return gm_cpu.reshape(*size[1:])
#     return gm.reshape(*size[1:])
#     #return(torch.where(gm[:] == torch.zeros((size[-2],size[-2])),
#     #                   scale_factor * torch.eye(size[-2], dtype=G.dtype), scale_factor * gm[:]).reshape(*size[1:]))

def get_karcher_mean(G, a, mask=None, scale_factor=1.0, filename='', device='cuda:0'):
  size = G.size()
  G = G.reshape(size[0], -1, *size[-2:])  # (T,-1,3,3)
  if mask is not None:
    mask = mask.reshape(-1)  # (-1,1)

  def get_karcher_mean_masked(G, a, scale_factor, filename, device):
      gm = G[0].to(device)

      for i in range(1, G.size(0)):
        #print('logm_invB_A, i', i, 'max gm', torch.max(gm))
        G_i = G[i].to(device)
        #U = logm_invB_A(gm, G_i)
        #U = logm_invB_A(make_pos_def(gm, mask.reshape(-1), 1.0e-10, skip_small_eval=True), G[i])
        #UTrless = U - torch.einsum("...ii,kl->...kl", U, torch.eye(size[-1], dtype=torch.double, device=device)) / size[-1]  # (...,2,2)

        #theta = ((torch.einsum("...ik,...ki->...", UTrless, UTrless) / a).sqrt() / 4 - np.pi)
        #tr_gm_b = torch.einsum("...kii->...k", U)
        #b = torch.einsum("...ik,...kj->...ij",gm, U)
        #bT = b - torch.einsum("...,...ij->...ij", tr_gm_b, gm) * a
        #theta = ((torch.einsum("...ik,...ki->...", bT, bT) / a).sqrt() / 4 - np.pi)

        logm_invgm_gi = logm_invB_A(gm, G_i)
        tr_gm_b = torch.einsum("...kii->...k", logm_invgm_gi)
        b = torch.einsum("...ik,...kj->...ij",gm, logm_invgm_gi)
        bT = b - torch.einsum("...,...ij->...ij", tr_gm_b, gm) * a
        gminv = torch.inverse(gm)
        gmbT = torch.einsum("...ik,...kj->...ij", gminv, bT)
        theta = torch.einsum("...ik,...ki->...", gmbT, gmbT)
        #print('tr(b):',tr_gm_b)
        #print('theta(b):',theta)
    

        #for di in dbgi:
        #  if U.shape[0] > di:
        #    print('\nG[',i,',',di,'] =\n', G[i,di],'\nU[',di,'] =\n', U[di],'\nUTrless[',di,'] =\n', UTrless[di],'\ntheta[',di,'] =\n', theta[di], '\n')
        if filename:
          sitk.WriteImage(sitk.GetImageFromArray(np.transpose(theta.reshape(*size[1:-2]).cpu(),(2,1,0))), filename+'_theta.nhdr')
          U_lin = np.zeros((6,*size[1:4])).cpu()
          #U_inv = torch.inverse(U) / tens_scale
          U_lin[0] = U.reshape(*size[1:])[:,:,:,0,0].cpu()
          U_lin[1] = U.reshape(*size[1:])[:,:,:,0,1].cpu()
          U_lin[2] = U.reshape(*size[1:])[:,:,:,0,2].cpu()
          U_lin[3] = U.reshape(*size[1:])[:,:,:,1,1].cpu()
          U_lin[4] = U.reshape(*size[1:])[:,:,:,1,2].cpu()
          U_lin[5] = U.reshape(*size[1:])[:,:,:,2,2].cpu()
          WriteTensorNPArray(np.transpose(U_lin,(3,2,1,0)), filename+'_U.nhdr')

        #thresh = 0
        thresh = a * (4*np.pi)**2
  
        Ind_inRange = (theta < thresh).nonzero().reshape(-1)  ## G[i] is in the range of the exponential map at gm
        Ind_notInRange = (theta >= thresh).nonzero().reshape(-1)  ## G[i] is not in the range

        # when g1 = 0, len(Ind_notInRange) and len(Ind_inRange) are both zero. So check len(Ind_notInRange) first
        if len(Ind_notInRange) == 0:  # all in the range
            #print('Before Rie_Exp_extended, i', i, 'max gm', torch.max(gm))
            #gm = Rie_Exp_extended(gm, inv_RieExp_extended(gm, G_i, a) / (i + 1), a)
            gm = Rie_Exp_extended(gm, inv_RieExp_extended(gm, G_i, a), a, 1.0 / (i + 1))
            #print('after Rie_Exp_extended, i', i, 'max gm', torch.max(gm))
        elif len(Ind_inRange) == 0:  # all not in range
            #print('Before ptPick_notInRange, i', i, 'max gm', torch.max(gm))
            gm = ptPick_notInRange(gm, G_i, logm_invgm_gi, i)
            #print('after ptPick_notInRange, i', i, 'max gm', torch.max(gm))
        else:
            #print('Before Rie_Exp_extended, ptPick_notInRange, i', i, 'max gm', torch.max(gm))
            #gm[Ind_inRange] = Rie_Exp_extended(gm[Ind_inRange],
            #                                   inv_RieExp_extended(gm[Ind_inRange], G_i[Ind_inRange], a) / (i + 1),
            #                                   a)  # stop here
            gm[Ind_inRange] = Rie_Exp_extended(gm[Ind_inRange],
                                               inv_RieExp_extended(gm[Ind_inRange], G_i[Ind_inRange], a),
                                                                   a, 1.0 / (i + 1))  # stop here
            #print('after Rie_Exp_extended, i', i, 'max gm', torch.max(gm))
            gm[Ind_notInRange] = ptPick_notInRange(gm[Ind_notInRange], G_i[Ind_notInRange], logm_invgm_gi[Ind_notInRange], i)
#             print('end')
            #print('after ptPick_notInRange, i', i, 'max gm', torch.max(gm))
        #print('get_karcher_mean num zeros', len(torch.where(gm[:] == torch.zeros((size[-2],size[-2])))[0]))
        #print("WARNING! Don't know why need to scale atlas by scale_factor")
        gm[:] = torch.where(gm[:] == torch.zeros((size[-2],size[-2]), device=device),
                       torch.eye(size[-2], dtype=G.dtype, device=device), gm[:])

        #del G_i
        #torch.cuda.empty_cache()
      return(gm)      
  # end def get_karcher_mean_masked

  if mask is None:
    gm = get_karcher_mean_masked(G, a, scale_factor, filename, device)
  else:
    Ind_inRange = (mask > 0.1).nonzero().reshape(-1)
    gm = torch.zeros_like(G[0])
    gm[Ind_inRange] = get_karcher_mean_masked(G[:,Ind_inRange], a, scale_factor, filename, device)

  #return gm.reshape(*size[1:])
  #for di in dbgi:
  #  if gm.shape[0] > di:
  #    print('\ngm[',di,'] =\n', gm[di])

  #gm_cpu = gm.cpu()
  #del gm
  torch.cuda.empty_cache()

  #return gm_cpu.reshape(*size[1:])
  return gm.reshape(*size[1:])
  #return(torch.where(gm[:] == torch.zeros((size[-2],size[-2])),
  #                   scale_factor * torch.eye(size[-2], dtype=G.dtype), scale_factor * gm[:]).reshape(*size[1:]))

def get_karcher_mean_scipy(G, a, mask=None, scale_factor=1.0, filename='', device='cuda:0'):
    size = G.size()
    G = G.reshape(size[0], -1, *size[-2:])  # (T,-1,3,3)
    if mask is not None:
      mask = mask.reshape(-1)

    def get_karcher_mean_scipy_masked(G, a, scale_factor, filename, device):  
      gm = G[0].to(device)

    
      for i in range(1, G.size(0)):
        #print('logm_invB_A, i', i, 'max gm', torch.max(gm))
        G_i = G[i].to(device)
        #U = scipy_logm_invB_A(gm, G_i)
        #U = logm_invB_A(make_pos_def(gm, mask.reshape(-1), 1.0e-10, skip_small_eval=True), G[i])
        #UTrless = U - torch.einsum("...ii,kl->...kl", U, torch.eye(size[-1], dtype=torch.double, device=device)) / size[-1]  # (...,2,2)

        #theta = ((torch.einsum("...ik,...ki->...", UTrless, UTrless) / a).sqrt() / 4 - np.pi)
        #tr_gm_b = torch.einsum("...kii->...k", U)
        #b = torch.einsum("...ik,...kj->...ij",gm, U)
        #bT = b - torch.einsum("...,...ij->...ij", tr_gm_b, gm) * a
        #theta = ((torch.einsum("...ik,...ki->...", bT, bT) / a).sqrt() / 4 - np.pi)

        logm_invgm_gi = scipy_logm_invB_A(gm, G_i)
        tr_gm_b = torch.einsum("...kii->...k", logm_invgm_gi)
        b = torch.einsum("...ik,...kj->...ij",gm, logm_invgm_gi)
        bT = b - torch.einsum("...,...ij->...ij", tr_gm_b, gm) * a
        gminv = torch.inverse(gm)
        gmbT = torch.einsum("...ik,...kj->...ij", gminv, bT)
        theta = torch.einsum("...ik,...ki->...", gmbT, gmbT)
        #print('tr(b):',tr_gm_b)
        #print('theta(b):',theta)

        #for di in dbgi:
        #  if U.shape[0] > di:
        #    print('\nG[',i,',',di,'] =\n', G[i,di],'\nU[',di,'] =\n', U[di],'\nUTrless[',di,'] =\n', UTrless[di],'\ntheta[',di,'] =\n', theta[di], '\n')
        if filename:
          sitk.WriteImage(sitk.GetImageFromArray(np.transpose(theta.reshape(*size[1:-2]).cpu(),(2,1,0))), filename+'_theta.nhdr')
          U_lin = np.zeros((6,*size[1:4])).cpu()
          #U_inv = torch.inverse(U) / tens_scale
          U_lin[0] = U.reshape(*size[1:])[:,:,:,0,0].cpu()
          U_lin[1] = U.reshape(*size[1:])[:,:,:,0,1].cpu()
          U_lin[2] = U.reshape(*size[1:])[:,:,:,0,2].cpu()
          U_lin[3] = U.reshape(*size[1:])[:,:,:,1,1].cpu()
          U_lin[4] = U.reshape(*size[1:])[:,:,:,1,2].cpu()
          U_lin[5] = U.reshape(*size[1:])[:,:,:,2,2].cpu()
          WriteTensorNPArray(np.transpose(U_lin,(3,2,1,0)), filename+'_U.nhdr')
        #thresh = 0
        thresh = a * (4*np.pi)**2

        Ind_inRange = (theta < thresh).nonzero().reshape(-1)  ## G[i] is in the range of the exponential map at gm
        Ind_notInRange = (theta >= thresh).nonzero().reshape(-1)  ## G[i] is not in the range

        # when g1 = 0, len(Ind_notInRange) and len(Ind_inRange) are both zero. So check len(Ind_notInRange) first
        if len(Ind_notInRange) == 0:  # all in the range
            #print('Before Rie_Exp_extended, i', i, 'max gm', torch.max(gm))
            #gm = Rie_Exp_extended(gm, inv_RieExp_extended_scipy(gm, G_i, a) / (i + 1), a)
            gm = Rie_Exp_extended(gm, inv_RieExp_extended_scipy(gm, G_i, a), a, 1.0 / (i + 1))
            #print('after Rie_Exp_extended, i', i, 'max gm', torch.max(gm))
        elif len(Ind_inRange) == 0:  # all not in range
            #print('Before ptPick_notInRange, i', i, 'max gm', torch.max(gm))
            gm = ptPick_notInRange(gm, G_i, logm_invgm_gi, i)
            #print('after ptPick_notInRange, i', i, 'max gm', torch.max(gm))
        else:
            #print('Before Rie_Exp_extended, ptPick_notInRange, i', i, 'max gm', torch.max(gm))
            #gm[Ind_inRange] = Rie_Exp_extended(gm[Ind_inRange],
            #                                   inv_RieExp_extended_scipy(gm[Ind_inRange], G_i[Ind_inRange], a) / (i + 1),
            #                                   a)  # stop here
            gm[Ind_inRange] = Rie_Exp_extended(gm[Ind_inRange],
                                               inv_RieExp_extended_scipy(gm[Ind_inRange], G_i[Ind_inRange], a),
                                               a, 1.0 / (i + 1))  # stop here
            #print('after Rie_Exp_extended, i', i, 'max gm', torch.max(gm))
            gm[Ind_notInRange] = ptPick_notInRange(gm[Ind_notInRange], G_i[Ind_notInRange], logm_invgm_gi[Ind_notInRange], i)
#             print('end')
            #print('after ptPick_notInRange, i', i, 'max gm', torch.max(gm))
        #print('get_karcher_mean num zeros', len(torch.where(gm[:] == torch.zeros((size[-2],size[-2])))[0]))
        #print("WARNING! Don't know why need to scale atlas by scale_factor")
        #gm[:] = torch.where(gm[:] == torch.zeros((size[-2],size[-2]), device=device),
        #               scale_factor * torch.eye(size[-2], dtype=G.dtype, device=device), scale_factor * gm[:])

        #del G_i
        #torch.cuda.empty_cache()

      return(gm)      
    # end def get_karcher_mean_masked

    if mask is None:
      gm = get_karcher_mean_scipy_masked(G, a, scale_factor, filename, device)
    else:
      Ind_inRange = (mask > 0.1).nonzero().reshape(-1)
      gm = torch.zeros_like(G[0])
      gm[Ind_inRange] = get_karcher_mean_scipy_masked(G[:,Ind_inRange], a, scale_factor, filename, device)

    #return gm.reshape(*size[1:])
    #for di in dbgi:
    #  if gm.shape[0] > di:
    #    print('\ngm[',di,'] =\n', gm[di])

    #gm_cpu = gm.cpu()
    #del gm
    torch.cuda.empty_cache()

    #return gm_cpu.reshape(*size[1:])
    return gm.reshape(*size[1:])
    #return(torch.where(gm[:] == torch.zeros((size[-2],size[-2])),
    #                   scale_factor * torch.eye(size[-2], dtype=G.dtype), scale_factor * gm[:]).reshape(*size[1:]))


# def get_karcher_mean_shuffle(G, a, device='cuda:0'):
#     size = G.size()
#     G = G.reshape(size[0], -1, *size[-2:])  # (T,-1,3,3)

#     orig_list = list(range(G.shape[0]))
#     shuffle_list = orig_list.copy()
#     random.shuffle(shuffle_list)
#     idx_shuffle_map = dict(zip(orig_list, shuffle_list))

#     gm = G[idx_shuffle_map[0]].to(device)

#     for i in range(1, G.size(0)):
#         G_i = G[idx_shuffle_map[i]].to(device)
#         #U = logm_invB_A(gm, G_i)
#         #UTrless = U - torch.einsum("...ii,kl->...kl", U, torch.eye(size[-1], dtype=torch.double, device=device)) / size[-1]  # (...,2,2)
#         #theta = ((torch.einsum("...ik,...ki->...", UTrless, UTrless) / a).sqrt() / 4 - np.pi)
#         #tr_gm_b = torch.einsum("...kii->...k", U)
#         #b = torch.einsum("...ik,...kj->...ij",gm, U)
#         #bT = b - torch.einsum("...,...ij->...ij", tr_gm_b, gm) * a
#         #theta = ((torch.einsum("...ik,...ki->...", bT, bT) / a).sqrt() / 4 - np.pi)
#         logm_invgm_gi = logm_invB_A(gm, G_i)
#         tr_gm_b = torch.einsum("...kii->...k", logm_invgm_gi)
#         b = torch.einsum("...ik,...kj->...ij",gm, logm_invgm_gi)
#         bT = b - torch.einsum("...,...ij->...ij", tr_gm_b, gm) * a
#         gminv = torch.inverse(gm)
#         gmbT = torch.einsum("...ik,...kj->...ij", gminv, bT)
#         theta = torch.einsum("...ik,...ki->...", gmbT, gmbT)
#         #print('tr(b):',tr_gm_b)
#         #print('theta(b):',theta)

#         #thresh = 0
#         thresh = a * (4*np.pi)**2
        
#         Ind_inRange = (theta < thresh).nonzero().reshape(-1)  ## G[i] is in the range of the exponential map at gm
#         Ind_notInRange = (theta >= thresh).nonzero().reshape(-1)  ## G[i] is not in the range

#         # when g1 = 0, len(Ind_notInRange) and len(Ind_inRange) are both zero. So check len(Ind_notInRange) first
#         if len(Ind_notInRange) == 0:  # all in the range
#             #gm = Rie_Exp_extended(gm, inv_RieExp_extended(gm, G_i, a) / (i + 1), a)
#             gm = Rie_Exp_extended(gm, inv_RieExp_extended(gm, G_i, a), a, 1.0 / (i + 1))
#         elif len(Ind_inRange) == 0:  # all not in range
#             gm = ptPick_notInRange(gm, G_i, logm_invgm_gi, i)
#         else:
#             #gm[Ind_inRange] = Rie_Exp_extended(gm[Ind_inRange],
#             #                                   inv_RieExp_extended(gm[Ind_inRange], G_i[Ind_inRange], a) / (i + 1),
#             #                                   a)  # stop here
#             gm[Ind_inRange] = Rie_Exp_extended(gm[Ind_inRange],
#                                                inv_RieExp_extended(gm[Ind_inRange], G_i[Ind_inRange], a),
#                                                a, 1.0 / (i + 1))  # stop here
#             gm[Ind_notInRange] = ptPick_notInRange(gm[Ind_notInRange], G_i[Ind_notInRange], logm_invgm_gi[Ind_notInRange], i)
#         gm[:] = torch.where(gm[:] == torch.zeros((size[-2],size[-2]), device=device),
#                        torch.eye(size[-2], dtype=G.dtype, device=device), gm[:])

#         #del G_i
#         #torch.cuda.empty_cache()
#     #gm_cpu = gm.cpu()
#     #del gm
#     torch.cuda.empty_cache()

#     #return gm_cpu.reshape(*size[1:])    
#     return gm.reshape(*size[1:])
#     #return(torch.where(gm[:] == torch.zeros((size[-2],size[-2]), device='cuda:0'),
#     #                   torch.eye(size[-2], dtype=G.dtype, device='cuda:0'), gm[:]).reshape(*size[1:]))

def get_karcher_mean_shuffle(G, a, mask=None, device='cuda:0'):
    size = G.size()
    G = G.reshape(size[0], -1, *size[-2:])  # (T,-1,3,3)
    if mask is not None:
      mask = mask.reshape(-1)  # (-1,1)

    def get_karcher_mean_shuffle_masked(G, a, device):  
      orig_list = list(range(G.shape[0]))
      shuffle_list = orig_list.copy()
      random.shuffle(shuffle_list)
      idx_shuffle_map = dict(zip(orig_list, shuffle_list))

      gm = G[idx_shuffle_map[0]].to(device)

      for i in range(1, G.size(0)):
        G_i = G[idx_shuffle_map[i]].to(device)
        #U = logm_invB_A(gm, G_i)
        #UTrless = U - torch.einsum("...ii,kl->...kl", U, torch.eye(size[-1], dtype=torch.double, device=device)) / size[-1]  # (...,2,2)
        #theta = ((torch.einsum("...ik,...ki->...", UTrless, UTrless) / a).sqrt() / 4 - np.pi)
        #tr_gm_b = torch.einsum("...kii->...k", U)
        #b = torch.einsum("...ik,...kj->...ij",gm, U)
        #bT = b - torch.einsum("...,...ij->...ij", tr_gm_b, gm) * a
        #theta = ((torch.einsum("...ik,...ki->...", bT, bT) / a).sqrt() / 4 - np.pi)
        logm_invgm_gi = logm_invB_A(gm, G_i)
        tr_gm_b = torch.einsum("...kii->...k", logm_invgm_gi)
        b = torch.einsum("...ik,...kj->...ij",gm, logm_invgm_gi)
        bT = b - torch.einsum("...,...ij->...ij", tr_gm_b, gm) * a
        gminv = torch.pinverse(gm)
        gmbT = torch.einsum("...ik,...kj->...ij", gminv, bT)
        theta = torch.einsum("...ik,...ki->...", gmbT, gmbT)
        #print('tr(b):',tr_gm_b)
        #print('theta(b):',theta)

        #thresh = 0
        thresh = a * (4*np.pi)**2
        
        Ind_inRange = (theta < thresh).nonzero().reshape(-1)  ## G[i] is in the range of the exponential map at gm
        Ind_notInRange = (theta >= thresh).nonzero().reshape(-1)  ## G[i] is not in the range

        # when g1 = 0, len(Ind_notInRange) and len(Ind_inRange) are both zero. So check len(Ind_notInRange) first
        if len(Ind_notInRange) == 0:  # all in the range
            #gm = Rie_Exp_extended(gm, inv_RieExp_extended(gm, G_i, a) / (i + 1), a)
            gm = Rie_Exp_extended(gm, inv_RieExp_extended(gm, G_i, a), a, 1.0 / (i + 1))
        elif len(Ind_inRange) == 0:  # all not in range
            gm = ptPick_notInRange(gm, G_i, logm_invgm_gi, i)
        else:
            #gm[Ind_inRange] = Rie_Exp_extended(gm[Ind_inRange],
            #                                   inv_RieExp_extended(gm[Ind_inRange], G_i[Ind_inRange], a) / (i + 1),
            #                                   a)  # stop here
            gm[Ind_inRange] = Rie_Exp_extended(gm[Ind_inRange],
                                               inv_RieExp_extended(gm[Ind_inRange], G_i[Ind_inRange], a),
                                               a, 1.0 / (i + 1))  # stop here
            gm[Ind_notInRange] = ptPick_notInRange(gm[Ind_notInRange], G_i[Ind_notInRange], logm_invgm_gi[Ind_notInRange], i)
        gm[:] = torch.where(gm[:] == torch.zeros((size[-2],size[-2]), device=device),
                       torch.eye(size[-2], dtype=G.dtype, device=device), gm[:])

        #del G_i
        #torch.cuda.empty_cache()
        #gm_cpu = gm.cpu()
        #del gm
      return(gm)      
    # end def get_karcher_mean_shuffle_masked

    if mask is None:
      gm = get_karcher_mean_shuffle_masked(G, a, device)
    else:
      #Ind_inRange = (mask > 0.1).nonzero().reshape(-1)
      Ind_inRange = (mask > 0).nonzero().reshape(-1)
      gm = torch.zeros_like(G[0])
      gm[Ind_inRange] = get_karcher_mean_shuffle_masked(G[:,Ind_inRange], a, device)
    torch.cuda.empty_cache()

    #return gm_cpu.reshape(*size[1:])    
    return gm.reshape(*size[1:])
    #return(torch.where(gm[:] == torch.zeros((size[-2],size[-2]), device='cuda:0'),
    #                   torch.eye(size[-2], dtype=G.dtype, device='cuda:0'), gm[:]).reshape(*size[1:]))


# def get_karcher_mean_shuffle_scipy(G, a, device='cuda:0'):
#     size = G.size()
#     G = G.reshape(size[0], -1, *size[-2:])  # (T,-1,3,3)

#     orig_list = list(range(G.shape[0]))
#     shuffle_list = orig_list.copy()
#     random.shuffle(shuffle_list)
#     idx_shuffle_map = dict(zip(orig_list, shuffle_list))

#     gm = G[idx_shuffle_map[0]].to(device)

#     for i in range(1, G.size(0)):
#         G_i = G[idx_shuffle_map[i]].to(device)
#         #U = scipy_logm_invB_A(gm, G_i)
#         #UTrless = U - torch.einsum("...ii,kl->...kl", U, torch.eye(size[-1], dtype=torch.double, device=device)) / size[-1]  # (...,2,2)
#         #theta = ((torch.einsum("...ik,...ki->...", UTrless, UTrless) / a).sqrt() / 4 - np.pi)
#         #tr_gm_b = torch.einsum("...kii->...k", U)
#         #b = torch.einsum("...ik,...kj->...ij",gm, U)
#         #bT = b - torch.einsum("...,...ij->...ij", tr_gm_b, gm) * a
#         #theta = ((torch.einsum("...ik,...ki->...", bT, bT) / a).sqrt() / 4 - np.pi)
#         logm_invgm_gi = scipy_logm_invB_A(gm, G_i)
#         tr_gm_b = torch.einsum("...kii->...k", logm_invgm_gi)
#         b = torch.einsum("...ik,...kj->...ij",gm, logm_invgm_gi)
#         bT = b - torch.einsum("...,...ij->...ij", tr_gm_b, gm) * a
#         gminv = torch.inverse(gm)
#         gmbT = torch.einsum("...ik,...kj->...ij", gminv, bT)
#         theta = torch.einsum("...ik,...ki->...", gmbT, gmbT)

#         #thresh = 0
#         thresh = a * (4*np.pi)**2
#         Ind_inRange = (theta < thresh).nonzero().reshape(-1)  ## G[i] is in the range of the exponential map at gm
#         Ind_notInRange = (theta >= thresh).nonzero().reshape(-1)  ## G[i] is not in the range

#         # when g1 = 0, len(Ind_notInRange) and len(Ind_inRange) are both zero. So check len(Ind_notInRange) first
#         if len(Ind_notInRange) == 0:  # all in the range
#             #gm = Rie_Exp_extended(gm, inv_RieExp_extended_scipy(gm, G_i, a) / (i + 1), a)
#             gm = Rie_Exp_extended(gm, inv_RieExp_extended_scipy(gm, G_i, a), a, 1.0 / (i + 1))
#         elif len(Ind_inRange) == 0:  # all not in range
#             gm = ptPick_notInRange(gm, G_i, logm_invgm_gi, i)
#         else:
#             #gm[Ind_inRange] = Rie_Exp_extended(gm[Ind_inRange],
#             #                                   inv_RieExp_extended_scipy(gm[Ind_inRange], G_i[Ind_inRange], a) / (i + 1),
#             #                                   a)  # stop here
#             gm[Ind_inRange] = Rie_Exp_extended(gm[Ind_inRange],
#                                                inv_RieExp_extended_scipy(gm[Ind_inRange], G_i[Ind_inRange], a),
#                                                a, 1.0 / (i + 1))  # stop here
#             gm[Ind_notInRange] = ptPick_notInRange(gm[Ind_notInRange], G_i[Ind_notInRange],  logm_invgm_gi[Ind_notInRange], i)
#         gm[:] = torch.where(gm[:] == torch.zeros((size[-2],size[-2]), device=device),
#                        torch.eye(size[-2], dtype=G.dtype, device=device), gm[:])

#         #del G_i
#         #torch.cuda.empty_cache()
#     #gm_cpu = gm.cpu()
#     #del gm
#     torch.cuda.empty_cache()

#     #return gm_cpu.reshape(*size[1:])    
#     return gm.reshape(*size[1:])
#     #return(torch.where(gm[:] == torch.zeros((size[-2],size[-2]), device='cuda:0'),
#     #                   torch.eye(size[-2], dtype=G.dtype, device='cuda:0'), gm[:]).reshape(*size[1:]))

def get_karcher_mean_shuffle_scipy(G, a, mask=None, device='cuda:0'):
    size = G.size()
    G = G.reshape(size[0], -1, *size[-2:])  # (T,-1,3,3)
    if mask is not None:
      mask = mask.reshape(-1)  # (-1,1)

    def get_karcher_mean_shuffle_scipy_masked(G, a, device):  
      orig_list = list(range(G.shape[0]))
      shuffle_list = orig_list.copy()
      random.shuffle(shuffle_list)
      idx_shuffle_map = dict(zip(orig_list, shuffle_list))

      gm = G[idx_shuffle_map[0]].to(device)

      for i in range(1, G.size(0)):
        G_i = G[idx_shuffle_map[i]].to(device)
        #U = scipy_logm_invB_A(gm, G_i)
        #UTrless = U - torch.einsum("...ii,kl->...kl", U, torch.eye(size[-1], dtype=torch.double, device=device)) / size[-1]  # (...,2,2)
        #theta = ((torch.einsum("...ik,...ki->...", UTrless, UTrless) / a).sqrt() / 4 - np.pi)
        #tr_gm_b = torch.einsum("...kii->...k", U)
        #b = torch.einsum("...ik,...kj->...ij",gm, U)
        #bT = b - torch.einsum("...,...ij->...ij", tr_gm_b, gm) * a
        #theta = ((torch.einsum("...ik,...ki->...", bT, bT) / a).sqrt() / 4 - np.pi)
        logm_invgm_gi = scipy_logm_invB_A(gm, G_i)
        tr_gm_b = torch.einsum("...kii->...k", logm_invgm_gi)
        b = torch.einsum("...ik,...kj->...ij",gm, logm_invgm_gi)
        bT = b - torch.einsum("...,...ij->...ij", tr_gm_b, gm) * a
        gminv = torch.inverse(gm)
        gmbT = torch.einsum("...ik,...kj->...ij", gminv, bT)
        theta = torch.einsum("...ik,...ki->...", gmbT, gmbT)

        #thresh = 0
        thresh = a * (4*np.pi)**2
        Ind_inRange = (theta < thresh).nonzero().reshape(-1)  ## G[i] is in the range of the exponential map at gm
        Ind_notInRange = (theta >= thresh).nonzero().reshape(-1)  ## G[i] is not in the range

        # when g1 = 0, len(Ind_notInRange) and len(Ind_inRange) are both zero. So check len(Ind_notInRange) first
        if len(Ind_notInRange) == 0:  # all in the range
            #gm = Rie_Exp_extended(gm, inv_RieExp_extended_scipy(gm, G_i, a) / (i + 1), a)
            gm = Rie_Exp_extended(gm, inv_RieExp_extended_scipy(gm, G_i, a), a, 1.0 / (i + 1))
        elif len(Ind_inRange) == 0:  # all not in range
            gm = ptPick_notInRange(gm, G_i, logm_invgm_gi, i)
        else:
            #gm[Ind_inRange] = Rie_Exp_extended(gm[Ind_inRange],
            #                                   inv_RieExp_extended_scipy(gm[Ind_inRange], G_i[Ind_inRange], a) / (i + 1),
            #                                   a)  # stop here
            gm[Ind_inRange] = Rie_Exp_extended(gm[Ind_inRange],
                                               inv_RieExp_extended_scipy(gm[Ind_inRange], G_i[Ind_inRange], a),
                                               a, 1.0 / (i + 1))  # stop here
            gm[Ind_notInRange] = ptPick_notInRange(gm[Ind_notInRange], G_i[Ind_notInRange],  logm_invgm_gi[Ind_notInRange], i)
        gm[:] = torch.where(gm[:] == torch.zeros((size[-2],size[-2]), device=device),
                       torch.eye(size[-2], dtype=G.dtype, device=device), gm[:])

        #del G_i
        #torch.cuda.empty_cache()
        #gm_cpu = gm.cpu()
        #del gm
        return(gm)      

    # end def get_karcher_mean_shuffle_scipy_masked

    if mask is None:
      gm = get_karcher_mean_shuffle_scipy_masked(G, a, device)
    else:
      Ind_inRange = (mask > 0.1).nonzero().reshape(-1)
      gm = torch.zeros_like(G[0])
      gm[Ind_inRange] = get_karcher_mean_shuffle_scipy_masked(G[:,Ind_inRange], a, device)

    torch.cuda.empty_cache()

    #return gm_cpu.reshape(*size[1:])    
    return gm.reshape(*size[1:])
    #return(torch.where(gm[:] == torch.zeros((size[-2],size[-2]), device='cuda:0'),
    #                   torch.eye(size[-2], dtype=G.dtype, device='cuda:0'), gm[:]).reshape(*size[1:]))

    
def update_karcher_mean(karcher_mean, gi, i, a, device='cuda:0'):
    size = gi.size()
    print('update_karcher_mean, size:', size)
    gi = gi.reshape(-1, *size[-2:])  # (-1,3,3)
    if i == 0:
        print('i == 0')
        gm = gi
    else:
        gm = karcher_mean.reshape(-1, *size[-2:])
#         print('logm_invB_A')
        U = logm_invB_A(gm, gi)
        UTrless = U - torch.einsum("...ii,kl->...kl", U, torch.eye(size[-1], dtype=torch.double, device=device)) / size[
            -1]  # (...,2,2)
        theta = ((torch.einsum("...ik,...ki->...", UTrless, UTrless) / a).sqrt() / 4 - np.pi)
        Ind_inRange = (theta < 0).nonzero().reshape(-1)  ## G[i] is in the range of the exponential map at gm
        Ind_notInRange = (theta >= 0).nonzero().reshape(-1)  ## G[i] is not in the range

        # when g1 = 0, len(Ind_notInRange) and len(Ind_inRange) are both zero. So check len(Ind_notInRange) first
        if len(Ind_notInRange) == 0:  # all in the range
#             print('Rie_Exp_extended')
            gm = Rie_Exp_extended(gm, inv_RieExp_extended(gm, G_i, a) / (i + 1), a)
        elif len(Ind_inRange) == 0:  # all not in range
#             print('ptPick_notInRange')
            gm = ptPick_notInRange(gm, G_i, U, i)
        else:
#             print('Rie_Exp_extended, ptPick_notInRange')
            gm[Ind_inRange] = Rie_Exp_extended(gm[Ind_inRange],
                                               inv_RieExp_extended(gm[Ind_inRange], G_i[Ind_inRange], a) / (i + 1),
                                               a)  # stop here
            gm[Ind_notInRange] = ptPick_notInRange(gm[Ind_notInRange], G_i[Ind_notInRange], U[Ind_notInRange], i)
        del G_i
        torch.cuda.empty_cache()
    del gm
    torch.cuda.empty_cache()

    return gm.reshape(*size[1:])

