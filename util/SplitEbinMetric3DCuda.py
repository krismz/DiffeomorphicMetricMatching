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
    theta = torch.min((trK0square.clamp(min=1.0e-15) / a).sqrt() / 4., torch.tensor(np.pi, dtype=torch.double))  # change 1e-40 to 1e-13, because there is only one negative error of 1e-15 in UKF brain experiment
    #theta = torch.min((trK0square.clamp(min=1.0e-15)).sqrt() / 4., torch.tensor(np.pi, dtype=torch.double))  # change 1e-40 to 1e-13, because there is only one negative error of 1e-15 in UKF brain experiment
    detg0 = torch.det(g0)
    detg1 = torch.det(g1)
    #print('len det(g0) < 0:',len(torch.where(detg0<=0)[0]))
    #print('len det(g1) < 0:',len(torch.where(detg1<=0)[0]))

    alpha, beta = detg0.clamp(min=1.0e-15).pow(1. / 4.), detg1.clamp(min=1.0e-15).pow(1. / 4.)
    E = 16 * a * (alpha ** 2 - 2 * alpha * beta * torch.cos(theta) + beta ** 2)
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
#    import SimpleITK as sitk
#     inputs: A/B.shape = [hxwxd, 3, 3]
#     output: shape = [hxwxd, 3, 3]
    # To convert back
    # zDirection = i % d
    # yDirection = (i / d) % w
    # xDirection = i / (w * d)
    # debug batch i = 19110
    #i=19108
    #i=686208
    #i=219
    #i=15449
    #if B.shape[0] > i:
    #  print('B[',i,'] =',B[i])
    #try:
    #  print(torch.linalg.cholesky(B[i]))
    #except:
    #  import time
    #  time.sleep(2)
    #G = torch.linalg.cholesky(B)
    G = batch_cholesky(B)
    nonpsd_idx = torch.where(torch.isnan(G))
    if len(nonpsd_idx[0]) > 0:
      print(len(nonpsd_idx[0]), 'non psd entries found in logm_invB_A', nonpsd_idx)
    for i in range(len(nonpsd_idx[0])):
      G[nonpsd_idx[0][i]] = torch.eye((3)).double().to(device=B.device)

    # KMC The following clamp reduces crashes, but adds many bad artifacts  
    #inv_G = torch.inverse(G.clamp(min=1.0e-10))
    #inv_G = torch.inverse(G)
    det_G = torch.det(G)
    inv_G = torch.zeros_like(G)
    inv_G[det_G.abs()>0.] = torch.inverse(G[det_G.abs()>0.])
    inv_G[det_G.abs()<=0.] = torch.eye((3)).double().to(device=B.device)

    W = torch.einsum("...ij,...jk,...lk->...il", inv_G, A, inv_G)
    #W_sym = (W + torch.transpose(W,len(W.shape)-2,len(W.shape)-1))/2
    # The eigenvector computation becomes inaccurate for matrices close to identity
    # Set those closer than float machine eps from identity matrix to identity
    #W[(torch.abs(W[:]-torch.eye((3)))<1.1921e-7).sum(dim=(1,2))<9] = torch.eye((3))
    lamda, Q = se.apply(W)
    #lamda, Q = se.apply(W_sym)
    #log_lamda = torch.zeros((*lamda.shape, lamda.shape[-1]),dtype=torch.double)
    #log_lamda = torch.diag_embed(torch.log(lamda))
    #log_lamda = torch.diag_embed(torch.log(torch.where(lamda>1.0e-20,lamda,1.0e-20)))
    log_lamda = torch.diag_embed(torch.log(lamda.clamp(min=1.0e-15)))
    V = torch.einsum('...ji,...jk->...ik', inv_G, Q)
    #inv_V = torch.inverse(V)
    det_V = torch.det(V)
    inv_V = torch.zeros_like(V)
    inv_V[det_V.abs()>0.] = torch.inverse(V[det_V.abs()>0.])
    inv_V[det_V.abs()<=0.] = torch.eye((3)).double().to(device=B.device)

    #print(' where inv_V > 1e22 ', torch.where(inv_V > 1e22))
    #print('logm_invB_A, B NaN?', B.isnan().any(), 'A NaN?', A.isnan().any(), 'inv_G NaN?', inv_G.isnan().any(),
    #      'lamda NaN?', lamda.isnan().any(), 'log_lamda NaN?', log_lamda.isnan().any(), 'Q NaN?', Q.isnan().any(), 'inv_V NaN?', inv_V.isnan().any())
    #print('logm_invB_A, B Inf?', B.isinf().any(), 'A Inf?', A.isinf().any(), 'inv_G Inf?', inv_G.isinf().any(),
    #      'lamda Inf?', lamda.isinf().any(), 'log_lamda Inf?', log_lamda.isinf().any(), 'Q Inf?', Q.isinf().any(), 'inv_V Inf?', inv_V.isinf().any())
    #print('logm_invB_A, max(G)', torch.max(G), 'max(inv_G)', torch.max(inv_G),
    #      'max(V)', torch.max(V), 'max(inv_V)', torch.max(inv_V), 'max(Q)', torch.max(Q),
    #      'max(log_lamda)', torch.max(log_lamda))
    #di=1398389
    #di=1365089
    #di=1147366 # inv_v > 1e22
    #dbgi = [296967, 488518, 489447, 958378,  14015, 958378] #max(u) > 1e15
    #dbgi = [26131]
    #for di in dbgi:
    #  if B.shape[0] > di:
    #    print('\nlogm_invB_A \nB[',di,'] =\n', B[di], '\nA[',di,'] =\n', A[di],
    #          '\nG[',di,'] =\n', G[di], '\ninv_G[',di,'] =\n', inv_G[di],
    #          '\nW[',di,'] =\n', W[di],'\nV[',di,'] =\n', V[di],
    #          '\ninv_V[',di,'] =\n', inv_V[di],'\nlog_lamda[',di,'] =\n', log_lamda[di])
    #return torch.einsum('...ij,...jk,...kl->...il', V, log_lamda, inv_V)
    result = torch.einsum('...ij,...jk,...kl->...il', V, log_lamda, inv_V)
    #ill_cond_idx = torch.where(inv_V > 1e16)
    ill_cond_idx = (inv_V > 1e20).nonzero().reshape(-1)
    #num_ill_cond = len(ill_cond_idx[0])
    num_ill_cond = len(ill_cond_idx)
    if num_ill_cond > 0:
      #dbg_ill = [ill_cond_idx[c][0] for c in range(len(ill_cond_idx))]
      dbg_ill = ill_cond_idx[0]
      print('Replacing', num_ill_cond, 'ill-conditioned results in logm_invB_A with cpu version. First index is', dbg_ill)
      # TODO this should be batchable
      #for ii in range(num_ill_cond):
      #  result[ill_cond_idx[0][ii]] = cpu_logm_invB_A(B[ill_cond_idx[0][ii]],A[ill_cond_idx[0][ii]])
      #result[ill_cond_idx] = cpu_logm_invB_A(B[ill_cond_idx],A[ill_cond_idx])
      result[ill_cond_idx] = torch.eye((3)).to(device=B.device)
    
    return result

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
    inv_g0_g1 = make_pos_def(torch.einsum("...ik,...kj->...ij", torch.inverse(g0), g1),None, 1.0e-10)  # (s,t,...,3,3)
    #print('inv_RieExp, max(inv_g0_g1)', torch.max(inv_g0_g1), 'max(inverse(g0))', torch.max(torch.inverse(g0)))
    
    #print('inv_RieExp after make_pos_def, g0 NaN?', g0.isnan().any(), 'g1 NaN?', g1.isnan().any(),'inv_g0_g1 NaN?',
    #      inv_g0_g1.isnan().any())
    #print('inv_RieExp after make_pos_def, g0 Inf?', g0.isinf().any(), 'g1 Inf?', g1.isinf().any(),'inv_g0_g1 Inf?',
    #      inv_g0_g1.isinf().any())
    
    def get_u_g0direction(g0, inv_g0_g1):  # (-1,3,3) first reshape g0,g1,inv_g..
        #         permute
        inv_g0_g1 = torch.einsum("...ij->ij...", inv_g0_g1)  # (3,3,-1)
        s = inv_g0_g1[0, 0].clamp(min=1.0e-15)  # (-1)
        u = 4 / n * (s ** (n / 4) - 1) * torch.einsum("...ij->ij...", g0)  # (-1)@(3,3,-1) -> (3,3,-1)

        #print('inv_RieExp.get_u_g0direction, g0 NaN?', g0.isnan().any(), 'inv_g0_g1 NaN?', inv_g0_g1.isnan().any(),
        #      's NaN?', s.isnan().any(), 'u NaN?', u.isnan().any())
        #print('inv_RieExp.get_u_g0direction, g0 Inf?', g0.isinf().any(), 'inv_g0_g1 Inf?', inv_g0_g1.isinf().any(),
        #      's Inf?', s.isinf().any(), 'u Inf?', u.isinf().any())
        #print('inv_RieExp get_u_g0direction max(u)', torch.max(u), 'max(s)', torch.max(s))
        return u.permute(2, 0, 1)  # (-1,3,3)

    def get_u_ng0direction(g0, g1, inv_g0_g1, a):  # (-1,3,3) first reshape g0,g1,inv_g..
        det_threshold=1e-11
        where_below = torch.where(torch.det(g0)<=det_threshold)
        num_below = len(where_below[0])
        if num_below > 0:
          print('inv_RieExp num det(g0) below thresh:', num_below)
        # TODO if this works, move to get_karcher_mean  
        #g0[torch.det(g0)<=det_threshold] = torch.eye((3))
        # It moved the problem in unexpected ways
        K = logm_invB_A(g0, g1)
        KTrless = K - torch.einsum("...ii,kl->...kl", K, torch.eye(n, dtype=torch.double, device=g0.device)) / n  # (-1,3,3)
        #         AA^T
        theta = ((1 / a * torch.einsum("...ik,...ki->...", KTrless, KTrless)).clamp(min=1.0e-15).sqrt() / 4).clamp(min=1.0e-15)  # (-1)
        gamma = torch.det(g1).pow(1 / 4) / (torch.det(g0).clamp(min=1.0e-15).pow(1 / 4))  # (-1)

        A = 4 / n * (gamma * torch.cos(theta) - 1)  # (-1)
        B = 1 / theta * gamma * torch.sin(theta)
        u = A * torch.einsum("...ij->ij...", g0) + B * torch.einsum("...ik,...kj->ij...", g0, KTrless)  # (-1)@(3,3,-1) -> (3,3,-1)
        #print('inv_RieExp.get_u_ng0direction, g0 NaN?', g0.isnan().any(), 'g1 NaN?', g1.isnan().any(),
        #      'inv_g0_g1 NaN?', inv_g0_g1.isnan().any(), 'K NaN?', K.isnan().any(),'theta NaN?', theta.isnan().any(),
        #      'gamma NaN?', gamma.isnan().any(), 'A NaN?', A.isnan().any(), 'B NaN?', B.isnan().any())
        #print('inv_RieExp.get_u_ng0direction, g0 Inf?', g0.isinf().any(), 'g1 Inf?', g1.isinf().any(),
        #      'inv_g0_g1 Inf?', inv_g0_g1.isinf().any(), 'K Inf?', K.isinf().any(),'theta Inf?', theta.isinf().any(),
        #      'gamma Inf?', gamma.isinf().any(), 'A Inf?', A.isinf().any(), 'B Inf?', B.isinf().any())
        #di=1398389
        #di=1365089
        #di=1147366 # inv_v > 1e22
        #di=26131
        #if g0.shape[0] > di:
        #  print('\nget_u_ng0direction \ng0[',di,'] =\n', g0[di], '\ng1[',di,'] =\n', g1[di],
        #        '\nK[',di,'] =\n', K[di], '\nKTrless[',di,'] =\n', KTrless[di],
        #        '\ntheta[',di,'] =\n', theta[di],'\ngamma[',di,'] =\n', gamma[di],
        #        '\nA[',di,'] =\n', A[di],'\nB[',di,'] =\n', B[di],
        #        '\nA term[',di,'] =\n', (A * torch.einsum("...ij->ij...", g0)).permute(2,0,1)[di],
        #        '\nB term[',di,'] =\n', (B * torch.einsum("...ik,...kj->ij...", g0, KTrless)).permute(2,0,1)[di])
        where_huge = torch.where(K > 6e14)
        if len(where_huge[0]) > 0:
          print('num K huge', len(where_huge[0]), 'first huge', where_huge[0][0], where_huge[1][0], where_huge[2][0])

        #print(' where u > 1e15 ', torch.where(u > 1e15))
        #print('inv_RieExp get_u_ng0direction max(u)', torch.max(u), 'max(K)', torch.max(K), 'max(KTrless)', torch.max(KTrless)
        #      , 'max(theta)', torch.max(theta), 'max(gamma)', torch.max(gamma), 'max(A)', torch.max(A), 'max(B)', torch.max(B))
        return u.permute(2, 0, 1)  # (-1,3,3)

    #inv_g0_g1_trless = inv_g0_g1 - torch.einsum("...ii,kl->...kl", inv_g0_g1, torch.eye(n, dtype=torch.double, device='cuda:0')) / n  #
    inv_g0_g1_trless = inv_g0_g1 - torch.einsum("...ii,kl->...kl", inv_g0_g1, torch.eye(n, dtype=torch.double, device=g0.device))  # (s,t,...,2,2)
    norm0 = torch.einsum("...ij,...ij->...", inv_g0_g1_trless, inv_g0_g1_trless).reshape(-1)  # (-1)

    # find the indices for which the entries are 0s and non0s
    Ind0 = (norm0 <= 1e-12).nonzero().reshape(-1)  # using squeeze results in [1,1]->[]
    Indn0 = (norm0 > 1e-12).nonzero().reshape(-1)

    u = torch.zeros(g0.reshape(-1, n, n).size(), dtype=torch.double, device=g0.device)  # (-1,3,3)
    if len(Indn0) == 0:
        u = get_u_g0direction(g0.reshape(-1, n, n), inv_g0_g1.reshape(-1, n, n))
        #di=1398389
        #di=1365089
        #di=1147366 # inv_v > 1e22
        #di=26131
        #if u.shape[0] > di:
        #    print('\nget_u_g0direction u[',di,'] =\n', u[di])
    elif len(Ind0) == 0:
        u = get_u_ng0direction(g0.reshape(-1, n, n), g1.reshape(-1, n, n), inv_g0_g1.reshape(-1, n, n), a)
        #di=1398389
        #di=1365089
        #di=1147366 # inv_v > 1e22
        #di=26131
        #if u.shape[0] > di:
        #    print('\nget_u_ng0direction u[',di,'] =\n', u[di])
    else:
        u[Ind0] = get_u_g0direction(g0.reshape(-1, n, n)[Ind0], inv_g0_g1.reshape(-1, n, n)[Ind0])
        u[Indn0] = get_u_ng0direction(g0.reshape(-1, n, n)[Indn0], g1.reshape(-1, n, n)[Indn0], inv_g0_g1.reshape(-1, n, n)[Indn0], a)
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


def Rie_Exp(g0, u, a):  # here g0 is of size (s,t,...,3,3) and u is of size (s,t,...,3,3), where g0\neq 0
    '''this function is to calculate the Riemannian exponential of u in the the maximal domain of the Riemannian exponential at g0
    '''
    n = g0.size(-1)
    #print('entering Rie_Exp, g0 Nan?', g0.isnan().any(), 'inverse(g0) Nan?', torch.inverse(g0).isnan().any(),
    #      'u Nan?', u.isnan().any())
    #print('entering Rie_Exp, g0 Inf?', g0.isinf().any(), 'inverse(g0) Inf?', torch.inverse(g0).isinf().any(),
    #      'u Inf?', u.isinf().any())
    #print('entering Rie_Exp, max(g0)', torch.max(g0), 'max(u)', torch.max(u))
    
    U = torch.einsum("...ik,...kj->...ij", torch.inverse(g0), u)  # (s,t,...,3,3)
    trU = torch.einsum("...ii->...", U)  # (s,t,...)
    UTrless = U - torch.einsum("...,ij->...ij", trU, torch.eye(n, n, dtype=torch.double, device=g0.device)) / n  # (s,t,...,3,3)
    #print('Rie_Exp, max(U)', torch.max(U), 'max(trU)', torch.max(trU), 'max(UTrless)', torch.max(UTrless))

    #     in g0 direction:K_0=0
    def get_g1_g0direction(g0, trU):  # first reshape g0 (-1,3,3) and trU (-1)
        g1 = (trU / 4 + 1).pow(4 / n) * torch.einsum("...ij->ij...", g0)  # (3,3,-1)
        #print('Rie_Exp, get_g1_g0direction, max(g1)', torch.max(g1))
        return g1.permute(2, 0, 1)  # (-1,3,3)

    #     not in g0 direction SplitEbinMetric.pdf Theorem 1 :K_0\not=0
    def get_g1_ng0direction(g0, trU, UTrless, a):  # first reshape g0,UTrless (-1,3,3) and trU (-1)
        if len((trU < -4).nonzero().reshape(-1)) != 0:
            warnings.warn('The tangent vector u is out of the maximal domain of the Riemannian exponential.', DeprecationWarning)

        q = trU / 4 + 1  # (-1)
        r = (1 / a * torch.einsum("...ik,...ki->...", UTrless, UTrless)).clamp(min=1.0e-15).sqrt() / 4  # (-1)
        ArctanUtrless = (torch.atan2(r, q) * torch.einsum("...ij->ij...", UTrless) / r.clamp(min=1.0e-15))  # use (2,2,-1) for computation

        # Get nans with matrix_exp sometimes
        # Use scaling/squaring trick as described here to avoid: https://github.com/scipy/scipy/issues/11839
        # SCALING TRICK TOO SLOW, just replace nans with 0
        ExpArctanUtrless = torch.nan_to_num(torch.matrix_exp(ArctanUtrless.permute(2, 0, 1)).permute(1, 2, 0))
        ExpArctanUtrless[torch.abs(ExpArctanUtrless) > 1e12] = 0
        #print('ArctanUtrless.shape',ArctanUtrless.shape)
        #norm0 = torch.einsum("...ij,...ij->...", ArctanUtrless.permute(2, 0, 1), ArctanUtrless.permute(2, 0, 1)).reshape(-1)  # (-1)
        ## find the indices for which the entries are 0s and non0s
        #large_idx = (norm0 > 1e12).nonzero().reshape(-1)  # using squeeze results in [1,1]->[]
  
        #if len(large_idx) > 0:
        #  print('found', len(large_idx), ' norms larger than 1e12, max norm', torch.max(norm0))
        #  #for lidx in large_idx:
        #  #  nn = torch.log2(norm0[lidx]).int().clamp(min=0)
        #  #  print('nn\n', nn)
        #  #  ExpArctanUtrless[lidx] = torch.matrix_power(torch.matrix_exp(ArctanUtrless.permute(2, 0, 1)[lidx]/(2**nn)),2**nn).permute(1, 2, 0)
        #  ExpArctanUtrless[large_idx] = torch.zeros((3,3)).permute(1, 2, 0)

        g1 = (q ** 2 + r ** 2).pow(2 / n) * torch.einsum("...ik,kj...->ij...", g0, ExpArctanUtrless)  # (2,2,-1)
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
    UMinusTrU = U - torch.einsum("...,ij->...ij", trU, torch.eye(n, n, dtype=torch.double, device=g0.device))  # (s,t,...,3,3)

    norm0 = torch.einsum("...ij,...ij->...", UMinusTrU, UMinusTrU).reshape(-1)  # (-1)

    # find the indices for which the entries are 0s and non0s
    #     k_0=0 or \not=0
    Ind0 = (norm0 <= 1e-12).nonzero().reshape(-1)
    Indn0 = (norm0 > 1e-12).nonzero().reshape(-1)

    g1 = torch.zeros(g0.reshape(-1, n, n).size(), dtype=torch.double, device=g0.device)  # (-1,2,2)
    if len(Indn0) == 0:
        g1 = get_g1_g0direction(g0.reshape(-1, n, n), trU.reshape(-1))
        #di=1398389
        #di=1365089
        #di=1147366 # inv_v > 1e22
        #di=26131
        #if g1.shape[0] > di:
        #    print('\nget_g1_g0 g1[',di,'] =\n', g1[di])
    elif len(Ind0) == 0:
        g1 = get_g1_ng0direction(g0.reshape(-1, n, n), trU.reshape(-1), UTrless.reshape(-1, n, n), a)
        #di=1398389
        #di=1365089
        #di=1147366 # inv_v > 1e22
        #di=26131
        #if g1.shape[0] > di:
        #    print('\nget_g1_ng0 g1[',di,'] =\n', g1[di])
    else:
        g1[Ind0] = get_g1_g0direction(g0.reshape(-1, n, n)[Ind0], trU.reshape(-1)[Ind0])
        g1[Indn0] = get_g1_ng0direction(g0.reshape(-1, n, n)[Indn0], trU.reshape(-1)[Indn0], UTrless.reshape(-1, n, n)[Indn0], a)
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
def Rie_Exp_extended(g0, u, a):  # here g0 is of size (s,t,...,3,3) and u is of size (s,t,...,3,3)
    size = g0.size()
    g0, u = g0.reshape(-1, *size[-2:]), u.reshape(-1, *size[-2:])  # (-1,3,3)
    detg0 = torch.det(g0)

    Ind_g0_is0 = (detg0 == 0).nonzero().reshape(-1)
    Ind_g0_isnot0 = (detg0 != 0).nonzero().reshape(-1)

    if len(Ind_g0_isnot0) == 0:  # g0x are 0s for all x
        g1 = u * g0.size(-1) / 4
    elif len(Ind_g0_is0) == 0:  # g0x are PD for all x
        g1 = Rie_Exp(g0, u, a)
    else:
        g1 = torch.zeros(g0.size(), dtype=torch.double, device=g0.device)
        g1[Ind_g0_is0] = u[Ind_g0_is0] * g0.size(-1) / 4
        g1[Ind_g0_isnot0] = Rie_Exp(g0[Ind_g0_isnot0], u[Ind_g0_isnot0], a)
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
            geo[i] = Rie_Exp_extended(g0, u * Time[i], a)
        return geo  # (Tpts,-1,2,2)

    def geo_not_in_range(g0, g1, a, Tpts):  # (-1,3,3)
        m0 = torch.zeros(g0.size(), dtype=torch.double, device=g0.device)
        u0 = inv_RieExp_extended(g0, m0, a)
        u1 = inv_RieExp_extended(g1, m0, a)

        geo = torch.zeros(Tpts, *g0.size(), dtype=torch.double, device=g0.device)  # (Tpts,-1,3,3)
        geo[0], geo[-1] = g0, g1

        for i in range(1, int((Tpts - 1) / 2)):
            geo[i] = Rie_Exp_extended(g0, u0 * Time[i], a)
        for j in range(-int((Tpts - 1) / 2), -1):
            geo[j] = Rie_Exp_extended(g1, u1 * (1 - Time[j]), a)
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


def ptPick_notInRange(g0, g1, i):  # (-1,3,3)
    alpha = torch.det(g1).clamp(min=1.0e-15).pow(1 / 4) / torch.det(g0).clamp(min=1.0e-15).pow(1 / 4)  # (-1)
    #print('ptPick_notInRange, g0 NaN?', g0.isnan().any(), 'g1 NaN?', g1.isnan().any(), 'alpha NaN?', alpha.isnan().any())
    #print('ptPick_notInRange, g0 Inf?', g0.isinf().any(), 'g1 Inf?', g1.isinf().any(), 'alpha Inf?', alpha.isinf().any())
    Ind_close_to_g0 = (alpha <= i).nonzero().reshape(-1)
    Ind_close_to_g1 = (alpha > i).nonzero().reshape(-1)

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

def get_karcher_mean(G, a, mask=None, scale_factor=1.0, filename='', device='cuda:0'):
    size = G.size()
    G = G.reshape(size[0], -1, *size[-2:])  # (T,-1,3,3)
    gm = G[0].to(device)
    #dbgi=1417992
    #dbgi=[1384951,1378791,1398389,1398390,1404550] # problem with logm_invB_A
    #dbgi=[1027390, 1365089, 1495564, 1499330, 1618888, 1635720, 1831681] # inv_V > 1e22
    #dbgi=[1147366]
    #dbgi = [26131]
    #for di in dbgi:
    #  if gm.shape[0] > di:
    #    print('\nG[0,',di,'] =\n', gm[di])

    for i in range(1, G.size(0)):
        #print('logm_invB_A, i', i, 'max gm', torch.max(gm))
        G_i = G[i].to(device)
        U = logm_invB_A(gm, G_i)
        #U = logm_invB_A(make_pos_def(gm, mask.reshape(-1), 1.0e-10, skip_small_eval=True), G[i])
        UTrless = U - torch.einsum("...ii,kl->...kl", U, torch.eye(size[-1], dtype=torch.double, device=device)) / size[-1]  # (...,2,2)

        theta = ((torch.einsum("...ik,...ki->...", UTrless, UTrless) / a).sqrt() / 4 - np.pi)
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
        Ind_inRange = (theta < 0).nonzero().reshape(-1)  ## G[i] is in the range of the exponential map at gm
        Ind_notInRange = (theta >= 0).nonzero().reshape(-1)  ## G[i] is not in the range

        # when g1 = 0, len(Ind_notInRange) and len(Ind_inRange) are both zero. So check len(Ind_notInRange) first
        if len(Ind_notInRange) == 0:  # all in the range
            print('Before Rie_Exp_extended, i', i, 'max gm', torch.max(gm))
            gm = Rie_Exp_extended(gm, inv_RieExp_extended(gm, G_i, a) / (i + 1), a)
            print('after Rie_Exp_extended, i', i, 'max gm', torch.max(gm))
        elif len(Ind_inRange) == 0:  # all not in range
            print('Before ptPick_notInRange, i', i, 'max gm', torch.max(gm))
            gm = ptPick_notInRange(gm, G_i, i)
            print('after ptPick_notInRange, i', i, 'max gm', torch.max(gm))
        else:
            print('Before Rie_Exp_extended, ptPick_notInRange, i', i, 'max gm', torch.max(gm))
            gm[Ind_inRange] = Rie_Exp_extended(gm[Ind_inRange],
                                               inv_RieExp_extended(gm[Ind_inRange], G_i[Ind_inRange], a) / (i + 1),
                                               a)  # stop here
            print('after Rie_Exp_extended, i', i, 'max gm', torch.max(gm))
            gm[Ind_notInRange] = ptPick_notInRange(gm[Ind_notInRange], G_i[Ind_notInRange], i)
#             print('end')
            print('after ptPick_notInRange, i', i, 'max gm', torch.max(gm))
        #print('get_karcher_mean num zeros', len(torch.where(gm[:] == torch.zeros((size[-2],size[-2])))[0]))
        gm[:] = torch.where(gm[:] == torch.zeros((size[-2],size[-2]), device=device),
                       scale_factor * torch.eye(size[-2], dtype=G.dtype, device=device), scale_factor * gm[:])

        del G_i
        torch.cuda.empty_cache()


    #return gm.reshape(*size[1:])
    print("WARNING! Don't know why need to scale atlas by scale_factor")
    #for di in dbgi:
    #  if gm.shape[0] > di:
    #    print('\ngm[',di,'] =\n', gm[di])

    gm_cpu = gm.cpu()
    del gm
    torch.cuda.empty_cache()

    return gm_cpu.reshape(*size[1:])
    #return(torch.where(gm[:] == torch.zeros((size[-2],size[-2])),
    #                   scale_factor * torch.eye(size[-2], dtype=G.dtype), scale_factor * gm[:]).reshape(*size[1:]))

def get_karcher_mean_shuffle(G, a, device='cuda:0'):
    size = G.size()
    G = G.reshape(size[0], -1, *size[-2:])  # (T,-1,3,3)

    orig_list = list(range(G.shape[0]))
    shuffle_list = orig_list.copy()
    random.shuffle(shuffle_list)
    idx_shuffle_map = dict(zip(orig_list, shuffle_list))

    gm = G[idx_shuffle_map[0]].to(device)

    for i in range(1, G.size(0)):
        G_i = G[idx_shuffle_map[i]].to(device)
        U = logm_invB_A(gm, G_i)
        UTrless = U - torch.einsum("...ii,kl->...kl", U, torch.eye(size[-1], dtype=torch.double, device=device)) / size[-1]  # (...,2,2)
        theta = ((torch.einsum("...ik,...ki->...", UTrless, UTrless) / a).sqrt() / 4 - np.pi)
        Ind_inRange = (theta < 0).nonzero().reshape(-1)  ## G[i] is in the range of the exponential map at gm
        Ind_notInRange = (theta >= 0).nonzero().reshape(-1)  ## G[i] is not in the range

        # when g1 = 0, len(Ind_notInRange) and len(Ind_inRange) are both zero. So check len(Ind_notInRange) first
        if len(Ind_notInRange) == 0:  # all in the range
            gm = Rie_Exp_extended(gm, inv_RieExp_extended(gm, G_i, a) / (i + 1), a)
        elif len(Ind_inRange) == 0:  # all not in range
            gm = ptPick_notInRange(gm, G_i, i)
        else:
            gm[Ind_inRange] = Rie_Exp_extended(gm[Ind_inRange],
                                               inv_RieExp_extended(gm[Ind_inRange], G_i[Ind_inRange], a) / (i + 1),
                                               a)  # stop here
            gm[Ind_notInRange] = ptPick_notInRange(gm[Ind_notInRange], G_i[Ind_notInRange], i)
        gm[:] = torch.where(gm[:] == torch.zeros((size[-2],size[-2]), device=device),
                       torch.eye(size[-2], dtype=G.dtype, device=device), gm[:])

        del G_i
        torch.cuda.empty_cache()
    gm_cpu = gm.cpu()
    del gm
    torch.cuda.empty_cache()

    return gm_cpu.reshape(*size[1:])    
    #return gm.reshape(*size[1:])
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
            gm = ptPick_notInRange(gm, G_i, i)
        else:
#             print('Rie_Exp_extended, ptPick_notInRange')
            gm[Ind_inRange] = Rie_Exp_extended(gm[Ind_inRange],
                                               inv_RieExp_extended(gm[Ind_inRange], G_i[Ind_inRange], a) / (i + 1),
                                               a)  # stop here
            gm[Ind_notInRange] = ptPick_notInRange(gm[Ind_notInRange], G_i[Ind_notInRange], i)
        del G_i
        torch.cuda.empty_cache()
    del gm
    torch.cuda.empty_cache()

    return gm.reshape(*size[1:])

