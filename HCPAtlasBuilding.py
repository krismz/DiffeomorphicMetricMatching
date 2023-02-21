import os 
# Turn on CPP Stacktraces for more debug detail
os.environ['TORCH_SHOW_CPP_STACKTRACES'] = "1"
import torch
from tqdm import tqdm
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import SimpleITK as sitk
import random
import gc
#from lazy_imports import itkwidgets
#from lazy_imports import itkview
#from lazy_imports import interactive
#from lazy_imports import ipywidgets
#from lazy_imports import pv

#from mtch.RegistrationFunc3DCuda import *
#from mtch.SplitEbinMetric3DCuda import *
#from mtch.GeoPlot import *
from util.RegistrationFunc3DCuda import *
from util.SplitEbinMetric3DCuda import *

# from Packages.disp.vis import show_2d, show_2d_tensors
from disp.vis import vis_tensors, vis_path, disp_scalar_to_file
from disp.vis import disp_vector_to_file, disp_tensor_to_file
from disp.vis import disp_gradG_to_file, disp_gradA_to_file
from disp.vis import view_3d_tensors, tensors_to_mesh
from data.io import readRaw, ReadScalars, ReadTensors, WriteTensorNPArray, WriteScalarNPArray, readPath3D
from data.convert import GetNPArrayFromSITK, GetSITKImageFromNP

#import algo.metricModSolver2d as mms
#import algo.geodesic as geo
#import algo.euler as euler
#import algo.dijkstra as dijkstra
from torch_sym3eig import Sym3Eig as se

#cuda_dev = 'cuda:0'
cuda_dev = 'cuda:1'

def phi_pullback(phi, g, mask=None):
#     input: phi.shape = [3, h, w, d]; g.shape = [h, w, d, 3, 3]
#     output: shape = [h, w, 2, 2]
#     torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    g = g.permute(3, 4, 0, 1, 2)
    idty = get_idty(*g.shape[-3:], device=cuda_dev)
    #     four layers of scalar field, of all 1, all 0, all 1, all 0, where the shape of each layer is g.shape[-2:]?
    eye = torch.eye(3, device=cuda_dev)
    ones = torch.ones(*g.shape[-3:], device=cuda_dev)
    d_phi = get_jacobian_matrix(phi - idty) + torch.einsum("ij,mno->ijmno", eye, ones)
    g_phi = compose_function(g, phi)
    return torch.einsum("ij...,ik...,kl...->...jl", d_phi, g_phi, d_phi)
    # Problems with .backward autograd for make_pos_def
    #return(make_pos_def(torch.einsum("ij...,ik...,kl...->...jl", d_phi, g_phi, d_phi), mask))
    
def energy_phi(phi):
    idty = get_idty(*phi.shape[1:], device=cuda_dev)
    #     four layers of scalar field, of all 1, all 0, all 1, all 0, where the shape of each layer is g.shape[-2:]?
    eye = torch.eye(3, device=cuda_dev)
    ones = torch.ones(*phi.shape[1:], device=cuda_dev)
    d_phi = get_jacobian_matrix(phi - idty) + torch.einsum("ij,mno->ijmno", eye, ones)
    dist = torch.sqrt(torch.det(d_phi.reshape((3,3,-1)).permute(2,0,1))).reshape(*ones.shape)-ones
    return(dist*dist)

def energy_ebin(phi, phi0, i0, i1, sigma, dim, mask): 
#     input: phi.shape = [3, h, w, d]; g0/g1/f0/f1.shape = [h, w, d, 3, 3]; sigma/dim = scalar; mask.shape = [1, h, w, d]
#     output: scalar
# the phi here is identity
    #phi_star_f1 = phi_pullback(phi, f1)# the compose operation in this step uses a couple of thousands MB of memory
    phi_star_i1 = compose_function(i1.unsqueeze(0), phi).squeeze()# the compose operation in this step uses a couple of thousands MB of memory

    phi_phi0 = compose_function(phi0, phi)# the compose operation in this step uses a couple of thousands MB of memory

    #E1 = Squared_distance_Ebin(f0, phi_star_f1, 1./dim, mask)
    E3 = torch.sum((i0 - phi_star_i1) ** 2)
    #E4 = torch.sum(energy_phi(phi0))
    E4 = torch.sum(energy_phi(phi_phi0))
    #print("Energy:",sigma*E4, E3*0.6e4)
    #return sigma*E4 + E3*0.6e4
    print("Energy:",sigma*E4, E3)
    return sigma*E4 + E3

    # Following was hdai version
    #return E1 + E2*2.5e2 + E3*0#1.5e-9


def energy_L2(phi, g0, g1, f0, f1, sigma, mask): 
#     input: phi.shape = [3, h, w, d]; g0/g1/f0/f1.shape = [h, w, d, 3, 3]; sigma = scalar; mask.shape = [1, h, w, d]
#     output: scalar
    phi_star_g1 = phi_pullback(phi, g1)
    phi_star_f1 = phi_pullback(phi, f1)
    E1 = sigma * torch.einsum("ijk...,lijk->", (f0 - phi_star_f1) ** 2, mask.unsqueeze(0))
    E2 = torch.einsum("ijk...,lijk->", (g0 - phi_star_g1) ** 2, mask.unsqueeze(0))
    # E = E1 + E2
#     del phi_star_g1, phi_star_f1
#     torch.cuda.empty_cache()
    return E1 + E2


def laplace_inverse(u):
#     input: u.shape = [3, h, w, d]
#     output: shape = [3, h, w, d]
    '''
    this function computes the laplacian inverse of a vector field u of size 3 x size_h x size_w x size_d
    '''
    size_h, size_w, size_d = u.shape[-3:]
    idty = get_idty(size_h, size_w, size_d, device='cpu').cpu().numpy()
    lap = 6. - 2. * (np.cos(2. * np.pi * idty[0] / size_h) +
                     np.cos(2. * np.pi * idty[1] / size_w) +
                     np.cos(2. * np.pi * idty[2] / size_d))
    lap[0, 0] = 1.
    lapinv = 1. / lap
    lap[0, 0] = 0.
    lapinv[0, 0] = 1.

    u = u.cpu().detach().numpy()
    fx = np.fft.fftn(u[0])
    fy = np.fft.fftn(u[1])
    fz = np.fft.fftn(u[2])
    fx *= lapinv
    fy *= lapinv
    fz *= lapinv
    vx = torch.from_numpy(np.real(np.fft.ifftn(fx)))
    vy = torch.from_numpy(np.real(np.fft.ifftn(fy)))
    vz = torch.from_numpy(np.real(np.fft.ifftn(fz)))

    return torch.stack((vx, vy, vz)).to(device=torch.device(cuda_dev))

def laplace_inverse_cuda(u):
#     input: u.shape = [3, h, w, d]
#     output: shape = [3, h, w, d]
    '''
    this function computes the laplacian inverse of a vector field u of size 3 x size_h x size_w x size_d
    '''
    size_h, size_w, size_d = u.shape[-3:]
    idty = get_idty(size_h, size_w, size_d, device=u.device)
    lap = 6. - 2. * (torch.cos(2. * np.pi * idty[0] / size_h) +
                     torch.cos(2. * np.pi * idty[1] / size_w) +
                     torch.cos(2. * np.pi * idty[2] / size_d))
    lap[0, 0] = 1.
    lapinv = 1. / lap
    lap[0, 0] = 0.
    lapinv[0, 0] = 1.

    #u = u.cpu().detach().numpy()
    fx = torch.fft.fftn(u[0])
    fy = torch.fft.fftn(u[1])
    fz = torch.fft.fftn(u[2])
    fx *= lapinv
    fy *= lapinv
    fz *= lapinv
    vx = torch.real(torch.fft.ifftn(fx))
    vy = torch.real(torch.fft.ifftn(fy))
    vz = torch.real(torch.fft.ifftn(fz))

    return torch.stack((vx, vy, vz)).to(device=torch.device(u.device))

        
def metric_matching(ii, im, height, width, depth, ith_mask, mask, iter_num, epsilon, sigma, dim):
    phi_inv = get_idty(height, width, depth,device=cuda_dev)
    #phi = get_idty(height, width, depth,device=cuda_dev)
    #h0 = torch.ones(height, width, depth).to(cuda_dev)
    #h1 = torch.ones(height, width, depth).to(cuda_dev)
    #f0 = torch.eye(int(dim)).repeat(height, width, depth, 1, 1).to(cuda_dev)
    #f1 = torch.eye(int(dim)).repeat(height, width, depth, 1, 1).to(cuda_dev)

    idty = get_idty(height, width, depth,device=cuda_dev)
    
    #phi_actsh0 = compose_function(h0.unsqueeze(0).to(cuda_dev), phi_inv).squeeze()
    #phi_actsf0 = make_pos_def(phi_pullback(phi_inv, f0.to(cuda_dev), ith_mask), ith_mask, 1.0e-10)
    #phi_actsf0 = phi_pullback(phi_inv, f0.to(cuda_dev), ith_mask)
    phi_actsi0 = compose_function(ii.unsqueeze(0).to(cuda_dev), phi_inv).squeeze()
    energy_prev = 2 * energy_ebin(idty, phi_inv, phi_actsi0, im.to(cuda_dev), sigma, dim, mask.to(cuda_dev)) # Double up so that first iteration does go down

    idty.requires_grad_()
    energy = []
      
    for j in range(iter_num):
        #print('\n\nmetric_matching, iter', j,', max phi acts:',torch.max(phi_actsg0), torch.max(phi_actsf0),'\n\n')
        # use atlas mask for energy calculation, since in atlas space (gm, im)
        #E = energy_ebin(idty, phi_actsh0, h1, phi_actsi0, im.to(cuda_dev), sigma, dim, mask.to(cuda_dev)) 
        E = energy_ebin(idty, phi_inv, phi_actsi0, im.to(cuda_dev), sigma, dim, mask.to(cuda_dev)) 
        print(E.item())
        energy.append(E.item())
        if torch.isnan(E):
            raise ValueError('NaN error')
        E.backward()
        with torch.no_grad():
            #print('metric_matching v NaN?', v.isnan().any())

            print('metric_matching, iter', j, 'energy is', E.item(), 'and epsilon is', epsilon)
            v = - laplace_inverse_cuda(idty.grad)
            #epsilon = min(1.0/E.item(), epsilon)
            #print('OVERRIDING epsilon to be min of epsilon, 1/E', epsilon) 
            psi =  idty + epsilon*v  
            psi[0][psi[0] > height - 1] = height - 1
            psi[1][psi[1] > width - 1] = width - 1
            psi[2][psi[2] > depth - 1] = depth - 1
            psi[psi < 0] = 0
            psi_inv =  idty - epsilon*v
            psi_inv[0][psi_inv[0] > height - 1] = height - 1
            psi_inv[1][psi_inv[1] > width - 1] = width - 1
            psi_inv[2][psi_inv[2] > depth - 1] = depth - 1
            psi_inv[psi_inv < 0] = 0
            # No mask needed when updating phi or phi_inv
            # because mask will be applied when composing phi,phi_inv with g or i
            if energy[-1] > energy_prev:
              print('Starting to diverge, stopping at iteration', j-1)
              break
            
            #phi = compose_function(psi, phi) 
            phi_inv = compose_function(phi_inv, psi_inv)
            del psi, psi_inv
            #phi_actsh0 = compose_function(h0.unsqueeze(0).to(cuda_dev), phi_inv).squeeze()
            #phi_actsf0 = make_pos_def(phi_pullback(phi_inv, f0.to(cuda_dev), ith_mask), ith_mask, 1.0e-10)
            #phi_actsf0 = phi_pullback(phi_inv, f0.to(cuda_dev), ith_mask)
            phi_actsi0 = compose_function(ii.unsqueeze(0).to(cuda_dev), phi_inv).squeeze()

            if energy_prev - energy[-1] < 1e-5:
              print('Converged at iteration',j, 'energy not changing more that 1e-5')
              break

            energy_prev = energy[-1]
            
            idty.grad.data.zero_()
            
            
    #gi = phi_pullback(phi_inv, gi, ith_mask)
    del phi_actsi0, v, idty
    ith_mask = compose_function(ith_mask, phi_inv)
    ii = compose_function(ii.unsqueeze(0), phi_inv)#, ith_mask, 0)
    #del phi_actsf0, phi_actsi0, psi, psi_inv, v, idty, f0, f1
    #del phi_actsi0, psi, psi_inv, v, idty# , h1 #, f0, f1, phi_actsf0
    torch.cuda.empty_cache()
    gc.collect()
    return ii.squeeze(), ith_mask, phi_inv, energy


def tensor_cleaning(g, mask, iso_tens, det_threshold=1e-11):
# 1e-8 matches CPU version
#def tensor_cleaning(g, det_threshold=1e-8):
    #g[torch.det(g)<=det_threshold] = torch.eye((3))
    #g[mask==0] = torch.eye((3))
    g[torch.det(g)<=det_threshold] = iso_tens
    g[mask==0] = iso_tens
    # Sylvester's criterion https://en.wikipedia.org/wiki/Sylvester%27s_criterion
    psd_map = torch.where(g[...,0,0]>0, 1, 0) + torch.where(torch.det(g[...,:2,:2])>0, 1, 0) + torch.where(torch.det(g)>0, 1, 0)
    nonpsd_idx = torch.where(psd_map!=3)
    # nonpsd_idx = torch.where(torch.isnan(torch.sum(batch_cholesky(g), (3,4))))
    for i in range(len(nonpsd_idx[0])):
        #g[nonpsd_idx[0][i], nonpsd_idx[1][i], nonpsd_idx[2][i]] = torch.eye((3))
        g[nonpsd_idx[0][i], nonpsd_idx[1][i], nonpsd_idx[2][i]] = iso_tens
    return g

    
def fractional_anisotropy(g):
    e, _ = torch.symeig(g)
    lambd1 = e[:,:,:,0]
    lambd2 = e[:,:,:,1]
    lambd3 = e[:,:,:,2]
    mean = torch.mean(e,dim=len(e.shape)-1)
    return torch.sqrt(3.*(torch.pow((lambd1-mean),2)+torch.pow((lambd2-mean),2)+torch.pow((lambd3-mean),2)))/\
    torch.sqrt(2.*(torch.pow(lambd1,2)+torch.pow(lambd2,2)+torch.pow(lambd3,2)))


def get_framework(arr):
      # return np or torch depending on type of array
    # also returns framework name as "numpy" or "torch"
    fw = None
    fw_name = ''
    if type(arr) == np.ndarray:
        fw = np
        fw_name = 'numpy'
    else:
        fw = torch
        fw_name = 'torch'
    return (fw, fw_name)


def batch_cholesky(tens):
    # from https://stackoverflow.com/questions/60230464/pytorch-torch-cholesky-ignoring-exception
    # will get NaNs instead of exception where cholesky is invalid
    fw, fw_name = get_framework(tens)
    L = fw.zeros_like(tens)

    for i in range(tens.shape[-1]):
      for j in range(i+1):
        s = 0.0
        for k in range(j):
          s = s + L[...,i,k].clone() * L[...,j,k].clone()

        L[...,i,j] = fw.sqrt((tens[...,i,i] - s).clamp(min=1.0e-15)) if (i == j) else \
                      (1.0 / L[...,j,j].clone().clamp(min=1.0e-15) * (tens[...,i,j] - s))
    return L


def test_make_pos_def(tens, mask, small_eval = 0.00005):
  # make any small or negative eigenvalues slightly positive and then reconstruct tensors

    det_threshold=1e-11
    tens[torch.det(tens)<=det_threshold] = torch.eye((3))
    fw, fw_name = get_framework(tens)
    if fw_name == 'numpy':
        sym_tens = (tens + tens.transpose(0,1,2,4,3))/2
        evals, evecs = np.linalg.eig(sym_tens)
    else:
        sym_tens = ((tens + torch.transpose(tens,len(tens.shape)-2,len(tens.shape)-1))/2).reshape((-1,3,3))
        evals, evecs = se.apply(sym_tens)
        evals = evals.reshape((*tens.shape[:-2],3))
        evecs = evecs.reshape((*tens.shape[:-2],3,3))

    idx = fw.where(evals < small_eval)
    small_map = fw.where(evals < small_eval,1,0)
    #idx = np.where(evals < 0)
    num_found = 0
    #print(len(idx[0]), 'tensors found with eigenvalues <', small_eval)
    for ee in range(len(idx[0])):
        if mask is None or mask[idx[0][ee], idx[1][ee], idx[2][ee]]:
            num_found += 1
            # If largest eigenvalue is negative, replace with identity
            eval_2 = (idx[3][ee]+1) % 3
            eval_3 = (idx[3][ee]+2) % 3
            if ((evals[idx[0][ee], idx[1][ee], idx[2][ee], eval_2] < 0) and 
             (evals[idx[0][ee], idx[1][ee], idx[2][ee], eval_3] < 0)):
                evecs[idx[0][ee], idx[1][ee], idx[2][ee]] = fw.eye(3, dtype=tens.dtype)
                evals[idx[0][ee], idx[1][ee], idx[2][ee], idx[3][ee]] = small_eval
            else:
                # otherwise just set this eigenvalue to small_eval
                evals[idx[0][ee], idx[1][ee], idx[2][ee], idx[3][ee]] = small_eval

    #print(num_found, 'tensors found with eigenvalues <', small_eval)
    mod_tens = fw.einsum('...ij,...jk,...k,...lk->...il',
                       evecs, fw.eye(3, dtype=tens.dtype), evals, evecs)
    print("WARNING!!!! Overriding small_eval fix in BrainAtlasBuilding3DUkfCudaImg.test_make_pos_def")
    mod_tens = tens.clone()
    chol = batch_cholesky(mod_tens)
    idx_nan = torch.where(torch.isnan(chol))
    nan_map = torch.where(torch.isnan(chol),1,0)
    #print('num nans found:', len(idx_nan[0]))
    iso_tens = small_eval * torch.eye((3))
    #iso_tens = torch.eye((3))
    for pt in range(len(idx_nan[0])):
        mod_tens[idx_nan[0][pt],idx_nan[1][pt],idx_nan[2][pt]] = iso_tens

    #return(mod_tens)
    mod_sym_tens = (mod_tens + torch.transpose(mod_tens,len(mod_tens.shape)-2,len(mod_tens.shape)-1))/2
    mod_sym_tens[torch.det(mod_sym_tens)<=det_threshold] = torch.eye((3))
    return(mod_sym_tens)

def make_pos_def(tens, mask, small_eval = 0.00005):
  # make any small or negative eigenvalues slightly positive and then reconstruct tensors
    #print('WARNING! Short-circuiting BrainAtlasBuilding3DUkfCudaImg.make_pos_def')
    #return(tens)

    det_threshold=1e-11
    tens[torch.det(tens)<=det_threshold] = torch.eye((3)).to(device=tens.device)

    fw, fw_name = get_framework(tens)
    # if fw_name == 'numpy':
    #     sym_tens = (tens + tens.transpose(0,1,2,4,3))/2
    #     evals, evecs = np.linalg.eig(sym_tens)
    # else:
    #     sym_tens = ((tens + torch.transpose(tens,len(tens.shape)-2,len(tens.shape)-1))/2).reshape((-1,3,3))
    #     # evals, evecs = torch.symeig(sym_tens,eigenvectors=True)
    #     #evals, evecs = se.apply(sym_tens.reshape((-1,3,3)))
    #     evals, evecs = se.apply(sym_tens)
    # evals = evals.reshape((*tens.shape[:-2],3))
    # evecs = evecs.reshape((*tens.shape[:-2],3,3))
    # #cmplx_evals, cmplx_evecs = fw.linalg.eig(sym_tens)
    # #evals = fw.real(cmplx_evals)
    # #evecs = fw.real(cmplx_evecs)
    # #np.abs(evals, out=evals)
    # idx = fw.where(evals < small_eval)
    # small_map = fw.where(evals < small_eval,1,0)
    # #idx = np.where(evals < 0)
    # num_found = 0
    # #print(len(idx[0]), 'tensors found with eigenvalues <', small_eval)
    # for ee in range(len(idx[0])):
    #     if mask is None or mask[idx[0][ee], idx[1][ee], idx[2][ee]]:
    #         num_found += 1
    #         # If largest eigenvalue is negative, replace with identity
    #         eval_2 = (idx[3][ee]+1) % 3
    #         eval_3 = (idx[3][ee]+2) % 3
    #         if ((evals[idx[0][ee], idx[1][ee], idx[2][ee], eval_2] < 0) and 
    #          (evals[idx[0][ee], idx[1][ee], idx[2][ee], eval_3] < 0)):
    #             evecs[idx[0][ee], idx[1][ee], idx[2][ee]] = fw.eye(3, dtype=tens.dtype).to(device=tens.device)
    #             evals[idx[0][ee], idx[1][ee], idx[2][ee], idx[3][ee]] = small_eval
    #         else:
    #             # otherwise just set this eigenvalue to small_eval
    #             evals[idx[0][ee], idx[1][ee], idx[2][ee], idx[3][ee]] = small_eval

    # print(num_found, 'tensors found with eigenvalues <', small_eval)
    # #print(num_found, 'tensors found with eigenvalues < 0')
    # mod_tens = fw.einsum('...ij,...jk,...k,...lk->...il',
    #                    evecs, fw.eye(3, dtype=tens.dtype).to(device=tens.device), evals, evecs)
    # #mod_tens = fw.einsum('...ij,...j,...jk->...ik',
    # #                     evecs, evals, evecs)

    # print("WARNING!!!! Overriding small_eval fix in BrainAtlasBuilding3DUkfCudaImg.make_pos_def")
    mod_tens = tens.clone()
    chol = batch_cholesky(mod_tens)
    idx_nan = torch.where(torch.isnan(chol))
    nan_map = torch.where(torch.isnan(chol),1,0)
    iso_tens = small_eval * torch.eye((3)).to(device=tens.device)
    for pt in range(len(idx_nan[0])):
        mod_tens[idx_nan[0][pt],idx_nan[1][pt],idx_nan[2][pt]] = iso_tens
    # if torch.norm(torch.transpose(mod_tens,3,4)-mod_tens)>0:
    #     print('asymmetric')
    #mod_tens[:,:,:,1,0]=mod_tens[:,:,:,0,1]
    #mod_tens[:,:,:,2,0]=mod_tens[:,:,:,0,2]
    #mod_tens[:,:,:,2,1]=mod_tens[:,:,:,1,2]
    mod_sym_tens = (mod_tens + torch.transpose(mod_tens,len(mod_tens.shape)-2,len(mod_tens.shape)-1))/2
    mod_sym_tens[torch.det(mod_sym_tens)<=det_threshold] = torch.eye((3)).to(device=tens.device)
    return(mod_sym_tens)


def get_euclidean_mean(img_list):
    mean = torch.zeros_like(img_list[0])
    for i in range(len(img_list)):
        mean += img_list[i]

    return mean/len(img_list)


if __name__ == "__main__":
    #print('setting torch print precision to 16')
    #torch.set_printoptions(precision=16)
    #print('WARNING turn off anomaly detection')
    #torch.autograd.set_detect_anomaly(True)
    if torch.cuda.is_available():
      device = torch.device(cuda_dev)
      torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    else:
      device = torch.device('cpu')
      torch.set_default_tensor_type('torch.DoubleTensor')
    # Keep subjects on CPU, and move each individually to/from GPU to update atlas
    #device = torch.device('cpu')
    

    input_dir = '/usr/sci/projects/HCP/Kris/2023Proposal/prepped_data/'
    #bval = 1000
    bval = 'all'
    output_dir = f'/usr/sci/scratch/kris/2023Proposal/HCPResults/PairwiseReg/'

    # TODO need more robust mechanism for working with BIDS data structure
    cases = [sbj for sbj in os.listdir(input_dir) if sbj[0] != '.']
    
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    # TODO read dimensions from input data

    sample_num = len(cases)
    mask_list, img_list, phi_inv_acc_list = [], [], []
    img_energy_list = []
    resume = False
   
    start_iter = 0
    #iter_num = 901#1801
    iter_num = 1001#1801
    inner_iter_num = 2

    #img_scale = 255.0

    # Orig
    #dim, sigma = 3., 1
    dim, sigma = 3., 1.0e-3 # For sqrt det (dphi) 1e-3 not restrictive enough on diffeo.  Set to 1e-2
    #dim, sigma = 3., 1.0e-10# For ebin dist to identity:
    if sigma == 0:
        print("WARNING!!! Regularizer on phi disabled, sigma set to 0")

    
    for s in range(len(cases)):
        subj = cases[s]
        in_prefix = os.path.join(input_dir, subj)
        #mask_np = ReadScalars(f'{in_prefix}/dti_{bval}_FA_mask_affreg.nhdr')
        img_np = np.transpose(ReadScalars(f'{in_prefix}/t1_to_reft1_affreg.nhdr'), (0,2,1))
        img_np /= np.max(img_np)
        mask_np = np.ones_like(img_np)

        mask_list.append(torch.from_numpy(mask_np).double().permute(2,1,0))
        img_list.append(torch.from_numpy(img_np).double().permute(2,1,0))

        if s == 0:
          height,width,depth = mask_list[0].shape
          #epsilon = 0.5 / (height*width*depth) # seems good convergence but slow
          #epsilon = 5 / (height*width*depth) #  better convergence, diverges after a while
          #epsilon = 50 / (height*width*depth) # better convergence, diverges after a while
          #epsilon = 100 / (height*width*depth) # 5000, 500 too much.  100 good for awhile then diverges
          epsilon = 5.0e-2 # 1e-2, decent but slow

        img_energy_list.append([])
        if resume==False:
            print('start from identity')
            phi_inv_acc_list.append(get_idty(height, width, depth,device=cuda_dev))
            #phi_acc_list.append(get_idty(height, width, depth,device=cuda_dev))
        else:
            print('start from checkpoint')
            phi_inv_acc_list.append(torch.from_numpy(sio.loadmat(f'{output_dir}/{subj}_{start_iter-1}_phi_inv.mat')['diffeo']).to(device))
            #phi_acc_list.append(torch.from_numpy(sio.loadmat(f'{output_dir}/{subj}_{start_iter-1}_phi.mat')['diffeo']).to(device))
            img_list[s] = compose_function(img_list[s], phi_inv_acc_list[s])


    for i in tqdm(range(start_iter, start_iter+iter_num)):
        G = torch.stack(tuple(img_list))    
        mask_union = (sum(mask_list)/len(mask_list)).to(device)

        atlas_img = get_euclidean_mean(img_list)

        if i%200==0:
            WriteScalarNPArray(np.transpose(atlas_img.cpu(),(2,0,1)), f'{output_dir}/atlas_{i}_t1.nhdr')
            WriteScalarNPArray(np.transpose(mask_union.cpu(),(2,0,1)), f'{output_dir}/atlas_{i}_mask.nhdr')
            
        
        print('\n\n main loop, iter', i, '\n\n')
        for s in range(len(cases)):
            print('subj',s)
            img_energy_list[s].append(torch.sum((img_list[s] - atlas_img)**2).item())
        
            img_list[s], mask_list[s], phi_inv, energy = metric_matching(img_list[s], atlas_img, height, width, depth, mask_list[s], mask_union, inner_iter_num, epsilon, sigma, dim)

            phi_inv_acc_list[s] = compose_function(phi_inv_acc_list[s], phi_inv)
            #phi_acc_list[s] = compose_function(phi, phi_acc_list[s])

            '''check point'''
            if i%200==0:
                subj = cases[s]
                sio.savemat(f'{output_dir}/{subj}_{i}_phi_inv.mat', {'diffeo': phi_inv.cpu().detach().numpy()})
                #sio.savemat(f'{output_dir}/{subj}_{i}_phi.mat', {'diffeo': phi.cpu().detach().numpy()})
                sio.savemat(f'{output_dir}/{subj}_{i}_img_energy.mat', {'energy': img_energy_list[s]})
                WriteScalarNPArray(np.transpose(img_list[s].cpu(),(2,0,1)), f'{output_dir}/{subj}_{i}_t1.nhdr')

            del phi_inv

        # end for each subject, register to atlas
            
    # end for each iteration i
    WriteScalarNPArray(np.transpose(atlas_img.cpu(),(2,0,1)), f'{output_dir}/atlas_t1.nhdr')
    WriteScalarNPArray(np.transpose(mask_union.cpu(),(2,0,1)), f'{output_dir}/atlas_mask.nhdr')
    for s in range(len(cases)):
        subj = cases[s]
        sio.savemat(f'{output_dir}/{subj}_atlas_phi_inv.mat', {'diffeo': phi_inv_acc_list[s].cpu().detach().numpy()})
        #sio.savemat(f'{output_dir}/{subj}_atlas_phi.mat', {'diffeo': phi_acc_list[s].cpu().detach().numpy()})
        sio.savemat(f'{output_dir}/{subj}_atlas_img_energy.mat', {'energy': img_energy_list[s]})
        WriteScalarNPArray(np.transpose(img_list[s].cpu(),(2,0,1)), f'{output_dir}/{subj}_atlas_t1.nhdr')

       

