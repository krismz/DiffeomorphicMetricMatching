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
    


def energy_ebin(phi, g0, g1, f0, f1, i0, i1, sigma, dim, mask): 
#     input: phi.shape = [3, h, w, d]; g0/g1/f0/f1.shape = [h, w, d, 3, 3]; sigma/dim = scalar; mask.shape = [1, h, w, d]
#     output: scalar
# the phi here is identity
    phi_star_g1 = phi_pullback(phi, g1)
    phi_star_f1 = phi_pullback(phi, f1)# the compose operation in this step uses a couple of thousands MB of memory
    phi_star_i1 = compose_function(i1.unsqueeze(0), phi, mask, 0).squeeze()# the compose operation in this step uses a couple of thousands MB of memory
    #print('energy_ebin phi NaN',phi.isnan().any(), 'g1 NaN?', g1.isnan().any(), 'phi_star_g1 NaN?', phi_star_g1.isnan().any(), 'phi_star_f1 NaN?', phi_star_f1.isnan().any())
    #print('energy_ebin phi Inf',phi.isinf().any(), 'g1 Inf?', g1.isinf().any(), 'phi_star_g1 Inf?', phi_star_g1.isinf().any(), 'phi_star_f1 Inf?', phi_star_f1.isinf().any())

    #print('\n\nenergy_ebin max phi star:',torch.max(phi_star_g1), torch.max(phi_star_f1),'\n\n')

    #E1 = sigma * Squared_distance_Ebin(f0, phi_star_f1, 1./dim, mask)
    E1 = Squared_distance_Ebin(f0, phi_star_f1, 1./dim, mask)
    E2 = Squared_distance_Ebin(g0, phi_star_g1, 1./dim, mask)
    # E3 = torch.einsum("ijk,ijk->", (i0 - phi_star_i1) ** 2, mask)
    # E3 = torch.einsum("ijk,ijk->", (i0 - phi_star_i1) ** 2, (1-mask)*brain_mask)
    E3 = torch.sum((i0 - phi_star_i1) ** 2)
    #print(E1, E2*2.5e2, E3*1.5e-9, 'DIFFERENT THAN HDAI VERSION')
    #return E1 + E2*2.5e2 + E3*1.5e-9
    # Use following when not scaling image by 255
    #print(E1, E2*2.5e2, E3*1.5e-1, 'DIFFERENT THAN HDAI VERSION')
    #return E1 + E2*2.5e2 + E3*1.5e-1
    # Use following when scaling image by 255
    #print(sigma*E1, E2*2.5e2, E3*1.5e4, 'DIFFERENT THAN HDAI VERSION')
    #return sigma*E1 + E2*2.5e2 + E3*1.5e4
    # Use following for 6 subj ABCD
    print(sigma*E1, E2*2.5e2, E3*0.6e4, 'DIFFERENT THAN HDAI VERSION')
    return sigma*E1 + E2*2.5e2 + E3*0.6e4
    
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

        
def metric_matching(gi, gm, ii, im, height, width, depth, ith_mask, mask, iter_num, epsilon, sigma, dim):
    phi_inv = get_idty(height, width, depth,device=cuda_dev)
    phi = get_idty(height, width, depth,device=cuda_dev)
    idty = get_idty(height, width, depth,device=cuda_dev)
    idty.requires_grad_()
    f0 = torch.eye(int(dim)).repeat(height, width, depth, 1, 1).to(cuda_dev)
    f1 = torch.eye(int(dim)).repeat(height, width, depth, 1, 1).to(cuda_dev)
    #with torch.no_grad():
      #phi_actsg0 = make_pos_def(phi_pullback(phi_inv, gi, ith_mask), ith_mask, 1.0e-10)
      #phi_actsf0 = make_pos_def(phi_pullback(phi_inv, f0, ith_mask), ith_mask, 1.0e-10) # TODO ith_mask or mask here?

    #print('metric_matching phi_actsg0 NaN?', phi_actsg0.isnan().any(), 'phi_actsf0 NaN?', phi_actsf0.isnan().any(),
    #      'phi_inv NaN?', phi_inv.isnan().any(), 'gm NaN?', gm.isnan().any(), 'f1 NaN?', f1.isnan().any(),
    #      'im NaN?', im.isnan().any(), 'idty NaN?', idty.isnan().any())
    #print('metric_matching phi_actsg0 Inf?', phi_actsg0.isinf().any(), 'phi_actsf0 Inf?', phi_actsf0.isinf().any(),
    #      'phi_inv Inf?', phi_inv.isinf().any(), 'gm Inf?', gm.isinf().any(), 'f1 Inf?', f1.isinf().any(),
    #      'im Inf?', im.isinf().any(), 'idty Inf?', idty.isinf().any())
      
    for j in range(iter_num):
        #phi_actsg0 = phi_pullback(phi_inv, gi, ith_mask)
        #phi_actsf0 = phi_pullback(phi_inv, f0, ith_mask) # TODO ith_mask or mask here?
        phi_actsg0 = make_pos_def(phi_pullback(phi_inv, gi.to(cuda_dev), ith_mask), ith_mask, 1.0e-10)
        phi_actsf0 = make_pos_def(phi_pullback(phi_inv, f0.to(cuda_dev), ith_mask), ith_mask, 1.0e-10) # TODO ith_mask or mask here?
        #print('\n\nmetric_matching, iter', j,', max phi acts:',torch.max(phi_actsg0), torch.max(phi_actsf0),'\n\n')
        phi_actsi0 = compose_function(ii.unsqueeze(0).to(cuda_dev), phi_inv, ith_mask, 0).squeeze()
        # use atlas mask for energy calculation, since in atlas space (gm, im)
        E = energy_ebin(idty, phi_actsg0, gm.to(cuda_dev), phi_actsf0, f1, phi_actsi0, im.to(cuda_dev), sigma, dim, mask.to(cuda_dev)) 
        print(E.item())
        if torch.isnan(E):
            raise ValueError('NaN error')
        E.backward()
        v = - laplace_inverse(idty.grad)
        with torch.no_grad():
            #print('metric_matching v NaN?', v.isnan().any())

            print('metric_matching, energy is', E.item(), 'and epsilon is', epsilon)
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
            phi = compose_function(psi, phi) 
            phi_inv = compose_function(phi_inv, psi_inv)
            phi_actsg0 = make_pos_def(phi_pullback(phi_inv, gi.to(cuda_dev), ith_mask), ith_mask, 1.0e-10)
            phi_actsf0 = make_pos_def(phi_pullback(phi_inv, f0.to(cuda_dev), ith_mask), ith_mask, 1.0e-10) # TODO ith_mask or mask here?
            idty.grad.data.zero_()
            
    #gi = phi_pullback(phi_inv, gi, ith_mask)
    ith_mask = compose_function(ith_mask, phi_inv)
    ii = compose_function(ii.unsqueeze(0), phi_inv)#, ith_mask, 0)
    gi = make_pos_def(phi_pullback(phi_inv, gi.to(cuda_dev), ith_mask), ith_mask, 1.0e-10)
    #return gi, ii.squeeze(), phi, phi_inv
    #return gi, ii.squeeze(), ith_mask, phi, phi_inv
    gi_cpu = gi.cpu()
    ii_cpu = ii.cpu()
    phi_cpu = phi.cpu()
    phi_inv_cpu = phi_inv.cpu()
    del gi, ii, phi_actsg0, phi_actsf0, phi_actsi0, psi, psi_inv, v, idty, f0, f1, phi, phi_inv
    torch.cuda.empty_cache()
    return gi_cpu, ii_cpu.squeeze(), ith_mask, phi_cpu, phi_inv_cpu


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
    if fw_name == 'numpy':
        sym_tens = (tens + tens.transpose(0,1,2,4,3))/2
        evals, evecs = np.linalg.eig(sym_tens)
    else:
        sym_tens = ((tens + torch.transpose(tens,len(tens.shape)-2,len(tens.shape)-1))/2).reshape((-1,3,3))
        # evals, evecs = torch.symeig(sym_tens,eigenvectors=True)
        #evals, evecs = se.apply(sym_tens.reshape((-1,3,3)))
        evals, evecs = se.apply(sym_tens)
    evals = evals.reshape((*tens.shape[:-2],3))
    evecs = evecs.reshape((*tens.shape[:-2],3,3))
    #cmplx_evals, cmplx_evecs = fw.linalg.eig(sym_tens)
    #evals = fw.real(cmplx_evals)
    #evecs = fw.real(cmplx_evecs)
    #np.abs(evals, out=evals)
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
                evecs[idx[0][ee], idx[1][ee], idx[2][ee]] = fw.eye(3, dtype=tens.dtype).to(device=tens.device)
                evals[idx[0][ee], idx[1][ee], idx[2][ee], idx[3][ee]] = small_eval
            else:
                # otherwise just set this eigenvalue to small_eval
                evals[idx[0][ee], idx[1][ee], idx[2][ee], idx[3][ee]] = small_eval

    print(num_found, 'tensors found with eigenvalues <', small_eval)
    #print(num_found, 'tensors found with eigenvalues < 0')
    mod_tens = fw.einsum('...ij,...jk,...k,...lk->...il',
                       evecs, fw.eye(3, dtype=tens.dtype).to(device=tens.device), evals, evecs)
    #mod_tens = fw.einsum('...ij,...j,...jk->...ik',
    #                     evecs, evals, evecs)

    print("WARNING!!!! Overriding small_eval fix in BrainAtlasBuilding3DUkfCudaImg.make_pos_def")
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
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # after switch device, you need restart the script
    #torch.cuda.set_device('cuda:0')
    #torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    #print('setting torch print precision to 16')
    #torch.set_printoptions(precision=16)
    #print('WARNING turn off anomaly detection')
    #torch.autograd.set_detect_anomaly(True)
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Keep subjects on CPU, and move each individually to/from GPU to update atlas
    device = torch.device('cpu')
    torch.set_default_tensor_type('torch.DoubleTensor')

    # file_name = []
    #file_name = [108222, 102715, 105923, 107422, 100206, 104416]
    input_dir = '/usr/sci/projects/abcd/anxiety_study/derivatives/metric_matching'
    #output_dir = '/usr/sci/projects/abcd/anxiety_study/derivatives/atlas_building_cuda_11controls_901iter'
    #output_dir = '/usr/sci/projects/abcd/anxiety_study/derivatives/atlas_building_cuda_34subjects_1001iter'
    output_dir = '/usr/sci/scratch/kris/abcd_results/atlas_building_cuda_34subjects_1001iter'
    
    # TODO need more robust mechanism for working with BIDS data structure
    cases = [sbj for sbj in os.listdir(input_dir) if sbj[:4] == 'sub-']
    #num=3
    #num=2
    offs=[0,4,5,6,7,8]
    offs=[0,4,5,6,7,8,9,10,11,12,13] # has an inversion problem, suspect it might be subject 13, but haven't confirmed yet
    offs=[1,3,5,6,8,9,10,11,12,14,17] # has an inversion problem
    offs=[0,4,5,6,7,8,9] # works
    offs=[0,4,5,6,7,8,9,10] # has an inversion problem
    offs=[0,4,5,6,7,8,9,11] # works
    offs=[0,4,5,6,7,8,9,11,12] # has an inversion problem
    offs=[0,4,5,6,7,8,9,11,13] # has an inversion problem
    offs=[0,4,5,6,7,8,9,11,14] # works
    offs=[0,4,5,6,7,8,9,11,14,15] # works sometimes, has an inversion problem depending on karcher mean order
    offs=[1,3,5,6,7,8,9,11,14,16,17] # 22 subject offsets
    offs=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17] 
    #offs=[0,1]
    #print(f"WARNING, using first {num} cases and {num} controls only!!")
    #cases = cases[0:num] + cases[18:18+num]
    #print(f"WARNING, using first {num} cases only!!")
    #cases = cases[0:num]
    #print(f"WARNING,building atlas from first subject repeated twice!!") # next do first 2 subjects
    #cases= cases[0:1] + cases[0:1]
    #print(f"WARNING, using", len(offs), 'cases and controls with offsets', offs)
    #print(f"WARNING, using", len(offs), 'cases vs CASES!! with offsets', offs)
    outc = []
    for offset in offs:
        outc.append(cases[offset])
        #outc.append(cases[offset])
        if (offset != 4) and (offset != 15): # the subjects at 22 and 33 are outliers to be excluded
          outc.append(cases[18+offset])
    cases = outc

    session = 'ses-baselineYear1Arm1'
    run = 'run-01'
    upsamp=''
    #upsamp='_upsamp'
    
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    # TODO read dimensions from input data
    if not upsamp:
      height, width, depth = 140,140,140
    else:
      height, width, depth = 256,256,256

    sample_num = len(cases)
    tensor_lin_list, tensor_met_list, mask_list, mask_thresh_list, fa_list, img_list, brain_mask_list = [], [], [], [], [], [], []
    mask_union = torch.zeros(height, width, depth).double().to(device)
    # brain_mask_union = torch.zeros(height, width, depth).double().to(device)
    phi_inv_acc_list, phi_acc_list, met_energy_list, img_energy_list = [], [], [], []
    resume = False
   
    start_iter = 0
    #iter_num = 901#1801
    iter_num = 1001#1801

    img_scale = 255.0
    tens_scale = 1000

    for s in range(len(cases)):
        subj = cases[s]
        print(f'{subj} is processing.')
        dwi_prefix = os.path.join(input_dir, subj, session,'dwi', f'{subj}_{session}')
        t1_prefix = os.path.join(input_dir, subj, session,'anat', f'{subj}_{session}')
        # tensor_np = sitk.GetArrayFromImage(sitk.ReadImage(f'{input_dir}/{file_name[s]}_scaled_unsmoothed_tensors.nhdr'))
        # mask_np = sitk.GetArrayFromImage(sitk.ReadImage(f'{input_dir}/{file_name[s]}_filt_mask.nhdr'))
        # brain_mask_np = sitk.GetArrayFromImage(sitk.ReadImage(f'{input_dir}/{file_name[s]}_brain_mask.nhdr'))
        # img_np = sitk.GetArrayFromImage(sitk.ReadImage(f'{input_dir}/{file_name[s]}_T1_flip_y.nhdr'))
        if not upsamp:
          # TODO determine if better to do unsmoothed or scaled_original tensors
          #tensor_np = sitk.GetArrayFromImage(sitk.ReadImage(f'{dwi_prefix}{upsamp}_scaled_unsmoothed_tensors.nhdr'))
          #tensor_np = sitk.GetArrayFromImage(sitk.ReadImage(f'{dwi_prefix}{upsamp}_scaled_orig_tensors_v2.nhdr'))
          #mask_np = sitk.GetArrayFromImage(sitk.ReadImage(f'{dwi_prefix}{upsamp}_filt_mask.nhdr'))
          # Use utilities for consistency
          tensor_np = ReadTensors(f'{dwi_prefix}{upsamp}_scaled_orig_tensors_v2.nhdr')
          mask_np = ReadScalars(f'{dwi_prefix}{upsamp}_filt_mask.nhdr')
        else:
          # Pad to match dimensions of T1 image
          # TODO determine if better to do unsmoothed or scaled_original tensors
          #tensor_np = np.pad(sitk.GetArrayFromImage(sitk.ReadImage(f'{dwi_prefix}{upsamp}_scaled_unsmoothed_tensors.nhdr')),[(9,9),(9,9),(9,9),(0,0)])
          # TODO would actually save space and time to chop 9 slices off of edges of T1 image instead of padding tensor image by 9 each side
          #tensor_np = np.pad(sitk.GetArrayFromImage(sitk.ReadImage(f'{dwi_prefix}{upsamp}_scaled_orig_tensors_v2.nhdr')),[(9,9),(9,9),(9,9),(0,0)])
          #mask_np = np.pad(sitk.GetArrayFromImage(sitk.ReadImage(f'{dwi_prefix}{upsamp}_filt_mask.nhdr')),[(9,9),(9,9),(9,9)])
          tensor_np = np.pad(ReadTensors(f'{dwi_prefix}{upsamp}_scaled_orig_tensors_v2.nhdr'),[(9,9),(9,9),(9,9),(0,0)])
          mask_np = np.pad(ReadScalars(f'{dwi_prefix}{upsamp}_filt_mask.nhdr'),[(9,9),(9,9),(9,9)])
        # brain_mask_np = sitk.GetArrayFromImage(sitk.ReadImage(f'{input_dir}/{file_name[s]}_brain_mask_rreg.nhdr'))
        #img_np = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(f'{t1_prefix}_T1_flip_y.nhdr')), (0,2,1))
        img_np = np.transpose(ReadScalars(f'{t1_prefix}_T1_flip_y.nhdr'), (0,2,1)) / img_scale
        if not upsamp:
          # Match resolution of tensor image
          t1_t_sitk = GetSITKImageFromNP(img_np)
          resample = sitk.ResampleImageFilter()
          resample.SetInterpolator(sitk.sitkLinear)
          resample.SetOutputDirection(t1_t_sitk.GetDirection())
          resample.SetOutputOrigin(t1_t_sitk.GetOrigin())
          new_spacing = [1.7, 1.7, 1.7]
          resample.SetOutputSpacing(new_spacing)

          orig_size = np.array(t1_t_sitk.GetSize(), dtype=np.int)
          orig_spacing = list(t1_t_sitk.GetSpacing())
          new_size = [orig_size[s]*(orig_spacing[s]/new_spacing[s]) for s in range(len(orig_size))]
          new_size = np.ceil(new_size).astype(np.int) #  Image dimensions are in integers
          new_size = [int(s) for s in new_size]
          resample.SetSize(new_size)
          # Remove extra slices of air to match dimensions and alignment with tensor image
          img_np = GetNPArrayFromSITK(resample.Execute(t1_t_sitk)[6:-5,6:-5,6:-5])
        # END if not upsamp

        print('subj', s, 'tensor.shape =', tensor_np.shape, 'img.shape =', img_np.shape, 'mask.shape =', mask_np.shape)

        # tensor_np = sitk.GetArrayFromImage(sitk.ReadImage(f'{input_dir}/{file_name[s]}_scaled_orig_tensors_rreg_v2.nhdr'))
        # mask_np = sitk.GetArrayFromImage(sitk.ReadImage(f'{input_dir}/{file_name[s]}_orig_mask_rreg.nhdr'))
        # # brain_mask_np = sitk.GetArrayFromImage(sitk.ReadImage(f'{input_dir}/{file_name[s]}_brain_mask_rreg.nhdr'))
        # img_np = sitk.GetArrayFromImage(sitk.ReadImage(f'{input_dir}/{file_name[s]}_t1_to_reft1_rreg.nhdr'))
        #tensor_lin_list.append(torch.from_numpy(tensor_np).double().permute(3,2,1,0))
        # KMC add scaling of tensors per http://dti-tk.sourceforge.net/pmwiki/pmwiki.php?n=Documentation.Diffusivity
        tensor_lin_list.append(torch.from_numpy(tens_scale * tensor_np).double().permute(3,2,1,0))

    #     create union of masks
        mask_union += torch.from_numpy(mask_np).double().permute(2,1,0).to(device)
        mask_list.append(torch.from_numpy(mask_np).double().permute(2,1,0))
        # brain_mask_list.append(torch.from_numpy(brain_mask_np).double().permute(2,1,0))
        img_list.append(torch.from_numpy(img_np).double().permute(2,1,0))
    #     rearrange tensor_lin to tensor_met
        tensor_met_zeros = torch.zeros(height,width,depth,3,3,dtype=torch.float64)
        tensor_met_zeros[:,:,:,0,0] = tensor_lin_list[s][0]
        tensor_met_zeros[:,:,:,0,1] = tensor_lin_list[s][1]
        tensor_met_zeros[:,:,:,0,2] = tensor_lin_list[s][2]
        tensor_met_zeros[:,:,:,1,0] = tensor_lin_list[s][1]
        tensor_met_zeros[:,:,:,1,1] = tensor_lin_list[s][3]
        tensor_met_zeros[:,:,:,1,2] = tensor_lin_list[s][4]
        tensor_met_zeros[:,:,:,2,0] = tensor_lin_list[s][2]
        tensor_met_zeros[:,:,:,2,1] = tensor_lin_list[s][4]
        tensor_met_zeros[:,:,:,2,2] = tensor_lin_list[s][5]
        # tensor_met_zeros = make_pos_def(tensor_met_zeros, torch.ones((height, width, depth)))
    #     balance the background and subject by rescaling
        # Choose size for isometric tensor of same order as input tensors
        if s == 0:
          scale_factor = 1.0
          max_tens = torch.max(tensor_met_zeros)
          while scale_factor * max_tens < 1:
            scale_factor = scale_factor * 10
          scale_factor = 1.0
          print("WARNING!! Overriding scale factor")
          print("Scaling isometric tensors by factor of", scale_factor)
 
          iso_tens = torch.zeros((3,3),dtype=torch.double)
          iso_tens[0,0] = 1.0 / scale_factor
          iso_tens[1,1] = 1.0 / scale_factor
          iso_tens[2,2] = 1.0 / scale_factor
        print(s,tensor_met_zeros[71,81,72])
        print(s,tensor_met_zeros[70,132,85])
        print('\n\nmax before tensor_cleaning:', torch.max(tensor_met_zeros), 'min:', torch.min(tensor_met_zeros))
        tensor_met_zeros = tensor_cleaning(tensor_met_zeros, mask_list[s], iso_tens)
        print(s,tensor_met_zeros[71,81,72])
        print(s,tensor_met_zeros[70,132,85])
        print('\n\nmax after tensor_cleaning:', torch.max(tensor_met_zeros), 'min:', torch.min(tensor_met_zeros))
        # fa_list.append(fractional_anisotropy(tensor_met_zeros))
        tensor_met_list.append(torch.inverse(tensor_met_zeros))
        print('\n\nmax after torch.inverse:', torch.max(tensor_met_list[s]))
        # fore_back_adaptor = torch.ones((height,width,depth))
        # fore_back_adaptor = torch.where(torch.det(tensor_met_list[s])>1e1, 5e-4, 1.)
        fore_back_adaptor = torch.where(torch.det(tensor_met_list[s])>1e2, 1e-3, 1.)#
        mask_thresh_list.append(fore_back_adaptor)
        # KMC Don't mask out the large determinant
        print("WARNING!!! DISABLING fore_back_adaptor")
        #tensor_met_list[s] = torch.einsum('ijk...,lijk->ijk...', tensor_met_list[s], mask_thresh_list[s].unsqueeze(0))
        # tensor_met_list[s][torch.abs(torch.det(tensor_met_list[s])-1)<=1e-3] = torch.eye((3))
    #     initialize the accumulative diffeomorphism    
        if resume==False:
            print('start from identity')
            phi_inv_acc_list.append(get_idty(height, width, depth,device='cpu'))
            phi_acc_list.append(get_idty(height, width, depth,device='cpu'))
        else:
            print('start from checkpoint')
            phi_inv_acc_list.append(torch.from_numpy(sio.loadmat(f'{output_dir}/{subj}_{session}_{start_iter-1}_phi_inv.mat')['diffeo']).to(device))
            phi_acc_list.append(torch.from_numpy(sio.loadmat(f'{output_dir}/{subj}_{session}_{start_iter-1}_phi.mat')['diffeo']).to(device))
            tensor_met_list[s] = phi_pullback(phi_inv_acc_list[s], tensor_met_list[s], mask_list[s])
        met_energy_list.append([])    
        img_energy_list.append([])    
        
    # mask_union[mask_union>0] = 1

    # Orig
    #dim, sigma, epsilon, inner_iter_num = 3., 0, 5e-3, 1 # epsilon = 3e-3 for orig tensor, 5e-3 for HCP
    # sigma = 0
    dim, sigma, epsilon, inner_iter_num = 3., 0, 5e-7, 2 # epsilon = 3e-3 for orig tensor, 5e-3 for HCP, 5e-9 not working for 2 ABCD subj, 5e-9 not working for 6 subj ABCD
    print("WARNING!!! Regularizer on phi disabled, sigma set to 0")
    # add regularizer on phi
    #dim, sigma, epsilon, inner_iter_num = 3., 1e10, 5e-8, 2 # epsilon = 3e-3 for orig tensor, 5e-3 for HCP
    
    print(f'Subject = {subj}, iter_num = {iter_num}, epsilon = {epsilon}')
    print(f'Starting from iteration {start_iter} to iteration {iter_num+start_iter}')

    for i in tqdm(range(start_iter, start_iter+iter_num)):
        G = torch.stack(tuple(tensor_met_list))
        # G[torch.abs(torch.det(G)-1)<=2e-4] = torch.eye((3))
        # G[torch.det(G)<=1e-1] = torch.eye((3))

        #mask_union = ((mask_list[0]+mask_list[1]+mask_list[2]+mask_list[3]+mask_list[4]+mask_list[5])/6).to(device)
        mask_union = (sum(mask_list)/len(mask_list)).to(device)

        print('\n\n main loop, iter', i, 'max G:', torch.max(G), 'max mask_union:', torch.max(mask_union), '\n\n')

        #print("WARNING!! Not shuffling Karcher mean!")
        #atlas = get_karcher_mean(G, 1./dim, mask_union, scale_factor,  f'{output_dir}/theta_{i}.nhdr')#_shuffle
        #atlas = get_karcher_mean(G, 1./dim, mask_union, scale_factor,  f'{output_dir}/{i}')#_shuffle
        #atlas = get_karcher_mean(G, 1./dim, mask_union, scale_factor, None)#_shuffle
        atlas = get_karcher_mean_shuffle(G, 1./dim, device=cuda_dev)
        mean_img = get_euclidean_mean(img_list)
        print('\n\n main loop, iter', i, 'max atlas:', torch.max(atlas), '\n\n')

        phi_inv_list, phi_list = [], []
        #mask_union = ((mask_list[0]+mask_list[1]+mask_list[2]+mask_list[3]+mask_list[4]+mask_list[5])/6).to(device)
        #mask_union = (sum(mask_list)/len(mask_list)).to(device)
        # brain_mask_union = ((brain_mask_list[0]+brain_mask_list[1]+brain_mask_list[2]+brain_mask_list[3]+brain_mask_list[4]+brain_mask_list[5])/6).to(device)
        for s in range(sample_num):
            met_energy_list[s].append(torch.einsum("ijk...,lijk->",[(tensor_met_list[s] - atlas)**2, mask_union.unsqueeze(0)]).item())
            img_energy_list[s].append(torch.sum((img_list[s] - mean_img)**2).item())
            old = tensor_met_list[s]
            #tensor_met_list[s], img_list[s], phi, phi_inv = metric_matching(tensor_met_list[s], atlas, img_list[s], mean_img, height, width, depth, mask_list[s], mask_union, inner_iter_num, epsilon, sigma, dim)
            tensor_met_list[s], img_list[s], mask_list[s], phi, phi_inv = metric_matching(tensor_met_list[s], atlas, img_list[s], mean_img, height, width, depth, mask_list[s], mask_union, inner_iter_num, epsilon, sigma, dim)
            #tensor_met_list[s], img_list[s], mask_list[s], phi, phi_inv = metric_matching(G[s], atlas, img_list[s], mean_img, height, width, depth, mask_list[s], mask_union, inner_iter_num, epsilon, sigma, dim)
            # tensor_met_list[s][torch.abs(torch.det(tensor_met_list[s])-1)<=1e-3] = torch.eye((3))
            phi_inv_list.append(phi_inv)
            phi_list.append(phi)
            phi_inv_acc_list[s] = compose_function(phi_inv_acc_list[s], phi_inv_list[s])
            phi_acc_list[s] = compose_function(phi_list[s], phi_acc_list[s])
            #mask_list[s] = compose_function(mask_list[s], phi_inv_list[s])
                
        '''check point'''
        if i%200==0:
        #if i%100==0:
        #if i%1==0:
            atlas_lin = np.zeros((6,height,width,depth))
            mask_acc = np.zeros((height,width,depth))
            #print('\n\n\natlas[1202815]',atlas.reshape(-1, 3, 3)[1202815])
            #print('atlas[70,132,85]',atlas[70,132,85])

            atlas_inv = torch.inverse(atlas) / tens_scale
            #print('atlas_inv[70,132,85]',atlas_inv[70,132,85])
            #print('atlas_inv[1390565]', atlas_inv.reshape(-1,3,3)[1390565],'\n\n\n')
            atlas_lin[0] = atlas_inv[:,:,:,0,0].cpu()
            atlas_lin[1] = atlas_inv[:,:,:,0,1].cpu()
            atlas_lin[2] = atlas_inv[:,:,:,0,2].cpu()
            atlas_lin[3] = atlas_inv[:,:,:,1,1].cpu()
            atlas_lin[4] = atlas_inv[:,:,:,1,2].cpu()
            atlas_lin[5] = atlas_inv[:,:,:,2,2].cpu()
            for s in range(sample_num):
                subj = cases[s]
                sio.savemat(f'{output_dir}/{subj}_{session}_{i}_phi_inv.mat', {'diffeo': phi_inv_acc_list[s].cpu().detach().numpy()})
                sio.savemat(f'{output_dir}/{subj}_{session}_{i}_phi.mat', {'diffeo': phi_acc_list[s].cpu().detach().numpy()})
                sio.savemat(f'{output_dir}/{subj}_{session}_{i}_met_energy.mat', {'energy': met_energy_list[s]})
                sio.savemat(f'{output_dir}/{subj}_{session}_{i}_img_energy.mat', {'energy': img_energy_list[s]})
                # tens_lin = np.zeros((6,height,width,depth))
                # phi_inv_G = test_make_pos_def(phi_pullback(phi_inv_list[s], G[s], mask_list[s]), mask_list[s], 1.0e-10)
                # #print('\n\n\ntens[',s,'][1202815]',tensor_met_list[s].reshape(-1, 3, 3)[1202815])
                # #print('tens[',s,'][70,132,85]',tensor_met_list[s][70,132,85])
                # #print('\n\n\nG[',s,'][1202815]',phi_inv_G.reshape(-1, 3, 3)[1202815])
                # #print('G[',s,'][70,132,85]',phi_inv_G[70,132,85])

                # tens_inv = torch.inverse(tensor_met_list[s]) / tens_scale
                # #print('tens_inv[',s,'][70,132,85]',tens_inv[70,132,85])
                # #print('tens_inv[',s,'][1390565]', tens_inv.reshape(-1,3,3)[1390565],'\n\n\n')
                # tens_lin[0] = tens_inv[:,:,:,0,0].cpu()
                # tens_lin[1] = tens_inv[:,:,:,0,1].cpu()
                # tens_lin[2] = tens_inv[:,:,:,0,2].cpu()
                # tens_lin[3] = tens_inv[:,:,:,1,1].cpu()
                # tens_lin[4] = tens_inv[:,:,:,1,2].cpu()
                # tens_lin[5] = tens_inv[:,:,:,2,2].cpu()
                # WriteTensorNPArray(np.transpose(tens_lin,(3,2,1,0)), f'{output_dir}/tens{s}_{i}_tens.nhdr')
                # G_inv = torch.inverse(phi_inv_G) / tens_scale
                # #print('Ginv[',s,'][70,132,85]',G_inv[70,132,85])
                # #print('Ginv[',s,'][1390565]', G_inv.reshape(-1,3,3)[1390565],'\n\n\n')
                # tens_lin[0] = G_inv[:,:,:,0,0].cpu()
                # tens_lin[1] = G_inv[:,:,:,0,1].cpu()
                # tens_lin[2] = G_inv[:,:,:,0,2].cpu()
                # tens_lin[3] = G_inv[:,:,:,1,1].cpu()
                # tens_lin[4] = G_inv[:,:,:,1,2].cpu()
                # tens_lin[5] = G_inv[:,:,:,2,2].cpu()
                # WriteTensorNPArray(np.transpose(tens_lin,(3,2,1,0)), f'{output_dir}/G{s}_{i}_tens.nhdr')

                
    #             plt.plot(energy_list[s])
                # mask_acc += mask_list[s].cpu().numpy()
            # mask_acc[mask_acc>0]=1
            #sitk.WriteImage(sitk.GetImageFromArray(np.transpose(atlas_lin,(3,2,1,0))), f'{output_dir}/atlas_{i}_tens.nhdr')
            #sitk.WriteImage(sitk.GetImageFromArray(np.transpose(mask_union.cpu(),(2,1,0))), f'{output_dir}/atlas_{i}_mask.nhdr')
            #sitk.WriteImage(sitk.GetImageFromArray(np.transpose(mean_img.cpu(),(2,1,0))), f'{output_dir}/atlas_{i}_img.nhdr')
            WriteTensorNPArray(np.transpose(atlas_lin,(3,2,1,0)), f'{output_dir}/atlas_{i}_tens.nhdr')
            WriteScalarNPArray(np.transpose(mask_union.cpu(),(2,1,0)), f'{output_dir}/atlas_{i}_mask.nhdr')
            WriteScalarNPArray(np.transpose(img_scale * mean_img.cpu(),(2,1,0)), f'{output_dir}/atlas_{i}_img.nhdr')

    atlas_lin = np.zeros((6,height,width,depth))
    # mask_acc = np.zeros((height,width,depth))

    for s in range(sample_num):
        subj = cases[s]
        sio.savemat(f'{output_dir}/{subj}_{session}_phi_inv.mat', {'diffeo': phi_inv_acc_list[s].cpu().detach().numpy()})
        sio.savemat(f'{output_dir}/{subj}_{session}_phi.mat', {'diffeo': phi_acc_list[s].cpu().detach().numpy()})
        sio.savemat(f'{output_dir}/{subj}_{session}_met_energy.mat', {'energy': met_energy_list[s]})
        sio.savemat(f'{output_dir}/{subj}_{session}_img_energy.mat', {'energy': img_energy_list[s]})
        
        plt.plot(met_energy_list[s])
        plt.plot(img_energy_list[s])
        # mask_acc += mask_list[s].cpu().numpy()

    atlas = torch.inverse(atlas) / tens_scale
    atlas_lin[0] = atlas[:,:,:,0,0].cpu()
    atlas_lin[1] = atlas[:,:,:,0,1].cpu()
    atlas_lin[2] = atlas[:,:,:,0,2].cpu()
    atlas_lin[3] = atlas[:,:,:,1,1].cpu()
    atlas_lin[4] = atlas[:,:,:,1,2].cpu()
    atlas_lin[5] = atlas[:,:,:,2,2].cpu()
    # mask_acc[mask_acc>0]=1
    #sitk.WriteImage(sitk.GetImageFromArray(np.transpose(atlas_lin,(3,2,1,0))), f'{output_dir}/atlas_tens.nhdr')
    #sitk.WriteImage(sitk.GetImageFromArray(np.transpose(mask_union.cpu(),(2,1,0))), f'{output_dir}/atlas_mask.nhdr')
    #sitk.WriteImage(sitk.GetImageFromArray(np.transpose(mean_img.cpu(),(2,1,0))), f'{output_dir}/atlas_img.nhdr')
    WriteTensorNPArray(np.transpose(atlas_lin,(3,2,1,0)), f'{output_dir}/atlas_tens.nhdr')
    WriteScalarNPArray(np.transpose(mask_union.cpu(),(2,1,0)), f'{output_dir}/atlas_mask.nhdr')
    WriteScalarNPArray(np.transpose(img_scale * mean_img.cpu(),(2,1,0)), f'{output_dir}/atlas_img.nhdr')
