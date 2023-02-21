import os
import sys
import torch
from tqdm import tqdm
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import SimpleITK as sitk
import random
from util.RegistrationFunc3DCuda import *
from util.SplitEbinMetric3DCuda import *

from disp.vis import vis_tensors, vis_path, disp_scalar_to_file
from disp.vis import disp_vector_to_file, disp_tensor_to_file
from disp.vis import disp_gradG_to_file, disp_gradA_to_file
from disp.vis import view_3d_tensors, tensors_to_mesh
from data.io import readRaw, ReadScalars, ReadTensors, WriteTensorNPArray, WriteScalarNPArray, readPath3D
from data.convert import GetNPArrayFromSITK, GetSITKImageFromNP
from torch_sym3eig import Sym3Eig as se

# After finding phi and phi_inv, apply to original tensors and recompute atlas
# This is done because there is a cumulative smoothing effect when applying the phi_inv repeatedly, incrementally to the tensors.  Which ends up with an overly blurry atlas.

cuda_dev = 'cuda:0'


def phi_pullback(phi, g, mask=None):
# #     input: phi.shape = [3, h, w, d]; g.shape = [h, w, d, 3, 3]
# #     output: shape = [h, w, 2, 2]
# #     torch.set_default_tensor_type('torch.cuda.DoubleTensor')
#     g = g.permute(3, 4, 0, 1, 2)
#     idty = get_idty(*g.shape[-3:])
#     #     four layers of scalar field, of all 1, all 0, all 1, all 0, where the shape of each layer is g.shape[-2:]?
#     eye = torch.eye(3)
#     ones = torch.ones(*g.shape[-3:])
#     d_phi = get_jacobian_matrix(phi - idty) + torch.einsum("ij,mno->ijmno", eye, ones)
#     g_phi = compose_function(g, phi, mask, eye)
#     return torch.einsum("ij...,ik...,kl...->...jl", d_phi, g_phi, d_phi)

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



def make_pos_def(tens, mask, small_eval = 0.00005):
  # make any small or negative eigenvalues slightly positive and then reconstruct tensors
    #print('WARNING! Short-circuiting BrainAtlasBuilding3DUkfCudaImg.make_pos_def')
    #return(tens)

    det_threshold=1e-11
    tens[torch.det(tens)<=det_threshold] = torch.eye((3))

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
                evecs[idx[0][ee], idx[1][ee], idx[2][ee]] = fw.eye(3, dtype=tens.dtype)
                evals[idx[0][ee], idx[1][ee], idx[2][ee], idx[3][ee]] = small_eval
            else:
                # otherwise just set this eigenvalue to small_eval
                evals[idx[0][ee], idx[1][ee], idx[2][ee], idx[3][ee]] = small_eval

    print(num_found, 'tensors found with eigenvalues <', small_eval)
    #print(num_found, 'tensors found with eigenvalues < 0')
    mod_tens = fw.einsum('...ij,...jk,...k,...lk->...il',
                       evecs, fw.eye(3, dtype=tens.dtype), evals, evecs)
    #mod_tens = fw.einsum('...ij,...j,...jk->...ik',
    #                     evecs, evals, evecs)

    print("WARNING!!!! Overriding small_eval fix in BrainAtlasBuilding3DUkfCudaImg.make_pos_def")
    mod_tens = tens.clone()
    chol = batch_cholesky(mod_tens)
    idx_nan = torch.where(torch.isnan(chol))
    nan_map = torch.where(torch.isnan(chol),1,0)
    iso_tens = small_eval * torch.eye((3))
    for pt in range(len(idx_nan[0])):
        mod_tens[idx_nan[0][pt],idx_nan[1][pt],idx_nan[2][pt]] = iso_tens
    # if torch.norm(torch.transpose(mod_tens,3,4)-mod_tens)>0:
    #     print('asymmetric')
    #mod_tens[:,:,:,1,0]=mod_tens[:,:,:,0,1]
    #mod_tens[:,:,:,2,0]=mod_tens[:,:,:,0,2]
    #mod_tens[:,:,:,2,1]=mod_tens[:,:,:,1,2]
    mod_sym_tens = (mod_tens + torch.transpose(mod_tens,len(mod_tens.shape)-2,len(mod_tens.shape)-1))/2
    mod_sym_tens[torch.det(mod_sym_tens)<=det_threshold] = torch.eye((3))
    return(mod_sym_tens)


def get_euclidean_mean(img_list):
    mean = torch.zeros_like(img_list[0])
    for i in range(len(img_list)):
        mean += img_list[i]

    return mean/len(img_list)


if __name__ == "__main__":
    device = torch.device(cuda_dev if torch.cuda.is_available() else 'cpu')
    # after switch device, you need restart the script
    torch.cuda.set_device(device)
    torch.set_default_tensor_type('torch.cuda.DoubleTensor' if torch.cuda.is_available() else 'torch.DoubleTensor')
    print('setting torch print precision to 16')
    torch.set_printoptions(precision=16)

    inroot = '/usr/sci/projects/abcd/simdata/3d_cubics/'
    outroot = '/usr/sci/projects/abcd/simresults/3d_cubics/'
    #sims = ['sim1', 'sim2', 'sim3']
    #num_subjs_in_sim = [10, 30, 30]
    #sims = ['noshape','sim1']
    #num_subjs_in_sim = [10, 10]
    #sims = ['sim1Img']
    #num_subjs_in_sim = [10]
    #sims = ['noshapeImg','sim1Img']
    #num_subjs_in_sim = [10, 10]
    sims = ['noshapeTex']
    num_subjs_in_sim = [10]

    prefix = 'metpy_3D_cubic'
    for sim, num_subjs in zip(sims, num_subjs_in_sim):
      indir = inroot + sim
      outdir = outroot + sim
      cases = [f'1_{cc}' for cc in range(num_subjs)]  + [f'2_{cc}' for cc in range(num_subjs)]

      tensor_lin_list, tensor_met_list, mask_list, mask_thresh_list, diffeo_list, img_list, brain_mask_list = [], [], [], [], [], [], []
      #img_scale = 255.0
      #tens_scale = 1000
      img_scale = 1
      tens_scale = 1

      first_time = True
      for s in range(len(cases)):
        subj = cases[s]
        print(f'{subj} is processing.')
        tensor_np = ReadTensors(f'{outdir}/{prefix}{subj}_scaled_orig_tensors_v2.nhdr')
        # Holes being added to cubic masks for some reason, so use originals instead
        #mask_np = ReadScalars(f'{outdir}/{prefix}{subj}_filt_mask.nhdr')
        mask_np = ReadScalars(f'{outdir}/{prefix}{subj}_orig_mask.nhdr')
        img_np = ReadScalars(f'{outdir}/{prefix}{subj}_T1_flip_y.nhdr') / img_scale

        if first_time:
          # since haven't permuted yet, shape is currently depth, width, height
          depth, width, height = mask_np.shape
          mask_union = torch.zeros(height, width, depth).double().to(device)
          first_time = False
   
        print('subj', s, 'tensor.shape =', tensor_np.shape, 'img.shape =', img_np.shape, 'mask.shape =', mask_np.shape)


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
        tensor_met_zeros = tensor_cleaning(tensor_met_zeros, mask_list[s], iso_tens)
        diffeo_list.append(torch.from_numpy(sio.loadmat(f'{outdir}/{prefix}{subj}_phi_inv.mat')['diffeo']).double().to(device))
        # fa_list.append(fractional_anisotropy(tensor_met_zeros))
        tensor_met_list.append(torch.inverse(tensor_met_zeros))

      for s in range(len(cases)):
        subj = cases[s]
        mask_list[s] = compose_function(mask_list[s].unsqueeze(0), diffeo_list[s]).squeeze()
        #tensor_met_list[s]= phi_pullback(diffeo_list[s], tensor_met_list[s])
        tensor_met_list[s]= make_pos_def(phi_pullback(diffeo_list[s], tensor_met_list[s], mask_list[s]), mask_list[s], 1.0e-10) 
        img_list[s] = compose_function(img_list[s].unsqueeze(0), diffeo_list[s]).squeeze()

        #tensor_met_list[s] = tensor_cleaning(tensor_met_list[s], mask_list[s], iso_tens)
        tens_lin = np.zeros((6,height,width,depth))
        tens_inv = torch.inverse(tensor_met_list[s]) / tens_scale
        tens_lin[0] = tens_inv[:,:,:,0,0].cpu()
        tens_lin[1] = tens_inv[:,:,:,0,1].cpu()
        tens_lin[2] = tens_inv[:,:,:,0,2].cpu()
        tens_lin[3] = tens_inv[:,:,:,1,1].cpu()
        tens_lin[4] = tens_inv[:,:,:,1,2].cpu()
        tens_lin[5] = tens_inv[:,:,:,2,2].cpu()
        WriteTensorNPArray(np.transpose(tens_lin,(3,2,1,0)), f'{outdir}/{prefix}{subj}_final_tens.nhdr')
        
      G = torch.stack(tuple(tensor_met_list))
      atlas_mask = (sum(mask_list)/len(mask_list)).to(device)
      atlas = get_karcher_mean_shuffle(G, 1./3.0) # 1./dim
      atlas_img = get_euclidean_mean(img_list)

      atlas_lin = np.zeros((6,height,width,depth))

      atlas_inv = torch.inverse(atlas) / tens_scale
      atlas_lin[0] = atlas_inv[:,:,:,0,0].cpu()
      atlas_lin[1] = atlas_inv[:,:,:,0,1].cpu()
      atlas_lin[2] = atlas_inv[:,:,:,0,2].cpu()
      atlas_lin[3] = atlas_inv[:,:,:,1,1].cpu()
      atlas_lin[4] = atlas_inv[:,:,:,1,2].cpu()
      atlas_lin[5] = atlas_inv[:,:,:,2,2].cpu()
      WriteTensorNPArray(np.transpose(atlas_lin,(3,2,1,0)), f'{outdir}/final_atlas_tens.nhdr')
      WriteScalarNPArray(np.transpose(atlas_mask.cpu(),(2,1,0)), f'{outdir}/final_atlas_mask.nhdr')
      WriteScalarNPArray(np.transpose(img_scale * atlas_img.cpu(),(2,1,0)), f'{outdir}/final_atlas_img.nhdr')
    # end for each sim directory  
