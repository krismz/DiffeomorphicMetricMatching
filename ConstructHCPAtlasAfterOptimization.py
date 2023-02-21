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

def phi_pullback(phi, g, mask=None):
#     input: phi.shape = [3, h, w, d]; g.shape = [h, w, d, 3, 3]
#     output: shape = [h, w, 2, 2]
#     torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    g = g.permute(3, 4, 0, 1, 2)
    idty = get_idty(*g.shape[-3:])
    #     four layers of scalar field, of all 1, all 0, all 1, all 0, where the shape of each layer is g.shape[-2:]?
    eye = torch.eye(3)
    ones = torch.ones(*g.shape[-3:])
    d_phi = get_jacobian_matrix(phi - idty) + torch.einsum("ij,mno->ijmno", eye, ones)
    g_phi = compose_function(g, phi, mask, eye)
    return torch.einsum("ij...,ik...,kl...->...jl", d_phi, g_phi, d_phi)

def energy_ebin(phi, g0, g1, f0, f1, i0, i1, sigma, dim, mask): 
#     input: phi.shape = [3, h, w, d]; g0/g1/f0/f1.shape = [h, w, d, 3, 3]; sigma/dim = scalar; mask.shape = [1, h, w, d]
#     output: scalar
# the phi here is identity
    phi_star_g1 = phi_pullback(phi, g1)
    phi_star_f1 = phi_pullback(phi, f1)# the compose operation in this step uses a couple of thousands MB of memory
    phi_star_i1 = compose_function(i1.unsqueeze(0), phi, mask, 0).squeeze()# the compose operation in this step uses a couple of thousands MB of memory

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
    idty = get_idty(size_h, size_w, size_d).cpu().numpy()
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

    return torch.stack((vx, vy, vz)).to(device=torch.device('cuda'))

        

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # after switch device, you need restart the script
    torch.cuda.set_device('cuda:1')
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    print('setting torch print precision to 16')
    torch.set_printoptions(precision=16)
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

    # brain_mask_union = torch.zeros(height, width, depth).double().to(device)


    for s in range(len(cases)):
        subj = cases[s]
        print(f'{subj} is processing.')
        in_prefix = os.path.join(input_dir, subj)
        #mask_np = ReadScalars(f'{in_prefix}/dti_{bval}_FA_mask_affreg.nhdr')
        img_np = np.transpose(ReadScalars(f'{in_prefix}/t1_to_reft1_affreg.nhdr'), (0,2,1))
        img_np /= np.max(img_np)
        mask_np = np.ones_like(img_np)

        mask = torch.from_numpy(mask_np).double().permute(2,1,0)
        img = torch.from_numpy(img_np).double().permute(2,1,0)
        if s == 0:
          height,width,depth = mask.shape
          atlas_img = torch.zeros_like(img).to(device)

        diffeo = torch.from_numpy(sio.loadmat(f'{output_dir}/{subj}_atlas_phi_inv.mat')['diffeo']).double().to(device)

        mask = compose_function(mask.unsqueeze(0), diffeo).squeeze()
        img = compose_function(img.unsqueeze(0), diffeo).squeeze()

        atlas_img += img

    atlas_img = atlas_img / len(cases)

    WriteScalarNPArray(np.transpose(img_scale * atlas_img.cpu(),(2,1,0)), f'{output_dir}/final_atlas_img.nhdr')
