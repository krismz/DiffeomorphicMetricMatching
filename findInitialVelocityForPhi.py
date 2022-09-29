import os 
import torch
from tqdm import tqdm
import gc
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import SimpleITK as sitk
import random
from util.RegistrationFunc3DCuda import *
from util.SplitEbinMetric3DCuda import *
from util.diffeo import shoot_geodesic_momenta_formulation as shoot_geodesic
#from util.diffeo import shoot_geodesic_velocity_formulation as shoot_geodesic

from disp.vis import vis_tensors, vis_path, disp_scalar_to_file
from disp.vis import disp_vector_to_file, disp_tensor_to_file
from disp.vis import disp_gradG_to_file, disp_gradA_to_file
from disp.vis import view_3d_tensors, tensors_to_mesh
from data.io import readRaw, ReadScalars, ReadTensors, WriteTensorNPArray, WriteScalarNPArray, readPath3D
from data.convert import GetNPArrayFromSITK, GetSITKImageFromNP
from torch_sym3eig import Sym3Eig as se

# Find the initial velocity, v0, that corresponds to the given phi at time t=1

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

def energy_phi_L2(phi0, phi1, mask):
#     input: phi0/phi1.shape = [3, h, w, d]; mask.shape = [1, h, w, d]
#     output: scalar
    E1 = torch.einsum("...ijk,...ijk->", (phi0 - phi1) ** 2, mask.unsqueeze(0))
    return E1
  
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

def diffeo_matching(phi_t, num_integration_iters = 10, max_iters = 50, step_size = 0.00001, min_change = 1.0e-5, min_E = 1.0e-5, init_with_zeros=False, phiinv_t = None):
  mask = torch.ones(*phi_t.shape[-3:])
  if init_with_zeros:
    # initial v0 guess is zeros
    init_v = torch.zeros_like(phi_t).requires_grad_()
  else:
    # initial v0 guess is (phi-idty) / N
    idty = get_idty(*phi_t.shape[-3:]).to(phi_t.device)
    init_v = ((phi_t - idty) / num_integration_iters).requires_grad_()
  keep_going = True
  cur_iter = 0
  reason = ""
  energy = []
  inv_energy = []
  prev_E = -1
  while keep_going:
    est_phi_t, est_phiinv_t, est_v_t = shoot_geodesic(init_v, num_integration_iters, True)
    E = energy_phi_L2(est_phi_t, phi_t, mask)
    E.backward()
    energy.append(E.item())
    print(E.item())

    v_grad = init_v.grad
    with torch.no_grad():
      if phiinv_t is not None:
        invE = energy_phi_L2(est_phiinv_t, phiinv_t, mask)
        inv_energy.append(invE.item())
        print(invE.item())
      new_init_v = init_v - step_size * v_grad

      # Check to see if we've converged...
      cur_iter += 1
      if cur_iter >= max_iters:
        keep_going = False
        reason += f"Reached {maxIters} iterations"
      Ediff = E - prev_E
      if (Ediff) < min_change:
        keep_going = False
        reason += f"E ({E}) - prev_E ({prev_E}) = {Ediff} < min change ({min_change})"
      if E < min_E:
        keep_going = False
        reason += f"E ({E}) < min E ({min_E})"
      
      # cleanup and reset for next time through
      del E, invE, init_v
      gc.collect()
      torch.cuda.empty_cache()

      init_v = new_init_v

    init_v.requires_grad_()
  # end for each iteration    
    
  return(new_init_v, est_phi_t, energy,
         est_phiinv_t, inv_energy, est_v_t, reason)
    

if __name__ == "__main__":
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # after switch device, you need restart the script
    #torch.cuda.set_device('cuda:0')
    #torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    device = torch.device('cpu')
    torch.set_default_tensor_type('torch.DoubleTensor')
    print('setting torch print precision to 16')
    torch.set_printoptions(precision=16)

    # file_name = []
    #file_name = [108222, 102715, 105923, 107422, 100206, 104416]
    input_dir = '/usr/sci/projects/abcd/anxiety_study/derivatives/metric_matching'
    output_dir = '/usr/sci/projects/abcd/anxiety_study/derivatives/atlas_building_cuda_22subj_901iter'
    #output_dir = '/usr/sci/projects/abcd/anxiety_study/derivatives/atlas_building_cuda_11cases_901iter'
    #output_dir = '/usr/sci/projects/abcd/anxiety_study/derivatives/atlas_building_cuda_11controls_901iter'

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
    offs=[1,3,5,6,7,8,9,11,14,16,17] 
    #offs=[0,1]
    #print(f"WARNING, using first {num} cases and {num} controls only!!")
    #cases = cases[0:num] + cases[18:18+num]
    #print(f"WARNING, using first {num} cases only!!")
    #cases = cases[0:num]
    #print(f"WARNING,building atlas from first subject repeated twice!!") # next do first 2 subjects
    #cases= cases[0:1] + cases[0:1]
    #print(f"WARNING, using", len(offs), 'cases and controls with offsets', offs)
    #print(f"WARNING, using", len(offs), 'cases vs CASES!! with offsets', offs)

    offs = [1]
    print(f"WARNING!!! only 1 subject to test method")
    outc = []
    for offset in offs:
        outc.append(cases[offset])
        #outc.append(cases[offset])
        #outc.append(cases[18+offset])
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

    for s in range(len(cases)):
        subj = cases[s]
        print(f'{subj} is processing.')
        dwi_prefix = os.path.join(input_dir, subj, session,'dwi', f'{subj}_{session}')

        phi = torch.from_numpy(sio.loadmat(f'{output_dir}/{subj}_{session}_phi.mat')['diffeo']).double().to(device)
        phiinv = torch.from_numpy(sio.loadmat(f'{output_dir}/{subj}_{session}_phi_inv.mat')['diffeo']).double().to(device)

        v0, phi_t, energy, phiinv_t, inv_energy, vt, reason = \
        diffeo_matching(phi, 10, 50, 1.0e-5, 1.0e-5, 1.0e-5, False,
                        phiinv)

        sio.savemat(f'{output_dir}/{subj}_{session}_v.mat', {'v0': v0.cpu().detach().numpy(),'vt': vt.cpu().detach().numpy()})
        sio.savemat(f'{output_dir}/{subj}_{session}_shoot_phi.mat', {'diffeo': phi_t.cpu().detach().numpy()})
        sio.savemat(f'{output_dir}/{subj}_{session}_shoot_phi_inv.mat', {'diffeo': phiinv_t.cpu().detach().numpy()})
        sio.savemat(f'{output_dir}/{subj}_{session}_est_v0_phi_energy.mat', {'energy': energy})
        sio.savemat(f'{output_dir}/{subj}_{session}_est_v0_phiinv_energy.mat', {'energy': inv_energy})
        
        print(f"{subj} finished processing, reason = {reason}")
