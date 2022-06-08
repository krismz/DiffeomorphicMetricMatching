import torch
from lazy_imports import np
import scipy.io as sio
from util.SplitEbinMetric3D import *
from util.RegistrationFunc3D import get_jacobian_determinant
from util.diffeo import coord_register_batch_3d, phi_pullback_3d, get_idty_3d
from util.tensors import tens_3x3_to_tens_6, tens_6_to_tens_3x3
from data.io import ReadTensors, ReadScalars, WriteTensorNPArray, WriteScalarNPArray
import nibabel as nib
import pathlib
import _pickle as pickle
import platform


###################
# Begin patch for getting large amount of data back from pool.map
# Error received was:
#  File "/usr/lib64/python3.6/multiprocessing/connection.py", line 393, in _send_bytes
#    header = struct.pack("!i", n)
# struct.error: 'i' format requires -2147483648 <= number <= 2147483647
#
# patch from https://stackoverflow.com/questions/47776486/python-struct-error-i-format-requires-2147483648-number-2147483647
###################
import functools
import logging
import struct
import sys

logger = logging.getLogger()

def patch_mp_connection_bpo_17560():
    """Apply PR-10305 / bpo-17560 connection send/receive max size update

    See the original issue at https://bugs.python.org/issue17560 and 
    https://github.com/python/cpython/pull/10305 for the pull request.

    This only supports Python versions 3.3 - 3.7, this function
    does nothing for Python versions outside of that range.

    """
    patchname = "Multiprocessing connection patch for bpo-17560"
    if not (3, 3) < sys.version_info < (3, 8):
        logger.info(
            patchname + " not applied, not an applicable Python version: %s",
            sys.version
        )
        return

    from multiprocessing.connection import Connection

    orig_send_bytes = Connection._send_bytes
    orig_recv_bytes = Connection._recv_bytes
    if (
        orig_send_bytes.__code__.co_filename == __file__
        and orig_recv_bytes.__code__.co_filename == __file__
    ):
        logger.info(patchname + " already applied, skipping")
        return

    @functools.wraps(orig_send_bytes)
    def send_bytes(self, buf):
        n = len(buf)
        if n > 0x7fffffff:
            pre_header = struct.pack("!i", -1)
            header = struct.pack("!Q", n)
            self._send(pre_header)
            self._send(header)
            self._send(buf)
        else:
            orig_send_bytes(self, buf)

    @functools.wraps(orig_recv_bytes)
    def recv_bytes(self, maxsize=None):
        buf = self._recv(4)
        size, = struct.unpack("!i", buf.getvalue())
        if size == -1:
            buf = self._recv(8)
            size, = struct.unpack("!Q", buf.getvalue())
        if maxsize is not None and size > maxsize:
            return None
        return self._recv(size)

    Connection._send_bytes = send_bytes
    Connection._recv_bytes = recv_bytes

    logger.info(patchname + " applied")

patch_mp_connection_bpo_17560()
###################
# End patch code
###################

import multiprocessing

if multiprocessing.connection.Connection._send_bytes.__code__.co_filename == __file__:
  print("Think patch worked")
else:
  print("Patch not detected")

all_results = {}
idty_scale = 0.001

def tensor_cleaning(g, det_threshold=1e-15):
  print('num below thresh:', torch.sum(torch.det(g)<=det_threshold))
  g[torch.det(g)<=det_threshold] = idty_scale * torch.eye((3))
  # Sylvester's criterion https://en.wikipedia.org/wiki/Sylvester%27s_criterion
  psd_map = torch.where(g[...,0,0]>0, 1, 0) + torch.where(torch.det(g[...,:2,:2])>0, 1, 0) + torch.where(torch.det(g)>0, 1, 0)
  nonpsd_idx = torch.where(psd_map!=3)
  # nonpsd_idx = torch.where(torch.isnan(torch.sum(batch_cholesky(g), (3,4))))
  print('num nonpsd:',len(nonpsd_idx[0]))
  for i in range(len(nonpsd_idx[0])):
    g[nonpsd_idx[0][i], nonpsd_idx[1][i], nonpsd_idx[2][i]] = idty_scale * torch.eye((3))
  return g


def torch_compute_Ebin_dist_from_met(met1, met2, num_in_mask, out_dist_file):
    """Compute the Ebin distance between metric field 1 and metric field 2, write distance image out to file
    """
    try:
      print('Computing squared Ebin distance')
      with torch.no_grad():
        dist_field = Squared_distance_Ebin_field(met1, met2, 1./3,
                                                 torch.Tensor(np.ones((met1.shape[0],met1.shape[1],met1.shape[2]))))
        print('num nan:', torch.sum(torch.isnan(dist_field)), 'for out file', out_dist_file)
        #torch.nan_to_num(dist_field, 10000000000, out=dist_field)
        # threshold the log distance to 7 to get rid of large errors on the outside border of brain due to poor masking for DTITK
        thresh = (np.exp(7)-1)**2
        torch.nan_to_num(dist_field, 0, out=dist_field)
        dist_field_thresh = torch.clone(dist_field)
        dist_field_thresh[dist_field >= thresh] = 0
        #dist = torch.einsum("hwd->", dist_field) / num_in_mask
        dist = torch.einsum("hwd->", dist_field_thresh) / num_in_mask
        dist = torch.log(torch.sqrt(dist)+1).detach().numpy()
        #dist_field = torch.log(torch.sqrt(dist_field)+1)
        dist_field = torch.log(torch.sqrt(dist_field_thresh)+1)
    except Exception as err:
      print('Caught error', err, 'while computing Ebin distance between metrics for out file', out_dist_file)
      dist = -1
      return(dist)

    try:
      WriteScalarNPArray(dist_field.detach().numpy(), out_dist_file)
    except Exception as err:
      print('Caught error', err, 'while writing Ebin distance field between metrics to file', out_dist_file)

    print('returning', dist, 'from compute_Ebin_dist')  
    return(dist)
# end torch_compute_Ebin_dist_from_met

def compute_Ebin_dist(tens1_file, mask1_file, tens2_file, mask2_file, out_dist_file):
    """Compute the Ebin distance between tensor field 1 and tensor field 2, write distance image out to file
    """
    try:
      print('Reading in tensor and mask files')
      if tens1_file[-4:]  == ".nii" or tens1_file[-7:] == ".nii.gz":
        tens_nib = nib.load(tens1_file)
        tens_np = tens_nib.get_fdata().squeeze()
        print('Swapping index 2 and 3 in tensor to match DTITK expectations.',  'DO NOT DO THIS if starting w/ NIFTI file w/ expected tensor ordering!')
        # Note dtitk is also off by one voxel in the x direction.  dtitk[0] = atlas[1]
        tens1 = tens_np.copy()
        #tens1[1:] = 0.001 * tens_np[0:-1]
        tens1[:,:,:,2] = tens_np[:,:,:,3]
        tens1[:,:,:,3] = tens_np[:,:,:,2]
      else:
        tens1 = ReadTensors(tens1_file)
        
      if tens2_file[-4:]  == ".nii" or tens2_file[-7:] == ".nii.gz":
        tens_nib = nib.load(tens2_file)
        tens_np = tens_nib.get_fdata().squeeze()
        print('Swapping index 2 and 3 in tensor to match DTITK expectations.',  'DO NOT DO THIS if starting w/ NIFTI file w/ expected tensor ordering!')
        # Note dtitk is also off by one voxel in the x direction.  dtitk[0] = atlas[1]
        tens2 = tens_np.copy()
        #tens2[1:] = 0.001 * tens_np[0:-1]
        tens2[:,:,:,2] = tens_np[:,:,:,3]
        tens2[:,:,:,3] = tens_np[:,:,:,2]
      else:
        tens2 = ReadTensors(tens2_file)
        
      mask1 = ReadScalars(mask1_file)
      mask2 = ReadScalars(mask2_file)

      print('tens1 shape:', tens1.shape, 'mask1 shape:', mask1.shape, 'num in mask:', np.sum(mask1))
      print('tens2 shape:', tens1.shape, 'mask2 shape:', mask1.shape, 'num in mask:', np.sum(mask2))
      
      met1 = np.zeros((tens1.shape[0], tens1.shape[1], tens1.shape[2], 3, 3))
      met2 = np.zeros((tens2.shape[0], tens2.shape[1], tens2.shape[2], 3, 3))
      num_in_mask = 0
      for xx in range(tens1.shape[0]):
        for yy in range(tens1.shape[1]):
          for zz in range(tens1.shape[2]):
            if mask1[xx,yy,zz] and mask2[xx,yy,zz]:
              met1[xx,yy,zz,0,0] = tens1[xx,yy,zz,0]
              met1[xx,yy,zz,0,1] = tens1[xx,yy,zz,1]
              met1[xx,yy,zz,1,0] = tens1[xx,yy,zz,1]
              met1[xx,yy,zz,0,2] = tens1[xx,yy,zz,2]
              met1[xx,yy,zz,2,0] = tens1[xx,yy,zz,2]
              met1[xx,yy,zz,1,1] = tens1[xx,yy,zz,3]
              met1[xx,yy,zz,1,2] = tens1[xx,yy,zz,4]
              met1[xx,yy,zz,2,1] = tens1[xx,yy,zz,4]
              met1[xx,yy,zz,2,2] = tens1[xx,yy,zz,5]
              met2[xx,yy,zz,0,0] = tens2[xx,yy,zz,0]
              met2[xx,yy,zz,0,1] = tens2[xx,yy,zz,1]
              met2[xx,yy,zz,1,0] = tens2[xx,yy,zz,1]
              met2[xx,yy,zz,0,2] = tens2[xx,yy,zz,2]
              met2[xx,yy,zz,2,0] = tens2[xx,yy,zz,2]
              met2[xx,yy,zz,1,1] = tens2[xx,yy,zz,3]
              met2[xx,yy,zz,1,2] = tens2[xx,yy,zz,4]
              met2[xx,yy,zz,2,1] = tens2[xx,yy,zz,4]
              met2[xx,yy,zz,2,2] = tens2[xx,yy,zz,5]
              num_in_mask += 1
            else:
              met1[xx,yy,zz,0,0] = 1 * idty_scale
              met1[xx,yy,zz,1,1] = 1 * idty_scale
              met1[xx,yy,zz,2,2] = 1 * idty_scale
              met2[xx,yy,zz,0,0] = 1 * idty_scale
              met2[xx,yy,zz,1,1] = 1 * idty_scale
              met2[xx,yy,zz,2,2] = 1 * idty_scale

      clean_met1 = tensor_cleaning(torch.Tensor(met1),1e-21)        
      clean_met2 = tensor_cleaning(torch.Tensor(met2),1e-21)
      #clean_met1 = torch.Tensor(met1)        
      #clean_met2 = torch.Tensor(met2) 
      #print('num met1:', np.sum(met1<1), 'num met2:', np.sum(met2<1), 'while computing Ebin distance between', tens1_file, 'and', tens2_file)
      #print('max det1:', torch.max(torch.det(torch.Tensor(met1))), 'max det2:', torch.max(torch.det(torch.Tensor(met2))), 'while computing Ebin distance between', tens1_file, 'and', tens2_file)
      #print('num clean met1:', torch.sum(clean_met1<1), 'num clean met2:', torch.sum(clean_met2<1), 'while computing Ebin distance between', tens1_file, 'and', tens2_file)
      
      dist = torch_compute_Ebin_dist_from_met(torch.inverse(clean_met1),
                                              torch.inverse(clean_met2),
                                              num_in_mask, out_dist_file)        
    except Exception as err:
      print('Caught error', err, 'while computing Ebin distance between', tens1_file, 'and', tens2_file)
      dist = -1
      return((dist, tens1_file, tens2_file))

    print('returning', dist, 'from compute_Ebin_dist')  
    return((dist, tens1_file, tens2_file))
# end compute_Ebin_dist

def compute_Ebin_dist_subj(tens1_file, mask1_file, diffeo_file, tens2_file, mask2_file, out_dist_file):
    """Compute the Ebin distance between tensor field 1 and tensor field 2 after deforming tensor field 1 by diffeo, write distance image out to file
    """
    try:
      print('Reading in tensor and mask files')
      if tens1_file[-4:]  == ".nii" or tens1_file[-7:] == ".nii.gz":
        tens_nib = nib.load(tens1_file)
        tens_np = tens_nib.get_fdata().squeeze()
        print('Swapping index 2 and 3 in tensor to match DTITK expectations.',  'DO NOT DO THIS if starting w/ NIFTI file w/ expected tensor ordering!')
        # Note dtitk is also off by one voxel in the x direction.  dtitk[0] = atlas[1]
        tens1 = tens_np.copy()
        #tens1[1:] = tens_np[0:-1]
        tens1[:,:,:,2] = tens_np[:,:,:,3]
        tens1[:,:,:,3] = tens_np[:,:,:,2]
      else:
        tens1 = ReadTensors(tens1_file)
        
      if tens2_file[-4:]  == ".nii" or tens2_file[-7:] == ".nii.gz":
        tens_nib = nib.load(tens2_file)
        tens_np = tens_nib.get_fdata().squeeze()
        print('Swapping index 2 and 3 in tensor to match DTITK expectations.',  'DO NOT DO THIS if starting w/ NIFTI file w/ expected tensor ordering!')
        # Note dtitk is also off by one voxel in the x direction.  dtitk[0] = atlas[1]
        tens2 = tens_np.copy()
        #tens2[1:] = 0.001 * tens_np[0:-1]
        tens2[:,:,:,2] = tens_np[:,:,:,3]
        tens2[:,:,:,3] = tens_np[:,:,:,2]
        disp = nib.load(diffeo_file).get_fdata().squeeze()
        if disp.shape[-1] == 3:
          disp = disp.transpose((3,0,1,2))
        diffeo = get_idty_3d(disp.shape[1],disp.shape[2], disp.shape[3]) + torch.Tensor(disp)
      else:
        tens2 = ReadTensors(tens2_file)
        diffeo = sio.loadmat(diffeo_file)['diffeo']
        
      if diffeo.shape[-1] == 3:
        diffeo = diffeo.transpose((3,0,1,2))
        
      mask1 = ReadScalars(mask1_file)
      mask2 = ReadScalars(mask2_file)

      print('tens1 shape:', tens1.shape, 'mask1 shape:', mask1.shape, 'diffeo shape:', diffeo.shape)
      print('tens2 shape:', tens2.shape, 'mask2 shape:', mask2.shape)
      
      met1 = np.zeros((tens1.shape[0], tens1.shape[1], tens1.shape[2], 3, 3))
      met2 = np.zeros((tens2.shape[0], tens2.shape[1], tens2.shape[2], 3, 3))
      num_in_mask = 0
      for xx in range(tens1.shape[0]):
        for yy in range(tens1.shape[1]):
          for zz in range(tens1.shape[2]):
            if mask1[xx,yy,zz] and mask2[xx,yy,zz]:
              met1[xx,yy,zz,0,0] = tens1[xx,yy,zz,0]
              met1[xx,yy,zz,0,1] = tens1[xx,yy,zz,1]
              met1[xx,yy,zz,1,0] = tens1[xx,yy,zz,1]
              met1[xx,yy,zz,0,2] = tens1[xx,yy,zz,2]
              met1[xx,yy,zz,2,0] = tens1[xx,yy,zz,2]
              met1[xx,yy,zz,1,1] = tens1[xx,yy,zz,3]
              met1[xx,yy,zz,1,2] = tens1[xx,yy,zz,4]
              met1[xx,yy,zz,2,1] = tens1[xx,yy,zz,4]
              met1[xx,yy,zz,2,2] = tens1[xx,yy,zz,5]
              met2[xx,yy,zz,0,0] = tens2[xx,yy,zz,0]
              met2[xx,yy,zz,0,1] = tens2[xx,yy,zz,1]
              met2[xx,yy,zz,1,0] = tens2[xx,yy,zz,1]
              met2[xx,yy,zz,0,2] = tens2[xx,yy,zz,2]
              met2[xx,yy,zz,2,0] = tens2[xx,yy,zz,2]
              met2[xx,yy,zz,1,1] = tens2[xx,yy,zz,3]
              met2[xx,yy,zz,1,2] = tens2[xx,yy,zz,4]
              met2[xx,yy,zz,2,1] = tens2[xx,yy,zz,4]
              met2[xx,yy,zz,2,2] = tens2[xx,yy,zz,5]
              num_in_mask += 1
            else:
              met1[xx,yy,zz,0,0] = 1 * idty_scale
              met1[xx,yy,zz,1,1] = 1 * idty_scale
              met1[xx,yy,zz,2,2] = 1 * idty_scale
              met2[xx,yy,zz,0,0] = 1 * idty_scale
              met2[xx,yy,zz,1,1] = 1 * idty_scale
              met2[xx,yy,zz,2,2] = 1 * idty_scale
      
      clean_met1 = tensor_cleaning(torch.Tensor(met1))        
      clean_met2 = tensor_cleaning(torch.Tensor(met2))        
      in_met = torch.inverse(clean_met1)
      with torch.no_grad():
        met_atlas_space = phi_pullback_3d(torch.Tensor(diffeo), in_met)
        met_atlas_space[mask1<0.3] = torch.eye(3) / idty_scale # since inverted already
        met_atlas_space[mask2<0.3] = torch.eye(3) / idty_scale

      dist = torch_compute_Ebin_dist_from_met(met_atlas_space, torch.inverse(clean_met2), num_in_mask, out_dist_file)        
    except Exception as err:
      print('Caught error', err, 'while computing Ebin distance between', tens1_file, 'and', tens2_file)
      dist = -1
      return((dist, tens1_file, tens2_file))

    print('returning', dist, 'from compute_Ebin_dist')  
    return((dist, tens1_file, tens2_file))
# end compute_Ebin_dist_subj

def compute_log_jac_det_subj(diffeo_file, jac_file):
    """Compute the log Jacobian determinant of diffeomorphism, write log jac det image out to file
    """
    try:
      print('Reading in diffeomorphism file')
      if diffeo_file[-4:]  == ".nii" or diffeo_file[-7:] == ".nii.gz":
        disp = nib.load(diffeo_file).get_fdata().squeeze()
        if disp.shape[-1] == 3:
          disp = disp.transpose((3,0,1,2))
        diffeo = get_idty_3d(disp.shape[1],disp.shape[2], disp.shape[3]) + torch.Tensor(disp) 
      else:
        diffeo = sio.loadmat(diffeo_file)['diffeo']
      if diffeo.shape[-1] == 3:
        print('transposing diffeo from shape:',diffeo.shape)
        diffeo = diffeo.transpose((3,0,1,2))


      # algo expects diffeo to be shape 3 x H x W x D
        print('diffeo shape:', diffeo.shape)
      det = get_jacobian_determinant(torch.Tensor(diffeo)).detach().numpy()
      log_det = np.log(det)
      WriteScalarNPArray(log_det, jac_file)
    except Exception as err:
      print('Caught error', err, 'while computing log Jacobian determinant of', diffeo_file)
    return()
# end compute_log_jac_det_subj    

# From https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
def shifted_data_variance(data):
    if len(data) < 2:
        return 0.0
    K = data[0]
    n = Ex = Ex2 = 0.0
    for x in data:
        n = n + 1
        Ex += x - K
        Ex2 += (x - K) * (x - K)
    variance = (Ex2 - (Ex * Ex) / n) / (n - 1)
    # use n instead of (n-1) if want to compute the exact variance of the given data
    # use (n-1) if data are samples of a larger population
    return variance

K = n = Ex = Ex2 = 0.0

def add_variable(x):
    global K, n, Ex, Ex2
    if n == 0:
        K = x
    n += 1
    Ex += x - K
    Ex2 += (x - K) * (x - K)

def remove_variable(x):
    global K, n, Ex, Ex2
    n -= 1
    Ex -= x - K
    Ex2 -= (x - K) * (x - K)

def get_mean():
    global K, n, Ex
    return K + Ex / n

def get_variance():
    global n, Ex, Ex2
    return (Ex2 - (Ex * Ex) / n) / (n - 1)
# End Wikipedia variance algorithms

def compute_variances(img_list, out_var_file, out_mean_file):
  """ Compute the mean and variances of each voxel across the image list."""
  # use the stable shifted data variance algorithm from
  # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
  num_imgs = len(img_list)
  K = ReadScalars(img_list[0])
  Ex = np.zeros_like(K)
  Ex2 = np.zeros_like(K)
  for idx in range(1,num_imgs):
    img_minus_K = ReadScalars(img_list[idx]) - K
    Ex += img_minus_K
    Ex2 += np.square(img_minus_K)
  mean = K + Ex / num_imgs
  var = (Ex2 - np.square(Ex) / num_imgs) / (num_imgs-1)
  WriteScalarNPArray(mean, out_mean_file)
  WriteScalarNPArray(var, out_var_file)
  return()
# end compute_variances    

def collect_result(result):
  # Expect a distance.
  print('collected distance between', result[1], 'and', result[2], ':', result[0])
  all_results[result[1] + ',' + result[2]] = result[0]

def empty_collect_result(result):
  # Do nothing
  print('in empty_collect_result')

  
if __name__ == "__main__":

  host = platform.node()
  if ('ardashir' in host) or ('lakota' in host) or ('kourosh' in host):
    pool = multiprocessing.Pool(7)
  elif 'beast' in host:
    pool = multiprocessing.Pool(7) # split into 4 batches to avoid hitting swap space
  else:
    print('Unknown host,', host, ' defaulting to 1 processes.  Increase if on capable host')
    pool = multiprocessing.Pool(1) # split into 1 batches to avoid hitting swap space

  
  subjs = []
  subjs.append('105923')
  subjs.append('108222')
  subjs.append('102715')
  subjs.append('100206')
  subjs.append('104416')
  subjs.append('107422')
  indir = '/usr/sci/projects/HCP/Kris/NSFCRCNS/prepped_UKF_data_with_grad_dev/'
  outsubjdir = f'/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/UKF_experiments/BallResults/'
  bval = 1000
  bval = "all"
  region_masks = []
  single_masks = []
  ars = []

  hd_septatlasname = 'BrainAtlasUkfBallImgMetDirectRegSept10'
  hd_janatlasname = 'BrainAtlasMetImg'
  hd_shuffleatlasname = 'BrainAtlasMetImgShuffle'
  dtitk_atlasname = 'DTITK_atlas'
  #dtitk_dir = '/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/UKF_experiments/BallResults/DTITKReg_wrongscale/'
  dtitk_dir = '/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/UKF_experiments/BallResults/DTITKReg/'

  atlas_names = []
  atlases = []
  masks = []
  atlas_names.append(hd_septatlasname)
  atlases.append(f'/home/sci/hdai/Projects/Atlas3D/output/{hd_septatlasname}/atlas_tens_phi_inv_img_met_rreg_800.nhdr')
  masks.append(f'/home/sci/hdai/Projects/Atlas3D/output/{hd_septatlasname}/atlas_mask_phi_inv_img_met_rreg_800.nhdr')
  
  atlas_names.append(hd_janatlasname)
  atlases.append(f'/home/sci/hdai/Projects/Atlas3D/output/{hd_janatlasname}/atlas_tens_phi_inv_met_rreg_800.nhdr')
  masks.append(f'/home/sci/hdai/Projects/Atlas3D/output/{hd_janatlasname}/atlas_mask_phi_inv_met_rreg_800.nhdr')

  atlas_names.append(hd_shuffleatlasname)
  atlases.append(f'/home/sci/hdai/Projects/Atlas3D/output/{hd_shuffleatlasname}/atlas_tens_phi_inv_met_rreg_800.nhdr')
  masks.append(f'/home/sci/hdai/Projects/Atlas3D/output/{hd_shuffleatlasname}/atlas_mask_phi_inv_met_rreg_800.nhdr')

  atlas_names.append(dtitk_atlasname)
  #atlases.append(f'{dtitk_dir}/mean_diffeomorphic_initial6_orig_dims_scaled_scaled.nii.gz')
  atlases.append(f'{dtitk_dir}/mean_diffeomorphic_initial6_orig_dims_scaled.nii.gz')
  # use same mask as septatlas since dtitk atlas has been registered to the same space
  #masks.append(f'/home/sci/hdai/Projects/Atlas3D/output/{hd_septatlasname}/atlas_mask_phi_inv_img_met_rreg_800.nhdr')
  #masks.append(f'{dtitk_dir}/mean_diffeomorphic_initial6_orig_dims_scaled_tr_mask.nii.gz')
  masks.append(f'{dtitk_dir}/mean_diffeomorphic_initial6_orig_dims_tr_mask.nii.gz')

  out_dist_dir = f'/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/atlas_distances/'
  pathlib.Path(out_dist_dir).mkdir(exist_ok=True)

  for aidx in range(len(atlases)):
    for bidx in range(aidx+1,len(atlases)):
      outfile = out_dist_dir + 'dist_' + atlas_names[aidx] + '_and_' + atlas_names[bidx] + '.nhdr'
      ar = pool.apply_async(compute_Ebin_dist, args=(atlases[aidx], masks[aidx], atlases[bidx], masks[bidx], outfile), callback=collect_result)
      ars.append(ar)
  
  # end for each atlas combination

  # Now compute distances between each subject and each atlas
  subj_dist_files = {}
  subj_jac_files = {}
  for atlas in atlas_names:
    subj_dist_files[atlas] = []
    subj_jac_files[atlas] = []

  for subj in subjs:
    #subj_tens = f'{outsubjdir}/{subj}_scaled_orig_tensors_rreg_v2.nhdr'
    subj_tens = f'{outsubjdir}/{subj}_scaled_orig_tensors_rreg.nhdr'
    subj_mask = f'{indir}/{subj}/dti_{bval}_FA_mask_0.20_rreg.nhdr'
    # use phi_inv to transform images and metrics from subj to atlas space
    # use phi to transform images and metrics from atlas to subj space
    # use phi to transform points from subj to atlas space
    # use phi_inv to transform points from atlas to subj space
    diffeos = []
    diffeos.append(f'/home/sci/hdai/Projects/Atlas3D/output/{hd_septatlasname}/{subj}_phi_inv.mat')
    diffeos.append(f'/home/sci/hdai/Projects/Atlas3D/output/{hd_janatlasname}/{subj}_phi_inv.mat')
    diffeos.append(f'/home/sci/hdai/Projects/Atlas3D/output/{hd_shuffleatlasname}/{subj}_phi_inv.mat')
    diffeos.append(f'{dtitk_dir}{subj}_padded_aff_aff_diffeo_orig_dims.df.nii.gz')
    
    for aidx in range(len(atlases)):
      outfile = out_dist_dir + 'dist_' + subj + '_and_' + atlas_names[aidx] + '.nhdr'
      subj_dist_files[atlas_names[aidx]].append(outfile)
      if atlas_names[aidx] == dtitk_atlasname:
        #subj_tens_file = f'{outsubjdir}/{subj}_scaled_orig_tensors_rreg.nhdr'
        #subj_tens_file = f'{outsubjdir}/{subj}_orig_tensors.nhdr'
        subj_tens_file = f'{dtitk_dir}/{subj}_padded_aff_aff_diffeo_orig_dims_scaled.nii.gz'
        ar = pool.apply_async(compute_Ebin_dist, args=(subj_tens_file, masks[aidx], atlases[aidx], masks[aidx], outfile), callback=collect_result)
        ars.append(ar)
      else:
        #ar = pool.apply_async(compute_Ebin_dist_subj, args=(subj_tens, subj_mask, diffeos[aidx], atlases[aidx], masks[aidx], outfile), callback=collect_result)
        # use atlas mask for fairer comparison
        ar = pool.apply_async(compute_Ebin_dist_subj, args=(subj_tens, masks[aidx], diffeos[aidx], atlases[aidx], masks[aidx], outfile), callback=collect_result)
        ars.append(ar)

      jacfile = out_dist_dir + f'log_jac_det_{subj}_atlas_' + atlas_names[aidx] + '.nhdr'
      subj_jac_files[atlas_names[aidx]].append(jacfile)
      ar = pool.apply_async(compute_log_jac_det_subj, args=(diffeos[aidx], jacfile), callback=empty_collect_result)
      ars.append(ar)

  for ar in ars:
    ar.wait()

  ars = []
  # Once all the distances and jacobian determinants have been calculated, compute the variances across subjects
  for aidx in range(len(atlases)):
    outmeanfile = out_dist_dir + 'dist_mean_' + atlas_names[aidx] + '.nhdr'
    outfile = out_dist_dir + 'dist_variances_' + atlas_names[aidx] + '.nhdr'
    ar = pool.apply_async(compute_variances, args=(subj_dist_files[atlas_names[aidx]],outfile,outmeanfile), callback=empty_collect_result)
    ars.append(ar)
    jacmeanfile = out_dist_dir + 'log_jac_det_mean_' + atlas_names[aidx] + '.nhdr'
    jacfile = out_dist_dir + 'log_jac_det_variances_' + atlas_names[aidx] + '.nhdr'
    ar = pool.apply_async(compute_variances, args=(subj_jac_files[atlas_names[aidx]],jacfile,jacmeanfile), callback=empty_collect_result)
    ars.append(ar)

  for ar in ars:
    ar.wait()
   
  pool.close()
  pool.join()

  print('all results:', all_results)

  # Save results to file

  fname = f'{out_dist_dir}atlas_Ebin_distances.pkl'
  with open(fname,'wb') as f:
    print('writing results to file:', fname)
    pickle.dump(all_results, f)

