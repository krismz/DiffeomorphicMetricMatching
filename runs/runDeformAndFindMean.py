# Apply diffeomorphisms to T1 and orig_scaled_tensors and find mean images for each

import pathlib

# Doing direct import of torch here because of some incompatibility between lazy_import and torch_sym3eig
import torch
# importing Sym3Eig here to avoid issue with finding libc10.so
from torch_sym3eig import Sym3Eig 
from lazy_imports import np
from util.SplitEbinMetric3DCuda import update_karcher_mean
import scipy.io as sio
from data.io import readRaw, ReadScalars, ReadTensors, WriteTensorNPArray, WriteScalarNPArray
from util.diffeo import compose_function_3d, phi_pullback_3d
from util.tensors import tens_3x3_to_tens_6, tens_6_to_tens_3x3, make_pos_def, smooth_tensors

import platform
import traceback


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

def tensor_cleaning(g, det_threshold=1e-15):
    g[torch.det(g)<=det_threshold] = torch.eye((3))
    # Sylvester's criterion https://en.wikipedia.org/wiki/Sylvester%27s_criterion
    psd_map = torch.where(g[...,0,0]>0, 1, 0) + torch.where(torch.det(g[...,:2,:2])>0, 1, 0) + torch.where(torch.det(g)>0, 1, 0)
    nonpsd_idx = torch.where(psd_map!=3)
    # nonpsd_idx = torch.where(torch.isnan(torch.sum(batch_cholesky(g), (3,4))))
    for i in range(len(nonpsd_idx[0])):
        g[nonpsd_idx[0][i], nonpsd_idx[1][i], nonpsd_idx[2][i]] = torch.eye((3))
    return g
  

def apply_transform_to_img(input_fname, diffeo_fname, output_fname):
  # use phi_inv and compose_function to take subj image into atlas space
  try:
    torch.set_default_tensor_type('torch.DoubleTensor')
    print('Applying transform', diffeo_fname, 'to image', input_fname,
          'and saving in', output_fname)
    in_img = ReadScalars(input_fname).astype(float)
    diffeo = sio.loadmat(diffeo_fname)['diffeo']

    with torch.no_grad():
      img_atlas_space = compose_function_3d(torch.from_numpy(in_img), torch.from_numpy(diffeo)).detach().numpy()

    WriteScalarNPArray(img_atlas_space, output_fname)
  except Exception as err:
    print('Caught', err, 'while applying transform', diffeo_fname, 'to image', input_fname)

def apply_transform_to_tensors(input_tens_fname, diffeo_fname, output_tens_fname):
  try:
    torch.set_default_tensor_type('torch.DoubleTensor')
    print('Applying transform', diffeo_fname, 'to tensors', input_tens_fname,
          'and saving in', output_tens_fname)
    in_tens = ReadTensors(input_tens_fname)
    diffeo = sio.loadmat(diffeo_fname)['diffeo']

    in_tens_full = tens_6_to_tens_3x3(in_tens)
    # in_metric = np.linalg.inv(in_tens_full)
    in_tens_full = tensor_cleaning(torch.from_numpy(in_tens_full))
    in_metric = torch.inverse(in_tens_full)

    with torch.no_grad():
      #met_atlas_space = phi_pullback_3d(torch.from_numpy(diffeo), torch.from_numpy(in_metric)).detach().numpy()
      met_atlas_space = phi_pullback_3d(torch.from_numpy(diffeo), in_metric).detach().numpy()

    out_tens_full = np.linalg.inv(met_atlas_space)
    out_tens = tens_3x3_to_tens_6(out_tens_full)

    WriteTensorNPArray(out_tens, output_tens_fname)
  except Exception as err:
    print('Caught', err, 'while applying transform', diffeo_fname, 'to tensors', input_tens_fname)

def collect_result(result):
  # Right now, empty list expected.
  print('collected result')
    
if __name__ == "__main__":
  torch.set_default_tensor_type('torch.DoubleTensor')

  host = platform.node()
  if ('ardashir' in host) or ('lakota' in host) or ('kourosh' in host):
    pool = multiprocessing.Pool(80)
  elif 'beast' in host:
    pool = multiprocessing.Pool(7) # split into 4 batches to avoid hitting swap space
  else:
    print('Unknown host,', host, ' defaulting to 7 processes.  Increase if on capable host')
    pool = multiprocessing.Pool(7) # split into 4 batches to avoid hitting swap space

  
  subjs = []
  subjs.append('105923')
  subjs.append('108222')
  subjs.append('102715')
  subjs.append('100206')
  subjs.append('104416')
  subjs.append('107422')
  dim = 3.
  #bval = 1000
  bval = 'all'
  sigma = 1.5
  prepdir = '/usr/sci/projects/HCP/Kris/NSFCRCNS/prepped_UKF_data_with_grad_dev/'
  indir = f'/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/UKF_experiments/B{bval}Results/'
  atlases = []
  atlasprefs = []
  #atlases.append('BrainAtlasUkfB1000Aug17')
  #atlasprefs.append(f'B1000_6subj')
  #atlases.append('BrainAtlasUkfBallAug16')
  #atlasprefs.append(f'Ball_6subj')
  #atlases.append('BrainAtlasUkfBallAug27Unsmoothed')
  #atlasprefs.append('Ball_unsmoothed_6subj')
  #atlases.append('BrainAtlasUkfBallAug27ScaledOrig')
  #atlasprefs.append('Ball_scaledorig_6subj')
  #atlases.append('BrainAtlasUkfBallMetDominated')
  #atlasprefs.append('Ball_met_dom_6subj')
  #atlases.append('BrainAtlasUkfBallImgMetBrainMaskSept1')
  #atlasprefs.append('Ball_met_img_mask_6subj')
  #atlases.append('BrainAtlasUkfBallImgMetBrainMaskSept3')
  #atlasprefs.append('Ball_met_img_brainmask_6subj')
  #atlases.append('BrainAtlasUkfBallMetDominatedBrainMaskSept3')
  #atlasprefs.append('Ball_met_dom_brainmask_6subj')
  atlases.append('BrainAtlasUkfBallImgMetDirectRegSept10')
  atlasprefs.append('Ball_met_img_rigid_6subj')
 
  # shorten subj list as appropriate before working with following 2 atlases
  #atlases.append('BrainAtlasUkf1B1000Aug13')
  #atlasprefs.append(f'Ukf1B1000Aug13')
  #atlases.append('BrainAtlasUkf2B1000Aug13')
  #atlasprefs.append(f'Ukf2B1000Aug13')
  do_transform=True
  do_tens_mean=False
  
  ars = []

  for atlas, atlaspref in zip(atlases, atlasprefs):
    atlas_dir = f'/home/sci/hdai/Projects/Atlas3D/output/{atlas}/'
    out_atlas_dir = f'/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/atlases/{atlas}/'
    pathlib.Path(out_atlas_dir).mkdir(exist_ok=True) 
    
    for subj in subjs:
      prep_prefix = prepdir + f'{subj}/dti_{bval}'
      in_prefix = indir + f'{subj}'
      out_prefix = out_atlas_dir + f'{subj}'
      #in_t1_fname = in_prefix + f'_T1_flip_y.nhdr'
      in_t1_fname = in_prefix + f'_t1_to_reft1_rreg.nhdr'
      out_t1_fname = out_prefix + f'_T1_rreg_atlas_space.nhdr'
      #in_tens_fname = in_prefix + f'_scaled_orig_tensors.nhdr'
      in_tens_fname = in_prefix + f'_scaled_orig_tensors_rreg.nhdr'
      out_tens_fname = out_prefix + f'_scaled_orig_tensors_rreg_atlas_space.nhdr'
      #in_mask_fname = in_prefix + f'_orig_mask.nhdr'
      in_mask_fname = in_prefix + f'_orig_mask_rreg.nhdr'
      out_mask_fname = out_prefix + f'_orig_mask_rreg_atlas_space.nhdr'
      #in_brain_mask_fname = in_prefix + f'_brain_mask.nhdr'
      in_brain_mask_fname = in_prefix + f'_brain_mask_rreg.nhdr'
      out_brain_mask_fname = out_prefix + f'_brain_mask_rreg_atlas_space.nhdr'
      #in_tens_fname = in_prefix + f'_scaled_unsmoothed_tensors.nhdr'
      #out_tens_fname = out_prefix + f'_scaled_unsmoothed_tensors_atlas_space.nhdr'
      #in_mask_fname = in_prefix + f'_filt_mask.nhdr'
      #out_mask_fname = out_prefix + f'_filt_mask_atlas_space.nhdr'
      in_fa_mask_fname = prep_prefix + f'_FA_mask_rreg.nhdr'
      out_fa_mask_fname = out_prefix + f'_FA_mask_rreg_atlas_space.nhdr'
      in_fa_mask_2_fname = prep_prefix + f'_FA_mask_0.20_rreg.nhdr'
      out_fa_mask_2_fname = out_prefix + f'_FA_mask_0.20_rreg_atlas_space.nhdr'
      
      # use phi_inv to transform images and metrics from subj to atlas space
      # use phi to transform images and metrics from atlas to subj space
      diffeo_fname = atlas_dir + f'{subj}_phi_inv.mat'
      if do_transform:
        ar = pool.apply_async(apply_transform_to_img, args=(in_t1_fname, diffeo_fname, out_t1_fname), callback=collect_result)
        ar = pool.apply_async(apply_transform_to_img, args=(in_mask_fname, diffeo_fname, out_mask_fname), callback=collect_result)
        ar = pool.apply_async(apply_transform_to_img, args=(in_fa_mask_fname, diffeo_fname, out_fa_mask_fname), callback=collect_result)
        ar = pool.apply_async(apply_transform_to_img, args=(in_fa_mask_2_fname, diffeo_fname, out_fa_mask_2_fname), callback=collect_result)
        ar = pool.apply_async(apply_transform_to_img, args=(in_brain_mask_fname, diffeo_fname, out_brain_mask_fname), callback=collect_result)
        ar = pool.apply_async(apply_transform_to_tensors, args=(in_tens_fname, diffeo_fname, out_tens_fname), callback=collect_result)
        ars.append(ar)
      else:
        print("Skipping Transforms!")

    # Once all the data is ready for an atlas, read in and compute means
    for ar in ars:
      ar.wait()

    print('Computing mean T1, mask, and metric for', atlas)
    try:
      num_t1s = len(subjs)
      for subjidx in range(num_t1s):
        print('Updating mean with subject', subjs[subjidx])
        subj = subjs[subjidx]
        out_prefix = out_atlas_dir + f'{subj}'
        out_t1_fname = out_prefix + f'_T1_atlas_space.nhdr'
        out_mask_fname = out_prefix + f'_orig_mask_atlas_space.nhdr'
        out_brain_mask_fname = out_prefix + f'_brain_mask_atlas_space.nhdr'
        out_tens_fname = out_prefix + f'_scaled_orig_tensors_atlas_space.nhdr'
        #out_mask_fname = out_prefix + f'_filt_mask_atlas_space.nhdr'
        #out_tens_fname = out_prefix + f'_scaled_unsmoothed_tensors_atlas_space.nhdr'
      
        if subjidx == 0:
          t1 = ReadScalars(out_t1_fname) # already in atlas space
          t1_mean = np.copy(t1)
          
          mask = ReadScalars(out_mask_fname)
          mask_mean = np.copy(mask)

          brain_mask = ReadScalars(out_brain_mask_fname)
          brain_mask_mean = np.copy(brain_mask)

          if do_tens_mean:
            ones_mask = np.ones_like(t1_mean)
      
            tens = ReadTensors(out_tens_fname) # already in atlas space
            #filt_tens = smooth_tensors(tens, sigma)
            #tens_full = tens_6_to_tens_3x3(filt_tens)
            tens_full = tens_6_to_tens_3x3(tens)
            tens_full[np.linalg.det(tens_full)<=1e-14] = np.eye((3))

            with torch.no_grad():
              met = np.linalg.inv(tens_full)
              #metpsd = make_pos_def(met, ones_mask)
              metric = torch.from_numpy(met)
              
              fore_back_adaptor = torch.where(torch.det(metric)>1e2, 1e-3, 1.)
              metric = torch.einsum('ijk...,lijk->ijk...', metric, fore_back_adaptor.unsqueeze(0))
              metric = make_pos_def(metric, torch.from_numpy(ones_mask))
              #metric[torch.det(metric)<=0] = torch.eye((3))
            
              metric_mean = update_karcher_mean(None, metric,
                                                subjidx, 1./dim)
              metric_mean[torch.det(metric_mean)<=0] = torch.eye((3))
              metric_mean = make_pos_def(metric_mean, ones_mask)

        else:
          t1 = ReadScalars(out_t1_fname) # already in atlas space
          t1_mean = t1_mean + t1

          mask = ReadScalars(out_mask_fname)
          mask_mean = mask_mean + mask

          brain_mask = ReadScalars(out_brain_mask_fname)
          brain_mask_mean = brain_mask_mean + brain_mask

          if do_tens_mean:
            tens = ReadTensors(out_tens_fname) # already in atlas space
            #filt_tens = smooth_tensors(tens, sigma)
            #tens_full = tens_6_to_tens_3x3(filt_tens)
            tens_full = tens_6_to_tens_3x3(tens)
            tens_full[np.linalg.det(tens_full)<=1e-14] = np.eye((3))

            with torch.no_grad():
              met = np.linalg.inv(tens_full)
              #metpsd = make_pos_def(met, ones_mask)
              metric = torch.from_numpy(met)

              fore_back_adaptor = torch.where(torch.det(metric)>1e2, 1e-3, 1.)
              metric = torch.einsum('ijk...,lijk->ijk...', metric, fore_back_adaptor.unsqueeze(0))
              metric = make_pos_def(metric, torch.from_numpy(ones_mask))
              #metric[torch.det(metric)<=0] = torch.eye((3))
            
              metric_mean = update_karcher_mean(metric_mean, metric,
                                                subjidx, 1./dim)

              metric_mean[torch.det(metric_mean)<=0] = torch.eye((3))
              metric_mean = make_pos_def(metric_mean, ones_mask)
            
      # end for each subject
      t1_mean = t1_mean / num_t1s
      mask_mean = mask_mean / num_t1s
      brain_mask_mean = brain_mask_mean / num_t1s
      
      if do_tens_mean: 
        tens_mean_full = np.linalg.inv(metric_mean.detach().numpy())
        tens_mean = tens_3x3_to_tens_6(tens_mean_full)
     
      out_t1_atlas_fname = out_atlas_dir + 'T1_mean.nhdr'
      out_mask_atlas_fname = out_atlas_dir + 'orig_mask_mean.nhdr'
      out_brain_mask_atlas_fname = out_atlas_dir + 'brain_mask_mean.nhdr'
      out_tens_atlas_fname = out_atlas_dir + 'scaled_orig_tensor_mean.nhdr'
      #out_mask_atlas_fname = out_atlas_dir + 'filt_mask_mean.nhdr'
      #out_tens_atlas_fname = out_atlas_dir + 'scaled_unsmoothed_tensor_mean.nhdr'
      if do_tens_mean:
        print('Writing means to files',out_t1_atlas_fname, out_mask_atlas_fname, out_brain_mask_atlas_fname, out_tens_atlas_fname)
      else:
        print('Writing means to files',out_t1_atlas_fname, out_mask_atlas_fname, out_brain_mask_atlas_fname)

      WriteScalarNPArray(t1_mean, out_t1_atlas_fname)
      WriteScalarNPArray(mask_mean, out_mask_atlas_fname)
      WriteScalarNPArray(brain_mask_mean, out_brain_mask_atlas_fname)
      if do_tens_mean:
        WriteTensorNPArray(tens_mean, out_tens_atlas_fname)
    except Exception as err:
      print('Caught exception', err, 'while computing means for', atlas)
      print(traceback.format_exc())
      
  # end for each atlas

  pool.close()
  pool.join()

