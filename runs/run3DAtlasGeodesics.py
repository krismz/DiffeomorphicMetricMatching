import os
from algo import geodesic
from lazy_imports import np
from lazy_imports import torch
from data.io import ReadTensors, ReadScalars, WriteTensorNPArray, WriteScalarNPArray
import gzip
import _pickle as pickle
import math
from functools import partial
import platform
import pathlib

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

def get_paths(offs, tens=None, seed_mask=None, end_mask=None, Gamma1=None, Gamma2=None, Gamma3=None, fileprefix=None):
  atlas_geos = []
  init_velocity = None
  xseedmin = 0
  xseedmax = seed_mask.shape[0]-1
  yseedmin = 0
  yseedmax = seed_mask.shape[1]-1
  zseedmin = 0
  zseedmax = seed_mask.shape[2]-1

  max_coords_at_once = 4000 #161000
  full_atlas_start_coords = []
  for xx in np.linspace(xseedmin, xseedmax, num=seed_mask.shape[0]):
    for yy in np.linspace(yseedmin, yseedmax, num=seed_mask.shape[1]):
      for zz in np.linspace(zseedmin, zseedmax, num=seed_mask.shape[2]):
        if seed_mask[math.floor(xx),math.floor(yy),math.floor(zz)] > 0.5:
          full_atlas_start_coords.append([xx+offs[0],yy+offs[1],zz+offs[2]])

  print('Processing offs(', offs[0], offs[1], offs[2], '), ', len(full_atlas_start_coords), 'geodesic paths to find.')

  atlas_geos = []
  num_paths = len(full_atlas_start_coords)
  num_blocks = math.floor(num_paths / max_coords_at_once)

  geo_delta_t = 0.1#0.01#0.005
  geo_iters = 3000 # 22000 for Kris annulus(delta_t=0.005), 32000 for cubic (delta_t=0.005)

  for block in range(num_blocks):
    geox, geoy, geoz = geodesic.batch_geodesicpath_3d(tens, end_mask,
                                                      full_atlas_start_coords[block*max_coords_at_once:(block+1)*max_coords_at_once], init_velocity,
                                                      geo_delta_t, iter_num=geo_iters, both_directions=True,
                                                      Gamma1=Gamma1, Gamma2=Gamma2, Gamma3=Gamma3)
    atlas_geos.append((geox, geoy, geoz))
  # now get last partial block
  geox, geoy, geoz = geodesic.batch_geodesicpath_3d(tens, end_mask,
                                                    full_atlas_start_coords[num_blocks*max_coords_at_once:], init_velocity,
                                                    geo_delta_t, iter_num=geo_iters, both_directions=True,
                                                    Gamma1=Gamma1, Gamma2=Gamma2, Gamma3=Gamma3)
  atlas_geos.append((geox, geoy, geoz))

  # Write to file
  with gzip.open(f'{fileprefix}{offs[0]}_{offs[1]}_{offs[2]}.pkl.gz','wb') as f:
    fname = f'{fileprefix}{offs[0]}_{offs[1]}_{offs[2]}.pkl.gz'
    print('writing results to file:', fname)
    pickle.dump(atlas_geos, f)

  del atlas_geos
  return([])

def collect_result(result):
  # Right now, empty list expected.
  print('collected result')

def compute_geodesics(atlas_tens_4_path, atlas_mask, end_mask, offset, fileprefix, pool):
  # Since calling batch_geodesic_3d multiple times, precompute gammas
  Gamma1, Gamma2, Gamma3 = geodesic.compute_gammas_3d(atlas_tens_4_path, atlas_mask)

  atlas_geos = []
  get_paths_tens = partial(get_paths, tens=atlas_tens_4_path, seed_mask=atlas_mask, end_mask=end_mask,
                           Gamma1=Gamma1, Gamma2=Gamma2, Gamma3=Gamma3, fileprefix=fileprefix)

  #pool.map_async(get_paths_tens, offs, callback=collect_result)
  for offs in offset:
    ar = pool.apply_async(get_paths_tens, args=(offs,), callback=collect_result)
  return(ar)



if __name__ == "__main__":
  steps = [0.00,0.33,0.67]
  steps = [0.00,0.33]
  offs = []
  for xoffs in steps:
    for yoffs in steps:
      for zoffs in steps:
        #if ((xoffs == 0.33 and yoffs == 0.33 and zoffs == 0.67) or
        #    (xoffs == 0.33 and yoffs == 0.67 and zoffs == 0.33) or
        #    (xoffs == 0.67 and yoffs == 0.67 and zoffs == 0.33)):
        offs.append((xoffs,yoffs,zoffs))

  host = platform.node()
  if ('ardashir' in host) or ('lakota' in host) or ('kourosh' in host):
    pool = multiprocessing.Pool(48) # 8 or 16 for atlas, 48 for subjs
  elif 'beast' in host:
    pool = multiprocessing.Pool(6) # split into 4 batches to avoid hitting swap space
  else:
    print('Unknown host,', host, ' defaulting to 6 processes.  Increase if on capable host')
    pool = multiprocessing.Pool(6) # split into 4 batches to avoid hitting swap space
    
  do_atlas=False
  subj_atlas_space=True
  print('DO ATLAS:', do_atlas)
  if not do_atlas:
    print("SUBJ_ATLAS_SPACE:", subj_atlas_space)
  if do_atlas:
    atlasdirs = []
    atlas_geo_dirs = []
    atlasnames = []
    ars = []
    #atlasname = atlasdir.split('/')[-2]
    #atlas_geo_dirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/b1000_UKF_6subj/')
    #atlasdirs.append('/home/sci/hdai/Projects/Atlas3D/output/BrainAtlasUkfB1000Aug17/')
    #atlasnames.append('B1000_6subj')
    #atlas_geo_dirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/ball_UKF_6subj/')
    #atlasdirs.append('/home/sci/hdai/Projects/Atlas3D/output/BrainAtlasUkfBallAug16/')
    #atlasnames.append('Ball_6subj')
    #atlas_geo_dirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/b1000_UKF_1/')
    #atlasdirs.append('/home/sci/hdai/Projects/Atlas3D/output/BrainAtlasUkf1B1000Aug13/')
    #atlasnames.append('Ukf1B1000Aug13')
    #atlas_geo_dirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/b1000_UKF_2/')
    #atlasdirs.append('/home/sci/hdai/Projects/Atlas3D/output/BrainAtlasUkf2B1000Aug13/')
    #atlasnames.append('Ukf2B1000Aug13')
    #atlas_geo_dirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/Ball_scaledorig_final_6subj/')
    #atlasdirs.append('/home/sci/hdai/Projects/Atlas3D/output/BrainAtlasUkfBallAug27ScaledOrig/')
    #atlasnames.append('Ball_scaledorig_6subj')
    #atlas_geo_dirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/Ball_unsmoothed_6subj/')
    #atlasdirs.append('/home/sci/hdai/Projects/Atlas3D/output/BrainAtlasUkfBallAug27Unsmoothed/')
    #atlasnames.append('Ball_unsmoothed_6subj')
    #atlas_geo_dirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/Ball_joint_img_6subj/')
    #atlasdirs.append('/home/sci/hdai/Projects/Atlas3D/output/BrainAtlasUkfBallImgAug28/')
    #atlasnames.append('Ball_joint_img_6subj')
    #atlas_geo_dirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/Ball_met_dom_6subj/')
    #atlasdirs.append('/home/sci/hdai/Projects/Atlas3D/output/BrainAtlasUkfBallMetDominated/')
    #atlasnames.append('Ball_met_dom_6subj')
    #atlas_geo_dirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/Ball_met_img_mask_6subj/')
    #atlasdirs.append('/home/sci/hdai/Projects/Atlas3D/output/BrainAtlasUkfBallImgMetBrainMaskSept1/')
    #atlasnames.append('Ball_met_img_mask_6subj')
    #atlas_geo_dirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/Ball_met_img_brainmask_v2_6subj/')
    #atlasdirs.append('/home/sci/hdai/Projects/Atlas3D/output/BrainAtlasUkfBallImgMetBrainMaskSept3/')
    #atlasnames.append('Ball_met_img_brainmask_6subj')
    #atlas_geo_dirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/Ball_met_dom_brainmask_v2_6subj/')
    #atlasdirs.append('/home/sci/hdai/Projects/Atlas3D/output/BrainAtlasUkfBallMetDominatedBrainMaskSept3/')
    #atlasnames.append('Ball_met_dom_brainmask_6subj')
    #atlas_geo_dirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/Ball_met_img_brainmask_iter0_6subj/')
    #atlasdirs.append('/home/sci/hdai/Projects/Atlas3D/output/BrainAtlasUkfBallImgMetBrainMaskSept3/')
    #atlasnames.append('Ball_met_img_brainmask_iter0_6subj')
    #atlas_geo_dirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/Ball_met_dom_brainmask_iter0_6subj/')
    #atlasdirs.append('/home/sci/hdai/Projects/Atlas3D/output/BrainAtlasUkfBallMetDominatedBrainMaskSept3/')
    #atlasnames.append('Ball_met_dom_brainmask_iter0_6subj')
    atlas_geo_dirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/Ball_met_img_rigid_6subj/')
    atlasdirs.append('/home/sci/hdai/Projects/Atlas3D/output/BrainAtlasUkfBallImgMetDirectRegSept10/')
    atlasnames.append('Ball_met_img_rigid_6subj')
    for atlas_geo_dir, atlasdir, atlasname in zip(atlas_geo_dirs, atlasdirs, atlasnames):
      pathlib.Path(atlas_geo_dir).mkdir(exist_ok=True) 
      niters = 800
      if atlasname == 'Ball_joint_img_6subj':
        niters = 500
      elif atlasname == 'Ball_met_dom_6subj':
        niters = 700
      elif atlasname == 'Ball_met_dom_brainmask_iter0_6subj':
        niters = 0
      elif atlasname == 'Ball_met_img_brainmask_iter0_6subj':
        niters = 0

      if ((atlasname == 'Ball_met_img_brainmask_6subj') or
         (atlasname == 'Ball_scaledorig_6subj') or
         (atlasname == 'Ball_met_img_rigid_6subj')):
        print('Computing geodesics for', atlasname, 'using tensors from', atlasdir + f'atlas_tens_phi_inv.nhdr')
        atlas_tens = ReadTensors(atlasdir + f'atlas_tens_phi_inv.nhdr')
        atlas_mask = ReadScalars(atlasdir + f'atlas_mask_phi_inv.nhdr')
      else:
        print('Computing geodesics for', atlasname, 'using tensors from', atlasdir + f'atlas_{niters}_tens.nhdr')
        atlas_tens = ReadTensors(atlasdir + f'atlas_{niters}_tens.nhdr')
        atlas_mask = ReadScalars(atlasdir + f'atlas_{niters}_mask.nhdr')
      atlas_mask[atlas_mask < 0.3] = 0
      atlas_mask[atlas_mask > 0] = 1
      end_mask = atlas_mask
      atlas_tens_4_path = np.transpose(atlas_tens,(3,0,1,2))
      ar = compute_geodesics(atlas_tens_4_path, atlas_mask, end_mask, offs, f'{atlas_geo_dir}{atlasname}_geos_', pool)
      ars.append(ar)
    for ar in ars:
      ar.wait()

  else:
    subjs = []
    subjs.append('105923')
    subjs.append('108222')
    subjs.append('102715')
    subjs.append('100206')
    subjs.append('104416')
    subjs.append('107422')
    bval = 1000
    bval = 'all'
    indir = '/usr/sci/projects/HCP/Kris/NSFCRCNS/prepped_UKF_data_with_grad_dev/'
    #outdir = '/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/working_3d_python/'
    #atlas_geo_dir = '/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/'
    outdir = f'/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/UKF_experiments/B{bval}Results/'
    outatlasdir = f'/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/atlases/BrainAtlasUkfBallImgMetDirectRegSept10/'
    #atlas_geo_dir = '/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/b1000_UKF/'
    atlas_geo_dir = f'/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/b{bval}_UKF_orig_scaled_tens_rreg/'
    #atlas_geo_dir = f'/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/b{bval}_UKF_orig_scaled_tens_rreg_v2/'
    if subj_atlas_space:
      outatlasdir = f'/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/atlases/BrainAtlasUkfBallImgMetDirectRegSept10/'  
      atlas_geo_dir = f'/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/b{bval}_UKF_orig_scaled_tens_rreg_atlas_space/'
    pathlib.Path(atlas_geo_dir).mkdir(exist_ok=True) 
    ars = []
    for subj in subjs:
      atlasname = subj + f'_{bval}_2masks_'

      if subj_atlas_space:
        atlas_tens = ReadTensors(f'{outatlasdir}/{subj}_scaled_orig_tensors_rreg_atlas_space.nhdr')
        #atlas_mask = ReadScalars(f'{outdir}/{subj}_filt_mask.nhdr')
        atlas_mask = ReadScalars(f'{outatlasdir}/{subj}_FA_mask_rreg_atlas_space.nhdr')
        end_mask = ReadScalars(f'{outatlasdir}/{subj}_FA_mask_0.20_rreg_atlas_space.nhdr')
      else:
        atlas_tens = ReadTensors(f'{outdir}/{subj}_scaled_orig_tensors_rreg.nhdr')
        #atlas_mask = ReadScalars(f'{outdir}/{subj}_filt_mask.nhdr')
        atlas_mask = ReadScalars(f'{indir}/{subj}/dti_{bval}_FA_mask_rreg.nhdr')
        end_mask = ReadScalars(f'{indir}/{subj}/dti_{bval}_FA_mask_0.20_rreg.nhdr')
      atlas_tens_4_path = np.transpose(atlas_tens,(3,0,1,2))
      atlas_mask[atlas_mask < 0.3] = 0
      end_mask[end_mask < 0.3] = 0
      atlas_mask[atlas_mask > 0] = 1
      end_mask[end_mask > 0] = 1
      ar = compute_geodesics(atlas_tens_4_path, atlas_mask, end_mask, offs, f'{atlas_geo_dir}{atlasname}_geos_', pool)
      ars.append(ar)
    for ar in ars:
      ar.wait()
    # end if do_atlas, else
    
  pool.close()
  pool.join()
      


  
  
