import gzip
import _pickle as pickle
import vtk
# Doing direct import of torch here because of some incompatibility between lazy_import and torch_sym3eig
import torch
# importing Sym3Eig here to avoid issue with finding libc10.so
from torch_sym3eig import Sym3Eig 
import numpy as np
import scipy.io as sio
from data.io import readRaw, ReadScalars, ReadTensors, WriteTensorNPArray, WriteScalarNPArray
from util.diffeo import compose_function_3d, phi_pullback_3d
import os
import platform
import whitematteranalysis as wma
from data.io import ReadScalars
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

    img_atlas_space[img_atlas_space > 0.1] = 1
    img_atlas_space[img_atlas_space < 0.9] = 0
    WriteScalarNPArray(img_atlas_space, output_fname)
  except Exception as err:
    print('Caught', err, 'while applying transform', diffeo_fname, 'to image', input_fname)

     
def collect_result(result):
  # Right now, empty list expected.
  print('collected result')


if __name__ == "__main__":
  print('DO NOT RUN this code using VTK 9.0 if want to read fibers in with Slicer 4.11, use VTK 8.* instead')
  
  host = platform.node()
  if ('ardashir' in host) or ('lakota' in host) or ('kourosh' in host):
    pool = multiprocessing.Pool(6)
  elif 'beast' in host:
    pool = multiprocessing.Pool(1) # split into 4 batches to avoid hitting swap space
  else:
    print('Unknown host,', host, ' defaulting to 1 process.  Increase if on capable host')
    pool = multiprocessing.Pool(1) # split into 4 batches to avoid hitting swap space

  # vtk tractography are in world spacing, bring back to voxel space for masking, then
  # convert back to world spacing.
  # Make sure to adjust tensor images etc back to this spacing when displaying
  # with these paths
  # Also correct origin to line up with UKF tractography (0,0,0) should be center of brain
  # TODO Confirm that this spacing and origin is appropriate for all subjects or read in from header appropriately
  spacing = [1.25,1.25,1.25]
  origin = [-90,-90.25,-72]
  
  atlasdirs = []
  region_masks = []
  single_masks = []
  diffeofiles = []
  do_atlas = False
  print("DO ATLAS:", do_atlas)
  if do_atlas:
    
    print('Nothing to do for atlas')

  else:
    subjs = []
    subjs.append('105923')
    subjs.append('108222')
    subjs.append('102715')
    subjs.append('100206')
    subjs.append('104416')
    subjs.append('107422')

    #region_masks.append('CST')
    #region_masks.append('Cingulum_Cor')
    #region_masks.append('Cingulum_Sag')
    ##region_masks.append('Cingulum')
    ##region_masks.append('SLFI')
    ##region_masks.append('SLFII')
    ##region_masks.append('SLFIII')
    #region_masks.append('SLF')
    ##region_masks.append('Post_Thalamus')
    ##single_masks.append('Thalamus_seed')
    #single_masks.append('AC_seed')
    #single_masks.append('CC_seed')
    #single_masks.append('CC_genu_seed')
    #region_masks.append('CST_v2')
    #region_masks.append('Cingulum_Cor_v2')
    #region_masks.append('Cingulum_Sag_v2')
    #region_masks.append('SLF_v2')
    #region_masks.append('CC_genu')
    #single_masks.append('AC_v2_seed')
    #single_masks.append('CC_v2_seed')
    #single_masks.append('CC_genu_v2_seed')
    region_masks.append('CST_v3')
    region_masks.append('Cing_cor_v3')
    region_masks.append('SLF_v3')
    single_masks.append('AC_v3_seed')
    single_masks.append('CC_v3_seed')
    single_masks.append('CC_genu_thick_seed')
    single_masks.append('CC_genu_thin_seed')
    bval = 1000
    bval = 'all'
    ars = []
    for subj in subjs:
      # WARNING!  This code only allows one of each of the following per subject!  
      #atlasdirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/atlases/BrainAtlasUkfBallImgMetBrainMaskSept3/')
      atlasdirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/atlases/BrainAtlasUkfBallImgMetDirectRegSept10/')
      # use phi_inv to transform images and metrics from subj to atlas space
      # use phi to transform images and metrics from atlas to subj space
      #diffeofiles.append(f'/home/sci/hdai/Projects/Atlas3D/output/BrainAtlasUkfBallImgMetBrainMaskSept3/{subj}_phi.mat')
      diffeofiles.append(f'/home/sci/hdai/Projects/Atlas3D/output/BrainAtlasUkfBallImgMetDirectRegSept10/{subj}_phi.mat')
      
    for atlasdir, diffeofile, subj in zip(atlasdirs, diffeofiles, subjs):
      outatlasdir = atlasdir + 'subjspace/'
      pathlib.Path(outatlasdir).mkdir(exist_ok=True) 

      for rmask in region_masks:
        ar = pool.apply_async(apply_transform_to_img, args=(atlasdir + f'/{rmask}_hemi1_start.nhdr', diffeofile, outatlasdir+f'/{subj}_{rmask}_hemi1_start.nhdr'), callback=collect_result)
        ars.append(ar)
        ar = pool.apply_async(apply_transform_to_img, args=(atlasdir + f'/{rmask}_hemi1_mid.nhdr', diffeofile, outatlasdir+f'/{subj}_{rmask}_hemi1_mid.nhdr'), callback=collect_result)
        ars.append(ar)
        ar = pool.apply_async(apply_transform_to_img, args=(atlasdir + f'/{rmask}_hemi2_start.nhdr', diffeofile, outatlasdir+f'/{subj}_{rmask}_hemi2_start.nhdr'), callback=collect_result)
        ars.append(ar)
        ar = pool.apply_async(apply_transform_to_img, args=(atlasdir + f'/{rmask}_hemi2_mid.nhdr', diffeofile, outatlasdir+f'/{subj}_{rmask}_hemi2_mid.nhdr'), callback=collect_result)
        ars.append(ar)
                              
      for smask in single_masks:
        ar = pool.apply_async(apply_transform_to_img, args=(atlasdir + f'/{smask}.nhdr', diffeofile, outatlasdir+f'/{subj}_{smask}.nhdr'), callback=collect_result)
        ars.append(ar)

    # end do_atlas, else

  print("All tasks launched, waiting for completion")
  for ar in ars:
    ar.wait()

  print("All waits returned, closing and joining")
  pool.close()
  pool.join()
  

  
