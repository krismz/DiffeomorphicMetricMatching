import gzip
import _pickle as pickle
import vtk
import numpy as np
import scipy.io as sio
from data.io import readRaw, ReadScalars, ReadTensors, WriteTensorNPArray, WriteScalarNPArray
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

    WriteScalarNPArray(img_atlas_space, output_fname)
  except Exception as err:
    print('Caught', err, 'while applying transform', diffeo_fname, 'to image', input_fname)

def do_masking(geodir, ingeofile, outgeoprefix, atlasdir, maskprefix, spacing, origin, thresh=0):
  try:
    maskfile = atlasdir + f'{maskprefix}.nhdr'
    print('Masking geo in directory:', geodir, 'from file:', ingeofile,
          'with mask', maskfile)
    
    # correct to world spacing since processing was all done in voxel space [1,1,1]
    # Note, do this for comparison to UKF tractography
    # Make sure to adjust tensor images etc back to this spacing when displaying
    # with these paths
    # Also correct origin to line up with UKF tractography (0,0,0) should be center of brain

    mask = ReadScalars(maskfile)

    mask_indices = []

    inpd = wma.io.read_polydata(geodir + ingeofile)

    # loop over lines
    inpd.GetLines().InitTraversal()
    num_lines = inpd.GetNumberOfLines()

    ptids = vtk.vtkIdList()
    inpoints = inpd.GetPoints()

    for lidx in range(num_lines):
      inpd.GetLines().GetNextCell(ptids)

      keep_curr_fiber_mask = False
      in_mask = False

      for ptnum in range(ptids.GetNumberOfIds()):
        ptid = ptids.GetId(ptnum)
        pt = inpoints.GetPoint(ptid)
        ptconvert = [int(np.floor((pt[0] - origin[0]) / spacing[0])),
                     int(np.floor((pt[1] - origin[1]) / spacing[1])),
                     int(np.floor((pt[2] - origin[2]) / spacing[2]))]
        if mask[ptconvert[0],ptconvert[1],ptconvert[2]] > thresh:
          in_mask = True

        if in_mask:
          keep_curr_fiber_mask = True
          break
      # end for each point in the line
      
      if keep_curr_fiber_mask:
        mask_indices.append(lidx)

    # end for each line

    
    fiber_mask = np.zeros(num_lines)
    fiber_mask[mask_indices] = 1

    print('Done finding fiber masks')

    outpd = wma.filter.mask(inpd, fiber_mask)
    fname = (f'{geodir}/{outgeoprefix}{maskprefix}.vtp')

    print("Writing data to file", fname, "...")

    wma.io.write_polydata(outpd, fname)
    print("Wrote output", fname)
    del outpd
    
    del inpd

  except Exception as err:
    print("Caught error", err, "while masking geos in ", geodir)

def do_region_masking(geodir, ingeofile, outgeoprefix, atlasdir, maskprefix, spacing, origin, thresh=0):
  # assumes maskprefix refers to 4 masks of form
  # maskprefix_hemi1_start, mask_prefix_hemi1_mid, maskprefix_hemi2_start, mask_prefix_hemi2_mid
  try:
    
    print('Doing region masking in directory:', geodir, 'from file:', ingeofile,
          'with mask', atlasdir+maskprefix)
    
    # correct to world spacing since processing was all done in voxel space [1,1,1]
    # Note, do this for comparison to UKF tractography
    # Make sure to adjust tensor images etc back to this spacing when displaying
    # with these paths
    # Also correct origin to line up with UKF tractography (0,0,0) should be center of brain

    #mask1_start = ReadScalars(atlasdir + '/CST_hemi1_start.nhdr')
    #mask1_mid = ReadScalars(atlasdir + '/CST_hemi1_mid.nhdr')
    #mask2_start = ReadScalars(atlasdir + '/CST_hemi2_start.nhdr')
    #mask2_mid = ReadScalars(atlasdir + '/CST_hemi2_mid.nhdr')
    mask1_start = ReadScalars(atlasdir + f'/{maskprefix}_hemi1_start.nhdr')
    mask1_mid = ReadScalars(atlasdir + f'/{maskprefix}_hemi1_mid.nhdr')
    mask2_start = ReadScalars(atlasdir + f'/{maskprefix}_hemi2_start.nhdr')
    mask2_mid = ReadScalars(atlasdir + f'/{maskprefix}_hemi2_mid.nhdr')

    mask1_indices = []
    mask2_indices = []

    inpd = wma.io.read_polydata(geodir + ingeofile)

    # loop over lines
    inpd.GetLines().InitTraversal()
    num_lines = inpd.GetNumberOfLines()

    ptids = vtk.vtkIdList()
    inpoints = inpd.GetPoints()

    for lidx in range(num_lines):
      inpd.GetLines().GetNextCell(ptids)

      keep_curr_fiber_mask1 = False
      keep_curr_fiber_mask2 = False
      in_mask1_start = False
      in_mask1_mid = False
      in_mask2_start = False
      in_mask2_mid = False

      for ptnum in range(ptids.GetNumberOfIds()):
        ptid = ptids.GetId(ptnum)
        pt = inpoints.GetPoint(ptid)
        ptconvert = [int(np.floor((pt[0] - origin[0]) / spacing[0])),
                     int(np.floor((pt[1] - origin[1]) / spacing[1])),
                     int(np.floor((pt[2] - origin[2]) / spacing[2]))]
        if mask1_start[ptconvert[0],ptconvert[1],ptconvert[2]] > thresh:
          in_mask1_start = True
        if mask1_mid[ptconvert[0],ptconvert[1],ptconvert[2]] > thresh:
          in_mask1_mid = True
        if mask2_start[ptconvert[0],ptconvert[1],ptconvert[2]] > thresh:
          in_mask2_start = True
        if mask2_mid[ptconvert[0],ptconvert[1],ptconvert[2]] > thresh:
          in_mask2_mid = True

        if in_mask1_start and in_mask1_mid:
          keep_curr_fiber_mask1 = True
          
        if in_mask2_start and in_mask2_mid:
          keep_curr_fiber_mask2 = True

        # mask1 and mask2 are mutually exclusive, so as soon as fiber is kept for one
        # that fiber won't be kept for the other
        # TODO add a check to confirm that mask1 and mask2 don't overlap and that
        # they're in separate hemispheres.  Or change the or in the if below to and
        # for a more general (and slower) masking setup
        if keep_curr_fiber_mask1 or keep_curr_fiber_mask2:
          break
      # end for each point in the line
      
      if keep_curr_fiber_mask1:
        mask1_indices.append(lidx)

      if keep_curr_fiber_mask2:
        mask2_indices.append(lidx)
    # end for each line

    
    fiber_mask1 = np.zeros(num_lines)
    fiber_mask2 = np.zeros(num_lines)
    fiber_mask1[mask1_indices] = 1
    fiber_mask2[mask2_indices] = 1

    print('Done finding fiber masks')

    outpd1 = wma.filter.mask(inpd, fiber_mask1)
    fname1 = (f'{geodir}/{outgeoprefix}{maskprefix}_hemi1.vtp')

    print("Writing data to file", fname1, "...")

    wma.io.write_polydata(outpd1, fname1)
    print("Wrote output", fname1)
    del outpd1
    
    outpd2 = wma.filter.mask(inpd, fiber_mask2)
    fname2 = (f'{geodir}/{outgeoprefix}{maskprefix}_hemi2.vtp')
    
    print("Writing data to file", fname2, "...")
    wma.io.write_polydata(outpd2, fname2)
    print("Wrote output", fname2)
    del outpd2
    
    del inpd

  except Exception as err:
    print("Caught error", err, "while masking geos in ", geodir)
      
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
  
  geodirs = []
  ingeofiles = []
  outgeoprefixes = []
  atlasdirs = []
  region_masks = []
  single_masks = []
  do_atlas = False
  print("DO ATLAS:", do_atlas)
  if do_atlas:
    #geodirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/Ball_scaledorig_6subj/')
    #prefixes.append('Ball_scaledorig_')
    #geodirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/Ball_joint_img_6subj/')
    #prefixes.append('Ball_joint_img_')
    #geodirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/Ball_met_dom_6subj/Preprocessed_100k/')
    #ingeofiles.append('Ball_met_dom_6subj_geodesics_pp.vtp')
    #outgeoprefixes.append('Ball_met_dom_6subj_geodesics_pp_masked_')
    #atlasdirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/atlases/BrainAtlasUkfBallMetDominated/')
    #geodirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/Ball_met_img_mask_6subj/')
    #prefixes.append('Ball_met_img_mask_6subj_')
    #geodirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/Ball_met_img_brainmask_v2_6subj/Preprocessed_100k/')
    #atlasdirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/atlases/BrainAtlasUkfBallImgMetBrainMaskSept3/')
    #ingeofiles.append('Ball_met_img_brainmask_6subj_geodesics_pp.vtp')
    #outgeoprefixes.append('Ball_met_img_brainmask_6subj_geodesics_pp_masked_')
    geodirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/Ball_met_img_rigid_6subj/Preprocessed_100k/')
    atlasdirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/atlases/BrainAtlasUkfBallImgMetDirectRegSept10/')
    ingeofiles.append('Ball_met_img_rigid_6subj_geodesics_pp.vtp')
    outgeoprefixes.append('Ball_met_img_rigid_6subj_geodesics_pp_masked_')
    #region_masks.append('CST')
    #region_masks.append('Cingulum_Cor')
    #region_masks.append('Cingulum_Sag')
    ##region_masks.append('SLFI')
    ##region_masks.append('SLFII')
    ##region_masks.append('SLFIII')
    #region_masks.append('SLF')
    ##region_masks.append('Post_Thalamus')
    ##single_masks.append('Thalamus_seed')
    #single_masks.append('AC_seed')
    #single_masks.append('CC_seed')
    #single_masks.append('CC_genu_seed')
    #
    #region_masks.append('CST_v2')
    #region_masks.append('Cingulum_Cor_v2')
    #region_masks.append('Cingulum_Sag_v2')
    #region_masks.append('SLF_v2')
    #region_masks.append('CC_genu')
    #single_masks.append('AC_v2_seed')
    #single_masks.append('CC_v2_seed')
    #single_masks.append('CC_genu_v2_seed')
    #
    region_masks.append('CST_v3')
    region_masks.append('Cing_cor_v3')
    region_masks.append('SLF_v3')
    single_masks.append('AC_v3_seed')
    single_masks.append('CC_v3_seed')
    single_masks.append('CC_genu_thick_seed')
    single_masks.append('CC_genu_thin_seed')

    ars = []
    for geodir, geofile, geoprefix, atlasdir in zip(geodirs, ingeofiles, outgeoprefixes, atlasdirs):
      for rmask in region_masks:
        ar = pool.apply_async(do_region_masking, args=(geodir, geofile, geoprefix, atlasdir, rmask, spacing, origin), callback=collect_result)
        ars.append(ar)
      for smask in single_masks:
        ar = pool.apply_async(do_masking, args=(geodir, geofile, geoprefix, atlasdir, smask, spacing, origin), callback=collect_result)
        ars.append(ar)

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
    #
    #region_masks.append('CST_v2')
    #region_masks.append('Cingulum_Cor_v2')
    #region_masks.append('Cingulum_Sag_v2')
    #region_masks.append('SLF_v2')
    #region_masks.append('CC_genu')
    #single_masks.append('AC_v2_seed')
    #single_masks.append('CC_v2_seed')
    #single_masks.append('CC_genu_v2_seed')
    #
    region_masks.append('CST_v3')
    region_masks.append('Cing_cor_v3')
    region_masks.append('SLF_v3')
    single_masks.append('AC_v3_seed')
    single_masks.append('CC_v3_seed')
    single_masks.append('CC_genu_thick_seed')
    single_masks.append('CC_genu_thin_seed')
    bval = 1000
    bval = 'all'
    #atlas_geo_dir = '/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/b1000_UKF/'
    #atlas_geo_dir = '/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/b1000_UKF_orig_scaled_tens/'
    #atlas_geo_dir = f'/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/b{bval}_UKF_orig_scaled_tens/Preprocessed_100k/'
    atlas_geo_dir = f'/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/b{bval}_UKF_orig_scaled_tens_rreg_v2/Preprocessed_100k/'
    ars = []
    for subj in subjs:
      # WARNING!  This code only allows one of each of the following per subject!  
      geodirs.append(atlas_geo_dir)
      ingeofiles.append(subj + f'_{bval}_2masks_geodesics_pp.vtp')
      #atlasdirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/atlases/BrainAtlasUkfBallImgMetBrainMaskSept3/')
      atlasdirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/atlases/BrainAtlasUkfBallImgMetDirectRegSept10/')
      outgeoprefixes.append(subj + f'_{bval}_2masks_geodesics_pp_masked_')
      
    for geodir, geofile, geoprefix, atlasdir, subj in zip(geodirs, ingeofiles, outgeoprefixes, atlasdirs, subjs):
      outatlasdir = atlasdir + 'subjspace/'
      for rmask in region_masks:
        ar = pool.apply_async(do_region_masking, args=(geodir, geofile, geoprefix, outatlasdir, f'{subj}_{rmask}', spacing, origin), callback=collect_result)
        ars.append(ar)
      for smask in single_masks:
        ar = pool.apply_async(do_masking, args=(geodir, geofile, geoprefix, outatlasdir, f'{subj}_{smask}', spacing, origin), callback=collect_result)
        ars.append(ar)
    # end do_atlas, else

  print("All tasks launched, waiting for completion")
  for ar in ars:
    ar.wait()

  print("All waits returned, closing and joining")
  pool.close()
  pool.join()
  

  
