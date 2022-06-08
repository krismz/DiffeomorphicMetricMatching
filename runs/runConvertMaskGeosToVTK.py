import gzip
import _pickle as pickle
import vtk
import os
import platform
import traceback
import whitematteranalysis as wma
import numpy as np
from data.io import ReadScalars


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



def do_conversion_with_mask(geodir, prefix, postfix, atlasdir, maskprefix, spacing, origin, thresh=0):
  try:
    minFiberLength = 40
    maxFiberLength = None
    retainData = False # matches default from wm_preprocess_all.py

    #print('do_conversion_with_mask', prefix, postfix)
    fs = os.listdir(geodir)
    ffs = [f for f in fs if (f[0:len(prefix)] == prefix and f[-len(postfix):] == postfix)]
    maskfile = atlasdir + f'{maskprefix}.nhdr'
    mask = ReadScalars(maskfile)
    print('Converting geos in directory:', geodir, 'from files:', ffs, 'and masking with', maskfile)
    
    # correct to world spacing since processing was all done in voxel space [1,1,1]
    # Note, do this for comparison to UKF tractography
    # Make sure to adjust tensor images etc back to this spacing when displaying
    # with these paths
    # Also correct origin to line up with UKF tractography (0,0,0) should be center of brain
    rng = np.random.default_rng()
    
    
    vtkp = vtk.vtkPoints()
    vtkc = vtk.vtkCellArray()
    vtkc.InitTraversal()
    first_file = True
    for fname in ffs:
      with gzip.open(geodir+fname,'rb') as f:
        geos = pickle.load(f)
      if first_file:
        # Estimate total number of points and cells
        num_files = len(ffs)
        estim_num_cells = num_files * len(geos) * len(geos[0][0])
        estim_max_pts_per_cell = len(geos[0][0][0])
        estim_num_pts = estim_num_cells * estim_max_pts_per_cell
        vtkp.Allocate(estim_num_pts)
        print("Preallocating", estim_num_pts, "points.")
      for b in range(len(geos)):
        for p in range(len(geos[b][0])):
          # only save this geodesic if some part of it crosses through the mask
          ids = vtk.vtkIdList()

          keep_curr_fiber_mask = False

          # TODO, leaving off all last points in path to avoid false connection to 0,0,0
          # Better option would be to detect that case and only toss sometimes

          for idx in range(len(geos[b][0][p])-1):
            if np.abs(geos[b][0][p][idx] - geos[b][0][p][idx+1]) > 1:
              print('Gap found for', fname, 'batch', b, 'geodesic', p, 'pt', idx)
            vid=vtkp.InsertNextPoint(geos[b][0][p][idx]*spacing[0]+origin[0], geos[b][1][p][idx]*spacing[1]+origin[1], geos[b][2][p][idx]*spacing[2]+origin[2])

            # check to see if this fiber passes through the mask at any point
            if mask[int(np.floor(geos[b][0][p][idx])),
                    int(np.floor(geos[b][1][p][idx])),
                    int(np.floor(geos[b][2][p][idx]))] > thresh:
              keep_curr_fiber_mask = True
              
            ids.InsertNextId(vid)
            #vid = vid + 1
          # end for each point

          if (ids.GetNumberOfIds() > 1) and keep_curr_fiber_mask:
            vtkc.InsertNextCell(ids)
        # end for each path in batch
      # end for each batch
      first_file = False
    # end for each file
    
    print('Created', vtkp.GetNumberOfPoints(), 'points and',
          vtkc.GetNumberOfCells(), 'cells')
    
    pd = vtk.vtkPolyData()
    pd.SetPoints(vtkp)
    pd.SetLines(vtkc)
    
    print('Done creating polydata')
    
  except Exception as err:
    print("Caught error", err, "while converting geos in ", geodir)
    print(traceback.format_exc())
      
  fname = (f'{geodir}/{prefix}geodesics.vtp')
  print("Writing data to file", fname, "...")
  try:
    #wma.io.write_polydata(wm3, fname)
    wma.io.write_polydata(pd, fname)
    print("Wrote output", fname)
  except Exception as err:
    print("Caught error", err, "while writing", fname)
    print(traceback.format_exc())

def do_conversion(geodir, prefix, postfix, spacing, origin):
  try:
    minFiberLength = 40
    maxFiberLength = None
    retainData = False # matches default from wm_preprocess_all.py
    #print('do_conversion', prefix, postfix)

    fs = os.listdir(geodir)
    ffs = [f for f in fs if (f[0:len(prefix)] == prefix and f[-len(postfix):] == postfix)]
    print('Converting geos in directory:', geodir, 'from files:', ffs)
    
    # correct to world spacing since processing was all done in voxel space [1,1,1]
    # Note, do this for comparison to UKF tractography
    # Make sure to adjust tensor images etc back to this spacing when displaying
    # with these paths
    # Also correct origin to line up with UKF tractography (0,0,0) should be center of brain
    rng = np.random.default_rng()
    
    
    vtkp = vtk.vtkPoints()
    vtkc = vtk.vtkCellArray()
    vtkc.InitTraversal()
    first_file = True
    for fname in ffs:
      with gzip.open(geodir+fname,'rb') as f:
        geos = pickle.load(f)
      if first_file:
        # Estimate total number of points and cells
        num_files = len(ffs)
        estim_num_cells = num_files * len(geos) * len(geos[0][0])
        estim_max_pts_per_cell = len(geos[0][0][0])
        estim_num_pts = estim_num_cells * estim_max_pts_per_cell
        vtkp.Allocate(estim_num_pts)
        print("Preallocating", estim_num_pts, "points.")
      for b in range(len(geos)):
        for p in range(len(geos[b][0])):
          # only save this geodesic if it's in the randomly chosen index list  
          ids = vtk.vtkIdList()
          # TODO, leaving off all last points in path to avoid false connection to 0,0,0
          # Better option would be to detect that case and only toss sometimes

          for idx in range(len(geos[b][0][p])-1):
            if np.abs(geos[b][0][p][idx] - geos[b][0][p][idx+1]) > 1:
              print('Gap found for', fname, 'batch', b, 'geodesic', p, 'pt', idx)
            vid=vtkp.InsertNextPoint(geos[b][0][p][idx]*spacing[0]+origin[0], geos[b][1][p][idx]*spacing[1]+origin[1], geos[b][2][p][idx]*spacing[2]+origin[2])
              
            ids.InsertNextId(vid)
            #vid = vid + 1
          # end for each point

          if ids.GetNumberOfIds() > 1:
            vtkc.InsertNextCell(ids)
        # end for each path in batch
      # end for each batch
      first_file = False
    # end for each file
    
    print('Created', vtkp.GetNumberOfPoints(), 'points and',
          vtkc.GetNumberOfCells(), 'cells')
    
    pd = vtk.vtkPolyData()
    pd.SetPoints(vtkp)
    pd.SetLines(vtkc)
    
    print('Done creating polydata')
    
  except Exception as err:
    print("Caught error", err, "while converting geos in ", geodir)
    print(traceback.format_exc())
      
  fname = (f'{geodir}/{prefix}geodesics.vtp')
  print("Writing data to file", fname, "...")
  try:
    wma.io.write_polydata(pd, fname)
    print("Wrote output", fname)
  except Exception as err:
    print("Caught error", err, "while writing", fname)
    print(traceback.format_exc())

def collect_result(result):
  # Right now, empty list expected.
  print('collected result')


if __name__ == "__main__":
  print('DO NOT RUN this code using VTK 9.0 if want to read fibers in with Slicer 4.11, use VTK 8.* instead')
  #geodir = '/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/'
  #prefix = 'Brain3AtlasMay24_'
  #fname = f'{geodir}/{prefix}atlas_geos_0.v1.pkl'
  #with open(fname,'rb') as f:
  #  geos = pickle.load(f)
  geodirs = []
  prefixes = []
  region_masks = []
  single_masks = []
  maskprefixes = []
  #prefix = '105923_1000_'
  do_atlas = False
  subj_atlas_space=True
  print('DO ATLAS:', do_atlas)
  if not do_atlas:
    print("SUBJ_ATLAS_SPACE:", subj_atlas_space)
  print("Remember geodesics between a start and mid region will still need to be filtered after the fact")
  if do_atlas:
    #geodirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/b1000_UKF_1/')
    #prefixes.append('Ukf1B1000Aug13_')
    #geodirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/b1000_UKF_2/')
    #prefixes.append('Ukf2B1000Aug13_')
    #geodirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/b1000_UKF_6subj/')
    #prefixes.append('B1000_6subj_')
    #geodirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/ball_UKF_6subj/')
    #prefixes.append('Ball_6subj_')
    #geodirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/Ball_scaledorig_6subj/')
    #prefixes.append('Ball_scaledorig_')
    #geodirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/Ball_unsmoothed_6subj/')
    #prefixes.append('Ball_unsmoothed_')
    #geodirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/Ball_joint_img_6subj/')
    #prefixes.append('Ball_joint_img_')
    #geodirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/Ball_met_dom_6subj/')
    #prefixes.append('Ball_met_dom_6subj_')
    #geodirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/Ball_met_img_mask_6subj/')
    #prefixes.append('Ball_met_img_mask_6subj_')
    #geodirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/Ball_met_img_brainmask_v2_6subj/')
    #prefixes.append('Ball_met_img_brainmask_6subj_')
    #geodirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/Ball_met_img_brainmask_0.5_6subj/')
    #prefixes.append('Ball_met_img_brainmask_6subj_')
    #geodirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/Ball_met_dom_brainmask_v2_6subj/')
    #prefixes.append('Ball_met_dom_brainmask_6subj_')
    #geodirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/Ball_met_dom_brainmask_iter0_6subj/')
    #prefixes.append('Ball_met_dom_brainmask_iter0_6subj_')
    #geodirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/Ball_met_img_brainmask_iter0_6subj/')
    #prefixes.append('Ball_met_img_brainmask_iter0_6subj_')
    atlasdir = '/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/atlases/BrainAtlasUkfBallImgMetDirectRegSept10/'

    geodirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/Ball_met_img_rigid_6subj/')
    prefixes.append('Ball_met_img_rigid_6subj')
    maskprefixes.append('')
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
    #atlas_geo_dir = '/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/b1000_UKF/'
    #atlas_geo_dir = '/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/b1000_UKF_orig_scaled_tens/'
    #atlas_geo_dir = f'/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/b{bval}_UKF_orig_scaled_tens/'
    if subj_atlas_space:
      atlas_geo_dir = f'/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/b{bval}_UKF_orig_scaled_tens_rreg_atlas_space/'
      atlasdir = f'/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/atlases/BrainAtlasUkfBallImgMetDirectRegSept10/'
    else:
      atlas_geo_dir = f'/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/b{bval}_UKF_orig_scaled_tens_rreg/'
      atlasdir = '/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/atlases/BrainAtlasUkfBallImgMetDirectRegSept10/subjspace/'
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
    for subj in subjs:
      geodirs.append(atlas_geo_dir)
      prefixes.append(subj + f'_{bval}_')
      if subj_atlas_space:
        maskprefixes.append('')
      else:
        maskprefixes.append(f'{subj}_')

  postfix = '.pkl.gz'
  # correct to world spacing since processing was all done in voxel space [1,1,1]
  # Note, do this for comparison to UKF tractography
  # Make sure to adjust tensor images etc back to this spacing when displaying
  # with these paths
  # Also correct origin to line up with UKF tractography (0,0,0) should be center of brain
  # TODO Confirm that this spacing and origin is appropriate for all subjects or read in from header appropriately
  spacing = [1.25,1.25,1.25]
  origin = [-90,-90.25,-72]

  host = platform.node()
  if ('ardashir' in host) or ('lakota' in host) or ('kourosh' in host):
    pool = multiprocessing.Pool(6)
  elif 'beast' in host:
    pool = multiprocessing.Pool(1) # split into 4 batches to avoid hitting swap space
  else:
    print('Unknown host,', host, ' defaulting to 7 processes.  Increase if on capable host')
    pool = multiprocessing.Pool(1) # split into 4 batches to avoid hitting swap space

  ars = []
  for geodir, prefix, maskprefix in zip(geodirs, prefixes, maskprefixes):
    for rmask in region_masks:
      ar = pool.apply_async(do_conversion_with_mask, args=(geodir, prefix+f'_geos_{rmask}_hemi1_', postfix, atlasdir, maskprefix+f'{rmask}_hemi1_mid', spacing, origin, 0.3), callback=collect_result)
      ars.append(ar)
      ar = pool.apply_async(do_conversion_with_mask, args=(geodir, prefix+f'_geos_{rmask}_hemi2_', postfix, atlasdir, maskprefix+f'{rmask}_hemi2_mid', spacing, origin, 0.3), callback=collect_result)
      ars.append(ar)
    for smask in single_masks:
      ar = pool.apply_async(do_conversion, args=(geodir, prefix+f'_geos_{smask}_', postfix, spacing, origin), callback=collect_result)
      ars.append(ar)
      
    print("All tasks launched, waiting for completion")
    
  for ar in ars:
    ar.wait()

  print("All waits returned, closing and joining")
  pool.close()
  pool.join()
  

  
