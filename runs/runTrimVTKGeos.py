# Shorten VTK geos to fit within mask region

import gzip
import _pickle as pickle
import vtk
import numpy as np
import os
import platform
import whitematteranalysis as wma
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

def do_trimming(geodir, ingeofile, outgeoprefix, maskfile, spacing, origin, thresh=0):
  try:
    
    print('Trimming geos in directory:', geodir, 'from file:', ingeofile,
          'with mask', maskfile)
    
    # correct to world spacing since processing was all done in voxel space [1,1,1]
    # Note, do this for comparison to UKF tractography
    # Make sure to adjust tensor images etc back to this spacing when displaying
    # with these paths
    # Also correct origin to line up with UKF tractography (0,0,0) should be center of brain

    mask = ReadScalars(maskfile)

    mask_indices = []

    inpd = wma.io.read_polydata(geodir + ingeofile)

    outpoints = vtk.vtkPoints()
    outcells = vtk.vtkCellArray()
    outcells.InitTraversal()

    # loop over lines
    inpd.GetLines().InitTraversal()
    num_lines = inpd.GetNumberOfLines()

    ptids = vtk.vtkIdList()
    inpoints = inpd.GetPoints()

    #Preallocate as best we can
    estim_num_pts = inpoints.GetNumberOfPoints()
    estim_num_cells = num_lines
    outpoints.Resize(estim_num_pts)
    outcells.Allocate(outcells.EstimateSize(estim_num_cells, int(estim_num_pts / estim_num_cells)))
    
    for lidx in range(num_lines):
      inpd.GetLines().GetNextCell(ptids)

      outids = vtk.vtkIdList()

      for ptnum in range(ptids.GetNumberOfIds()):
        ptid = ptids.GetId(ptnum)
        pt = inpoints.GetPoint(ptid)
        ptconvert = [int(np.floor((pt[0] - origin[0]) / spacing[0])),
                     int(np.floor((pt[1] - origin[1]) / spacing[1])),
                     int(np.floor((pt[2] - origin[2]) / spacing[2]))]
        if mask[ptconvert[0],ptconvert[1],ptconvert[2]] > thresh:
          vid=outpoints.InsertNextPoint(pt[0], pt[1], pt[2])
          outids.InsertNextId(vid)
        else:
          #line is now outside mask so don't add any more points
          #save this one off and move on to the next line
          break
      # end for each point in the line

      if outids.GetNumberOfIds() > 0:
        outcells.InsertNextCell(outids)

    # end for each line

    print('Done trimming fibers')
    del inpd

    outpd = vtk.vtkPolyData()
    outpd.SetPoints(outpoints)
    outpd.SetLines(outcells)
    fname = (f'{geodir}/{outgeoprefix}_trimmed.vtp')

    print("Writing data to file", fname, "...")

    wma.io.write_polydata(outpd, fname)
    print("Wrote output", fname)
    del outpd
    
  except Exception as err:
    print("Caught error", err, "while masking geos in ", geodir)

      
def collect_result(result):
  # Right now, empty list expected.
  print('collected result')


if __name__ == "__main__":
  print('DO NOT RUN this code using VTK 9.0 if want to read fibers in with Slicer 4.11, use VTK 8.* instead')
  geodirs = []
  ingeofiles = []
  outgeoprefixes = []
  atlasdirs = []
  do_atlas = True
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
    geodirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/Ball_met_img_brainmask_6subj/Preprocessed_100k/')
    ingeofiles.append('Ball_met_img_brainmask_6subj_geodesics_pp.vtp')
    outgeoprefixes.append('Ball_met_img_brainmask_6subj_geodesics_pp')
    # Use following atlasdir for full masking
    atlasdirs.append('/home/sci/hdai/Projects/Atlas3D/output/BrainAtlasUkfBallImgMetBrainMaskSept3/')
    geodirs.append('/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/Ball_met_dom_brainmask_6subj/Preprocessed_100k/')
    ingeofiles.append('Ball_met_dom_brainmask_6subj_geodesics_pp.vtp')
    outgeoprefixes.append('Ball_met_dom_brainmask_6subj_geodesics_pp')
    # Use following atlasdir for full masking
    atlasdirs.append('/home/sci/hdai/Projects/Atlas3D/output/BrainAtlasUkfBallMetDominatedBrainMaskSept3/')
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
    atlas_geo_dir = f'/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/b{bval}_UKF_orig_scaled_tens/'
    for subj in subjs:
      geodirs.append(atlas_geo_dir)
      prefixes.append(subj + f'_{bval}_2masks_')

  # vtk tractography are in world spacing, bring back to voxel space for masking, then
  # convert back to world spacing.
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
  for geodir, geofile, geoprefix, atlasdir in zip(geodirs, ingeofiles, outgeoprefixes, atlasdirs):
    
    maskfile = atlasdir + 'atlas_800_mask.nhdr'
    ar = pool.apply_async(do_trimming, args=(geodir, geofile, geoprefix, maskfile, spacing, origin, 0.3), callback=collect_result)
    ars.append(ar)

  print("All tasks launched, waiting for completion")
  for ar in ars:
    ar.wait()

  print("All waits returned, closing and joining")
  pool.close()
  pool.join()
  

  
