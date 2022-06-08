# Apply a diffeomorphism to all points in VTK polydata
import pathlib

from lazy_imports import np
import scipy.io as sio
from util.diffeo import coord_register_batch_3d
import vtk
import whitematteranalysis as wma

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



def apply_transform_to_polydata(input_polydata_fname, diffeo_fname, output_polydata_fname, spacing, origin):
  inpd = wma.io.read_polydata(input_polydata_fname)
  diffeo_mat = sio.loadmat(diffeo_fname)

  inpoints = inpd.GetPoints()

  # output and temporary objects
  ptids = vtk.vtkIdList()
  outpd = vtk.vtkPolyData()
  outlines = vtk.vtkCellArray()
  outpoints = vtk.vtkPoints()
                
  # loop over lines
  inpd.GetLines().InitTraversal()
  outlines.InitTraversal()

  print('applying', diffeo_fname, 'to', inpd.GetNumberOfLines(), ' geodesics from', input_polydata_fname,
        'saving results in', output_polydata_fname)
  try:
    for lidx in range(inpd.GetNumberOfLines()):
      inpd.GetLines().GetNextCell(ptids)
  
  
      # get points for each ptid and add to output polydata
      cellptids = vtk.vtkIdList()
  
      numpts = ptids.GetNumberOfIds()
      curx = np.zeros((numpts))
      cury = np.zeros((numpts))
      curz = np.zeros((numpts))
      for pidx in range(numpts):
        pt = inpoints.GetPoint(ptids.GetId(pidx))
        curx[pidx] = (pt[0] - origin[0]) / spacing[0]
        cury[pidx] = (pt[1] - origin[1]) / spacing[1]
        curz[pidx] = (pt[2] - origin[2]) / spacing[2]
  
      try:
        newx, newy, newz = coord_register_batch_3d(curx,cury,curz,diffeo_mat['diffeo'])
      except Exception as err:
        print('Caught', err, 'while trying to transform', numpts, 'points for line', lidx, 'from', input_polydata_fname,
              'using diffeo', diffeo_fname, '. Stopping processing for this data set.')
        del inpd
        del outpd
        return
  
      for pidx in range(numpts):
        idx = outpoints.InsertNextPoint(newx[pidx]*spacing[0]+origin[0],
                                        newy[pidx]*spacing[1]+origin[1],
                                        newz[pidx]*spacing[2]+origin[2])
        cellptids.InsertNextId(idx)
  
      outlines.InsertNextCell(cellptids)
  except Exception as err:
    print('Caught', err, 'while processing', lidx, 'th line from', input_polydata_fname)
    return
    # end for each input line

  del inpd

  # put data into output polydata
  outpd.SetLines(outlines)
  outpd.SetPoints(outpoints)

  print("Writing data to file", output_polydata_fname, "...")
  try:
    wma.io.write_polydata(outpd, output_polydata_fname)
    print("Wrote output", output_polydata_fname)
  except Exception as err:
    print("Caught error", err, "while writing", output_polydata_fname)
  del outpd
# end apply_transform_to_polydata

def collect_result(result):
  # Right now, empty list expected.
  print('collected result')

if __name__ == "__main__":
  print('DO NOT RUN this code using VTK 9.0 if want to read fibers in with Slicer 4.11, use VTK 8.* instead')

  host = platform.node()
  if ('ardashir' in host) or ('lakota' in host) or ('kourosh' in host):
    pool = multiprocessing.Pool(80)
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
  bval = 1000
  bval = "all"
  atlases = []
  atlasprefs = []
  region_masks = []
  single_masks = []
  #atlases.append('BrainAtlasUkfB1000Aug17')
  #atlasprefs.append(f'B1000_6subj')
  #atlases.append('BrainAtlasUkfBallAug16')
  #atlasprefs.append(f'Ball_6subj')
  #atlases.append('BrainAtlasUkf1B1000Aug13')
  #atlasprefs.append(f'Ukf1B1000Aug13')
  #atlases.append('BrainAtlasUkf2B1000Aug13')
  #atlasprefs.append(f'Ukf2B1000Aug13')
  #atlases.append('BrainAtlasUkfBallAug27ScaledOrig')
  #atlasprefs.append('Ball_scaledorig_6subj')
  #atlases.append('BrainAtlasUkfBallMetDominated')
  #atlasprefs.append('Ball_met_dom_6subj')
  #atlases.append('BrainAtlasUkfBallImgMetBrainMaskSept3')
  #atlasprefs.append('Ball_met_img_mask_6subj')
  atlases.append('BrainAtlasUkfBallImgMetDirectRegSept10')
  atlasprefs.append('Ball_met_img_rigid_6subj')
  from_atlas_space = False
  ars = []

  spacing = [1.25,1.25,1.25]
  origin = [-90,-90.25,-72]

  for atlas, atlaspref in zip(atlases, atlasprefs):
    atlas_dir = f'/home/sci/hdai/Projects/Atlas3D/output/{atlas}/'
    if from_atlas_space:
      geo_dir = '/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/atlases/Preprocessed_100k/'
      out_geo_dir = geo_dir + f'tfmed_from_{atlas}_to_subj/'
      pathlib.Path(out_geo_dir).mkdir(exist_ok=True)
      for subj in subjs:
        in_pd_fname = geo_dir + f'{atlaspref}_geodesics_pp.vtp'
        out_pd_fname = out_geo_dir + f'{atlaspref}_to_{subj}_{bval}_2masks_geodesics_pp.vtp'
        # use phi to transform from subj to atlas space
        # use phi_inv to transform from atlas to subj space
        diffeo_fname = atlas_dir + f'{subj}_phi_inv.mat'
        ar = pool.apply_async(apply_transform_to_polydata, args=(in_pd_fname, diffeo_fname, out_pd_fname), callback=collect_result)
        ars.append(ar)
    
    else:
      #geo_dir = '/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/b1000_UKF/Preprocessed_100k/'
      #geo_dir = f'/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/b{bval}_UKF_orig_scaled_tens/Preprocessed_100k/'
      #geo_dir = '/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/b1000_UKF/Preprocessed_100k/'
      #geo_dir = f'/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/b{bval}_UKF_orig_scaled_tens_rreg_v2/'
      geo_dir = f'/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/b{bval}_UKF_orig_scaled_tens_rreg/'
      out_geo_dir = geo_dir + f'tfmed_to_{atlas}/'
      pathlib.Path(out_geo_dir).mkdir(exist_ok=True)

      region_masks.append('CST_v3')
      region_masks.append('Cing_cor_v3')
      region_masks.append('SLF_v3')
      single_masks.append('AC_v3_seed')
      single_masks.append('CC_v3_seed')
      single_masks.append('CC_genu_thick_seed')
      single_masks.append('CC_genu_thin_seed')
    
      for subj in subjs:
        in_pd_fname = geo_dir + f'{subj}_{bval}_2masks_geodesics_pp.vtp'
        out_pd_fname = out_geo_dir + f'{subj}_{bval}_2masks_geodesics_pp.vtp'
        # use phi to transform points from subj to atlas space
        # use phi_inv to transform points from atlas to subj space
        diffeo_fname = atlas_dir + f'{subj}_phi.mat'
        ar = pool.apply_async(apply_transform_to_polydata, args=(in_pd_fname, diffeo_fname, out_pd_fname, spacing, origin), callback=collect_result)
        ars.append(ar)

        for rmask in region_masks:
          for hemi in ["hemi1", "hemi2"]:
            in_pd_fname = geo_dir + f'{subj}_{bval}__geos_{rmask}_{hemi}_geodesics.vtp'
            out_pd_fname = out_geo_dir + f'{subj}_{bval}__geos_{rmask}_{hemi}_geodesics.vtp'
            ar = pool.apply_async(apply_transform_to_polydata, args=(in_pd_fname, diffeo_fname, out_pd_fname, spacing, origin), callback=collect_result)
            ars.append(ar)
        for smask in single_masks:
          in_pd_fname = geo_dir + f'{subj}_{bval}__geos_{smask}_geodesics.vtp'
          out_pd_fname = out_geo_dir + f'{subj}_{bval}__geos_{smask}_geodesics.vtp'
          ar = pool.apply_async(apply_transform_to_polydata, args=(in_pd_fname, diffeo_fname, out_pd_fname, spacing, origin), callback=collect_result)
          ars.append(ar)


  # end for each atlas
  for ar in ars:
    ar.wait()

  pool.close()
  pool.join()
