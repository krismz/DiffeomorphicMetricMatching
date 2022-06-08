# Apply a diffeomorphism to all points in VTK polydata
import pathlib
import torch
from lazy_imports import np
import scipy.io as sio
from util.diffeo import coord_register_batch_3d, get_idty_3d
import vtk
import whitematteranalysis as wma
import nibabel as nib

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



def apply_transform_to_polydata(input_polydata_fname, diffeo, output_polydata_fname, spacing, origin):
  inpd = wma.io.read_polydata(input_polydata_fname)

  inpoints = inpd.GetPoints()

  # output and temporary objects
  ptids = vtk.vtkIdList()
  outpd = vtk.vtkPolyData()
  outlines = vtk.vtkCellArray()
  outpoints = vtk.vtkPoints()
                
  # loop over lines
  inpd.GetLines().InitTraversal()
  outlines.InitTraversal()

  print('applying diffeo to', inpd.GetNumberOfLines(), ' geodesics from', input_polydata_fname,
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
        newx, newy, newz = coord_register_batch_3d(curx,cury,curz,diffeo)
      except Exception as err:
        print('Caught', err, 'while trying to transform', numpts, 'points for line', lidx, 'from', input_polydata_fname,
              '. Stopping processing for this data set.')
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
  subjs.append('100206')
  subjs.append('102715')
  subjs.append('104416')
  subjs.append('105923')
  subjs.append('107422')
  subjs.append('108222')
  bval = 1000
  bval = "all"
  atlases = []
  atlasprefs = []
  region_masks = []
  single_masks = []
  
  from_atlas_space = False
  ars = []

  spacing = [1.25,1.25,1.25]
  origin = [-90,-90.25,-72]
  spacing = [1,1,1]
  origin = [0,0,0]

  hd_atlasname = 'BrainAtlasUkfBallImgMetDirectRegSept10'
  dtitk_atlasname = 'DTITKReg'
  kc_atlasname = 'Ball_met_img_rigid_6subj'
  out_tract_dir = f'/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/{kc_atlasname}_and_subj_tracts/'
  tracts_of_dtitk_subjects_dir = f'{out_tract_dir}dtitk_subj_tracts/'
  tracts_of_phi_of_dtitk_subjects_dir = f'{out_tract_dir}subj_tracts_computed_in_dtitk_atlas_space/'
  dtitkatlasdir = f'/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/UKF_experiments/BallResults/{dtitk_atlasname}/'
  tracts_of_dtitk_subjects_dir = f'{dtitkatlasdir}/'
  tracts_of_phi_of_dtitk_subjects_dir = f'{dtitkatlasdir}/'
  
  pathlib.Path(tracts_of_phi_of_dtitk_subjects_dir).mkdir(exist_ok=True)

  region_masks.append('CST_v3')
  region_masks.append('Cing_cor_v3')
  region_masks.append('SLF_v3')
  single_masks.append('AC_v3_seed')
  single_masks.append('CC_v3_seed')
  single_masks.append('CC_genu_thick_seed')
  single_masks.append('CC_genu_thin_seed')

  for subj in subjs:
    in_pd_fname = tracts_of_dtitk_subjects_dir + f'{subj}_whole_brain_in_subj_space.vtk'
    out_pd_fname = tracts_of_phi_of_dtitk_subjects_dir + f'{subj}_whole_brain_to_atlas_space.vtk'

    # For DTITK use df to transform images and metrics from subj to atlas space
    # For DTITK use df_inv to transform images and metrics from atlas to subj space
    # For DTITK use df_inv to transform points from subj to atlas space
    # For DTITK use df to transform points from atlas to subj space
    diffeo_to_dtitk_atlas_fname = dtitkatlasdir + f'{subj}_padded_combined_aff_aff_diffeo.df_inv.nii.gz'
    disp = nib.load(diffeo_to_dtitk_atlas_fname).get_fdata().squeeze()
    if disp.shape[-1] == 3:
      disp = disp.transpose((3,0,1,2))
    dtitk_diffeo_to_atlas = (get_idty_3d(disp.shape[1],disp.shape[2], disp.shape[3]) + torch.Tensor(disp)).detach().numpy()
    
    ar = pool.apply_async(apply_transform_to_polydata, args=(in_pd_fname, dtitk_diffeo_to_atlas, out_pd_fname, spacing, origin), callback=collect_result)
    ars.append(ar)

    for rmask in region_masks:
      for hemi in ["hemi1", "hemi2"]:
        in_pd_fname = tracts_of_dtitk_subjects_dir + f'{subj}_{rmask}_{hemi}_in_subj_space.vtk'
        out_pd_fname = tracts_of_phi_of_dtitk_subjects_dir + f'{subj}_{rmask}_{hemi}_to_atlas_space.vtk'
        ar = pool.apply_async(apply_transform_to_polydata, args=(in_pd_fname, dtitk_diffeo_to_atlas, out_pd_fname, spacing, origin), callback=collect_result)
        ars.append(ar)
    for smask in single_masks:
      in_pd_fname = tracts_of_dtitk_subjects_dir + f'{subj}_{smask}_in_subj_space.vtk'
      out_pd_fname = tracts_of_phi_of_dtitk_subjects_dir + f'{subj}_{smask}_to_atlas_space.vtk'
      ar = pool.apply_async(apply_transform_to_polydata, args=(in_pd_fname, dtitk_diffeo_to_atlas, out_pd_fname, spacing, origin), callback=collect_result)
      ars.append(ar)


  for ar in ars:
    ar.wait()

  pool.close()
  pool.join()
