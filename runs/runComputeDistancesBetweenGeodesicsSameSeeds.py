# Apply a diffeomorphism to all points in VTK polydata
import pathlib

from lazy_imports import np
import scipy.io as sio
from util.diffeo import coord_register_batch_3d
import vtk
import whitematteranalysis as wma
import gzip
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

def mean_of_min_dist_of_curves(curve1, curve2):
  min_list = []
  for ii in range(curve1.shape[0]):
    min_list.append(np.min(np.sqrt((curve2[0,:]-curve1[0,ii])**2+(curve2[1,:]-curve1[1,ii])**2+(curve2[2,:]-curve1[2,ii])**2)))
  return np.mean(np.array(min_list))

def corresponding_distance_sets_of_curves(fixed, moving):
    """Compute the distance between corresponding geodesics.  We know they correspond because
they started from the same / equivalent seed points.  Only include paths less than 176 mm long
    """

    spacing = 1.25 # since image is 1.25mm^3
    longest_path_allowed = 176 # to match 220 mm used for DTITK with spacing = 1
    num_paths_too_long = 0
    
    num_batches = len(fixed)
    if num_batches != len(moving):
      print("Error, different number of batches between fixed (", num_batches, ") and moving(", len(moving),".")
      return(-4)
    total_paths = 0
    dist = 0
    for b in range(num_batches):
      if len(fixed[b][0]) != len(moving[b][0]):
        print("Error, different number of paths for batch", b, ", between fixed(", len(fixed[b][0]), ") and moving(", len(moving[b][0]),".")
        return(-5)
      for p in range(len(fixed[b][0])):
        # Find total distance between each pair of fibers.
        # Assumes that fibers of same idx in moving and fixed correspond to each other. But not necessarily same number of points along each fiber
        mean_dist = 0
        num_pts = len(fixed[b][0][p])
        if num_pts > 0:
          prev_pt = (fixed[b][0][p][0],fixed[b][1][p][0],fixed[b][2][p][0])
        path_len = 0
        for ff in range(num_pts):
          path_len += np.sqrt((prev_pt[0]-fixed[b][0][p][ff])**2 +
                              (prev_pt[1]-fixed[b][1][p][ff])**2 +
                              (prev_pt[2]-fixed[b][2][p][ff])**2)
          prev_pt = (fixed[b][0][p][ff],fixed[b][1][p][ff],fixed[b][2][p][ff])
          
          # compute min dist from point in fixed path to all points in corresponding moving path
          if len(moving[b][0][p][:]) > 0:  
            d = np.min(np.sqrt((moving[b][0][p][:]-fixed[b][0][p][ff])**2 +
                               (moving[b][1][p][:]-fixed[b][1][p][ff])**2 +
                               (moving[b][2][p][:]-fixed[b][2][p][ff])**2))
          else:
            d = 0  
          mean_dist += d
        if num_pts > 0:
          mean_dist /= num_pts
        if path_len * spacing <= longest_path_allowed:
          dist += mean_dist
          total_paths += 1
        else:
          num_paths_too_long += 1  

    if total_paths > 0:    
      avg_dist = dist / total_paths

    print('Found', num_paths_too_long, ' paths longer than', longest_path_allowed, 'mm')
    return(avg_dist)

def compute_distance_between_geodesics(input_geo1_fname, input_geo2_fname):
  print('Reading data for', input_geo1_fname, 'and', input_geo2_fname)
  try:
   with gzip.open(input_geo1_fname,'rb') as f:
      geo1 = pickle.load(f)
  except Exception as e:
    print("Error,", e, "while reading file", input_geo1_fname, ".")
    return((-1, input_geo11_fname, input_geo2_fname))
      
  try:  
    with gzip.open(input_geo2_fname,'rb') as f:
      geo2 = pickle.load(f)
  except Exception as e:
    print("Error,", e, "while reading file", input_geo2_fname, ".")
    return((-2, input_geo11_fname, input_geo2_fname))


  print('computing distance between', input_geo1_fname, 'and', input_geo2_fname)
  try:
    dist = corresponding_distance_sets_of_curves(geo1, geo2)
  except Exception as err:
    print('Caught', err, 'while computing distance between', input_geo1_fname, 'and', input_geo2_fname)
    dist = -3

  print('Distance between', input_geo1_fname, 'and', input_geo2_fname, 'is', dist)
  return((dist, input_geo1_fname, input_geo2_fname))
# end compute_distance_between_geodesics

def collect_result(result):
  # Expect a distance.
  print('collected distance between', result[1], 'and', result[2], ':', result[0])
  all_results[result[1] + ',' + result[2]] = (result[0],)
  
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
  region_masks = []
  single_masks = []
  ars = []

  region_masks.append('CST_v3')
  region_masks.append('Cing_cor_v3')
  region_masks.append('SLF_v3')
  single_masks.append('AC_v3_seed')
  single_masks.append('CC_v3_seed')
  single_masks.append('CC_genu_thick_seed')
  single_masks.append('CC_genu_thin_seed')
  hd_atlasname = 'BrainAtlasUkfBallImgMetDirectRegSept10'
  dtitk_atlasname = 'DTITKReg'
  kc_atlasname = 'Ball_met_img_rigid_6subj'
  out_tract_dir = f'/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/{kc_atlasname}_and_subj_tracts/'

  atlas_tract_dir = f'{out_tract_dir}atlas_tracts/'
  # dtitk_atlas_tract_dir = f'{out_tract_dir}dtitk_atlas_tracts/'
  dtitk_atlas_tract_dir = f'/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/UKF_experiments/BallResults/DTITKReg/'
  tracts_of_subjects_dir = f'{out_tract_dir}subj_tracts/'
  phi_of_tracts_of_subjects_dir = f'{out_tract_dir}subj_tracts_deformed_to_atlas_space/'
  tracts_of_phi_of_subjects_dir = f'{out_tract_dir}subj_tracts_computed_in_atlas_space/'
  #tracts_of_dtitk_subjects_dir = f'{out_tract_dir}dtitk_subj_tracts/'
  #phi_of_tracts_of_dtitk_subjects_dir = f'{out_tract_dir}subj_tracts_deformed_to_dtitk_atlas_space/'
  #tracts_of_phi_of_dtitk_subjects_dir = f'{out_tract_dir}subj_tracts_computed_in_dtitk_atlas_space/'
  tracts_of_dtitk_subjects_dir = f'/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/UKF_experiments/BallResults/DTITKReg/'
  phi_of_tracts_of_dtitk_subjects_dir = f'/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/UKF_experiments/BallResults/DTITKReg/'
  tracts_of_phi_of_dtitk_subjects_dir = f'/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/UKF_experiments/BallResults/DTITKReg/'
  pp=''
  #atlas_tract_dir = f'{out_tract_dir}atlas_tracts/Preprocessed_l20_lmax100/'
  #dtitk_atlas_tract_dir = f'{out_tract_dir}dtitk_atlas_tracts/Preprocessed_l20_lmax100/'
  #tracts_of_subjects_dir = f'{out_tract_dir}subj_tracts/Preprocessed_l20_lmax100/'
  #phi_of_tracts_of_subjects_dir = f'{out_tract_dir}subj_tracts_deformed_to_atlas_space/Preprocessed_l20_lmax100/'
  #tracts_of_phi_of_subjects_dir = f'{out_tract_dir}subj_tracts_computed_in_atlas_space/Preprocessed_l20_lmax100/'
  #pp='_pp'

  # Compute distances between T(subject) and T(atlas)
  for subj in subjs:
    subjname = subj + f'_{bval}_'  
    subj_pd_fname = tracts_of_subjects_dir + f'{subjname}_all_geos_geodesics{pp}.pkl.gz'
    atlas_pd_fname = atlas_tract_dir + f'{kc_atlasname}_geos_all_geodesics{pp}.pkl.gz'
    #dtitk_subj_pd_fname = tracts_of_dtitk_subjects_dir + f'{subjname}_all_geos_geodesics{pp}.pkl.gz'
    #dtitk_atlas_pd_fname = dtitk_atlas_tract_dir + f'dtitk_all_geos_geodesics{pp}.pkl.gz'
    dtitk_subj_pd_fname = tracts_of_dtitk_subjects_dir + f'{subj}_whole_brain_in_subj_space.vtk'
    dtitk_atlas_pd_fname = dtitk_atlas_tract_dir + f'dtitk_atlas_whole_brain.vtk'
    
    ar = pool.apply_async(compute_distance_between_geodesics, args=(subj_pd_fname, atlas_pd_fname), callback=collect_result)
    ars.append(ar)
    #ar = pool.apply_async(compute_distance_between_geodesics, args=(dtitk_subj_pd_fname, dtitk_atlas_pd_fname), callback=collect_result)
    #ars.append(ar)

    for rmask in region_masks:
      for hemi in ["hemi1", "hemi2"]:
        subj_pd_fname = tracts_of_subjects_dir + f'{subjname}_geos_{rmask}_{hemi}_geodesics{pp}.pkl.gz'
        atlas_pd_fname = atlas_tract_dir + f'{kc_atlasname}_geos_{rmask}_{hemi}_geodesics{pp}.pkl.gz'
        #dtitk_subj_pd_fname = tracts_of_dtitk_subjects_dir + f'{subjname}_geos_{rmask}_{hemi}_geodesics{pp}.pkl.gz'
        #dtitk_atlas_pd_fname = dtitk_atlas_tract_dir + f'dtitk_geos_{rmask}_{hemi}_geodesics{pp}.pkl.gz'
        dtitk_subj_pd_fname = tracts_of_dtitk_subjects_dir + f'{subj}_{rmask}_{hemi}_in_subj_space.vtk'
        dtitk_atlas_pd_fname = dtitk_atlas_tract_dir + f'dtitk_atlas_{rmask}_{hemi}.vtk'
        ar = pool.apply_async(compute_distance_between_geodesics, args=(subj_pd_fname, atlas_pd_fname), callback=collect_result)
        ars.append(ar)
        #ar = pool.apply_async(compute_distance_between_geodesics, args=(dtitk_subj_pd_fname, dtitk_atlas_pd_fname), callback=collect_result)
        #ars.append(ar)
        # Add following if want to check that we get 0 distances
        #ar = pool.apply_async(compute_distance_between_geodesics, args=(subj_pd_fname, subj_pd_fname), callback=collect_result)
        #ars.append(ar)
        #ar = pool.apply_async(compute_distance_between_geodesics, args=(atlas_pd_fname, atlas_pd_fname), callback=collect_result)
        #ars.append(ar)

    for smask in single_masks:
      subj_pd_fname = tracts_of_subjects_dir + f'{subjname}_geos_{smask}_geodesics{pp}.pkl.gz'
      atlas_pd_fname = atlas_tract_dir + f'{kc_atlasname}_geos_{smask}_geodesics{pp}.pkl.gz'
      #dtitk_subj_pd_fname = tracts_of_dtitk_subjects_dir + f'{subjname}_geos_{smask}_geodesics{pp}.pkl.gz'
      #dtitk_atlas_pd_fname = dtitk_atlas_tract_dir + f'dtitk_geos_{smask}_geodesics{pp}.pkl.gz'
      dtitk_subj_pd_fname = tracts_of_dtitk_subjects_dir + f'{subj}_{smask}_in_subj_space.vtk'
      dtitk_atlas_pd_fname = dtitk_atlas_tract_dir + f'dtitk_atlas_{smask}.vtk'
      ar = pool.apply_async(compute_distance_between_geodesics, args=(subj_pd_fname, atlas_pd_fname), callback=collect_result)
      ars.append(ar)
      #ar = pool.apply_async(compute_distance_between_geodesics, args=(dtitk_subj_pd_fname, dtitk_atlas_pd_fname), callback=collect_result)
      #ars.append(ar)
  # end for each subject
      

  # Compute distances between T(\phi(subject)) and T(atlas)
  for subj in subjs:
    subjname = subj + f'_{bval}_'  
    subj_pd_fname = tracts_of_phi_of_subjects_dir + f'{subjname}_all_geos_geodesics{pp}.pkl.gz'
    atlas_pd_fname = atlas_tract_dir + f'{kc_atlasname}_geos_all_geodesics{pp}.pkl.gz'
    #dtitk_subj_pd_fname = tracts_of_phi_of_dtitk_subjects_dir + f'{subjname}_all_geos_geodesics{pp}.pkl.gz'
    #dtitk_atlas_pd_fname = dtitk_atlas_tract_dir + f'dtitk_all_geos_geodesics{pp}.pkl.gz'
    dtitk_subj_pd_fname = tracts_of_phi_of_dtitk_subjects_dir + f'{subj}_whole_brain_in_atlas_space.vtk'
    dtitk_atlas_pd_fname = dtitk_atlas_tract_dir + f'dtitk_atlas_whole_brain.vtk'
    
    ar = pool.apply_async(compute_distance_between_geodesics, args=(subj_pd_fname, atlas_pd_fname), callback=collect_result)
    ars.append(ar)
    #ar = pool.apply_async(compute_distance_between_geodesics, args=(dtitk_subj_pd_fname, dtitk_atlas_pd_fname), callback=collect_result)
    #ars.append(ar)

    for rmask in region_masks:
      for hemi in ["hemi1", "hemi2"]:
        subj_pd_fname = tracts_of_phi_of_subjects_dir + f'{subjname}_geos_{rmask}_{hemi}_geodesics{pp}.pkl.gz'
        atlas_pd_fname = atlas_tract_dir + f'{kc_atlasname}_geos_{rmask}_{hemi}_geodesics{pp}.pkl.gz'
        #dtitk_subj_pd_fname = tracts_of_phi_of_dtitk_subjects_dir + f'{subjname}_geos_{rmask}_{hemi}_geodesics{pp}.pkl.gz'
        #dtitk_atlas_pd_fname = dtitk_atlas_tract_dir + f'dtitk_geos_{rmask}_{hemi}_geodesics{pp}.pkl.gz'
        dtitk_subj_pd_fname = tracts_of_phi_of_dtitk_subjects_dir + f'{subj}_{rmask}_{hemi}_in_atlas_space.vtk'
        dtitk_atlas_pd_fname = dtitk_atlas_tract_dir + f'dtitk_atlas_{rmask}_{hemi}.vtk'
        ar = pool.apply_async(compute_distance_between_geodesics, args=(subj_pd_fname, atlas_pd_fname), callback=collect_result)
        ars.append(ar)
        #ar = pool.apply_async(compute_distance_between_geodesics, args=(dtitk_subj_pd_fname, dtitk_atlas_pd_fname), callback=collect_result)
        #ars.append(ar)
        # Add following if want to check that we get 0 distances
        #ar = pool.apply_async(compute_distance_between_geodesics, args=(subj_pd_fname, subj_pd_fname), callback=collect_result)
        #ars.append(ar)
        #ar = pool.apply_async(compute_distance_between_geodesics, args=(atlas_pd_fname, atlas_pd_fname), callback=collect_result)
        #ars.append(ar)

    for smask in single_masks:
      subj_pd_fname = tracts_of_phi_of_subjects_dir + f'{subjname}_geos_{smask}_geodesics{pp}.pkl.gz'
      atlas_pd_fname = atlas_tract_dir + f'{kc_atlasname}_geos_{smask}_geodesics{pp}.pkl.gz'
      #dtitk_subj_pd_fname = tracts_of_phi_of_dtitk_subjects_dir + f'{subjname}_geos_{smask}_geodesics{pp}.pkl.gz'
      #dtitk_atlas_pd_fname = dtitk_atlas_tract_dir + f'dtitk_geos_{smask}_geodesics{pp}.pkl.gz'
      dtitk_subj_pd_fname = tracts_of_phi_of_dtitk_subjects_dir + f'{subj}_{smask}_in_atlas_space.vtk'
      dtitk_atlas_pd_fname = dtitk_atlas_tract_dir + f'dtitk_atlas_{smask}.vtk'
      ar = pool.apply_async(compute_distance_between_geodesics, args=(subj_pd_fname, atlas_pd_fname), callback=collect_result)
      ars.append(ar)
      #ar = pool.apply_async(compute_distance_between_geodesics, args=(dtitk_subj_pd_fname, dtitk_atlas_pd_fname), callback=collect_result)
      #ars.append(ar)
  # end for each subject

  
  # Compute distances between \phi(T(subject)) and T(atlas)
  for subj in subjs:
    subjname = subj + f'_{bval}_'  
    subj_pd_fname = phi_of_tracts_of_subjects_dir + f'{subjname}_all_geos_geodesics{pp}.pkl.gz'
    atlas_pd_fname = atlas_tract_dir + f'{kc_atlasname}_geos_all_geodesics{pp}.pkl.gz'
    #dtitk_subj_pd_fname = phi_of_tracts_of_dtitk_subjects_dir + f'{subjname}_all_geos_geodesics{pp}.pkl.gz'
    #dtitk_atlas_pd_fname = dtitk_atlas_tract_dir + f'dtitk_all_geos_geodesics{pp}.pkl.gz'
    dtitk_subj_pd_fname =  phi_of_tracts_of_dtitk_subjects_dir + f'{subj}_whole_brain_to_atlas_space.vtk'
    dtitk_atlas_pd_fname = dtitk_atlas_tract_dir + f'dtitk_atlas_whole_brain.vtk'

    ar = pool.apply_async(compute_distance_between_geodesics, args=(subj_pd_fname, atlas_pd_fname), callback=collect_result)
    ars.append(ar)
    #ar = pool.apply_async(compute_distance_between_geodesics, args=(dtitk_subj_pd_fname, dtitk_atlas_pd_fname), callback=collect_result)
    #ars.append(ar)

    for rmask in region_masks:
      for hemi in ["hemi1", "hemi2"]:
        subj_pd_fname = phi_of_tracts_of_subjects_dir + f'{subjname}_geos_{rmask}_{hemi}_geodesics{pp}.pkl.gz'
        atlas_pd_fname = atlas_tract_dir + f'{kc_atlasname}_geos_{rmask}_{hemi}_geodesics{pp}.pkl.gz'
        #dtitk_subj_pd_fname = phi_of_tracts_of_dtitk_subjects_dir + f'{subjname}_geos_{rmask}_{hemi}_geodesics{pp}.pkl.gz'
        #dtitk_atlas_pd_fname = dtitk_atlas_tract_dir + f'dtitk_geos_{rmask}_{hemi}_geodesics{pp}.pkl.gz'
        dtitk_subj_pd_fname = phi_of_tracts_of_dtitk_subjects_dir + f'{subj}_{rmask}_{hemi}_to_atlas_space.vtk'
        dtitk_atlas_pd_fname = dtitk_atlas_tract_dir + f'dtitk_atlas_{rmask}_{hemi}.vtk'

        ar = pool.apply_async(compute_distance_between_geodesics, args=(subj_pd_fname, atlas_pd_fname), callback=collect_result)
        ars.append(ar)
        #ar = pool.apply_async(compute_distance_between_geodesics, args=(dtitk_subj_pd_fname, dtitk_atlas_pd_fname), callback=collect_result)
        #ars.append(ar)
        # Add following if want to check that we get 0 distances
        #ar = pool.apply_async(compute_distance_between_geodesics, args=(subj_pd_fname, subj_pd_fname), callback=collect_result)
        #ars.append(ar)
        #ar = pool.apply_async(compute_distance_between_geodesics, args=(atlas_pd_fname, atlas_pd_fname), callback=collect_result)
        #ars.append(ar)

    for smask in single_masks:
      subj_pd_fname = phi_of_tracts_of_subjects_dir + f'{subjname}_geos_{smask}_geodesics{pp}.pkl.gz'
      atlas_pd_fname = atlas_tract_dir + f'{kc_atlasname}_geos_{smask}_geodesics{pp}.pkl.gz'
      #dtitk_subj_pd_fname = phi_of_tracts_of_dtitk_subjects_dir + f'{subjname}_geos_{smask}_geodesics{pp}.pkl.gz'
      #dtitk_atlas_pd_fname = dtitk_atlas_tract_dir + f'dtitk_geos_{smask}_geodesics{pp}.pkl.gz'
      dtitk_subj_pd_fname = phi_of_tracts_of_dtitk_subjects_dir + f'{subj}_{smask}_to_atlas_space.vtk'
      dtitk_atlas_pd_fname = dtitk_atlas_tract_dir + f'dtitk_atlas_{smask}.vtk'
      ar = pool.apply_async(compute_distance_between_geodesics, args=(subj_pd_fname, atlas_pd_fname), callback=collect_result)
      ars.append(ar)
      #ar = pool.apply_async(compute_distance_between_geodesics, args=(dtitk_subj_pd_fname, dtitk_atlas_pd_fname), callback=collect_result)
      #ars.append(ar)
  # end for each subject

  # Compute distances between \phi(T(subject)) and T(\phi(subject))
  for subj in subjs:
    subjname = subj + f'_{bval}_'  
    subj_pd_fname = phi_of_tracts_of_subjects_dir + f'{subjname}_all_geos_geodesics{pp}.pkl.gz'
    atlas_pd_fname = tracts_of_phi_of_subjects_dir + f'{subjname}_all_geos_geodesics{pp}.pkl.gz'
    #dtitk_subj_pd_fname = phi_of_tracts_of_dtitk_subjects_dir + f'{subjname}_all_geos_geodesics{pp}.pkl.gz'
    #dtitk_atlas_pd_fname = tracts_of_phi_of_dtitk_subjects_dir + f'{subjname}_all_geos_geodesics{pp}.pkl.gz'
    dtitk_subj_pd_fname = phi_of_tracts_of_dtitk_subjects_dir + f'{subj}_whole_brain_to_atlas_space.vtk'
    dtitk_atlas_pd_fname = tracts_of_phi_of_dtitk_subjects_dir + f'{subj}_whole_brain_in_atlas_space.vtk'

    ar = pool.apply_async(compute_distance_between_geodesics, args=(subj_pd_fname, atlas_pd_fname), callback=collect_result)
    ars.append(ar)
    #ar = pool.apply_async(compute_distance_between_geodesics, args=(dtitk_subj_pd_fname, dtitk_atlas_pd_fname), callback=collect_result)
    #ars.append(ar)

    for rmask in region_masks:
      for hemi in ["hemi1", "hemi2"]:
        subj_pd_fname = phi_of_tracts_of_subjects_dir + f'{subjname}_geos_{rmask}_{hemi}_geodesics{pp}.pkl.gz'
        atlas_pd_fname = tracts_of_phi_of_subjects_dir + f'{subjname}_geos_{rmask}_{hemi}_geodesics{pp}.pkl.gz'
        #dtitk_subj_pd_fname = phi_of_tracts_of_dtitk_subjects_dir + f'{subjname}_geos_{rmask}_{hemi}_geodesics{pp}.pkl.gz'
        #dtitk_atlas_pd_fname = tracts_of_phi_of_dtitk_subjects_dir + f'{subjname}_geos_{rmask}_{hemi}_geodesics{pp}.pkl.gz'
        dtitk_subj_pd_fname = phi_of_tracts_of_dtitk_subjects_dir + f'{subj}_{rmask}_{hemi}_to_atlas_space.vtk'
        dtitk_atlas_pd_fname = tracts_of_phi_of_dtitk_subjects_dir + f'{subj}_{rmask}_{hemi}_in_atlas_space.vtk'
        ar = pool.apply_async(compute_distance_between_geodesics, args=(subj_pd_fname, atlas_pd_fname), callback=collect_result)
        ars.append(ar)
        #ar = pool.apply_async(compute_distance_between_geodesics, args=(dtitk_subj_pd_fname, dtitk_atlas_pd_fname), callback=collect_result)
        #ars.append(ar)
        # Add following if want to check that we get 0 distances
        #ar = pool.apply_async(compute_distance_between_geodesics, args=(subj_pd_fname, subj_pd_fname), callback=collect_result)
        #ars.append(ar)
        #ar = pool.apply_async(compute_distance_between_geodesics, args=(atlas_pd_fname, atlas_pd_fname), callback=collect_result)
        #ars.append(ar)

    for smask in single_masks:
      subj_pd_fname = phi_of_tracts_of_subjects_dir + f'{subjname}_geos_{smask}_geodesics{pp}.pkl.gz'
      atlas_pd_fname = tracts_of_phi_of_subjects_dir + f'{subjname}_geos_{smask}_geodesics{pp}.pkl.gz'
      #dtitk_subj_pd_fname = phi_of_tracts_of_dtitk_subjects_dir + f'{subjname}_geos_{smask}_geodesics{pp}.pkl.gz'
      #dtitk_atlas_pd_fname = tracts_of_phi_of_dtitk_subjects_dir + f'{subjname}_geos_{smask}_geodesics{pp}.pkl.gz'
      dtitk_subj_pd_fname = phi_of_tracts_of_dtitk_subjects_dir + f'{subj}_{smask}_to_atlas_space.vtk'
      dtitk_atlas_pd_fname = tracts_of_phi_of_dtitk_subjects_dir + f'{subj}_{smask}_in_atlas_space.vtk'
      ar = pool.apply_async(compute_distance_between_geodesics, args=(subj_pd_fname, atlas_pd_fname), callback=collect_result)
      ars.append(ar)
      #ar = pool.apply_async(compute_distance_between_geodesics, args=(dtitk_subj_pd_fname, dtitk_atlas_pd_fname), callback=collect_result)
      #ars.append(ar)
  # end for each subject

  # Compute distances between T(dtitk_atlas) and T(atlas)
  #subj_pd_fname = dtitk_atlas_tract_dir + f'dtitk_all_geos_geodesics{pp}.pkl.gz'
  subj_pd_fname = dtitk_atlas_tract_dir + f'dtitk_atlas_whole_brain.vtk'
  atlas_pd_fname = atlas_tract_dir + f'{kc_atlasname}_geos_all_geodesics{pp}.pkl.gz'
    
  #ar = pool.apply_async(compute_distance_between_geodesics, args=(subj_pd_fname, atlas_pd_fname), callback=collect_result)
  #ars.append(ar)

  for rmask in region_masks:
    for hemi in ["hemi1", "hemi2"]:
      #subj_pd_fname = dtitk_atlas_tract_dir + f'dtitk_geos_{rmask}_{hemi}_geodesics{pp}.pkl.gz'
      subj_pd_fname = dtitk_atlas_tract_dir + f'dtitk_atlas_{rmask}_{hemi}.vtk'
      atlas_pd_fname = atlas_tract_dir + f'{kc_atlasname}_geos_{rmask}_{hemi}_geodesics{pp}.pkl.gz'
      #ar = pool.apply_async(compute_distance_between_geodesics, args=(subj_pd_fname, atlas_pd_fname), callback=collect_result)
      #ars.append(ar)
        # Add following if want to check that we get 0 distances
        #ar = pool.apply_async(compute_distance_between_geodesics, args=(subj_pd_fname, subj_pd_fname), callback=collect_result)
        #ars.append(ar)
        #ar = pool.apply_async(compute_distance_between_geodesics, args=(atlas_pd_fname, atlas_pd_fname), callback=collect_result)
        #ars.append(ar)

  for smask in single_masks:
    #subj_pd_fname = dtitk_atlas_tract_dir + f'dtitk_geos_{smask}_geodesics{pp}.pkl.gz'
    subj_pd_fname = dtitk_atlas_tract_dir + f'dtitk_atlas_{smask}.vtk'
    atlas_pd_fname = atlas_tract_dir + f'{kc_atlasname}_geos_{smask}_geodesics{pp}.pkl.gz'
    #ar = pool.apply_async(compute_distance_between_geodesics, args=(subj_pd_fname, atlas_pd_fname), callback=collect_result)
    #ars.append(ar)


  # end for each atlas
  for ar in ars:
    ar.wait()

  pool.close()
  pool.join()

  print('all results:', all_results)

  # Save results to file
  pathlib.Path(out_tract_dir).mkdir(exist_ok=True)

  fname = f'{out_tract_dir}geo_distances.pkl'
  with open(fname,'wb') as f:
    print('writing results to file:', fname)
    pickle.dump(all_results, f)
