# Apply a diffeomorphism to all points in VTK polydata
import pathlib

from lazy_imports import np
import scipy.io as sio
from util.diffeo import coord_register_batch_3d
import vtk
import whitematteranalysis as wma
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
    """Compute the distance between corresponding fibers.  We know they correspond because
they started from the same / equivalent seed points.  Be careful not to use randomly downsampled fibers
here, because the downsampling process will lose the correspondences
    """

    (dims, number_of_fibers_moving, points_per_fiber) = moving.shape
    (dims, number_of_fibers_fixed, points_per_fiber) = fixed.shape

    distance = np.zeros(number_of_fibers_moving)

    if number_of_fibers_moving != number_of_fibers_fixed:
      print("Error, when computing corresponding distance, need same number of fibers.  Instead, fixed has", number_of_fibers_fixed,
            "fibers, while moving has", number_of_fibers_moving, "fibers.")
      return(-3)

      
    # Loop over fibers in moving and fixed. Find total distance between each pair of fibers.
    # Assumes that fibers of same idx in moving and fixed correspond to each other.
    for idx in range(number_of_fibers_moving):
      distance[idx] = mean_of_min_dist_of_curves(moving[:,idx,:], fixed[:,idx,:])
 
    avg_dist = np.sum(distance) / number_of_fibers_moving
    return(avg_dist)
    


def inner_loop_objective(fixed, moving, sigmasq):
    """The code called within the objective_function to find the negative log
    probability of one brain given all other brains.
    """

    (dims, number_of_fibers_moving, points_per_fiber) = moving.shape
    # number of compared fibers (normalization factor)
    (dims, number_of_fibers_fixed, points_per_fiber) = fixed.shape
    #print('moving shape:', moving.shape)
    #print('fixed shape:', fixed.shape)

    probability = np.zeros(number_of_fibers_moving) + 1e-20
    probability2 = np.zeros(number_of_fibers_fixed) + 1e-20
    distance1 = np.zeros(number_of_fibers_moving)
    distance2 = np.zeros(number_of_fibers_fixed)
    
    # Loop over fibers in moving. Find total probability of
    # each fiber using all fibers from fixed.
    for idx in range(number_of_fibers_moving):
        #probability[idx] += total_probability_numpy(moving[:,idx,:], fixed,
        #        sigmasq)
        distance1[idx] = total_probability_numpy(moving[:,idx,:], fixed,
                sigmasq)
        probability[idx] += distance1[idx]

    for idx in range(number_of_fibers_fixed):
        #probability[idx] += total_probability_numpy(fixed[:,idx,:], moving,
        #        sigmasq)
        distance2[idx] = total_probability_numpy(fixed[:,idx,:], moving,
                sigmasq)
        probability2[idx] += distance2[idx]
    
        
    # commented out by KMC
    # # Divide total probability by number of fibers in the atlas ("mean
    # # brain").  This neglects Z, the normalization constant for the
    # # pdf, which would not affect the optimization.
    # probability /= number_of_fibers_fixed

    # added by KMC
    #print('probability',np.min(probability), np.max(probability))
    #print('probability2',np.min(probability2), np.max(probability2))
    #print('distance1',np.min(distance1), np.max(distance1))
    #print('distance2',np.min(distance2), np.max(distance2))

    # # add negative log probabilities of all fibers in this brain.
    # entropy = numpy.sum(- numpy.log(probability))
    # return entropy
    return np.maximum(np.max(distance1)/number_of_fibers_moving, np.max(distance2)/number_of_fibers_moving)

def total_probability_numpy(moving_fiber, fixed_fibers, sigmasq):
    """Compute total probability for moving fiber when compared to all fixed
    fibers.
    """
    distance = fiber_distance_numpy(moving_fiber, fixed_fibers)
    # This part seems wrong according to Wikipedia https://en.wikipedia.org/wiki/Hausdorff_distance, not sure thought
    # the difference between the probability formulation and the direct distance calculation
    #probability = numpy.exp(numpy.divide(-distance, sigmasq))
    #return numpy.sum(probability)
    # Here we want the smallest distance between x \in X and the set Y
    return(np.min(distance))

def fiber_distance_numpy(moving_fiber, fixed_fibers):
    """
    Find pairwise fiber distance from fixed fiber to all moving fibers.
    """
    # compute pairwise fiber distances along fibers
    distance_1 = _fiber_distance_internal_use_numpy(moving_fiber, fixed_fibers)
    distance_2 = _fiber_distance_internal_use_numpy(moving_fiber, fixed_fibers, reverse_fiber_order=True)
    
    # choose the lowest distance, corresponding to the optimal fiber
    # representation (either forward or reverse order)
    return np.minimum(distance_1, distance_2)



def _fiber_distance_internal_use_numpy(moving_fiber, fixed_fibers, reverse_fiber_order=False):
    """Compute the total fiber distance from one fiber to an array of many
    fibers.  This function does not handle equivalent fiber
    representations. For that use fiber_distance, above.
    """
    
    # compute the distance from this fiber to the array of other fibers
    if reverse_fiber_order:
        ddx = fixed_fibers[0,:,:] - moving_fiber[0,::-1]
        ddy = fixed_fibers[1,:,:] - moving_fiber[1,::-1]
        ddz = fixed_fibers[2,:,:] - moving_fiber[2,::-1]
    else:
        ddx = fixed_fibers[0,:,:] - moving_fiber[0,:]
        ddy = fixed_fibers[1,:,:] - moving_fiber[1,:]
        ddz = fixed_fibers[2,:,:] - moving_fiber[2,:]

    #print "MAX abs ddx:", numpy.max(numpy.abs(ddx)), "MAX ddy:", numpy.max(numpy.abs(ddy)), "MAX ddz:", numpy.max(numpy.abs(ddz))
    #print "MIN abs ddx:", numpy.min(numpy.abs(ddx)), "MIN ddy:", numpy.min(numpy.abs(ddy)), "MIN ddz:", numpy.min(numpy.abs(ddz))
    
    distance = np.square(ddx)
    distance += np.square(ddy)
    distance += np.square(ddz)

    # Use the mean distance as it works better than Hausdorff-like distance
    return np.mean(distance, 1)

def compute_distance_between_polydatas(input_polydata1_fname, input_polydata2_fname):
  print('Reading and converting data for', input_polydata1_fname, 'and', input_polydata2_fname)
  try:  
    inpd1 = wma.io.read_polydata(input_polydata1_fname)
    inpd2 = wma.io.read_polydata(input_polydata2_fname)

    # Filter to keep only 10,000 geodesics
    num_to_keep = 10000
    # use same random seed for all datasets to maintain any correspondences between them that may exist
    # need to use a random seed to sample all across the image.
    # Choosing the first n geodesics, would likely select geodesics that only go through a smaller region of the brain
    rand_seed=159
    print('Keeping no more than', num_to_keep, 'geodesics of', inpd1.GetNumberOfLines(), 'total geodesics for', input_polydata1_fname)
    filtpd1 = wma.filter.downsample(inpd1, num_to_keep, random_seed=rand_seed, verbose=False)
    print('Keeping no more than', num_to_keep, 'geodesics of', inpd2.GetNumberOfLines(), 'total geodesics for', input_polydata2_fname)
    filtpd2 = wma.filter.downsample(inpd2, num_to_keep, random_seed=rand_seed, verbose=False)

    # code taken by following wm_register_multisubject_faster.py into the depths to find the objective function
    points_per_fiber = 10 # 15 from wma congeal_multisubject.py, 10 from affine setting and 3 from nonrigid setting in wm_register_multisubject_faster

    # I get unexpected results where bundles that are more different have a smaller distance.
    # I suspect this may be due to undersampling along the length of the fiber, so will increase points_per_fiber here to see how that changes the distances
    points_per_fiber = 40 # 15 from wma congeal_multisubject.py, 10 from affine setting and 3 from nonrigid setting in wm_register_multisubject_faster
    points_per_fiber = 100 # 15 from wma congeal_multisubject.py, 10 from affine setting and 3 from nonrigid setting in wm_register_multisubject_faster
    sigma = 2 # 2 nonrigid, largest grid resolution, 5, 10 or 20 other possible values
    #pd1_ds = wma.filter.downsample(inpd1, self.subject_brain_size, verbose=False, random_seed=self.random_seed)
    fibers1 = wma.fibers.FiberArray()
    fibers1.convert_from_polydata(filtpd1, points_per_fiber)
    fibers_array1 = np.array([fibers1.fiber_array_r,fibers1.fiber_array_a,fibers1.fiber_array_s])
    #pd2_ds = wma.filter.downsample(inpd2, self.subject_brain_size, verbose=False, random_seed=self.random_seed)
    fibers2 = wma.fibers.FiberArray()
    fibers2.convert_from_polydata(filtpd2, points_per_fiber)
    fibers_array2 = np.array([fibers2.fiber_array_r,fibers2.fiber_array_a,fibers2.fiber_array_s])
  except Exception as err:
    print('Caught', err, 'while reading and converting data for', input_polydata1_fname, 'and', input_polydata2_fname)
    distil = -1
    distwma = -1
    distcorr = -1
    return((distcorr, distil, distwma, input_polydata1_fname, input_polydata2_fname))

  print('computing distance between', input_polydata1_fname, 'and', input_polydata2_fname)
  try:
    # We get large distances 1444, 2850, etc when comparing fiber array to itself using wma.register_two_subjects.inner_loop_objective
    (dims, number_of_fibers_moving, points_per_fiber) = fibers_array2.shape
    distwma = wma.register_two_subjects.inner_loop_objective(fibers_array1, fibers_array2, sigma * sigma) / number_of_fibers_moving
    distil = inner_loop_objective(fibers_array1, fibers_array2, sigma * sigma)
    distcorr = corresponding_distance_sets_of_curves(fibers_array1, fibers_array2)
    print('wma vs ours vs corresp', distwma, distil,distcorr)
  except Exception as err:
    print('Caught', err, 'while computing distance between', input_polydata1_fname, 'and', input_polydata2_fname)
    distil = -2
    distwma = -2
    distcorr = -2


  del inpd1
  del inpd2

  print('Distance between', input_polydata1_fname, 'and', input_polydata2_fname, 'is', distcorr)
  return((distcorr, distil, distwma, input_polydata1_fname, input_polydata2_fname))
# end compute_distance_between_polydatas

def collect_result(result):
  # Expect a distance.
  print('collected corresponding, our and wma distances between', result[3], 'and', result[4], ':', result[0], result[1], result[2])
  all_results[result[3] + ',' + result[4]] = (result[0], result[1], result[2])
  
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
    subj_pd_fname = tracts_of_subjects_dir + f'{subjname}_all_geos_geodesics{pp}.vtp'
    atlas_pd_fname = atlas_tract_dir + f'{kc_atlasname}_geos_all_geodesics{pp}.vtp'
    #dtitk_subj_pd_fname = tracts_of_dtitk_subjects_dir + f'{subjname}_all_geos_geodesics{pp}.vtp'
    #dtitk_atlas_pd_fname = dtitk_atlas_tract_dir + f'dtitk_all_geos_geodesics{pp}.vtp'
    dtitk_subj_pd_fname = tracts_of_dtitk_subjects_dir + f'{subj}_whole_brain_in_subj_space.vtk'
    dtitk_atlas_pd_fname = dtitk_atlas_tract_dir + f'dtitk_atlas_whole_brain.vtk'
    
    ar = pool.apply_async(compute_distance_between_polydatas, args=(subj_pd_fname, atlas_pd_fname), callback=collect_result)
    ars.append(ar)
    # Skipping DTITK whole brain because expensive
    # Downsampling to 10,000 points because distance computation is expensive
    ar = pool.apply_async(compute_distance_between_polydatas, args=(dtitk_subj_pd_fname, dtitk_atlas_pd_fname), callback=collect_result)
    ars.append(ar)

    for rmask in region_masks:
      for hemi in ["hemi1", "hemi2"]:
        subj_pd_fname = tracts_of_subjects_dir + f'{subjname}_geos_{rmask}_{hemi}_geodesics{pp}.vtp'
        atlas_pd_fname = atlas_tract_dir + f'{kc_atlasname}_geos_{rmask}_{hemi}_geodesics{pp}.vtp'
        #dtitk_subj_pd_fname = tracts_of_dtitk_subjects_dir + f'{subjname}_geos_{rmask}_{hemi}_geodesics{pp}.vtp'
        #dtitk_atlas_pd_fname = dtitk_atlas_tract_dir + f'dtitk_geos_{rmask}_{hemi}_geodesics{pp}.vtp'
        dtitk_subj_pd_fname = tracts_of_dtitk_subjects_dir + f'{subj}_{rmask}_{hemi}_in_subj_space.vtk'
        dtitk_atlas_pd_fname = dtitk_atlas_tract_dir + f'dtitk_atlas_{rmask}_{hemi}.vtk'
        ar = pool.apply_async(compute_distance_between_polydatas, args=(subj_pd_fname, atlas_pd_fname), callback=collect_result)
        ars.append(ar)
        ar = pool.apply_async(compute_distance_between_polydatas, args=(dtitk_subj_pd_fname, dtitk_atlas_pd_fname), callback=collect_result)
        ars.append(ar)
        # Add following if want to check that we get 0 distances
        #ar = pool.apply_async(compute_distance_between_polydatas, args=(subj_pd_fname, subj_pd_fname), callback=collect_result)
        #ars.append(ar)
        #ar = pool.apply_async(compute_distance_between_polydatas, args=(atlas_pd_fname, atlas_pd_fname), callback=collect_result)
        #ars.append(ar)

    for smask in single_masks:
      subj_pd_fname = tracts_of_subjects_dir + f'{subjname}_geos_{smask}_geodesics{pp}.vtp'
      atlas_pd_fname = atlas_tract_dir + f'{kc_atlasname}_geos_{smask}_geodesics{pp}.vtp'
      #dtitk_subj_pd_fname = tracts_of_dtitk_subjects_dir + f'{subjname}_geos_{smask}_geodesics{pp}.vtp'
      #dtitk_atlas_pd_fname = dtitk_atlas_tract_dir + f'dtitk_geos_{smask}_geodesics{pp}.vtp'
      dtitk_subj_pd_fname = tracts_of_dtitk_subjects_dir + f'{subj}_{smask}_in_subj_space.vtk'
      dtitk_atlas_pd_fname = dtitk_atlas_tract_dir + f'dtitk_atlas_{smask}.vtk'
      ar = pool.apply_async(compute_distance_between_polydatas, args=(subj_pd_fname, atlas_pd_fname), callback=collect_result)
      ars.append(ar)
      ar = pool.apply_async(compute_distance_between_polydatas, args=(dtitk_subj_pd_fname, dtitk_atlas_pd_fname), callback=collect_result)
      ars.append(ar)
  # end for each subject
      

  # Compute distances between T(\phi(subject)) and T(atlas)
  for subj in subjs:
    subjname = subj + f'_{bval}_'  
    subj_pd_fname = tracts_of_phi_of_subjects_dir + f'{subjname}_all_geos_geodesics{pp}.vtp'
    atlas_pd_fname = atlas_tract_dir + f'{kc_atlasname}_geos_all_geodesics{pp}.vtp'
    #dtitk_subj_pd_fname = tracts_of_phi_of_dtitk_subjects_dir + f'{subjname}_all_geos_geodesics{pp}.vtp'
    #dtitk_atlas_pd_fname = dtitk_atlas_tract_dir + f'dtitk_all_geos_geodesics{pp}.vtp'
    dtitk_subj_pd_fname = tracts_of_phi_of_dtitk_subjects_dir + f'{subj}_whole_brain_in_atlas_space.vtk'
    dtitk_atlas_pd_fname = dtitk_atlas_tract_dir + f'dtitk_atlas_whole_brain.vtk'
    
    ar = pool.apply_async(compute_distance_between_polydatas, args=(subj_pd_fname, atlas_pd_fname), callback=collect_result)
    ars.append(ar)
    # Skipping DTITK whole brain because expensive
    # Downsampling to 10,000 points because distance computation is expensive
    ar = pool.apply_async(compute_distance_between_polydatas, args=(dtitk_subj_pd_fname, dtitk_atlas_pd_fname), callback=collect_result)
    ars.append(ar)

    for rmask in region_masks:
      for hemi in ["hemi1", "hemi2"]:
        subj_pd_fname = tracts_of_phi_of_subjects_dir + f'{subjname}_geos_{rmask}_{hemi}_geodesics{pp}.vtp'
        atlas_pd_fname = atlas_tract_dir + f'{kc_atlasname}_geos_{rmask}_{hemi}_geodesics{pp}.vtp'
        #dtitk_subj_pd_fname = tracts_of_phi_of_dtitk_subjects_dir + f'{subjname}_geos_{rmask}_{hemi}_geodesics{pp}.vtp'
        #dtitk_atlas_pd_fname = dtitk_atlas_tract_dir + f'dtitk_geos_{rmask}_{hemi}_geodesics{pp}.vtp'
        dtitk_subj_pd_fname = tracts_of_phi_of_dtitk_subjects_dir + f'{subj}_{rmask}_{hemi}_in_atlas_space.vtk'
        dtitk_atlas_pd_fname = dtitk_atlas_tract_dir + f'dtitk_atlas_{rmask}_{hemi}.vtk'
        ar = pool.apply_async(compute_distance_between_polydatas, args=(subj_pd_fname, atlas_pd_fname), callback=collect_result)
        ars.append(ar)
        ar = pool.apply_async(compute_distance_between_polydatas, args=(dtitk_subj_pd_fname, dtitk_atlas_pd_fname), callback=collect_result)
        ars.append(ar)
        # Add following if want to check that we get 0 distances
        #ar = pool.apply_async(compute_distance_between_polydatas, args=(subj_pd_fname, subj_pd_fname), callback=collect_result)
        #ars.append(ar)
        #ar = pool.apply_async(compute_distance_between_polydatas, args=(atlas_pd_fname, atlas_pd_fname), callback=collect_result)
        #ars.append(ar)

    for smask in single_masks:
      subj_pd_fname = tracts_of_phi_of_subjects_dir + f'{subjname}_geos_{smask}_geodesics{pp}.vtp'
      atlas_pd_fname = atlas_tract_dir + f'{kc_atlasname}_geos_{smask}_geodesics{pp}.vtp'
      #dtitk_subj_pd_fname = tracts_of_phi_of_dtitk_subjects_dir + f'{subjname}_geos_{smask}_geodesics{pp}.vtp'
      #dtitk_atlas_pd_fname = dtitk_atlas_tract_dir + f'dtitk_geos_{smask}_geodesics{pp}.vtp'
      dtitk_subj_pd_fname = tracts_of_phi_of_dtitk_subjects_dir + f'{subj}_{smask}_in_atlas_space.vtk'
      dtitk_atlas_pd_fname = dtitk_atlas_tract_dir + f'dtitk_atlas_{smask}.vtk'
      ar = pool.apply_async(compute_distance_between_polydatas, args=(subj_pd_fname, atlas_pd_fname), callback=collect_result)
      ars.append(ar)
      ar = pool.apply_async(compute_distance_between_polydatas, args=(dtitk_subj_pd_fname, dtitk_atlas_pd_fname), callback=collect_result)
      ars.append(ar)
  # end for each subject

  
  # Compute distances between \phi(T(subject)) and T(atlas)
  for subj in subjs:
    subjname = subj + f'_{bval}_'  
    subj_pd_fname = phi_of_tracts_of_subjects_dir + f'{subjname}_all_geos_geodesics{pp}.vtp'
    atlas_pd_fname = atlas_tract_dir + f'{kc_atlasname}_geos_all_geodesics{pp}.vtp'
    #dtitk_subj_pd_fname = phi_of_tracts_of_dtitk_subjects_dir + f'{subjname}_all_geos_geodesics{pp}.vtp'
    #dtitk_atlas_pd_fname = dtitk_atlas_tract_dir + f'dtitk_all_geos_geodesics{pp}.vtp'
    dtitk_subj_pd_fname =  phi_of_tracts_of_dtitk_subjects_dir + f'{subj}_whole_brain_to_atlas_space.vtk'
    dtitk_atlas_pd_fname = dtitk_atlas_tract_dir + f'dtitk_atlas_whole_brain.vtk'

    ar = pool.apply_async(compute_distance_between_polydatas, args=(subj_pd_fname, atlas_pd_fname), callback=collect_result)
    ars.append(ar)
    # Skipping DTITK whole brain because expensive
    # Downsampling to 10,000 points because distance computation is expensive
    ar = pool.apply_async(compute_distance_between_polydatas, args=(dtitk_subj_pd_fname, dtitk_atlas_pd_fname), callback=collect_result)
    ars.append(ar)

    for rmask in region_masks:
      for hemi in ["hemi1", "hemi2"]:
        subj_pd_fname = phi_of_tracts_of_subjects_dir + f'{subjname}_geos_{rmask}_{hemi}_geodesics{pp}.vtp'
        atlas_pd_fname = atlas_tract_dir + f'{kc_atlasname}_geos_{rmask}_{hemi}_geodesics{pp}.vtp'
        #dtitk_subj_pd_fname = phi_of_tracts_of_dtitk_subjects_dir + f'{subjname}_geos_{rmask}_{hemi}_geodesics{pp}.vtp'
        #dtitk_atlas_pd_fname = dtitk_atlas_tract_dir + f'dtitk_geos_{rmask}_{hemi}_geodesics{pp}.vtp'
        dtitk_subj_pd_fname = phi_of_tracts_of_dtitk_subjects_dir + f'{subj}_{rmask}_{hemi}_to_atlas_space.vtk'
        dtitk_atlas_pd_fname = dtitk_atlas_tract_dir + f'dtitk_atlas_{rmask}_{hemi}.vtk'

        ar = pool.apply_async(compute_distance_between_polydatas, args=(subj_pd_fname, atlas_pd_fname), callback=collect_result)
        ars.append(ar)
        ar = pool.apply_async(compute_distance_between_polydatas, args=(dtitk_subj_pd_fname, dtitk_atlas_pd_fname), callback=collect_result)
        ars.append(ar)
        # Add following if want to check that we get 0 distances
        #ar = pool.apply_async(compute_distance_between_polydatas, args=(subj_pd_fname, subj_pd_fname), callback=collect_result)
        #ars.append(ar)
        #ar = pool.apply_async(compute_distance_between_polydatas, args=(atlas_pd_fname, atlas_pd_fname), callback=collect_result)
        #ars.append(ar)

    for smask in single_masks:
      subj_pd_fname = phi_of_tracts_of_subjects_dir + f'{subjname}_geos_{smask}_geodesics{pp}.vtp'
      atlas_pd_fname = atlas_tract_dir + f'{kc_atlasname}_geos_{smask}_geodesics{pp}.vtp'
      #dtitk_subj_pd_fname = phi_of_tracts_of_dtitk_subjects_dir + f'{subjname}_geos_{smask}_geodesics{pp}.vtp'
      #dtitk_atlas_pd_fname = dtitk_atlas_tract_dir + f'dtitk_geos_{smask}_geodesics{pp}.vtp'
      dtitk_subj_pd_fname = phi_of_tracts_of_dtitk_subjects_dir + f'{subj}_{smask}_to_atlas_space.vtk'
      dtitk_atlas_pd_fname = dtitk_atlas_tract_dir + f'dtitk_atlas_{smask}.vtk'
      ar = pool.apply_async(compute_distance_between_polydatas, args=(subj_pd_fname, atlas_pd_fname), callback=collect_result)
      ars.append(ar)
      ar = pool.apply_async(compute_distance_between_polydatas, args=(dtitk_subj_pd_fname, dtitk_atlas_pd_fname), callback=collect_result)
      ars.append(ar)
  # end for each subject

  # Compute distances between \phi(T(subject)) and T(\phi(subject))
  for subj in subjs:
    subjname = subj + f'_{bval}_'  
    subj_pd_fname = phi_of_tracts_of_subjects_dir + f'{subjname}_all_geos_geodesics{pp}.vtp'
    atlas_pd_fname = tracts_of_phi_of_subjects_dir + f'{subjname}_all_geos_geodesics{pp}.vtp'
    #dtitk_subj_pd_fname = phi_of_tracts_of_dtitk_subjects_dir + f'{subjname}_all_geos_geodesics{pp}.vtp'
    #dtitk_atlas_pd_fname = tracts_of_phi_of_dtitk_subjects_dir + f'{subjname}_all_geos_geodesics{pp}.vtp'
    dtitk_subj_pd_fname = phi_of_tracts_of_dtitk_subjects_dir + f'{subj}_whole_brain_to_atlas_space.vtk'
    dtitk_atlas_pd_fname = tracts_of_phi_of_dtitk_subjects_dir + f'{subj}_whole_brain_in_atlas_space.vtk'

    ar = pool.apply_async(compute_distance_between_polydatas, args=(subj_pd_fname, atlas_pd_fname), callback=collect_result)
    ars.append(ar)
    # Skipping DTITK whole brain because expensive
    # Downsampling to 10,000 points because distance computation is expensive
    ar = pool.apply_async(compute_distance_between_polydatas, args=(dtitk_subj_pd_fname, dtitk_atlas_pd_fname), callback=collect_result)
    ars.append(ar)

    for rmask in region_masks:
      for hemi in ["hemi1", "hemi2"]:
        subj_pd_fname = phi_of_tracts_of_subjects_dir + f'{subjname}_geos_{rmask}_{hemi}_geodesics{pp}.vtp'
        atlas_pd_fname = tracts_of_phi_of_subjects_dir + f'{subjname}_geos_{rmask}_{hemi}_geodesics{pp}.vtp'
        #dtitk_subj_pd_fname = phi_of_tracts_of_dtitk_subjects_dir + f'{subjname}_geos_{rmask}_{hemi}_geodesics{pp}.vtp'
        #dtitk_atlas_pd_fname = tracts_of_phi_of_dtitk_subjects_dir + f'{subjname}_geos_{rmask}_{hemi}_geodesics{pp}.vtp'
        dtitk_subj_pd_fname = phi_of_tracts_of_dtitk_subjects_dir + f'{subj}_{rmask}_{hemi}_to_atlas_space.vtk'
        dtitk_atlas_pd_fname = tracts_of_phi_of_dtitk_subjects_dir + f'{subj}_{rmask}_{hemi}_in_atlas_space.vtk'
        ar = pool.apply_async(compute_distance_between_polydatas, args=(subj_pd_fname, atlas_pd_fname), callback=collect_result)
        ars.append(ar)
        ar = pool.apply_async(compute_distance_between_polydatas, args=(dtitk_subj_pd_fname, dtitk_atlas_pd_fname), callback=collect_result)
        ars.append(ar)
        # Add following if want to check that we get 0 distances
        #ar = pool.apply_async(compute_distance_between_polydatas, args=(subj_pd_fname, subj_pd_fname), callback=collect_result)
        #ars.append(ar)
        #ar = pool.apply_async(compute_distance_between_polydatas, args=(atlas_pd_fname, atlas_pd_fname), callback=collect_result)
        #ars.append(ar)

    for smask in single_masks:
      subj_pd_fname = phi_of_tracts_of_subjects_dir + f'{subjname}_geos_{smask}_geodesics{pp}.vtp'
      atlas_pd_fname = tracts_of_phi_of_subjects_dir + f'{subjname}_geos_{smask}_geodesics{pp}.vtp'
      #dtitk_subj_pd_fname = phi_of_tracts_of_dtitk_subjects_dir + f'{subjname}_geos_{smask}_geodesics{pp}.vtp'
      #dtitk_atlas_pd_fname = tracts_of_phi_of_dtitk_subjects_dir + f'{subjname}_geos_{smask}_geodesics{pp}.vtp'
      dtitk_subj_pd_fname = phi_of_tracts_of_dtitk_subjects_dir + f'{subj}_{smask}_to_atlas_space.vtk'
      dtitk_atlas_pd_fname = tracts_of_phi_of_dtitk_subjects_dir + f'{subj}_{smask}_in_atlas_space.vtk'
      ar = pool.apply_async(compute_distance_between_polydatas, args=(subj_pd_fname, atlas_pd_fname), callback=collect_result)
      ars.append(ar)
      ar = pool.apply_async(compute_distance_between_polydatas, args=(dtitk_subj_pd_fname, dtitk_atlas_pd_fname), callback=collect_result)
      ars.append(ar)
  # end for each subject

  # Compute distances between T(dtitk_atlas) and T(atlas)
  #subj_pd_fname = dtitk_atlas_tract_dir + f'dtitk_all_geos_geodesics{pp}.vtp'
  subj_pd_fname = dtitk_atlas_tract_dir + f'dtitk_atlas_whole_brain.vtk'
  atlas_pd_fname = atlas_tract_dir + f'{kc_atlasname}_geos_all_geodesics{pp}.vtp'
    
  # Skipping DTITK whole brain because expensive
  # Downsampling to 10,000 points because distance computation is expensive
  ar = pool.apply_async(compute_distance_between_polydatas, args=(subj_pd_fname, atlas_pd_fname), callback=collect_result)
  ars.append(ar)

  for rmask in region_masks:
    for hemi in ["hemi1", "hemi2"]:
      #subj_pd_fname = dtitk_atlas_tract_dir + f'dtitk_geos_{rmask}_{hemi}_geodesics{pp}.vtp'
      subj_pd_fname = dtitk_atlas_tract_dir + f'dtitk_atlas_{rmask}_{hemi}.vtk'
      atlas_pd_fname = atlas_tract_dir + f'{kc_atlasname}_geos_{rmask}_{hemi}_geodesics{pp}.vtp'
      ar = pool.apply_async(compute_distance_between_polydatas, args=(subj_pd_fname, atlas_pd_fname), callback=collect_result)
      ars.append(ar)
        # Add following if want to check that we get 0 distances
        #ar = pool.apply_async(compute_distance_between_polydatas, args=(subj_pd_fname, subj_pd_fname), callback=collect_result)
        #ars.append(ar)
        #ar = pool.apply_async(compute_distance_between_polydatas, args=(atlas_pd_fname, atlas_pd_fname), callback=collect_result)
        #ars.append(ar)

  for smask in single_masks:
    #subj_pd_fname = dtitk_atlas_tract_dir + f'dtitk_geos_{smask}_geodesics{pp}.vtp'
    subj_pd_fname = dtitk_atlas_tract_dir + f'dtitk_atlas_{smask}.vtk'
    atlas_pd_fname = atlas_tract_dir + f'{kc_atlasname}_geos_{smask}_geodesics{pp}.vtp'
    ar = pool.apply_async(compute_distance_between_polydatas, args=(subj_pd_fname, atlas_pd_fname), callback=collect_result)
    ars.append(ar)
 

  # end for each atlas
  for ar in ars:
    ar.wait()

  pool.close()
  pool.join()

  print('all results:', all_results)

  # Save results to file
  pathlib.Path(out_tract_dir).mkdir(exist_ok=True)

  fname = f'{out_tract_dir}distances.pkl'
  with open(fname,'wb') as f:
    print('writing results to file:', fname)
    pickle.dump(all_results, f)
