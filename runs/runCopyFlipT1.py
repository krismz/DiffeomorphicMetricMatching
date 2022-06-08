import os
from lazy_imports import np
from lazy_imports import savemat
from data.io import ReadTensors, ReadScalars, WriteTensorNPArray, WriteScalarNPArray
from algo import euler, geodesic
import algo.metricModSolver as mms

if __name__ == "__main__":
  inroot = '/usr/sci/projects/HCP/Kris/NSFCRCNS/prepped_UKF_data_with_grad_dev/'
  #outdir = '/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/UKF_experiments/'
  #inroot = '/usr/sci/projects/HCP/Kris/NSFCRCNS/prepped_data/'
  cases=[sbj for sbj in os.listdir(inroot) if sbj[0] != '.']

  bval = 1000
  bval = 'all'
  outdir = f'/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/UKF_experiments/B{bval}Results/'
  
  for cc in cases:
    try:
      run_case = f'{cc}'
      print("Running", run_case)
      for_mat_file = {}
      out_prefix = outdir + run_case
      
      subj = run_case[0:6]
      indir = inroot + subj + '/'
      #tens_file = f'dti_{bval}_tensor.nhdr'
      #mask_file = f'dti_{bval}_FA_mask.nhdr'
      ##t1_file = 't1_stripped_irescaled.nhdr'
      t1_file = 't1_to_reft1_rreg.nhdr'
      #num_iters = 1000 # Way too much, but want to understand convergence for all brain cases
      #save_intermediate_results = True
      ## TODO Something funky is going on with thresh_ratio.  Seems better to skip thresholding altogether
      ## Valid values are supposed to be between 0 and 1, with 1 meaning don't threshold anything.
      ## And yet, we need a value of at least 2.0 to get decent results for cubics (or even better is to
      ## skip the thresholding altogether).
      ## TODO test thresholds with real data, and decide whether to skip altogether.
      #thresh_ratio = 1.0
      #small_eval = 5e-3
      #sigma = 1.5
      
      #in_tens = ReadTensors(indir+'/'+tens_file)
      #in_mask = ReadScalars(indir+'/'+mask_file)
      if t1_file:
        in_T1 = ReadScalars(indir+'/'+t1_file)
      else:
        in_T1 = in_mask
      
      xsz=in_mask.shape[0]
      ysz=in_mask.shape[1]
      zsz=in_mask.shape[2]
      
      #alpha, out_tens, out_mask, rks, intermed_results = mms.solve_3d(in_tens, in_mask, num_iters, [-2,2],
      #                                                                thresh_ratio, save_intermediate_results, small_eval, sigma)
      
      #out_tens_tri = np.zeros((xsz,ysz,zsz,6))
      #out_tens_tri[:,:,:,0] = out_tens[:,:,:,0,0]
      #out_tens_tri[:,:,:,1] = out_tens[:,:,:,0,1]
      #out_tens_tri[:,:,:,2] = out_tens[:,:,:,0,2]
      #out_tens_tri[:,:,:,3] = out_tens[:,:,:,1,1]
      #out_tens_tri[:,:,:,4] = out_tens[:,:,:,1,2]
      #out_tens_tri[:,:,:,5] = out_tens[:,:,:,2,2]
      
      #if save_intermediate_results:
      #  scaled_tens_tri = np.zeros((xsz,ysz,zsz,6))
      #  scaled_tens_tri[:,:,:,0] = intermed_results['scaled_tensors'][:,:,:,0,0]
      #  scaled_tens_tri[:,:,:,1] = intermed_results['scaled_tensors'][:,:,:,0,1]
      #  scaled_tens_tri[:,:,:,2] = intermed_results['scaled_tensors'][:,:,:,0,2]
      #  scaled_tens_tri[:,:,:,3] = intermed_results['scaled_tensors'][:,:,:,1,1]
      #  scaled_tens_tri[:,:,:,4] = intermed_results['scaled_tensors'][:,:,:,1,2]
      #  scaled_tens_tri[:,:,:,5] = intermed_results['scaled_tensors'][:,:,:,2,2]

      out_T1 = in_T1.astype(float)[:,::-1,:].copy()
        
      #WriteTensorNPArray(out_tens_tri, out_prefix + f'_thresh_{thresh_ratio}_tensors.nhdr')
      #WriteTensorNPArray(in_tens, out_prefix + '_orig_tensors.nhdr')
      #WriteScalarNPArray(out_mask, out_prefix + '_filt_mask.nhdr')
      #WriteScalarNPArray(alpha, out_prefix + '_alpha.nhdr')
      if t1_file:
        WriteScalarNPArray(in_T1, out_prefix + '_T1.nhdr')
        WriteScalarNPArray(out_T1, out_prefix + '_T1_flip_y.nhdr')
      #if save_intermediate_results:
      #  WriteTensorNPArray(scaled_tens_tri, out_prefix + f'_scaled_tensors.nhdr')
      
      #for_mat_file['orig_tensors'] = in_tens
      #for_mat_file['thresh_tensors'] = out_tens
      #for_mat_file['alpha'] = alpha
      #for_mat_file['T1'] = in_T1
      #for_mat_file['filt_mask'] = out_mask
      #for_mat_file['rks'] = rks
      #if save_intermediate_results:
      #  for_mat_file['scaled_tensors'] = intermed_results['scaled_tensors']
        
      
      #tens_4_path = np.transpose(in_tens,(3,0,1,2))
      #thresh_tens_4_path = np.transpose(out_tens_tri,(3,0,1,2))
      #if save_intermediate_results:
      #  scaled_tens_4_path = np.transpose(scaled_tens_tri,(3,0,1,2))
      
      #for_mat_file['tens_4_path'] = tens_4_path
      #if save_intermediate_results:
      #  for_mat_file['scaled_tens_4_path'] = scaled_tens_4_path
        
      #savemat(out_prefix + '_results.mat',for_mat_file)
    except Exception as err:
      print('Exception', err, 'caught while processing subj', cc, '. Moving to next subject.')

  # end for each case
  #savemat(outdir + 'brain_results.mat',for_mat_file)

