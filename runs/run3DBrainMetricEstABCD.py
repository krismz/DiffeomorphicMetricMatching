import os
from lazy_imports import np
from lazy_imports import savemat
from data.io import ReadTensors, ReadScalars, WriteTensorNPArray, WriteScalarNPArray
from algo import euler, geodesic
import algo.metricModSolver as mms

if __name__ == "__main__":
  inroot = '/usr/sci/projects/abcd/anxiety_study/derivatives/metric_matching/'
  cases=[sbj for sbj in os.listdir(inroot) if sbj[:4] == 'sub-']
  upsamp=''
  #upsamp='_upsamp'

  for cc in cases:
    try:
      run_case = f'{cc}'
      print("Running", run_case)
      for_mat_file = {}
      t1_prefix = os.path.join(inroot, run_case, 'ses-baselineYear1Arm1','anat', run_case + '_ses-baselineYear1Arm1')
      dwi_prefix = os.path.join(inroot, run_case, 'ses-baselineYear1Arm1','dwi', run_case + '_ses-baselineYear1Arm1')
      
      subj = run_case
      #tens_file = f'{dwi_prefix}_dti_tensor.nhdr'
      #mask_file = f'{dwi_prefix}_dti_FA_mask.nhdr'
      tens_file = f'{dwi_prefix}_dti{upsamp}_tensor.nhdr'
      mask_file = f'{dwi_prefix}_dti{upsamp}_FA_mask.nhdr'
      t1_file = f'{t1_prefix}_run-01_T1w.nii'
      num_iters = 1000 # Way too much, but want to understand convergence for all brain cases
      save_intermediate_results = True
      # TODO Something funky is going on with thresh_ratio.  Seems better to skip thresholding altogether
      # Valid values are supposed to be between 0 and 1, with 1 meaning don't threshold anything.
      # And yet, we need a value of at least 2.0 to get decent results for cubics (or even better is to
      # skip the thresholding altogether).
      # TODO test thresholds with real data, and decide whether to skip altogether.
      thresh_ratio = 1.0
      small_eval = 5e-3
      sigma = 1.5
      
      in_tens = ReadTensors(tens_file)
      in_mask = ReadScalars(mask_file)
      if t1_file:
        in_T1 = ReadScalars(t1_file)
      else:
        in_T1 = in_mask
      
      xsz=in_mask.shape[0]
      ysz=in_mask.shape[1]
      zsz=in_mask.shape[2]
      
      alpha, out_tens, out_mask, rks, intermed_results = mms.solve_3d(in_tens, in_mask, num_iters, [-2,2],
                                                                      thresh_ratio, save_intermediate_results, small_eval, sigma)
      
      out_tens_tri = np.zeros((xsz,ysz,zsz,6))
      out_tens_tri[:,:,:,0] = out_tens[:,:,:,0,0]
      out_tens_tri[:,:,:,1] = out_tens[:,:,:,0,1]
      out_tens_tri[:,:,:,2] = out_tens[:,:,:,0,2]
      out_tens_tri[:,:,:,3] = out_tens[:,:,:,1,1]
      out_tens_tri[:,:,:,4] = out_tens[:,:,:,1,2]
      out_tens_tri[:,:,:,5] = out_tens[:,:,:,2,2]
      
      if save_intermediate_results:
        scaled_tens_tri = np.zeros((xsz,ysz,zsz,6))
        scaled_tens_tri[:,:,:,0] = intermed_results['scaled_tensors'][:,:,:,0,0]
        scaled_tens_tri[:,:,:,1] = intermed_results['scaled_tensors'][:,:,:,0,1]
        scaled_tens_tri[:,:,:,2] = intermed_results['scaled_tensors'][:,:,:,0,2]
        scaled_tens_tri[:,:,:,3] = intermed_results['scaled_tensors'][:,:,:,1,1]
        scaled_tens_tri[:,:,:,4] = intermed_results['scaled_tensors'][:,:,:,1,2]
        scaled_tens_tri[:,:,:,5] = intermed_results['scaled_tensors'][:,:,:,2,2]

      out_T1 = in_T1.astype(float)[:,::-1,:].copy()
        
      WriteTensorNPArray(out_tens_tri, dwi_prefix + f'_thresh_{thresh_ratio}{upsamp}_tensors.nhdr')
      WriteTensorNPArray(in_tens, dwi_prefix + f'{upsamp}_orig_tensors.nhdr')
      WriteScalarNPArray(out_mask, dwi_prefix + f'{upsamp}_filt_mask.nhdr')
      WriteScalarNPArray(alpha, dwi_prefix + f'{upsamp}_alpha.nhdr')
      if t1_file:
        #WriteScalarNPArray(in_T1, out_prefix + '_T1.nhdr')
        WriteScalarNPArray(out_T1, t1_prefix + '_T1_flip_y.nhdr')
      if save_intermediate_results:
        WriteTensorNPArray(scaled_tens_tri, dwi_prefix + f'{upsamp}_scaled_tensors.nhdr')
      
      for_mat_file['orig_tensors'] = in_tens
      for_mat_file['thresh_tensors'] = out_tens
      for_mat_file['alpha'] = alpha
      for_mat_file['T1'] = in_T1
      for_mat_file['filt_mask'] = out_mask
      for_mat_file['rks'] = rks
      if save_intermediate_results:
        for_mat_file['scaled_tensors'] = intermed_results['scaled_tensors']
        
      
      tens_4_path = np.transpose(in_tens,(3,0,1,2))
      thresh_tens_4_path = np.transpose(out_tens_tri,(3,0,1,2))
      if save_intermediate_results:
        scaled_tens_4_path = np.transpose(scaled_tens_tri,(3,0,1,2))
      
      for_mat_file['tens_4_path'] = tens_4_path
      if save_intermediate_results:
        for_mat_file['scaled_tens_4_path'] = scaled_tens_4_path
        
      savemat(dwi_prefix + f'{upsamp}_results.mat',for_mat_file)
    except Exception as err:
      print('Exception', err, 'caught while processing subj', cc, '. Moving to next subject.')

  # end for each case
  #savemat(outdir + 'brain_results.mat',for_mat_file)

