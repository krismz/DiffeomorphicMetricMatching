import os
from lazy_imports import np
from lazy_imports import savemat
from data.io import ReadTensors, ReadScalars, WriteTensorNPArray, WriteScalarNPArray
from algo import euler, geodesic
import algo.metricModSolver as mms
from util.tensors import tens_6_to_tens_3x3

if __name__ == "__main__":
  inroot = '/usr/sci/projects/abcd/simdata/3d_cubics/'
  outroot = '/usr/sci/projects/abcd/simresults/3d_cubics/'
  #sims = ['sim1', 'sim2', 'sim3']
  #num_subjs_in_sim = [10, 30, 30]
  sims = ['noshape','sim1']
  num_subjs_in_sim = [10,10]
  prefix = 'metpy_3D_cubic'
  for sim, num_subjs in zip(sims, num_subjs_in_sim):
    indir = inroot + sim
    outdir = outroot + sim
    cases = ['1_novar', '2_novar', '_1_2_mean'] + [f'1_{cc}' for cc in range(num_subjs)]  + [f'2_{cc}' for cc in range(num_subjs)]

    for cc in cases:
      #try:
        run_case = f'{cc}'
        print("Running", run_case)
        for_mat_file = {}
        subj = run_case
  
        tens_file = f'{indir}/{prefix}{cc}_tens.nhdr'
        mask_file = f'{indir}/{prefix}{cc}_mask.nhdr'

        num_iters = 1500 
        save_intermediate_results = True
        # TODO Something funky is going on with thresh_ratio.  Seems better to skip thresholding altogether
        # Valid values are supposed to be between 0 and 1, with 1 meaning don't threshold anything.
        # And yet, we need a value of at least 2.0 to get decent results for cubics (or even better is to
        # skip the thresholding altogether).
        # TODO test thresholds with real data, and decide whether to skip altogether.
        thresh_ratio = 1.0
        small_eval = 5e-3
        sigma = 1.5 # smooth to compute alpha, then use ApplyAlphaToTensors afterward to scale the original unsmoothed tensors by alpha
      
        in_tens = ReadTensors(tens_file)
        in_mask = ReadScalars(mask_file)
        in_T1 = in_mask
      
        xsz=in_mask.shape[0]
        ysz=in_mask.shape[1]
        zsz=in_mask.shape[2]
      
        alpha, out_tens, out_mask, rks, intermed_results = mms.solve_3d(in_tens, in_mask, num_iters, [-2,2],
                                                                        thresh_ratio, save_intermediate_results, small_eval, sigma, do_mask_open = False)
      
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

        #out_T1 = in_T1.astype(float)[:,::-1,:].copy()
        out_T1 = in_T1.astype(float).copy()
        
        WriteTensorNPArray(out_tens_tri, f'{outdir}/{prefix}{cc}_thresh_{thresh_ratio}_tensors.nhdr')
        WriteTensorNPArray(in_tens, f'{outdir}/{prefix}{cc}_orig_tensors.nhdr')
        WriteScalarNPArray(in_mask, f'{outdir}/{prefix}{cc}_orig_mask.nhdr')
        WriteScalarNPArray(out_mask, f'{outdir}/{prefix}{cc}_filt_mask.nhdr')
        WriteScalarNPArray(alpha, f'{outdir}/{prefix}{cc}_alpha.nhdr')

        WriteScalarNPArray(out_T1, f'{outdir}/{prefix}{cc}_T1_flip_y.nhdr')
        if save_intermediate_results:
          WriteTensorNPArray(scaled_tens_tri, f'{outdir}/{prefix}{cc}_scaled_tensors.nhdr')
      
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
        
        savemat(f'{outdir}/{prefix}{cc}_results.mat',for_mat_file)
      #except Exception as err:
      #  print('Exception', err, 'caught while processing subj', cc, '. Moving to next subject.')

  # end for each case
  #savemat(outdir + 'brain_results.mat',for_mat_file)

