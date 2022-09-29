import os
from util import tensors # put this import first to avoid ImportError: libc10.so: cannot open shared object file
from lazy_imports import np
from lazy_imports import savemat
from data.io import ReadTensors, ReadScalars, WriteTensorNPArray, WriteScalarNPArray

if __name__ == "__main__":
  originroot = '/usr/sci/projects/abcd/anxiety_study/derivatives/mproc/'
  inroot = '/usr/sci/projects/abcd/anxiety_study/derivatives/metric_matching/'
  cases=[sbj for sbj in os.listdir(inroot) if sbj[:4] == 'sub-']

  session = 'ses-baselineYear1Arm1'
  run = 'run-01'
  upsamp=''
  #upsamp='_upsamp'
  
  do_apply=True
  do_orig_scale=True # else do unsmoothed w/ filt mask
  
  for cc in cases:
    try:
      run_case = f'{cc}'
      print("Running", run_case)
      for_mat_file = {}
      
      subj = cc
      indir = f'{inroot}{subj}/{session}/dwi/'
      tens_file = f'{subj}_{session}_dti{upsamp}_tensor.nhdr'
      mask_file = f'{subj}_{session}_dti{upsamp}_FA_mask.nhdr'
      filt_mask_file = f'{subj}_{session}{upsamp}_filt_mask.nhdr'
      
      in_tens = ReadTensors(indir+'/'+tens_file)
      if do_orig_scale:
        in_mask = ReadScalars(indir+'/'+mask_file)
      else:
        in_mask = ReadScalars(indir+'/' + filt_mask_file)

      brain_mask_file = f'{subj}_{session}_{run}_dwi{upsamp}_bet_mask.nii'
      in_brain_mask = ReadScalars(indir + brain_mask_file)

      t1indir = f'{inroot}{subj}/{session}/anat/'
      #t1_file = f'{subj}_{session}_{run}_T1w.nii'
      t1_file = f'{subj}_{session}_T1_flip_y.nhdr'
      in_t1 = ReadScalars(t1indir + t1_file)

      alpha_file = f'{subj}_{session}{upsamp}_alpha.nhdr'
      alpha = ReadScalars(indir + alpha_file)

      xsz=in_mask.shape[0]
      ysz=in_mask.shape[1]
      zsz=in_mask.shape[2]

      if do_apply:
        scale_factor = 1.0
        max_tens = np.max(in_tens)
        while scale_factor * max_tens < 1:
          scale_factor = scale_factor * 10
        
        iso_tens = np.zeros((3,3))
        iso_tens[0,0] = 1.0 / scale_factor
        iso_tens[1,1] = 1.0 / scale_factor
        iso_tens[2,2] = 1.0 / scale_factor 
        
        tens_full = np.zeros((xsz,ysz,zsz,3,3))
        if do_orig_scale:
          tens_full[:,:,:,0,0] = in_tens[:,:,:,0]
          tens_full[:,:,:,0,1] = in_tens[:,:,:,1]
          tens_full[:,:,:,0,2] = in_tens[:,:,:,2]
          tens_full[:,:,:,1,0] = tens_full[:,:,:,0,1]
          tens_full[:,:,:,2,0] = tens_full[:,:,:,0,2]
          tens_full[:,:,:,1,1] = in_tens[:,:,:,3]
          tens_full[:,:,:,1,2] = in_tens[:,:,:,4]
          tens_full[:,:,:,2,1] = tens_full[:,:,:,1,2]
          tens_full[:,:,:,2,2] = in_tens[:,:,:,5]
        else:
          tens_full[:,:,:,0,0] = in_mask * in_tens[:,:,:,0]
          tens_full[:,:,:,0,1] = in_mask * in_tens[:,:,:,1]
          tens_full[:,:,:,0,2] = in_mask * in_tens[:,:,:,2]
          tens_full[:,:,:,1,0] = tens_full[:,:,:,0,1]
          tens_full[:,:,:,2,0] = tens_full[:,:,:,0,2]
          tens_full[:,:,:,1,1] = in_mask * in_tens[:,:,:,3]
          tens_full[:,:,:,1,2] = in_mask * in_tens[:,:,:,4]
          tens_full[:,:,:,2,1] = tens_full[:,:,:,1,2]
          tens_full[:,:,:,2,2] = in_mask * in_tens[:,:,:,5]
        
        abs_tens = np.abs(tens_full)
        for xx in range(xsz):
          for yy in range(ysz):
            for zz in range(zsz):
              if np.sum(abs_tens[xx,yy,zz,:,:] < 1e-10):
                # if tensor is all 0, replace with scaled identity (ie isotropic tensor)
                tens_full[xx,yy,zz] = iso_tens
        
        scaled_tensors = tensors.scale_by_alpha(tens_full, alpha)
        
        scaled_tens_tri = np.zeros((xsz,ysz,zsz,6))
        scaled_tens_tri[:,:,:,0] = scaled_tensors[:,:,:,0,0]
        scaled_tens_tri[:,:,:,1] = scaled_tensors[:,:,:,0,1]
        scaled_tens_tri[:,:,:,2] = scaled_tensors[:,:,:,0,2]
        scaled_tens_tri[:,:,:,3] = scaled_tensors[:,:,:,1,1]
        scaled_tens_tri[:,:,:,4] = scaled_tensors[:,:,:,1,2]
        scaled_tens_tri[:,:,:,5] = scaled_tensors[:,:,:,2,2]

      out_t1 = in_t1.astype(float)[:,::-1,:].copy()

      if do_orig_scale:
        WriteScalarNPArray(in_mask, indir + '/{subj}_{session}{upsamp}_orig_mask.nhdr')
        
      #WriteScalarNPArray(out_t1, out_prefix + '_T1_flip_y.nhdr')
      WriteScalarNPArray(in_brain_mask, indir + '/{subj}_{session}{upsamp}_brain_mask.nhdr')
      if do_apply:
        if do_orig_scale:
          WriteTensorNPArray(scaled_tens_tri, indir + f'/{subj}_{session}{upsamp}_scaled_orig_tensors_v2.nhdr')
        else:
          WriteTensorNPArray(scaled_tens_tri, indir + f'{subj}_{session}{upsamp}_scaled_unsmoothed_tensors.nhdr')
      
    except Exception as err:
      print('Exception', err, 'caught while processing subj', cc, '. Moving to next subject.')

  # end for each case
  #savemat(outdir + 'brain_results.mat',for_mat_file)

