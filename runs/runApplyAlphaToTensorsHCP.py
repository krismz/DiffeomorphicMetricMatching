import os
from util import tensors # put this import first to avoid ImportError: libc10.so: cannot open shared object file
from lazy_imports import np
from lazy_imports import savemat
from data.io import ReadTensors, ReadScalars, WriteTensorNPArray, WriteScalarNPArray

if __name__ == "__main__":
  inroot = '/usr/sci/projects/HCP/Kris/2023Proposal/prepped_data/'
  cases=[sbj for sbj in os.listdir(inroot) if sbj[0] != '.']
  bval = 1000
  #bval = 'all'
  outdir = f'/usr/sci/scratch/kris/2023Proposal/HCPResults/B{bval}Results/'

  do_apply=True
  do_orig_scale=True # else do unsmoothed w/ filt mask
  
  for cc in cases:
    try:
      run_case = f'{cc}'
      print("Running", run_case)
      for_mat_file = {}
      out_prefix = outdir + run_case
      
      subj = run_case[0:6]
      indir = inroot + subj + '/'
      tens_file = f'{indir}/dti_{bval}_tensor_rreg.nhdr'
      mask_file = f'{indir}/dti_{bval}_FA_mask_rreg.nhdr'
      t1_file = '{indir}/t1_to_reft1_affreg.nhdr'   

      filt_mask_file = mask_file
      
      in_tens = ReadTensors(tens_file)
      print('in_tens shape:', in_tens.shape)
      if do_orig_scale:
        in_mask = ReadScalars(mask_file)
      else:
        in_mask = ReadScalars(filt_mask_file)

      alpha_file = f'{out_prefix}_alpha_rreg.nhdr'
      alpha = ReadScalars(alpha_file)

      xsz=in_mask.shape[0]
      ysz=in_mask.shape[1]
      zsz=in_mask.shape[2]

      if do_apply:
        scale_factor = 1.0
        max_tens = np.max(in_tens)
        while scale_factor * max_tens < 1:
          scale_factor = scale_factor * 10
      
        iso_tens = np.zeros((3,3))
        #iso_tens = np.eye(3)
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
              if np.sum(abs_tens[xx,yy,zz,:,:]) < 1e-10:
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

      if do_orig_scale:
        WriteScalarNPArray(in_mask, f'{out_prefix}_orig_mask_rreg.nhdr')
      
      if do_apply:
        if do_orig_scale:
          WriteTensorNPArray(scaled_tens_tri, f'{out_prefix}_scaled_orig_tensors_v2_rreg.nhdr')
        else:
          WriteTensorNPArray(scaled_tens_tri, f'{out_prefix}_scaled_unsmoothed_tensors_rreg.nhdr')
    
    except Exception as err:
      print('Exception', err, 'caught while processing subj', cc, '. Moving to next subject.')

  # end for each case

