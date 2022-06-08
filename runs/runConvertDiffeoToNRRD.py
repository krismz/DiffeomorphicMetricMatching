from lazy_imports import np
import scipy.io as sio
from data.io import readRaw, ReadScalars, ReadTensors, WriteTensorNPArray, WriteScalarNPArray
import pathlib

if __name__ == "__main__":
  subjs = []
  subjs.append('105923')
  subjs.append('108222')
  subjs.append('102715')
  subjs.append('100206')
  subjs.append('104416')
  subjs.append('107422')
  dim = 3.
  #bval = 1000
  bval = 'all'
  sigma = 1.5
  indir = f'/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/UKF_experiments/B{bval}Results/'
  atlases = []
  atlasprefs = []
  #atlases.append('BrainAtlasUkfB1000Aug17')
  #atlasprefs.append(f'B1000_6subj')
  #atlases.append('BrainAtlasUkfBallAug16')
  #atlasprefs.append(f'Ball_6subj')
  #atlases.append('BrainAtlasUkfBallAug27Unsmoothed')
  #atlasprefs.append('Ball_unsmoothed_6subj')
  atlases.append('BrainAtlasUkfBallAug27ScaledOrig')
  atlasprefs.append('Ball_scaledorig_6subj')
 
  for atlas, atlaspref in zip(atlases, atlasprefs):
    atlas_dir = f'/home/sci/hdai/Projects/Atlas3D/output/{atlas}/'
    out_atlas_dir = f'/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/atlases/{atlas}/'
    pathlib.Path(out_atlas_dir).mkdir(exist_ok=True) 
    
    for subj in subjs:
      in_prefix = atlas_dir + f'{subj}'
      out_prefix = out_atlas_dir + f'{subj}'

      

      # use phi_inv to transform images and metrics from subj to atlas space
      # use phi to transform images and metrics from atlas to subj space
      in_diffeo_fname = in_prefix + f'_phi_inv.mat'
      out_diffeo_fname = out_prefix + f'_phi_inv.nhdr'
  
      diffeo = sio.loadmat(in_diffeo_fname)['diffeo']
      WriteTensorNPArray(diffeo.transpose(1,2,3,0), out_diffeo_fname)

      in_diffeo_fname = in_prefix + f'_phi.mat'
      out_diffeo_fname = out_prefix + f'_phi.nhdr'
  
      diffeo = sio.loadmat(in_diffeo_fname)['diffeo']
      WriteTensorNPArray(diffeo.transpose(1,2,3,0), out_diffeo_fname)
