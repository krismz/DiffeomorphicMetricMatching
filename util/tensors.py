import math
#from lazy_imports import torch
# currently an incompatibility between lazy import of torch and importing torch_sym3eig
#import torch
#import torch_sym3eig # installed from https://github.com/nnaisense/pytorch_sym3eig
# need to be imported after torch is imported
#from torch_sym3eig import Sym3Eig as se
from lazy_imports import torch
from lazy_imports import se
#from lazy_imports import torch_sym3eig # installed from https://github.com/nnaisense/pytorch_sym3eig
from lazy_imports import np
from lazy_imports import sitk
from data.convert import get_framework
from data.convert import GetNPArrayFromSITK, GetSITKImageFromNP
from numba import jit, njit, prange
# from interp3d import interp_3d # installed from https://github.com/jglaser/interp3d/blob/master/interp3d/interp_3d.py

# uncomment this for legit @profile when not using kernprof
def profile(blah):                
  return blah

def batch_cholesky_v2(tens):
  fw, fw_name = get_framework(tens)
  if fw_name == 'numpy':
    nan = fw.nan
  else:
    nan = fw.tensor(float('nan'))
  L = fw.zeros_like(tens)
  for xx in range(tens.shape[0]):
    for yy in range(tens.shape[1]):
      for zz in range(tens.shape[2]):
        try:
          L[xx,yy,zz] = fw.linalg.cholesky(tens[xx,yy,zz])
        except:
          L[xx,yy,zz] = nan * fw.ones((tens.shape[-2:]))
  return L

def batch_cholesky(tens):
  # from https://stackoverflow.com/questions/60230464/pytorch-torch-cholesky-ignoring-exception
  # will get NaNs instead of exception where cholesky is invalid
  fw, fw_name = get_framework(tens)
  L = fw.zeros_like(tens)

  for i in range(tens.shape[-1]):
    for j in range(i+1):
      s = 0.0
      for k in range(j):
        s = s + L[...,i,k].clone() * L[...,j,k].clone()

      L[...,i,j] = fw.sqrt((tens[...,i,i] - s).clamp(min=1.0e-15)) if (i == j) else \
                      (1.0 / L[...,j,j].clone().clamp(min=1.0e-15) * (tens[...,i,j] - s))
  return L

def smooth_tensors(tens, sigma):
  fw, fw_name = get_framework(tens)
  if fw_name == 'numpy':
    filt_tens = GetNPArrayFromSITK(
                sitk.RecursiveGaussian(sitk.RecursiveGaussian(sitk.RecursiveGaussian(
                  GetSITKImageFromNP(tens,True), sigma=sigma,direction=0), sigma=sigma,direction=1), sigma=sigma, direction=2),True)
  else:
    filt_tens = torch.from_numpy(GetNPArrayFromSITK(
                sitk.RecursiveGaussian(sitk.RecursiveGaussian(sitk.RecursiveGaussian(
                GetSITKImageFromNP(tens.cpu().numpy(),True), sigma=sigma,direction=0), sigma=sigma,direction=1), sigma=sigma,direction=2),True))
  return(filt_tens)

def tens_6_to_tens_3x3(tens):
  tens_full = np.zeros((tens.shape[0], tens.shape[1], tens.shape[2], 3, 3))
  tens_full[:,:,:,0,0] = tens[:,:,:,0]
  tens_full[:,:,:,0,1] = tens[:,:,:,1]
  tens_full[:,:,:,1,0] = tens[:,:,:,1]
  tens_full[:,:,:,0,2] = tens[:,:,:,2]
  tens_full[:,:,:,2,0] = tens[:,:,:,2]
  tens_full[:,:,:,1,1] = tens[:,:,:,3]
  tens_full[:,:,:,1,2] = tens[:,:,:,4]
  tens_full[:,:,:,2,1] = tens[:,:,:,4]
  tens_full[:,:,:,2,2] = tens[:,:,:,5]
  return(tens_full)

def tens_3x3_to_tens_6(tens):
  tens_tri = np.zeros((tens.shape[0], tens.shape[1], tens.shape[2], 6))
  tens_tri[:,:,:,0] = tens[:,:,:,0,0]
  tens_tri[:,:,:,1] = tens[:,:,:,0,1]
  tens_tri[:,:,:,2] = tens[:,:,:,0,2]
  tens_tri[:,:,:,3] = tens[:,:,:,1,1]
  tens_tri[:,:,:,4] = tens[:,:,:,1,2]
  tens_tri[:,:,:,5] = tens[:,:,:,2,2]
  return(tens_tri)

def direction(coordinate, tensor_field):
  tens = tens_interp(coordinate[0], coordinate[1], tensor_field)
  u, v = eigen_vec(tens)
  return (np.array([u, v]))

def direction_torch(coordinate, tensor_field):
  tens = tens_interp_torch(coordinate[0], coordinate[1], tensor_field)
  u, v = eigen_vec_torch(tens)
  return (torch.tensor([u, v]))

def direction_3d(coordinate, tensor_field):
  tens = tens_interp_3d(coordinate[0], coordinate[1], coordinate[2], tensor_field)
  u, v, w = eigen_vec_3d(tens)
  return (np.array([u, v, w]))

def direction_3d_torch(coordinate, tensor_field):
  tens = tens_interp_3d_torch(coordinate[0], coordinate[1], coordinate[2], tensor_field)
  u, v, w = eigen_vec_3d_torch(tens)
  return (torch.tensor((u, v, w)))

#@jit(nopython=True)
def batch_direction_3d(coordinates, tensor_field):
  tens = batch_tens_interp_3d(coordinates[:,0], coordinates[:,1], coordinates[:,2], tensor_field)
  directions = batch_eigen_vec_3d(tens)
  return (directions)

def batch_direction_3d_torch(coordinates, tensor_field):
  tens = batch_tens_interp_3d_torch(coordinates[:,0], coordinates[:,1], coordinates[:,2], tensor_field)
  directions = batch_eigen_vec_3d_torch(tens)
  return (directions)


def fractional_anisotropy(g):
    e, _ = torch.symeig(g)
    lambd1 = e[:,:,:,0]
    lambd2 = e[:,:,:,1]
    lambd3 = e[:,:,:,2]
    mean = torch.mean(e,dim=len(e.shape)-1)
    return torch.sqrt(3.*(torch.pow((lambd1-mean),2)+torch.pow((lambd2-mean),2)+torch.pow((lambd3-mean),2)))/\
    torch.sqrt(2.*(torch.pow(lambd1,2)+torch.pow(lambd2,2)+torch.pow(lambd3,2)))

def eigen_vec(tens):
  evals, evecs = np.linalg.eigh(tens)
  [u, v] = evecs[:, evals.argmax()]
  return (u, v)

def eigen_vec_torch(tens):
  #evals, evecs =torch.symeig(tens, eigenvectors=True)
  #[u, v] = evecs[:, evals.argmax()]
  evals, evecs = torch_sym3eig.Sym3Eig.apply(tens.reshape((-1,3,3)))
  [u, v] = evecs[:, :, evals.argmax()]
  return (u,v)

def eigen_vec_3d(tens):
  evals, evecs = np.linalg.eigh(tens)
  [u, v, w] = evecs[:, evals.argmax()]
  return (u, v, w)

def eigen_vec_3d_torch(tens):
  #evals, evecs = torch.symeig(tens,eigenvectors=True)
  #evals, evecs = torch.linalg.eigh(tens)
  #[u, v, w] = evecs[:, evals.argmax()]
  evals, evecs = torch_sym3eig.Sym3Eig.apply(tens.reshape((-1,3,3)))
  [u, v, w] = evecs[0, :, evals.argmax()]
  return (u,v,w)

#@jit(nopython=True)
def batch_eigen_vec_3d(tens):
  evals, evecs = np.linalg.eigh(tens)
  #return (evecs[:, :, evals.argmax(axis=1)])
  idx = np.expand_dims(np.expand_dims(evals.argmax(axis=1),axis=-1),axis=-1)
  return(np.take_along_axis(evecs, idx, axis=2).reshape((-1,3)))

def batch_eigen_vec_3d_torch(tens):
  #evals, evecs = torch.symeig(tens,eigenvectors=True)
  #evals, evecs = torch.linalg.eigh(tens)
  #[u, v, w] = evecs[:, evals.argmax()]
  evals, evecs = torch_sym3eig.Sym3Eig.apply(tens.reshape((-1,3,3)))
  #return (evecs[:, :, evals.argmax(axis=1)])
  idx = torch.unsqueeze(torch.unsqueeze(torch.argmax(evals, dim=1),-1),-1)
  return(torch.take_along_dim(evecs, idx, dim = 2).reshape((-1,3)))


def circ_shift(I, shift):
  I = np.roll(I, shift[0], axis=0)
  I = np.roll(I, shift[1], axis=1)
  return (I)

def circ_shift_torch(I, shift):
  I = torch.roll(I, shift[0], dims=0)
  I = torch.roll(I, shift[1], dims=1)
  return (I)

def circ_shift_3d(I, shift):
  I = np.roll(I, shift[0], axis=0)
  I = np.roll(I, shift[1], axis=1)
  I = np.roll(I, shift[2], axis=2)
  return (I)

def circ_shift_3d_torch(I, shift):
  I = torch.roll(I, shift[0], dims=0)
  I = torch.roll(I, shift[1], dims=1)
  I = torch.roll(I, shift[2], dims=2)
  return (I)

def tens_interp(x, y, tensor_field):
  tens = np.zeros((2, 2))
  eps11 = tensor_field[0, :, :]
  eps12 = tensor_field[1, :, :]
  eps22 = tensor_field[2, :, :]
  if (math.floor(x) < 0) or (math.ceil(x) >= eps11.shape[0]) or (math.floor(y) < 0) or (math.ceil(y) >= eps11.shape[1]):
     # data is out of bounds, return identity tensor
     tens[0,0] = 1
     tens[1,1] = 1
     return(tens)
  if x == math.floor(x) and y == math.floor(y):
    tens[0, 0] = eps11[int(x), int(y)]
    tens[0, 1] = eps12[int(x), int(y)]
    tens[1, 0] = eps12[int(x), int(y)]
    tens[1, 1] = eps22[int(x), int(y)]
  elif x == math.floor(x) and y != math.floor(y):
    tens[0, 0] = abs(y - math.floor(y)) * eps11[int(x), math.ceil(y)] + \
           abs(y - math.ceil(y)) * eps11[int(x), math.floor(y)]
    tens[0, 1] = abs(y - math.floor(y)) * eps12[int(x), math.ceil(y)] + \
           abs(y - math.ceil(y)) * eps12[int(x), math.floor(y)]
    tens[1, 0] = abs(y - math.floor(y)) * eps12[int(x), math.ceil(y)] + \
           abs(y - math.ceil(y)) * eps12[int(x), math.floor(y)]
    tens[1, 1] = abs(y - math.floor(y)) * eps22[int(x), math.ceil(y)] + \
           abs(y - math.ceil(y)) * eps22[int(x), math.floor(y)]
  elif x != math.floor(x) and y == math.floor(y):
    tens[0, 0] = abs(x - math.floor(x)) * eps11[math.ceil(x), int(y)] + \
           abs(x - math.ceil(x)) * eps11[math.floor(x), int(y)]
    tens[0, 1] = abs(x - math.floor(x)) * eps12[math.ceil(x), int(y)] + \
           abs(x - math.ceil(x)) * eps12[math.floor(x), int(y)]
    tens[1, 0] = abs(x - math.floor(x)) * eps12[math.ceil(x), int(y)] + \
           abs(x - math.ceil(x)) * eps12[math.floor(x), int(y)]
    tens[1, 1] = abs(x - math.floor(x)) * eps22[math.ceil(x), int(y)] + \
           abs(x - math.ceil(x)) * eps22[math.floor(x), int(y)]
  else:
    tens[0, 0] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps11[math.ceil(x), math.ceil(y)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps11[math.ceil(x), math.floor(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps11[math.floor(x), math.ceil(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps11[math.floor(x), math.floor(y)]
    tens[0, 1] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps12[math.ceil(x), math.ceil(y)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps12[math.ceil(x), math.floor(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps12[math.floor(x), math.ceil(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps12[math.floor(x), math.floor(y)]
    tens[1, 0] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps12[math.ceil(x), math.ceil(y)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps12[math.ceil(x), math.floor(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps12[math.floor(x), math.ceil(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps12[math.floor(x), math.floor(y)]
    tens[1, 1] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps22[math.ceil(x), math.ceil(y)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps22[math.ceil(x), math.floor(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps22[math.floor(x), math.ceil(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps22[math.floor(x), math.floor(y)]

  return (tens)
# end tens_interp

def tens_interp_torch(x, y, tensor_field):
  tens = torch.zeros((2, 2))
  eps11 = tensor_field[0, :, :]
  eps12 = tensor_field[1, :, :]
  eps22 = tensor_field[2, :, :]
  if x == math.floor(x) and y == math.floor(y):
    tens[0, 0] = eps11[int(x), int(y)]
    tens[0, 1] = eps12[int(x), int(y)]
    tens[1, 0] = eps12[int(x), int(y)]
    tens[1, 1] = eps22[int(x), int(y)]
  elif x == math.floor(x) and y != math.floor(y):
    tens[0, 0] = abs(y - math.floor(y)) * eps11[int(x), math.ceil(y)] + \
           abs(y - math.ceil(y)) * eps11[int(x), math.floor(y)]
    tens[0, 1] = abs(y - math.floor(y)) * eps12[int(x), math.ceil(y)] + \
           abs(y - math.ceil(y)) * eps12[int(x), math.floor(y)]
    tens[1, 0] = abs(y - math.floor(y)) * eps12[int(x), math.ceil(y)] + \
           abs(y - math.ceil(y)) * eps12[int(x), math.floor(y)]
    tens[1, 1] = abs(y - math.floor(y)) * eps22[int(x), math.ceil(y)] + \
           abs(y - math.ceil(y)) * eps22[int(x), math.floor(y)]
  elif x != math.floor(x) and y == math.floor(y):
    tens[0, 0] = abs(x - math.floor(x)) * eps11[math.ceil(x), int(y)] + \
           abs(x - math.ceil(x)) * eps11[math.floor(x), int(y)]
    tens[0, 1] = abs(x - math.floor(x)) * eps12[math.ceil(x), int(y)] + \
           abs(x - math.ceil(x)) * eps12[math.floor(x), int(y)]
    tens[1, 0] = abs(x - math.floor(x)) * eps12[math.ceil(x), int(y)] + \
           abs(x - math.ceil(x)) * eps12[math.floor(x), int(y)]
    tens[1, 1] = abs(x - math.floor(x)) * eps22[math.ceil(x), int(y)] + \
           abs(x - math.ceil(x)) * eps22[math.floor(x), int(y)]
  else:
    tens[0, 0] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps11[math.ceil(x), math.ceil(y)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps11[math.ceil(x), math.floor(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps11[math.floor(x), math.ceil(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps11[math.floor(x), math.floor(y)]
    tens[0, 1] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps12[math.ceil(x), math.ceil(y)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps12[math.ceil(x), math.floor(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps12[math.floor(x), math.ceil(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps12[math.floor(x), math.floor(y)]
    tens[1, 0] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps12[math.ceil(x), math.ceil(y)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps12[math.ceil(x), math.floor(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps12[math.floor(x), math.ceil(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps12[math.floor(x), math.floor(y)]
    tens[1, 1] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps22[math.ceil(x), math.ceil(y)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps22[math.ceil(x), math.floor(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps22[math.floor(x), math.ceil(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps22[math.floor(x), math.floor(y)]

  return (tens)

def tens_interp_3d(x, y, z, tensor_field):
  tens = np.zeros((3, 3))
  eps11 = tensor_field[0, :, :]
  eps12 = tensor_field[1, :, :]
  eps13 = tensor_field[2, :, :]
  eps22 = tensor_field[3, :, :]
  eps23 = tensor_field[4, :, :]
  eps33 = tensor_field[5, :, :]
  if ((math.floor(x) < 0) or (math.ceil(x) >= eps11.shape[0])
      or (math.floor(y) < 0) or (math.ceil(y) >= eps11.shape[1])
      or (math.floor(z) < 0) or (math.ceil(z) >= eps11.shape[2])):
     # data is out of bounds, return identity tensor
     tens[0,0] = 1
     tens[1,1] = 1
     tens[2,2] = 1
     return(tens)

   
  if x == math.floor(x) and y == math.floor(y) and z == math.floor(z):
    tens[0, 0] = eps11[int(x), int(y), int(z)]
    tens[0, 1] = eps12[int(x), int(y), int(z)]
    tens[1, 0] = eps12[int(x), int(y), int(z)]
    tens[0, 2] = eps13[int(x), int(y), int(z)]
    tens[2, 0] = eps13[int(x), int(y), int(z)]
    tens[1, 1] = eps22[int(x), int(y), int(z)]
    tens[1, 2] = eps23[int(x), int(y), int(z)]
    tens[2, 1] = eps23[int(x), int(y), int(z)]
    tens[2, 2] = eps33[int(x), int(y), int(z)]
  elif x == math.floor(x) and y != math.floor(y) and z != math.floor(z):
    tens[0, 0] = abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps11[int(x), math.ceil(y), math.ceil(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps11[int(x), math.floor(y), math.ceil(z)] \
                 + abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps11[int(x), math.ceil(y), math.floor(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps11[int(x), math.floor(y), math.floor(z)] 
    tens[0, 1] = abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps12[int(x), math.ceil(y), math.ceil(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps12[int(x), math.floor(y), math.ceil(z)] \
                 + abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps12[int(x), math.ceil(y), math.floor(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps12[int(x), math.floor(y), math.floor(z)] 
    tens[1, 0] = tens[0,1]
    tens[0, 2] = abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps13[int(x), math.ceil(y), math.ceil(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps13[int(x), math.floor(y), math.ceil(z)] \
                 + abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps13[int(x), math.ceil(y), math.floor(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps13[int(x), math.floor(y), math.floor(z)] 
    tens[2, 0] = tens[0,2]
    tens[1, 1] = abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps22[int(x), math.ceil(y), math.ceil(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps22[int(x), math.floor(y), math.ceil(z)] \
                 + abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps22[int(x), math.ceil(y), math.floor(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps22[int(x), math.floor(y), math.floor(z)] 
    tens[1, 2] = abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps23[int(x), math.ceil(y), math.ceil(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps23[int(x), math.floor(y), math.ceil(z)] \
                 + abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps23[int(x), math.ceil(y), math.floor(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps23[int(x), math.floor(y), math.floor(z)] 
    tens[2, 1] = tens[1,2]
    tens[2, 2] = abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps33[int(x), math.ceil(y), math.ceil(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps33[int(x), math.floor(y), math.ceil(z)] \
                 + abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps33[int(x), math.ceil(y), math.floor(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps33[int(x), math.floor(y), math.floor(z)]
  elif x == math.floor(x) and y == math.floor(y) and z != math.floor(z):
    tens[0, 0] = abs(z - math.floor(z)) * eps11[int(x), int(y), math.ceil(z)] \
                 + abs(z - math.ceil(z)) * eps11[int(x), int(y), math.floor(z)] 
    tens[0, 1] = abs(z - math.floor(z)) * eps12[int(x), int(y), math.ceil(z)] \
                 + abs(z - math.ceil(z)) * eps12[int(x), int(y), math.floor(z)] 
    tens[1, 0] = tens[0,1]
    tens[0, 2] = abs(z - math.floor(z)) * eps13[int(x), int(y), math.ceil(z)] \
                 + abs(z - math.ceil(z)) * eps13[int(x), int(y), math.floor(z)] 
    tens[2, 0] = tens[0,2]
    tens[1, 1] = abs(z - math.floor(z)) * eps22[int(x), int(y), math.ceil(z)] \
                 + abs(z - math.ceil(z)) * eps22[int(x), int(y), math.floor(z)] 
    tens[1, 2] = abs(z - math.floor(z)) * eps23[int(x), int(y), math.ceil(z)] \
                 + abs(z - math.ceil(z)) * eps23[int(x), int(y), math.floor(z)] 
    tens[2, 1] = tens[1,2]
    tens[2, 2] = abs(z - math.floor(z)) * eps33[int(x), int(y), math.ceil(z)] \
                 + abs(z - math.ceil(z)) * eps33[int(x), int(y), math.floor(z)]   
  elif x != math.floor(x) and y == math.floor(y) and z != math.floor(z):
    tens[0, 0] = abs(x - math.floor(x)) * abs(z - math.floor(z)) * eps11[math.ceil(x), int(y), math.ceil(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.floor(z)) * eps11[math.floor(x), int(y), math.ceil(z)] \
                 + abs(x - math.floor(x)) * abs(z - math.ceil(z)) * eps11[math.ceil(x), int(y), math.floor(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.ceil(z)) * eps11[math.floor(x), int(y), math.floor(z)]
    tens[0, 1] = abs(x - math.floor(x)) * abs(z - math.floor(z)) * eps12[math.ceil(x), int(y), math.ceil(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.floor(z)) * eps12[math.floor(x), int(y), math.ceil(z)] \
                 + abs(x - math.floor(x)) * abs(z - math.ceil(z)) * eps12[math.ceil(x), int(y), math.floor(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.ceil(z)) * eps12[math.floor(x), int(y), math.floor(z)]
    tens[1, 0] = tens[0,1]
    tens[0, 2] = abs(x - math.floor(x)) * abs(z - math.floor(z)) * eps13[math.ceil(x), int(y), math.ceil(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.floor(z)) * eps13[math.floor(x), int(y), math.ceil(z)] \
                 + abs(x - math.floor(x)) * abs(z - math.ceil(z)) * eps13[math.ceil(x), int(y), math.floor(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.ceil(z)) * eps13[math.floor(x), int(y), math.floor(z)]
    tens[2, 0] = tens[0,2]
    tens[1, 1] = abs(x - math.floor(x)) * abs(z - math.floor(z)) * eps22[math.ceil(x), int(y), math.ceil(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.floor(z)) * eps22[math.floor(x), int(y), math.ceil(z)] \
                 + abs(x - math.floor(x)) * abs(z - math.ceil(z)) * eps22[math.ceil(x), int(y), math.floor(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.ceil(z)) * eps22[math.floor(x), int(y), math.floor(z)]
    tens[1, 2] = abs(x - math.floor(x)) * abs(z - math.floor(z)) * eps23[math.ceil(x), int(y), math.ceil(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.floor(z)) * eps23[math.floor(x), int(y), math.ceil(z)] \
                 + abs(x - math.floor(x)) * abs(z - math.ceil(z)) * eps23[math.ceil(x), int(y), math.floor(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.ceil(z)) * eps23[math.floor(x), int(y), math.floor(z)]
    tens[2, 1] = tens[1,2]
    tens[2, 2] = abs(x - math.floor(x)) * abs(z - math.floor(z)) * eps33[math.ceil(x), int(y), math.ceil(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.floor(z)) * eps33[math.floor(x), int(y), math.ceil(z)] \
                 + abs(x - math.floor(x)) * abs(z - math.ceil(z)) * eps33[math.ceil(x), int(y), math.floor(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.ceil(z)) * eps33[math.floor(x), int(y), math.floor(z)]
  elif x != math.floor(x) and y == math.floor(y) and z == math.floor(z):
    tens[0, 0] = abs(x - math.floor(x)) * eps11[math.ceil(x), int(y),int(z)] \
                 + abs(x - math.ceil(x)) * eps11[math.floor(x), int(y), int(z)]
    tens[0, 1] = abs(x - math.floor(x)) * eps12[math.ceil(x), int(y),int(z)] \
                 + abs(x - math.ceil(x)) * eps12[math.floor(x), int(y), int(z)]
    tens[1, 0] = tens[0,1]
    tens[0, 2] = abs(x - math.floor(x)) * eps13[math.ceil(x), int(y),int(z)] \
                 + abs(x - math.ceil(x)) * eps13[math.floor(x), int(y), int(z)]
    tens[2, 0] = tens[0,2]
    tens[1, 1] = abs(x - math.floor(x)) * eps22[math.ceil(x), int(y),int(z)] \
                 + abs(x - math.ceil(x)) * eps22[math.floor(x), int(y), int(z)]
    tens[1, 2] = abs(x - math.floor(x)) * eps23[math.ceil(x), int(y),int(z)] \
                 + abs(x - math.ceil(x)) * eps23[math.floor(x), int(y), int(z)]
    tens[2, 1] = tens[1,2]
    tens[2, 2] = abs(x - math.floor(x)) * eps33[math.ceil(x), int(y),int(z)] \
                 + abs(x - math.ceil(x)) * eps33[math.floor(x), int(y), int(z)]
  elif x != math.floor(x) and y != math.floor(y) and z == math.floor(z):
    tens[0, 0] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps11[math.ceil(x), math.ceil(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps11[math.floor(x), math.ceil(y), int(z)] \
                 + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps11[math.ceil(x), math.floor(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps11[math.floor(x), math.floor(y), int(z)]  
    tens[0, 1] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps12[math.ceil(x), math.ceil(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps12[math.floor(x), math.ceil(y), int(z)] \
                 + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps12[math.ceil(x), math.floor(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps12[math.floor(x), math.floor(y), int(z)]  
    tens[1, 0] = tens[0, 1]
    tens[0, 2] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps13[math.ceil(x), math.ceil(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps13[math.floor(x), math.ceil(y), int(z)] \
                 + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps13[math.ceil(x), math.floor(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps13[math.floor(x), math.floor(y), int(z)]  
    tens[2, 0] = tens[0, 2]
    tens[1, 1] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps22[math.ceil(x), math.ceil(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps22[math.floor(x), math.ceil(y), int(z)] \
                 + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps22[math.ceil(x), math.floor(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps22[math.floor(x), math.floor(y), int(z)]  
    tens[1, 2] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps23[math.ceil(x), math.ceil(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps23[math.floor(x), math.ceil(y), int(z)] \
                 + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps23[math.ceil(x), math.floor(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps23[math.floor(x), math.floor(y), int(z)]  
    tens[2, 1] = tens[1, 2]
    tens[2, 2] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps33[math.ceil(x), math.ceil(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps33[math.floor(x), math.ceil(y), int(z)] \
                 + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps33[math.ceil(x), math.floor(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps33[math.floor(x), math.floor(y), int(z)]  
  elif x == math.floor(x) and y != math.floor(y) and z == math.floor(z):
    tens[0, 0] = abs(y - math.floor(y)) * eps11[int(x), math.ceil(y), int(z)] \
                 + abs(y - math.ceil(y)) * eps11[int(x), math.floor(y), int(z)]
    tens[0, 1] = abs(y - math.floor(y)) * eps12[int(x), math.ceil(y), int(z)] \
                 + abs(y - math.ceil(y)) * eps12[int(x), math.floor(y), int(z)]
    tens[1, 0] = tens[0,1]
    tens[0, 2] = abs(y - math.floor(y)) * eps13[int(x), math.ceil(y), int(z)] \
                 + abs(y - math.ceil(y)) * eps13[int(x), math.floor(y), int(z)]
    tens[2, 0] = tens[0,2]
    tens[1, 1] = abs(y - math.floor(y)) * eps22[int(x), math.ceil(y), int(z)] \
                 + abs(y - math.ceil(y)) * eps22[int(x), math.floor(y), int(z)]
    tens[1, 2] = abs(y - math.floor(y)) * eps23[int(x), math.ceil(y), int(z)] \
                 + abs(y - math.ceil(y)) * eps23[int(x), math.floor(y), int(z)]
    tens[2, 1] = tens[1,2]
    tens[2, 2] = abs(y - math.floor(y)) * eps33[int(x), math.ceil(y), int(z)] \
                 + abs(y - math.ceil(y)) * eps33[int(x), math.floor(y), int(z)]
  else:
    tens[0, 0] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps11[math.ceil(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps11[math.ceil(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps11[math.floor(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps11[math.floor(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps11[math.ceil(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps11[math.ceil(x), math.floor(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps11[math.floor(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps11[math.floor(x), math.floor(y), math.floor(z)]
    tens[0, 1] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps12[math.ceil(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps12[math.ceil(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps12[math.floor(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps12[math.floor(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps12[math.ceil(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps12[math.ceil(x), math.floor(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps12[math.floor(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps12[math.floor(x), math.floor(y), math.floor(z)]
    tens[1, 0] = tens[0,1]
    tens[0, 2] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps13[math.ceil(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps13[math.ceil(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps13[math.floor(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps13[math.floor(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps13[math.ceil(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps13[math.ceil(x), math.floor(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps13[math.floor(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps13[math.floor(x), math.floor(y), math.floor(z)]
    tens[2, 0] = tens[0,2]
    tens[1, 1] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps22[math.ceil(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps22[math.ceil(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps22[math.floor(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps22[math.floor(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps22[math.ceil(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps22[math.ceil(x), math.floor(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps22[math.floor(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps22[math.floor(x), math.floor(y), math.floor(z)]
    tens[1, 2] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps23[math.ceil(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps23[math.ceil(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps23[math.floor(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps23[math.floor(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps23[math.ceil(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps23[math.ceil(x), math.floor(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps23[math.floor(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps23[math.floor(x), math.floor(y), math.floor(z)]
    tens[2,1] = tens[1,2]
    tens[2, 2] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps33[math.ceil(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps33[math.ceil(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps33[math.floor(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps33[math.floor(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps33[math.ceil(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps33[math.ceil(x), math.floor(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps33[math.floor(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps33[math.floor(x), math.floor(y), math.floor(z)]

  return (tens)
# end tens_interp_3d

def tens_interp_3d_torch(x, y, z, tensor_field):
  tens = torch.zeros((3, 3))
  eps11 = tensor_field[0, :, :]
  eps12 = tensor_field[1, :, :]
  eps13 = tensor_field[2, :, :]
  eps22 = tensor_field[3, :, :]
  eps23 = tensor_field[4, :, :]
  eps33 = tensor_field[5, :, :]
  if ((math.floor(x) < 0) or (math.ceil(x) >= eps11.shape[0])
      or (math.floor(y) < 0) or (math.ceil(y) >= eps11.shape[1])
      or (math.floor(z) < 0) or (math.ceil(z) >= eps11.shape[2])):
     # data is out of bounds, return identity tensor
     tens[0,0] = 1
     tens[1,1] = 1
     tens[2,2] = 1
     return(tens)

   
  if x == math.floor(x) and y == math.floor(y) and z == math.floor(z):
    tens[0, 0] = eps11[int(x), int(y), int(z)]
    tens[0, 1] = eps12[int(x), int(y), int(z)]
    tens[1, 0] = eps12[int(x), int(y), int(z)]
    tens[0, 2] = eps13[int(x), int(y), int(z)]
    tens[2, 0] = eps13[int(x), int(y), int(z)]
    tens[1, 1] = eps22[int(x), int(y), int(z)]
    tens[1, 2] = eps23[int(x), int(y), int(z)]
    tens[2, 1] = eps23[int(x), int(y), int(z)]
    tens[2, 2] = eps33[int(x), int(y), int(z)]
  elif x == math.floor(x) and y != math.floor(y) and z != math.floor(z):
    tens[0, 0] = abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps11[int(x), math.ceil(y), math.ceil(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps11[int(x), math.floor(y), math.ceil(z)] \
                 + abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps11[int(x), math.ceil(y), math.floor(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps11[int(x), math.floor(y), math.floor(z)] 
    tens[0, 1] = abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps12[int(x), math.ceil(y), math.ceil(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps12[int(x), math.floor(y), math.ceil(z)] \
                 + abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps12[int(x), math.ceil(y), math.floor(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps12[int(x), math.floor(y), math.floor(z)] 
    tens[1, 0] = tens[0,1]
    tens[0, 2] = abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps13[int(x), math.ceil(y), math.ceil(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps13[int(x), math.floor(y), math.ceil(z)] \
                 + abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps13[int(x), math.ceil(y), math.floor(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps13[int(x), math.floor(y), math.floor(z)] 
    tens[2, 0] = tens[0,2]
    tens[1, 1] = abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps22[int(x), math.ceil(y), math.ceil(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps22[int(x), math.floor(y), math.ceil(z)] \
                 + abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps22[int(x), math.ceil(y), math.floor(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps22[int(x), math.floor(y), math.floor(z)] 
    tens[1, 2] = abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps23[int(x), math.ceil(y), math.ceil(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps23[int(x), math.floor(y), math.ceil(z)] \
                 + abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps23[int(x), math.ceil(y), math.floor(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps23[int(x), math.floor(y), math.floor(z)] 
    tens[2, 1] = tens[1,2]
    tens[2, 2] = abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps33[int(x), math.ceil(y), math.ceil(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps33[int(x), math.floor(y), math.ceil(z)] \
                 + abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps33[int(x), math.ceil(y), math.floor(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps33[int(x), math.floor(y), math.floor(z)]
  elif x == math.floor(x) and y == math.floor(y) and z != math.floor(z):
    tens[0, 0] = abs(z - math.floor(z)) * eps11[int(x), int(y), math.ceil(z)] \
                 + abs(z - math.ceil(z)) * eps11[int(x), int(y), math.floor(z)] 
    tens[0, 1] = abs(z - math.floor(z)) * eps12[int(x), int(y), math.ceil(z)] \
                 + abs(z - math.ceil(z)) * eps12[int(x), int(y), math.floor(z)] 
    tens[1, 0] = tens[0,1]
    tens[0, 2] = abs(z - math.floor(z)) * eps13[int(x), int(y), math.ceil(z)] \
                 + abs(z - math.ceil(z)) * eps13[int(x), int(y), math.floor(z)] 
    tens[2, 0] = tens[0,2]
    tens[1, 1] = abs(z - math.floor(z)) * eps22[int(x), int(y), math.ceil(z)] \
                 + abs(z - math.ceil(z)) * eps22[int(x), int(y), math.floor(z)] 
    tens[1, 2] = abs(z - math.floor(z)) * eps23[int(x), int(y), math.ceil(z)] \
                 + abs(z - math.ceil(z)) * eps23[int(x), int(y), math.floor(z)] 
    tens[2, 1] = tens[1,2]
    tens[2, 2] = abs(z - math.floor(z)) * eps33[int(x), int(y), math.ceil(z)] \
                 + abs(z - math.ceil(z)) * eps33[int(x), int(y), math.floor(z)]   
  elif x != math.floor(x) and y == math.floor(y) and z != math.floor(z):
    tens[0, 0] = abs(x - math.floor(x)) * abs(z - math.floor(z)) * eps11[math.ceil(x), int(y), math.ceil(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.floor(z)) * eps11[math.floor(x), int(y), math.ceil(z)] \
                 + abs(x - math.floor(x)) * abs(z - math.ceil(z)) * eps11[math.ceil(x), int(y), math.floor(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.ceil(z)) * eps11[math.floor(x), int(y), math.floor(z)]
    tens[0, 1] = abs(x - math.floor(x)) * abs(z - math.floor(z)) * eps12[math.ceil(x), int(y), math.ceil(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.floor(z)) * eps12[math.floor(x), int(y), math.ceil(z)] \
                 + abs(x - math.floor(x)) * abs(z - math.ceil(z)) * eps12[math.ceil(x), int(y), math.floor(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.ceil(z)) * eps12[math.floor(x), int(y), math.floor(z)]
    tens[1, 0] = tens[0,1]
    tens[0, 2] = abs(x - math.floor(x)) * abs(z - math.floor(z)) * eps13[math.ceil(x), int(y), math.ceil(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.floor(z)) * eps13[math.floor(x), int(y), math.ceil(z)] \
                 + abs(x - math.floor(x)) * abs(z - math.ceil(z)) * eps13[math.ceil(x), int(y), math.floor(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.ceil(z)) * eps13[math.floor(x), int(y), math.floor(z)]
    tens[2, 0] = tens[0,2]
    tens[1, 1] = abs(x - math.floor(x)) * abs(z - math.floor(z)) * eps22[math.ceil(x), int(y), math.ceil(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.floor(z)) * eps22[math.floor(x), int(y), math.ceil(z)] \
                 + abs(x - math.floor(x)) * abs(z - math.ceil(z)) * eps22[math.ceil(x), int(y), math.floor(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.ceil(z)) * eps22[math.floor(x), int(y), math.floor(z)]
    tens[1, 2] = abs(x - math.floor(x)) * abs(z - math.floor(z)) * eps23[math.ceil(x), int(y), math.ceil(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.floor(z)) * eps23[math.floor(x), int(y), math.ceil(z)] \
                 + abs(x - math.floor(x)) * abs(z - math.ceil(z)) * eps23[math.ceil(x), int(y), math.floor(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.ceil(z)) * eps23[math.floor(x), int(y), math.floor(z)]
    tens[2, 1] = tens[1,2]
    tens[2, 2] = abs(x - math.floor(x)) * abs(z - math.floor(z)) * eps33[math.ceil(x), int(y), math.ceil(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.floor(z)) * eps33[math.floor(x), int(y), math.ceil(z)] \
                 + abs(x - math.floor(x)) * abs(z - math.ceil(z)) * eps33[math.ceil(x), int(y), math.floor(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.ceil(z)) * eps33[math.floor(x), int(y), math.floor(z)]
  elif x != math.floor(x) and y == math.floor(y) and z == math.floor(z):
    tens[0, 0] = abs(x - math.floor(x)) * eps11[math.ceil(x), int(y),int(z)] \
                 + abs(x - math.ceil(x)) * eps11[math.floor(x), int(y), int(z)]
    tens[0, 1] = abs(x - math.floor(x)) * eps12[math.ceil(x), int(y),int(z)] \
                 + abs(x - math.ceil(x)) * eps12[math.floor(x), int(y), int(z)]
    tens[1, 0] = tens[0,1]
    tens[0, 2] = abs(x - math.floor(x)) * eps13[math.ceil(x), int(y),int(z)] \
                 + abs(x - math.ceil(x)) * eps13[math.floor(x), int(y), int(z)]
    tens[2, 0] = tens[0,2]
    tens[1, 1] = abs(x - math.floor(x)) * eps22[math.ceil(x), int(y),int(z)] \
                 + abs(x - math.ceil(x)) * eps22[math.floor(x), int(y), int(z)]
    tens[1, 2] = abs(x - math.floor(x)) * eps23[math.ceil(x), int(y),int(z)] \
                 + abs(x - math.ceil(x)) * eps23[math.floor(x), int(y), int(z)]
    tens[2, 1] = tens[1,2]
    tens[2, 2] = abs(x - math.floor(x)) * eps33[math.ceil(x), int(y),int(z)] \
                 + abs(x - math.ceil(x)) * eps33[math.floor(x), int(y), int(z)]
  elif x != math.floor(x) and y != math.floor(y) and z == math.floor(z):
    tens[0, 0] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps11[math.ceil(x), math.ceil(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps11[math.floor(x), math.ceil(y), int(z)] \
                 + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps11[math.ceil(x), math.floor(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps11[math.floor(x), math.floor(y), int(z)]  
    tens[0, 1] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps12[math.ceil(x), math.ceil(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps12[math.floor(x), math.ceil(y), int(z)] \
                 + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps12[math.ceil(x), math.floor(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps12[math.floor(x), math.floor(y), int(z)]  
    tens[1, 0] = tens[0, 1]
    tens[0, 2] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps13[math.ceil(x), math.ceil(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps13[math.floor(x), math.ceil(y), int(z)] \
                 + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps13[math.ceil(x), math.floor(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps13[math.floor(x), math.floor(y), int(z)]  
    tens[2, 0] = tens[0, 2]
    tens[1, 1] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps22[math.ceil(x), math.ceil(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps22[math.floor(x), math.ceil(y), int(z)] \
                 + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps22[math.ceil(x), math.floor(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps22[math.floor(x), math.floor(y), int(z)]  
    tens[1, 2] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps23[math.ceil(x), math.ceil(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps23[math.floor(x), math.ceil(y), int(z)] \
                 + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps23[math.ceil(x), math.floor(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps23[math.floor(x), math.floor(y), int(z)]  
    tens[2, 1] = tens[1, 2]
    tens[2, 2] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps33[math.ceil(x), math.ceil(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps33[math.floor(x), math.ceil(y), int(z)] \
                 + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps33[math.ceil(x), math.floor(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps33[math.floor(x), math.floor(y), int(z)]  
  elif x == math.floor(x) and y != math.floor(y) and z == math.floor(z):
    tens[0, 0] = abs(y - math.floor(y)) * eps11[int(x), math.ceil(y), int(z)] \
                 + abs(y - math.ceil(y)) * eps11[int(x), math.floor(y), int(z)]
    tens[0, 1] = abs(y - math.floor(y)) * eps12[int(x), math.ceil(y), int(z)] \
                 + abs(y - math.ceil(y)) * eps12[int(x), math.floor(y), int(z)]
    tens[1, 0] = tens[0,1]
    tens[0, 2] = abs(y - math.floor(y)) * eps13[int(x), math.ceil(y), int(z)] \
                 + abs(y - math.ceil(y)) * eps13[int(x), math.floor(y), int(z)]
    tens[2, 0] = tens[0,2]
    tens[1, 1] = abs(y - math.floor(y)) * eps22[int(x), math.ceil(y), int(z)] \
                 + abs(y - math.ceil(y)) * eps22[int(x), math.floor(y), int(z)]
    tens[1, 2] = abs(y - math.floor(y)) * eps23[int(x), math.ceil(y), int(z)] \
                 + abs(y - math.ceil(y)) * eps23[int(x), math.floor(y), int(z)]
    tens[2, 1] = tens[1,2]
    tens[2, 2] = abs(y - math.floor(y)) * eps33[int(x), math.ceil(y), int(z)] \
                 + abs(y - math.ceil(y)) * eps33[int(x), math.floor(y), int(z)]
  else:
    tens[0, 0] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps11[math.ceil(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps11[math.ceil(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps11[math.floor(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps11[math.floor(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps11[math.ceil(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps11[math.ceil(x), math.floor(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps11[math.floor(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps11[math.floor(x), math.floor(y), math.floor(z)]
    tens[0, 1] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps12[math.ceil(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps12[math.ceil(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps12[math.floor(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps12[math.floor(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps12[math.ceil(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps12[math.ceil(x), math.floor(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps12[math.floor(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps12[math.floor(x), math.floor(y), math.floor(z)]
    tens[1, 0] = tens[0,1]
    tens[0, 2] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps13[math.ceil(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps13[math.ceil(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps13[math.floor(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps13[math.floor(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps13[math.ceil(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps13[math.ceil(x), math.floor(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps13[math.floor(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps13[math.floor(x), math.floor(y), math.floor(z)]
    tens[2, 0] = tens[0,2]
    tens[1, 1] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps22[math.ceil(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps22[math.ceil(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps22[math.floor(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps22[math.floor(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps22[math.ceil(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps22[math.ceil(x), math.floor(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps22[math.floor(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps22[math.floor(x), math.floor(y), math.floor(z)]
    tens[1, 2] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps23[math.ceil(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps23[math.ceil(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps23[math.floor(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps23[math.floor(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps23[math.ceil(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps23[math.ceil(x), math.floor(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps23[math.floor(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps23[math.floor(x), math.floor(y), math.floor(z)]
    tens[2,1] = tens[1,2]
    tens[2, 2] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps33[math.ceil(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps33[math.ceil(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps33[math.floor(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps33[math.floor(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps33[math.ceil(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps33[math.ceil(x), math.floor(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps33[math.floor(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps33[math.floor(x), math.floor(y), math.floor(z)]

  return (tens)
# end tens_interp_3d_torch


# @jit(nopython=False,parallel=True)
# def make_tens_interpolators(tensor_field):
#   x = np.linspace(0,tensor_field.shape[1]-1,tensor_field.shape[1])
#   y = np.linspace(0,tensor_field.shape[2]-1,tensor_field.shape[2])
#   z = np.linspace(0,tensor_field.shape[3]-1,tensor_field.shape[3])

#   interpolators = [interp_3d.Interp3D(tensor_field[i], x,y,z) for i in range(tensor_field.shape[0])]
#   return(interpolators)
# # end make_tens_interpolators

# @jit(nopython=False,parallel=True)
# def batch_interpolate_3d(x,y,z, interpolators):
#   num_tens = x.shape[0]
#   tens = np.zeros((num_tens, 3, 3))
#   for p in prange(num_tens):
#     tens[p,0,0] = interpolators[0]((x[p],y[p],z[p]))
#     tens[p,0,1] = interpolators[1]((x[p],y[p],z[p]))
#     tens[p,1,0] = tens[p,0,1]
#     tens[p,0,2] = interpolators[2]((x[p],y[p],z[p]))
#     tens[p,2,0] = tens[p,0,2]
#     tens[p,1,1] = interpolators[3]((x[p],y[p],z[p]))
#     tens[p,1,2] = interpolators[4]((x[p],y[p],z[p]))
#     tens[p,2,1] = tens[p,1,2]
#     tens[p,2,2] = interpolators[5]((x[p],y[p],z[p]))
#   return(tens)
# # end batch_interpolate_3d

##@profile
##@jit(nopython=True,parallel=True)
##@njit(parallel=True)
@njit()
def batch_tens_interp_3d(x, y, z, tensor_field):
  #print("Warning! reenable njit")
  num_tens = x.shape[0]
  tens = np.zeros((num_tens, 3, 3),dtype=np.float_)
  eps11 = tensor_field[0, :, :, :]
  eps12 = tensor_field[1, :, :, :]
  eps13 = tensor_field[2, :, :, :]
  eps22 = tensor_field[3, :, :, :]
  eps23 = tensor_field[4, :, :, :]
  eps33 = tensor_field[5, :, :, :]

  x = np.where(x<0,0,x)
  x = np.where(x>=eps11.shape[0]-1,eps11.shape[0]-1,x)
  y = np.where(y<0,0,y)
  y = np.where(y>=eps11.shape[1]-1,eps11.shape[1]-1,y)
  z = np.where(z<0,0,z)
  z = np.where(z>=eps11.shape[2]-1,eps11.shape[2]-1,z)

  ceil_x = np.ceil(x).astype(np.int_)
  floor_x = np.floor(x).astype(np.int_)
  ceil_y = np.ceil(y).astype(np.int_)
  floor_y = np.floor(y).astype(np.int_)
  ceil_z = np.ceil(z).astype(np.int_)
  floor_z = np.floor(z).astype(np.int_)
  x_minus_floor_x = np.abs(x - floor_x)
  x_minus_ceil_x = np.abs(x - ceil_x)
  y_minus_floor_y = np.abs(y - floor_y)
  y_minus_ceil_y = np.abs(y - ceil_y)
  z_minus_floor_z = np.abs(z - floor_z)
  z_minus_ceil_z = np.abs(z - ceil_z)
  
  # # Find index where interpolation is needed and interpolate
  # intidx = np.where((x_minus_floor_x + x_minus_ceil_x
  #              + y_minus_floor_y + y_minus_ceil_y
  #              + z_minus_floor_z + z_minus_ceil_z) >= 1e-14)
  
  # if len(intidx[0]) > 0:
  #   tens[intidx,0,0] = x_minus_floor_x[intidx] * y_minus_floor_y[intidx] * z_minus_floor_z[intidx] * eps11[ceil_x[intidx], ceil_y[intidx], ceil_z[intidx]] \
  #          + x_minus_floor_x[intidx] * y_minus_ceil_y[intidx] * z_minus_floor_z[intidx] * eps11[ceil_x[intidx], floor_y[intidx], ceil_z[intidx]] \
  #          + x_minus_ceil_x[intidx] * y_minus_floor_y[intidx] * z_minus_floor_z[intidx] * eps11[floor_x[intidx], ceil_y[intidx], ceil_z[intidx]] \
  #          + x_minus_ceil_x[intidx] * y_minus_ceil_y[intidx] * z_minus_floor_z[intidx] * eps11[floor_x[intidx], floor_y[intidx], ceil_z[intidx]] \
  #          + x_minus_floor_x[intidx] * y_minus_floor_y[intidx] * z_minus_ceil_z[intidx] * eps11[ceil_x[intidx], ceil_y[intidx], floor_z[intidx]] \
  #          + x_minus_floor_x[intidx] * y_minus_ceil_y[intidx] * z_minus_ceil_z[intidx] * eps11[ceil_x[intidx], floor_y[intidx], floor_z[intidx]] \
  #          + x_minus_ceil_x[intidx] * y_minus_floor_y[intidx] * z_minus_ceil_z[intidx] * eps11[floor_x[intidx], ceil_y[intidx], floor_z[intidx]] \
  #          + x_minus_ceil_x[intidx] * y_minus_ceil_y[intidx] * z_minus_ceil_z[intidx] * eps11[floor_x[intidx], floor_y[intidx], floor_z[intidx]]
  #   tens[intidx,0,1] = x_minus_floor_x[intidx] * y_minus_floor_y[intidx] * z_minus_floor_z[intidx] * eps12[ceil_x[intidx], ceil_y[intidx], ceil_z[intidx]] \
  #          + x_minus_floor_x[intidx] * y_minus_ceil_y[intidx] * z_minus_floor_z[intidx] * eps12[ceil_x[intidx], floor_y[intidx], ceil_z[intidx]] \
  #          + x_minus_ceil_x[intidx] * y_minus_floor_y[intidx] * z_minus_floor_z[intidx] * eps12[floor_x[intidx], ceil_y[intidx], ceil_z[intidx]] \
  #          + x_minus_ceil_x[intidx] * y_minus_ceil_y[intidx] * z_minus_floor_z[intidx] * eps12[floor_x[intidx], floor_y[intidx], ceil_z[intidx]] \
  #          + x_minus_floor_x[intidx] * y_minus_floor_y[intidx] * z_minus_ceil_z[intidx] * eps12[ceil_x[intidx], ceil_y[intidx], floor_z[intidx]] \
  #          + x_minus_floor_x[intidx] * y_minus_ceil_y[intidx] * z_minus_ceil_z[intidx] * eps12[ceil_x[intidx], floor_y[intidx], floor_z[intidx]] \
  #          + x_minus_ceil_x[intidx] * y_minus_floor_y[intidx] * z_minus_ceil_z[intidx] * eps12[floor_x[intidx], ceil_y[intidx], floor_z[intidx]] \
  #          + x_minus_ceil_x[intidx] * y_minus_ceil_y[intidx] * z_minus_ceil_z[intidx] * eps12[floor_x[intidx], floor_y[intidx], floor_z[intidx]]
  #   tens[intidx,1,0] = tens[intidx,0,1]
  #   tens[intidx,0,2] = x_minus_floor_x[intidx] * y_minus_floor_y[intidx] * z_minus_floor_z[intidx] * eps13[ceil_x[intidx], ceil_y[intidx], ceil_z[intidx]] \
  #          + x_minus_floor_x[intidx] * y_minus_ceil_y[intidx] * z_minus_floor_z[intidx] * eps13[ceil_x[intidx], floor_y[intidx], ceil_z[intidx]] \
  #          + x_minus_ceil_x[intidx] * y_minus_floor_y[intidx] * z_minus_floor_z[intidx] * eps13[floor_x[intidx], ceil_y[intidx], ceil_z[intidx]] \
  #          + x_minus_ceil_x[intidx] * y_minus_ceil_y[intidx] * z_minus_floor_z[intidx] * eps13[floor_x[intidx], floor_y[intidx], ceil_z[intidx]] \
  #          + x_minus_floor_x[intidx] * y_minus_floor_y[intidx] * z_minus_ceil_z[intidx] * eps13[ceil_x[intidx], ceil_y[intidx], floor_z[intidx]] \
  #          + x_minus_floor_x[intidx] * y_minus_ceil_y[intidx] * z_minus_ceil_z[intidx] * eps13[ceil_x[intidx], floor_y[intidx], floor_z[intidx]] \
  #          + x_minus_ceil_x[intidx] * y_minus_floor_y[intidx] * z_minus_ceil_z[intidx] * eps13[floor_x[intidx], ceil_y[intidx], floor_z[intidx]] \
  #          + x_minus_ceil_x[intidx] * y_minus_ceil_y[intidx] * z_minus_ceil_z[intidx] * eps13[floor_x[intidx], floor_y[intidx], floor_z[intidx]]
  #   tens[intidx,2,0] = tens[intidx,0,2]
  #   tens[intidx,1,1] = x_minus_floor_x[intidx] * y_minus_floor_y[intidx] * z_minus_floor_z[intidx] * eps22[ceil_x[intidx], ceil_y[intidx], ceil_z[intidx]] \
  #          + x_minus_floor_x[intidx] * y_minus_ceil_y[intidx] * z_minus_floor_z[intidx] * eps22[ceil_x[intidx], floor_y[intidx], ceil_z[intidx]] \
  #          + x_minus_ceil_x[intidx] * y_minus_floor_y[intidx] * z_minus_floor_z[intidx] * eps22[floor_x[intidx], ceil_y[intidx], ceil_z[intidx]] \
  #          + x_minus_ceil_x[intidx] * y_minus_ceil_y[intidx] * z_minus_floor_z[intidx] * eps22[floor_x[intidx], floor_y[intidx], ceil_z[intidx]] \
  #          + x_minus_floor_x[intidx] * y_minus_floor_y[intidx] * z_minus_ceil_z[intidx] * eps22[ceil_x[intidx], ceil_y[intidx], floor_z[intidx]] \
  #          + x_minus_floor_x[intidx] * y_minus_ceil_y[intidx] * z_minus_ceil_z[intidx] * eps22[ceil_x[intidx], floor_y[intidx], floor_z[intidx]] \
  #          + x_minus_ceil_x[intidx] * y_minus_floor_y[intidx] * z_minus_ceil_z[intidx] * eps22[floor_x[intidx], ceil_y[intidx], floor_z[intidx]] \
  #          + x_minus_ceil_x[intidx] * y_minus_ceil_y[intidx] * z_minus_ceil_z[intidx] * eps22[floor_x[intidx], floor_y[intidx], floor_z[intidx]]
  #   tens[intidx,1,2] = x_minus_floor_x[intidx] * y_minus_floor_y[intidx] * z_minus_floor_z[intidx] * eps23[ceil_x[intidx], ceil_y[intidx], ceil_z[intidx]] \
  #          + x_minus_floor_x[intidx] * y_minus_ceil_y[intidx] * z_minus_floor_z[intidx] * eps23[ceil_x[intidx], floor_y[intidx], ceil_z[intidx]] \
  #          + x_minus_ceil_x[intidx] * y_minus_floor_y[intidx] * z_minus_floor_z[intidx] * eps23[floor_x[intidx], ceil_y[intidx], ceil_z[intidx]] \
  #          + x_minus_ceil_x[intidx] * y_minus_ceil_y[intidx] * z_minus_floor_z[intidx] * eps23[floor_x[intidx], floor_y[intidx], ceil_z[intidx]] \
  #          + x_minus_floor_x[intidx] * y_minus_floor_y[intidx] * z_minus_ceil_z[intidx] * eps23[ceil_x[intidx], ceil_y[intidx], floor_z[intidx]] \
  #          + x_minus_floor_x[intidx] * y_minus_ceil_y[intidx] * z_minus_ceil_z[intidx] * eps23[ceil_x[intidx], floor_y[intidx], floor_z[intidx]] \
  #          + x_minus_ceil_x[intidx] * y_minus_floor_y[intidx] * z_minus_ceil_z[intidx] * eps23[floor_x[intidx], ceil_y[intidx], floor_z[intidx]] \
  #          + x_minus_ceil_x[intidx] * y_minus_ceil_y[intidx] * z_minus_ceil_z[intidx] * eps23[floor_x[intidx], floor_y[intidx], floor_z[intidx]]
  #   tens[intidx,2,1] = tens[intidx,1,2]
  #   tens[intidx,2,2] = x_minus_floor_x[intidx] * y_minus_floor_y[intidx] * z_minus_floor_z[intidx] * eps33[ceil_x[intidx], ceil_y[intidx], ceil_z[intidx]] \
  #          + x_minus_floor_x[intidx] * y_minus_ceil_y[intidx] * z_minus_floor_z[intidx] * eps33[ceil_x[intidx], floor_y[intidx], ceil_z[intidx]] \
  #          + x_minus_ceil_x[intidx] * y_minus_floor_y[intidx] * z_minus_floor_z[intidx] * eps33[floor_x[intidx], ceil_y[intidx], ceil_z[intidx]] \
  #          + x_minus_ceil_x[intidx] * y_minus_ceil_y[intidx] * z_minus_floor_z[intidx] * eps33[floor_x[intidx], floor_y[intidx], ceil_z[intidx]] \
  #          + x_minus_floor_x[intidx] * y_minus_floor_y[intidx] * z_minus_ceil_z[intidx] * eps33[ceil_x[intidx], ceil_y[intidx], floor_z[intidx]] \
  #          + x_minus_floor_x[intidx] * y_minus_ceil_y[intidx] * z_minus_ceil_z[intidx] * eps33[ceil_x[intidx], floor_y[intidx], floor_z[intidx]] \
  #          + x_minus_ceil_x[intidx] * y_minus_floor_y[intidx] * z_minus_ceil_z[intidx] * eps33[floor_x[intidx], ceil_y[intidx], floor_z[intidx]] \
  #          + x_minus_ceil_x[intidx] * y_minus_ceil_y[intidx] * z_minus_ceil_z[intidx] * eps33[floor_x[intidx], floor_y[intidx], floor_z[intidx]]

  # # Find index where no interpolation is needed and just copy the values
  # nointidx = np.where((x_minus_floor_x + x_minus_ceil_x
  #              + y_minus_floor_y + y_minus_ceil_y
  #              + z_minus_floor_z + z_minus_ceil_z) < 1e-14)
  # if len(nointidx[0]) > 0:
  #   tens[nointidx,0,0] = eps11[x[nointidx].astype(np.int64),y[nointidx].astype(np.int64),z[nointidx].astype(np.int64)]
  #   tens[nointidx,0,1] = eps12[x[nointidx].astype(np.int64),y[nointidx].astype(np.int64),z[nointidx].astype(np.int64)]
  #   tens[nointidx,1,0] = tens[nointidx,0,1]
  #   tens[nointidx,0,2] = eps13[x[nointidx].astype(np.int64),y[nointidx].astype(np.int64),z[nointidx].astype(np.int64)]
  #   tens[nointidx,2,0] = tens[nointidx,0,2]
  #   tens[nointidx,1,1] = eps22[x[nointidx].astype(np.int64),y[nointidx].astype(np.int64),z[nointidx].astype(np.int64)]
  #   tens[nointidx,1,2] = eps23[x[nointidx].astype(np.int64),y[nointidx].astype(np.int64),z[nointidx].astype(np.int64)]
  #   tens[nointidx,2,1] = tens[nointidx,1,2]
  #   tens[nointidx,2,2] = eps33[x[nointidx].astype(np.int64),y[nointidx].astype(np.int64),z[nointidx].astype(np.int64)]

  # floor_x_floor_y = x_minus_floor_x * y_minus_floor_y
  # floor_x_floor_y_floor_z = floor_x_floor_y * z_minus_floor_z
  # floor_x_floor_y_ceil_z = floor_x_floor_y * z_minus_ceil_z
  # floor_x_ceil_y = x_minus_floor_x * y_minus_ceil_y
  # floor_x_ceil_y_floor_z = floor_x_ceil_y * z_minus_floor_z
  # floor_x_ceil_y_ceil_z = floor_x_ceil_y * z_minus_ceil_z
  # ceil_x_floor_y = x_minus_ceil_x * y_minus_floor_y
  # ceil_x_floor_y_floor_z = ceil_x_floor_y * z_minus_floor_z
  # ceil_x_floor_y_ceil_z = ceil_x_floor_y * z_minus_ceil_z
  # ceil_x_ceil_y = x_minus_ceil_x * y_minus_ceil_y
  # ceil_x_ceil_y_floor_z = ceil_x_ceil_y * z_minus_floor_z
  # ceil_x_ceil_y_ceil_z = ceil_x_ceil_y * z_minus_ceil_z

  # Find index where interpolation is needed and interpolate
  # This for loop is way too slow without numba.  Use above indexing if numba is not available
  #for p in prange(num_tens):
  for p in range(num_tens):
    if x[p] == floor_x[p] and y[p] == floor_y[p] and z[p] == floor_z[p]:
      # Find index where no interpolation is needed and just copy the values
      tens[p,0,0] = eps11[floor_x[p],floor_y[p],floor_z[p]]
      tens[p,0,1] = eps12[floor_x[p],floor_y[p],floor_z[p]]
      tens[p,1,0] = tens[p,0,1]
      tens[p,0,2] = eps13[floor_x[p],floor_y[p],floor_z[p]]
      tens[p,2,0] = tens[p,0,2]
      tens[p,1,1] = eps22[floor_x[p],floor_y[p],floor_z[p]]
      tens[p,1,2] = eps23[floor_x[p],floor_y[p],floor_z[p]]
      tens[p,2,1] = tens[p,1,2]
      tens[p,2,2] = eps33[floor_x[p],floor_y[p],floor_z[p]]
    elif x[p] == floor_x[p] and y[p] != floor_y[p] and z[p] != floor_z[p]:
      floor_y_floor_z = y_minus_floor_y[p] * z_minus_floor_z[p]
      floor_y_ceil_z = y_minus_floor_y[p] * z_minus_ceil_z[p]
      ceil_y_floor_z = y_minus_ceil_y[p] * z_minus_floor_z[p]
      ceil_y_ceil_z = y_minus_ceil_y[p] * z_minus_ceil_z[p]
      
      tens[p,0,0] = floor_y_floor_z * eps11[np.int64(x[p]), ceil_y[p], ceil_z[p]] \
           + ceil_y_floor_z * eps11[np.int64(x[p]), floor_y[p], ceil_z[p]] \
           + floor_y_ceil_z * eps11[np.int64(x[p]), ceil_y[p], floor_z[p]] \
           + ceil_y_ceil_z * eps11[np.int64(x[p]), floor_y[p], floor_z[p]] 
      tens[p,0,1] = floor_y_floor_z * eps12[np.int64(x[p]), ceil_y[p], ceil_z[p]] \
           + ceil_y_floor_z * eps12[np.int64(x[p]), floor_y[p], ceil_z[p]] \
           + floor_y_ceil_z * eps12[np.int64(x[p]), ceil_y[p], floor_z[p]] \
           + ceil_y_ceil_z * eps12[np.int64(x[p]), floor_y[p], floor_z[p]] 
      tens[p,1,0] = tens[p,0,1]
      tens[p,0,2] = floor_y_floor_z * eps13[np.int64(x[p]), ceil_y[p], ceil_z[p]] \
           + ceil_y_floor_z * eps13[np.int64(x[p]), floor_y[p], ceil_z[p]] \
           + floor_y_ceil_z * eps13[np.int64(x[p]), ceil_y[p], floor_z[p]] \
           + ceil_y_ceil_z * eps13[np.int64(x[p]), floor_y[p], floor_z[p]] 
      tens[p,2,0] = tens[p,0,2]
      tens[p,1,1] = floor_y_floor_z * eps22[np.int64(x[p]), ceil_y[p], ceil_z[p]] \
           + ceil_y_floor_z * eps22[np.int64(x[p]), floor_y[p], ceil_z[p]] \
           + floor_y_ceil_z * eps22[np.int64(x[p]), ceil_y[p], floor_z[p]] \
           + ceil_y_ceil_z * eps22[np.int64(x[p]), floor_y[p], floor_z[p]] 
      tens[p,1,2] = floor_y_floor_z * eps23[np.int64(x[p]), ceil_y[p], ceil_z[p]] \
           + ceil_y_floor_z * eps23[np.int64(x[p]), floor_y[p], ceil_z[p]] \
           + floor_y_ceil_z * eps23[np.int64(x[p]), ceil_y[p], floor_z[p]] \
           + ceil_y_ceil_z * eps23[np.int64(x[p]), floor_y[p], floor_z[p]] 
      tens[p,2,1] = tens[p,1,2]
      tens[p,2,2] = floor_y_floor_z * eps33[np.int64(x[p]), ceil_y[p], ceil_z[p]] \
           + ceil_y_floor_z * eps33[np.int64(x[p]), floor_y[p], ceil_z[p]] \
           + floor_y_ceil_z * eps33[np.int64(x[p]), ceil_y[p], floor_z[p]] \
           + ceil_y_ceil_z * eps33[np.int64(x[p]), floor_y[p], floor_z[p]] 
    elif x[p] == floor_x[p] and y[p] == floor_y[p] and z[p] != floor_z[p]:
      tens[p,0,0] = z_minus_floor_z[p] * eps11[np.int64(x[p]), np.int64(y[p]), ceil_z[p]] \
           + z_minus_ceil_z[p] * eps11[np.int64(x[p]), np.int64(y[p]), floor_z[p]]
      tens[p,0,1] = z_minus_floor_z[p] * eps12[np.int64(x[p]), np.int64(y[p]), ceil_z[p]] \
           + z_minus_ceil_z[p] * eps12[np.int64(x[p]), np.int64(y[p]), floor_z[p]]
      tens[p,1,0] = tens[p,0,1]
      tens[p,0,2] = z_minus_floor_z[p] * eps13[np.int64(x[p]), np.int64(y[p]), ceil_z[p]] \
           + z_minus_ceil_z[p] * eps13[np.int64(x[p]), np.int64(y[p]), floor_z[p]]
      tens[p,2,0] = tens[p,0,2]
      tens[p,1,1] = z_minus_floor_z[p] * eps22[np.int64(x[p]), np.int64(y[p]), ceil_z[p]] \
           + z_minus_ceil_z[p] * eps22[np.int64(x[p]), np.int64(y[p]), floor_z[p]]
      tens[p,1,2] = z_minus_floor_z[p] * eps23[np.int64(x[p]), np.int64(y[p]), ceil_z[p]] \
           + z_minus_ceil_z[p] * eps23[np.int64(x[p]), np.int64(y[p]), floor_z[p]]
      tens[p,2,1] = tens[p,1,2]
      tens[p,2,2] = z_minus_floor_z[p] * eps33[np.int64(x[p]), np.int64(y[p]), ceil_z[p]] \
           + z_minus_ceil_z[p] * eps33[np.int64(x[p]), np.int64(y[p]), floor_z[p]]
    elif x[p] != floor_x[p] and y[p] == floor_y[p] and z[p] != floor_z[p]:
      floor_x_floor_z = x_minus_floor_x[p] * z_minus_floor_z[p]
      floor_x_ceil_z = x_minus_floor_x[p] * z_minus_ceil_z[p]
      ceil_x_floor_z = x_minus_ceil_x[p] * z_minus_floor_z[p]
      ceil_x_ceil_z = x_minus_ceil_x[p] * z_minus_ceil_z[p]
      
      tens[p,0,0] = floor_x_floor_z * eps11[ceil_x[p], np.int64(y[p]), ceil_z[p]] \
           + ceil_x_floor_z * eps11[floor_x[p], np.int64(y[p]), ceil_z[p]] \
           + floor_x_ceil_z * eps11[ceil_x[p], np.int64(y[p]), floor_z[p]] \
           + ceil_x_ceil_z * eps11[floor_x[p], np.int64(y[p]), floor_z[p]] 
      tens[p,0,1] = floor_x_floor_z * eps12[ceil_x[p], np.int64(y[p]), ceil_z[p]] \
           + ceil_x_floor_z * eps12[floor_x[p], np.int64(y[p]), ceil_z[p]] \
           + floor_x_ceil_z * eps12[ceil_x[p], np.int64(y[p]), floor_z[p]] \
           + ceil_x_ceil_z * eps12[floor_x[p], np.int64(y[p]), floor_z[p]] 
      tens[p,1,0] = tens[p,0,1]
      tens[p,0,2] = floor_x_floor_z * eps13[ceil_x[p], np.int64(y[p]), ceil_z[p]] \
           + ceil_x_floor_z * eps13[floor_x[p], np.int64(y[p]), ceil_z[p]] \
           + floor_x_ceil_z * eps13[ceil_x[p], np.int64(y[p]), floor_z[p]] \
           + ceil_x_ceil_z * eps13[floor_x[p], np.int64(y[p]), floor_z[p]] 
      tens[p,2,0] = tens[p,0,2]
      tens[p,1,1] = floor_x_floor_z * eps22[ceil_x[p], np.int64(y[p]), ceil_z[p]] \
           + ceil_x_floor_z * eps22[floor_x[p], np.int64(y[p]), ceil_z[p]] \
           + floor_x_ceil_z * eps22[ceil_x[p], np.int64(y[p]), floor_z[p]] \
           + ceil_x_ceil_z * eps22[floor_x[p], np.int64(y[p]), floor_z[p]] 
      tens[p,1,2] = floor_x_floor_z * eps23[ceil_x[p], np.int64(y[p]), ceil_z[p]] \
           + ceil_x_floor_z * eps23[floor_x[p], np.int64(y[p]), ceil_z[p]] \
           + floor_x_ceil_z * eps23[ceil_x[p], np.int64(y[p]), floor_z[p]] \
           + ceil_x_ceil_z * eps23[floor_x[p], np.int64(y[p]), floor_z[p]] 
      tens[p,2,1] = tens[p,1,2]
      tens[p,2,2] = floor_x_floor_z * eps33[ceil_x[p], np.int64(y[p]), ceil_z[p]] \
           + ceil_x_floor_z * eps33[floor_x[p], np.int64(y[p]), ceil_z[p]] \
           + floor_x_ceil_z * eps33[ceil_x[p], np.int64(y[p]), floor_z[p]] \
           + ceil_x_ceil_z * eps33[floor_x[p], np.int64(y[p]), floor_z[p]] 
    elif x[p] != floor_x[p] and y[p] == floor_y[p] and z[p] == floor_z[p]:
      tens[p,0,0] = x_minus_floor_x[p] * eps11[ceil_x[p], np.int64(y[p]), np.int64(z[p])] \
           + x_minus_ceil_x[p] * eps11[floor_x[p], np.int64(y[p]), np.int64(z[p])]
      tens[p,0,1] = x_minus_floor_x[p] * eps12[ceil_x[p], np.int64(y[p]), np.int64(z[p])] \
           + x_minus_ceil_x[p] * eps12[floor_x[p], np.int64(y[p]), np.int64(z[p])]
      tens[p,1,0] = tens[p,0,1]
      tens[p,0,2] = x_minus_floor_x[p] * eps13[ceil_x[p], np.int64(y[p]), np.int64(z[p])] \
           + x_minus_ceil_x[p] * eps13[floor_x[p], np.int64(y[p]), np.int64(z[p])]
      tens[p,2,0] = tens[p,0,2]
      tens[p,1,1] = x_minus_floor_x[p] * eps22[ceil_x[p], np.int64(y[p]), np.int64(z[p])] \
           + x_minus_ceil_x[p] * eps22[floor_x[p], np.int64(y[p]), np.int64(z[p])]
      tens[p,1,2] = x_minus_floor_x[p] * eps23[ceil_x[p], np.int64(y[p]), np.int64(z[p])] \
           + x_minus_ceil_x[p] * eps23[floor_x[p], np.int64(y[p]), np.int64(z[p])]
      tens[p,2,1] = tens[p,1,2]
      tens[p,2,2] = x_minus_floor_x[p] * eps33[ceil_x[p], np.int64(y[p]), np.int64(z[p])] \
           + x_minus_ceil_x[p] * eps33[floor_x[p], np.int64(y[p]), np.int64(z[p])]
    elif x[p] != floor_x[p] and y[p] != floor_y[p] and z[p] == floor_z[p]:
      floor_x_floor_y = x_minus_floor_x[p] * y_minus_floor_y[p]
      floor_x_ceil_y = x_minus_floor_x[p] * y_minus_ceil_y[p]
      ceil_x_floor_y = x_minus_ceil_x[p] * y_minus_floor_y[p]
      ceil_x_ceil_y = x_minus_ceil_x[p] * y_minus_ceil_y[p]
      
      tens[p,0,0] = floor_x_floor_y * eps11[ceil_x[p], ceil_y[p], np.int64(z[p])] \
           + ceil_x_floor_y * eps11[floor_x[p], ceil_y[p], np.int64(z[p])] \
           + floor_x_ceil_y * eps11[ceil_x[p], floor_y[p], np.int64(z[p])] \
           + ceil_x_ceil_y * eps11[floor_x[p], floor_y[p], np.int64(z[p])] 
      tens[p,0,1] = floor_x_floor_y * eps12[ceil_x[p], ceil_y[p], np.int64(z[p])] \
           + ceil_x_floor_y * eps12[floor_x[p], ceil_y[p], np.int64(z[p])] \
           + floor_x_ceil_y * eps12[ceil_x[p], floor_y[p], np.int64(z[p])] \
           + ceil_x_ceil_y * eps12[floor_x[p], floor_y[p], np.int64(z[p])] 
      tens[p,1,0] = tens[p,0,1]
      tens[p,0,2] = floor_x_floor_y * eps13[ceil_x[p], ceil_y[p], np.int64(z[p])] \
           + ceil_x_floor_y * eps13[floor_x[p], ceil_y[p], np.int64(z[p])] \
           + floor_x_ceil_y * eps13[ceil_x[p], floor_y[p], np.int64(z[p])] \
           + ceil_x_ceil_y * eps13[floor_x[p], floor_y[p], np.int64(z[p])] 
      tens[p,2,0] = tens[p,0,2]
      tens[p,1,1] = floor_x_floor_y * eps22[ceil_x[p], ceil_y[p], np.int64(z[p])] \
           + ceil_x_floor_y * eps22[floor_x[p], ceil_y[p], np.int64(z[p])] \
           + floor_x_ceil_y * eps22[ceil_x[p], floor_y[p], np.int64(z[p])] \
           + ceil_x_ceil_y * eps22[floor_x[p], floor_y[p], np.int64(z[p])] 
      tens[p,1,2] = floor_x_floor_y * eps23[ceil_x[p], ceil_y[p], np.int64(z[p])] \
           + ceil_x_floor_y * eps23[floor_x[p], ceil_y[p], np.int64(z[p])] \
           + floor_x_ceil_y * eps23[ceil_x[p], floor_y[p], np.int64(z[p])] \
           + ceil_x_ceil_y * eps23[floor_x[p], floor_y[p], np.int64(z[p])] 
      tens[p,2,1] = tens[p,1,2]
      tens[p,2,2] = floor_x_floor_y * eps33[ceil_x[p], ceil_y[p], np.int64(z[p])] \
           + ceil_x_floor_y * eps33[floor_x[p], ceil_y[p], np.int64(z[p])] \
           + floor_x_ceil_y * eps33[ceil_x[p], floor_y[p], np.int64(z[p])] \
           + ceil_x_ceil_y * eps33[floor_x[p], floor_y[p], np.int64(z[p])] 
    elif x[p] == floor_x[p] and y[p] != floor_y[p] and z[p] == floor_z[p]:
      tens[p,0,0] = y_minus_floor_y[p] * eps11[np.int64(x[p]), ceil_y[p], np.int64(z[p])] \
           + y_minus_ceil_y[p] * eps11[np.int64(x[p]), floor_y[p], np.int64(z[p])]
      tens[p,0,1] = y_minus_floor_y[p] * eps12[np.int64(x[p]), ceil_y[p], np.int64(z[p])] \
           + y_minus_ceil_y[p] * eps12[np.int64(x[p]), floor_y[p], np.int64(z[p])]
      tens[p,1,0] = tens[p,0,1]
      tens[p,0,2] = y_minus_floor_y[p] * eps13[np.int64(x[p]), ceil_y[p], np.int64(z[p])] \
           + y_minus_ceil_y[p] * eps13[np.int64(x[p]), floor_y[p], np.int64(z[p])]
      tens[p,2,0] = tens[p,0,2]
      tens[p,1,1] = y_minus_floor_y[p] * eps22[np.int64(x[p]), ceil_y[p], np.int64(z[p])] \
           + y_minus_ceil_y[p] * eps22[np.int64(x[p]), floor_y[p], np.int64(z[p])]
      tens[p,1,2] = y_minus_floor_y[p] * eps23[np.int64(x[p]), ceil_y[p], np.int64(z[p])] \
           + y_minus_ceil_y[p] * eps23[np.int64(x[p]), floor_y[p], np.int64(z[p])]
      tens[p,2,1] = tens[p,1,2]
      tens[p,2,2] = y_minus_floor_y[p] * eps33[np.int64(x[p]), ceil_y[p], np.int64(z[p])] \
           + y_minus_ceil_y[p] * eps33[np.int64(x[p]), floor_y[p], np.int64(z[p])]
    else:
      floor_x_floor_y = x_minus_floor_x[p] * y_minus_floor_y[p]
      floor_x_floor_y_floor_z = floor_x_floor_y * z_minus_floor_z[p]
      floor_x_floor_y_ceil_z = floor_x_floor_y * z_minus_ceil_z[p]
      floor_x_ceil_y = x_minus_floor_x[p] * y_minus_ceil_y[p]
      floor_x_ceil_y_floor_z = floor_x_ceil_y * z_minus_floor_z[p]
      floor_x_ceil_y_ceil_z = floor_x_ceil_y * z_minus_ceil_z[p]
      ceil_x_floor_y = x_minus_ceil_x[p] * y_minus_floor_y[p]
      ceil_x_floor_y_floor_z = ceil_x_floor_y * z_minus_floor_z[p]
      ceil_x_floor_y_ceil_z = ceil_x_floor_y * z_minus_ceil_z[p]
      ceil_x_ceil_y = x_minus_ceil_x[p] * y_minus_ceil_y[p]
      ceil_x_ceil_y_floor_z = ceil_x_ceil_y * z_minus_floor_z[p]
      ceil_x_ceil_y_ceil_z = ceil_x_ceil_y * z_minus_ceil_z[p]

      tens[p,0,0] = floor_x_floor_y_floor_z * eps11[ceil_x[p], ceil_y[p], ceil_z[p]] \
           + floor_x_ceil_y_floor_z * eps11[ceil_x[p], floor_y[p], ceil_z[p]] \
           + ceil_x_floor_y_floor_z * eps11[floor_x[p], ceil_y[p], ceil_z[p]] \
           + ceil_x_ceil_y_floor_z * eps11[floor_x[p], floor_y[p], ceil_z[p]] \
           + floor_x_floor_y_ceil_z * eps11[ceil_x[p], ceil_y[p], floor_z[p]] \
           + floor_x_ceil_y_ceil_z * eps11[ceil_x[p], floor_y[p], floor_z[p]] \
           + ceil_x_floor_y_ceil_z * eps11[floor_x[p], ceil_y[p], floor_z[p]] \
           + ceil_x_ceil_y_ceil_z * eps11[floor_x[p], floor_y[p], floor_z[p]]
      tens[p,0,1] = floor_x_floor_y_floor_z * eps12[ceil_x[p], ceil_y[p], ceil_z[p]] \
           + floor_x_ceil_y_floor_z * eps12[ceil_x[p], floor_y[p], ceil_z[p]] \
           + ceil_x_floor_y_floor_z * eps12[floor_x[p], ceil_y[p], ceil_z[p]] \
           + ceil_x_ceil_y_floor_z * eps12[floor_x[p], floor_y[p], ceil_z[p]] \
           + floor_x_floor_y_ceil_z * eps12[ceil_x[p], ceil_y[p], floor_z[p]] \
           + floor_x_ceil_y_ceil_z * eps12[ceil_x[p], floor_y[p], floor_z[p]] \
           + ceil_x_floor_y_ceil_z * eps12[floor_x[p], ceil_y[p], floor_z[p]] \
           + ceil_x_ceil_y_ceil_z * eps12[floor_x[p], floor_y[p], floor_z[p]]
      tens[p,1,0] = tens[p,0,1]
      tens[p,0,2] = floor_x_floor_y_floor_z * eps13[ceil_x[p], ceil_y[p], ceil_z[p]] \
           + floor_x_ceil_y_floor_z * eps13[ceil_x[p], floor_y[p], ceil_z[p]] \
           + ceil_x_floor_y_floor_z * eps13[floor_x[p], ceil_y[p], ceil_z[p]] \
           + ceil_x_ceil_y_floor_z * eps13[floor_x[p], floor_y[p], ceil_z[p]] \
           + floor_x_floor_y_ceil_z * eps13[ceil_x[p], ceil_y[p], floor_z[p]] \
           + floor_x_ceil_y_ceil_z * eps13[ceil_x[p], floor_y[p], floor_z[p]] \
           + ceil_x_floor_y_ceil_z * eps13[floor_x[p], ceil_y[p], floor_z[p]] \
           + ceil_x_ceil_y_ceil_z * eps13[floor_x[p], floor_y[p], floor_z[p]]
      tens[p,2,0] = tens[p,0,2]
      tens[p,1,1] = floor_x_floor_y_floor_z * eps22[ceil_x[p], ceil_y[p], ceil_z[p]] \
           + floor_x_ceil_y_floor_z * eps22[ceil_x[p], floor_y[p], ceil_z[p]] \
           + ceil_x_floor_y_floor_z * eps22[floor_x[p], ceil_y[p], ceil_z[p]] \
           + ceil_x_ceil_y_floor_z * eps22[floor_x[p], floor_y[p], ceil_z[p]] \
           + floor_x_floor_y_ceil_z * eps22[ceil_x[p], ceil_y[p], floor_z[p]] \
           + floor_x_ceil_y_ceil_z * eps22[ceil_x[p], floor_y[p], floor_z[p]] \
           + ceil_x_floor_y_ceil_z * eps22[floor_x[p], ceil_y[p], floor_z[p]] \
           + ceil_x_ceil_y_ceil_z * eps22[floor_x[p], floor_y[p], floor_z[p]]
      tens[p,1,2] = floor_x_floor_y_floor_z * eps23[ceil_x[p], ceil_y[p], ceil_z[p]] \
           + floor_x_ceil_y_floor_z * eps23[ceil_x[p], floor_y[p], ceil_z[p]] \
           + ceil_x_floor_y_floor_z * eps23[floor_x[p], ceil_y[p], ceil_z[p]] \
           + ceil_x_ceil_y_floor_z * eps23[floor_x[p], floor_y[p], ceil_z[p]] \
           + floor_x_floor_y_ceil_z * eps23[ceil_x[p], ceil_y[p], floor_z[p]] \
           + floor_x_ceil_y_ceil_z * eps23[ceil_x[p], floor_y[p], floor_z[p]] \
           + ceil_x_floor_y_ceil_z * eps23[floor_x[p], ceil_y[p], floor_z[p]] \
           + ceil_x_ceil_y_ceil_z * eps23[floor_x[p], floor_y[p], floor_z[p]]
      tens[p,2,1] = tens[p,1,2]
      tens[p,2,2] = floor_x_floor_y_floor_z * eps33[ceil_x[p], ceil_y[p], ceil_z[p]] \
           + floor_x_ceil_y_floor_z * eps33[ceil_x[p], floor_y[p], ceil_z[p]] \
           + ceil_x_floor_y_floor_z * eps33[floor_x[p], ceil_y[p], ceil_z[p]] \
           + ceil_x_ceil_y_floor_z * eps33[floor_x[p], floor_y[p], ceil_z[p]] \
           + floor_x_floor_y_ceil_z * eps33[ceil_x[p], ceil_y[p], floor_z[p]] \
           + floor_x_ceil_y_ceil_z * eps33[ceil_x[p], floor_y[p], floor_z[p]] \
           + ceil_x_floor_y_ceil_z * eps33[floor_x[p], ceil_y[p], floor_z[p]] \
           + ceil_x_ceil_y_ceil_z * eps33[floor_x[p], floor_y[p], floor_z[p]]
  
  return (tens)
# end batch_tens_interp_3d

def batch_tens_interp_3d_torch(x, y, z, tensor_field):
  num_tens = x.shape[0]
  tens = torch.zeros((num_tens, 3, 3),dtype=tensor_field.dtype)
  eps11 = tensor_field[0, :, :, :]
  eps12 = tensor_field[1, :, :, :]
  eps13 = tensor_field[2, :, :, :]
  eps22 = tensor_field[3, :, :, :]
  eps23 = tensor_field[4, :, :, :]
  eps33 = tensor_field[5, :, :, :]

  try:
    # Want to do torch.where(x<0,0,x), but get strange type promotion errors ala
    # https://github.com/pytorch/pytorch/issues/9190
    # hence the messy torch.tensor syntax
    x = torch.where(x<0,torch.tensor(0,dtype=x.dtype),x)
    x = torch.where(x>=eps11.shape[0]-1,torch.tensor(eps11.shape[0]-1,dtype=x.dtype),x)
    y = torch.where(y<0,torch.tensor(0,dtype=y.dtype),y)
    y = torch.where(y>=eps11.shape[1]-1,torch.tensor(eps11.shape[1]-1,dtype=y.dtype),y)
    z = torch.where(z<0,torch.tensor(0,dtype=z.dtype),z)
    z = torch.where(z>=eps11.shape[2]-1,torch.tensor(eps11.shape[2]-1,dtype=z.dtype),z)
  except Exception as err:
    print('Caught Exception:', err)
    print('x dtype',x.dtype)
    print('torch.where(x<0,0,x).dtype', torch.where(x<0,0,x).dtype)
    raise

  # Casting to double here because of this torch issue
  # https://github.com/pytorch/pytorch/issues/51199
  ceil_x = torch.ceil(x.double()).long()
  floor_x = torch.floor(x.double()).long()
  ceil_y = torch.ceil(y.double()).long()
  floor_y = torch.floor(y.double()).long()
  ceil_z = torch.ceil(z.double()).long()
  floor_z = torch.floor(z.double()).long()
  x_minus_floor_x = torch.abs(x - floor_x)
  x_minus_ceil_x = torch.abs(x - ceil_x)
  y_minus_floor_y = torch.abs(y - floor_y)
  y_minus_ceil_y = torch.abs(y - ceil_y)
  z_minus_floor_z = torch.abs(z - floor_z)
  z_minus_ceil_z = torch.abs(z - ceil_z)

  # Find index where interpolation is needed and interpolate
  intidx = torch.where((x_minus_floor_x + x_minus_ceil_x
               + y_minus_floor_y + y_minus_ceil_y
               + z_minus_floor_z + z_minus_ceil_z) >= 1e-14)
  try:
    if len(intidx[0]) > 0:
      tens[intidx[0][:],0,0] = x_minus_floor_x[intidx] * y_minus_floor_y[intidx] * z_minus_floor_z[intidx] * eps11[ceil_x[intidx], ceil_y[intidx], ceil_z[intidx]] \
           + x_minus_floor_x[intidx] * y_minus_ceil_y[intidx] * z_minus_floor_z[intidx] * eps11[ceil_x[intidx], floor_y[intidx], ceil_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_floor_y[intidx] * z_minus_floor_z[intidx] * eps11[floor_x[intidx], ceil_y[intidx], ceil_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_ceil_y[intidx] * z_minus_floor_z[intidx] * eps11[floor_x[intidx], floor_y[intidx], ceil_z[intidx]] \
           + x_minus_floor_x[intidx] * y_minus_floor_y[intidx] * z_minus_ceil_z[intidx] * eps11[ceil_x[intidx], ceil_y[intidx], floor_z[intidx]] \
           + x_minus_floor_x[intidx] * y_minus_ceil_y[intidx] * z_minus_ceil_z[intidx] * eps11[ceil_x[intidx], floor_y[intidx], floor_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_floor_y[intidx] * z_minus_ceil_z[intidx] * eps11[floor_x[intidx], ceil_y[intidx], floor_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_ceil_y[intidx] * z_minus_ceil_z[intidx] * eps11[floor_x[intidx], floor_y[intidx], floor_z[intidx]]
      tens[intidx[0][:],0,1] = x_minus_floor_x[intidx] * y_minus_floor_y[intidx] * z_minus_floor_z[intidx] * eps12[ceil_x[intidx], ceil_y[intidx], ceil_z[intidx]] \
           + x_minus_floor_x[intidx] * y_minus_ceil_y[intidx] * z_minus_floor_z[intidx] * eps12[ceil_x[intidx], floor_y[intidx], ceil_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_floor_y[intidx] * z_minus_floor_z[intidx] * eps12[floor_x[intidx], ceil_y[intidx], ceil_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_ceil_y[intidx] * z_minus_floor_z[intidx] * eps12[floor_x[intidx], floor_y[intidx], ceil_z[intidx]] \
           + x_minus_floor_x[intidx] * y_minus_floor_y[intidx] * z_minus_ceil_z[intidx] * eps12[ceil_x[intidx], ceil_y[intidx], floor_z[intidx]] \
           + x_minus_floor_x[intidx] * y_minus_ceil_y[intidx] * z_minus_ceil_z[intidx] * eps12[ceil_x[intidx], floor_y[intidx], floor_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_floor_y[intidx] * z_minus_ceil_z[intidx] * eps12[floor_x[intidx], ceil_y[intidx], floor_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_ceil_y[intidx] * z_minus_ceil_z[intidx] * eps12[floor_x[intidx], floor_y[intidx], floor_z[intidx]]
      tens[intidx[0][:],1,0] = tens[intidx[0][:],0,1]
      tens[intidx[0][:],0,2] = x_minus_floor_x[intidx] * y_minus_floor_y[intidx] * z_minus_floor_z[intidx] * eps13[ceil_x[intidx], ceil_y[intidx], ceil_z[intidx]] \
           + x_minus_floor_x[intidx] * y_minus_ceil_y[intidx] * z_minus_floor_z[intidx] * eps13[ceil_x[intidx], floor_y[intidx], ceil_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_floor_y[intidx] * z_minus_floor_z[intidx] * eps13[floor_x[intidx], ceil_y[intidx], ceil_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_ceil_y[intidx] * z_minus_floor_z[intidx] * eps13[floor_x[intidx], floor_y[intidx], ceil_z[intidx]] \
           + x_minus_floor_x[intidx] * y_minus_floor_y[intidx] * z_minus_ceil_z[intidx] * eps13[ceil_x[intidx], ceil_y[intidx], floor_z[intidx]] \
           + x_minus_floor_x[intidx] * y_minus_ceil_y[intidx] * z_minus_ceil_z[intidx] * eps13[ceil_x[intidx], floor_y[intidx], floor_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_floor_y[intidx] * z_minus_ceil_z[intidx] * eps13[floor_x[intidx], ceil_y[intidx], floor_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_ceil_y[intidx] * z_minus_ceil_z[intidx] * eps13[floor_x[intidx], floor_y[intidx], floor_z[intidx]]
      tens[intidx[0][:],2,0] = tens[intidx[0][:],0,2]
      tens[intidx[0][:],1,1] = x_minus_floor_x[intidx] * y_minus_floor_y[intidx] * z_minus_floor_z[intidx] * eps22[ceil_x[intidx], ceil_y[intidx], ceil_z[intidx]] \
           + x_minus_floor_x[intidx] * y_minus_ceil_y[intidx] * z_minus_floor_z[intidx] * eps22[ceil_x[intidx], floor_y[intidx], ceil_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_floor_y[intidx] * z_minus_floor_z[intidx] * eps22[floor_x[intidx], ceil_y[intidx], ceil_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_ceil_y[intidx] * z_minus_floor_z[intidx] * eps22[floor_x[intidx], floor_y[intidx], ceil_z[intidx]] \
           + x_minus_floor_x[intidx] * y_minus_floor_y[intidx] * z_minus_ceil_z[intidx] * eps22[ceil_x[intidx], ceil_y[intidx], floor_z[intidx]] \
           + x_minus_floor_x[intidx] * y_minus_ceil_y[intidx] * z_minus_ceil_z[intidx] * eps22[ceil_x[intidx], floor_y[intidx], floor_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_floor_y[intidx] * z_minus_ceil_z[intidx] * eps22[floor_x[intidx], ceil_y[intidx], floor_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_ceil_y[intidx] * z_minus_ceil_z[intidx] * eps22[floor_x[intidx], floor_y[intidx], floor_z[intidx]]
      tens[intidx[0][:],1,2] = x_minus_floor_x[intidx] * y_minus_floor_y[intidx] * z_minus_floor_z[intidx] * eps23[ceil_x[intidx], ceil_y[intidx], ceil_z[intidx]] \
           + x_minus_floor_x[intidx] * y_minus_ceil_y[intidx] * z_minus_floor_z[intidx] * eps23[ceil_x[intidx], floor_y[intidx], ceil_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_floor_y[intidx] * z_minus_floor_z[intidx] * eps23[floor_x[intidx], ceil_y[intidx], ceil_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_ceil_y[intidx] * z_minus_floor_z[intidx] * eps23[floor_x[intidx], floor_y[intidx], ceil_z[intidx]] \
           + x_minus_floor_x[intidx] * y_minus_floor_y[intidx] * z_minus_ceil_z[intidx] * eps23[ceil_x[intidx], ceil_y[intidx], floor_z[intidx]] \
           + x_minus_floor_x[intidx] * y_minus_ceil_y[intidx] * z_minus_ceil_z[intidx] * eps23[ceil_x[intidx], floor_y[intidx], floor_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_floor_y[intidx] * z_minus_ceil_z[intidx] * eps23[floor_x[intidx], ceil_y[intidx], floor_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_ceil_y[intidx] * z_minus_ceil_z[intidx] * eps23[floor_x[intidx], floor_y[intidx], floor_z[intidx]]
      tens[intidx[0][:],2,1] = tens[intidx[0][:],1,2]
      tens[intidx[0][:],2,2] = x_minus_floor_x[intidx] * y_minus_floor_y[intidx] * z_minus_floor_z[intidx] * eps33[ceil_x[intidx], ceil_y[intidx], ceil_z[intidx]] \
           + x_minus_floor_x[intidx] * y_minus_ceil_y[intidx] * z_minus_floor_z[intidx] * eps33[ceil_x[intidx], floor_y[intidx], ceil_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_floor_y[intidx] * z_minus_floor_z[intidx] * eps33[floor_x[intidx], ceil_y[intidx], ceil_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_ceil_y[intidx] * z_minus_floor_z[intidx] * eps33[floor_x[intidx], floor_y[intidx], ceil_z[intidx]] \
           + x_minus_floor_x[intidx] * y_minus_floor_y[intidx] * z_minus_ceil_z[intidx] * eps33[ceil_x[intidx], ceil_y[intidx], floor_z[intidx]] \
           + x_minus_floor_x[intidx] * y_minus_ceil_y[intidx] * z_minus_ceil_z[intidx] * eps33[ceil_x[intidx], floor_y[intidx], floor_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_floor_y[intidx] * z_minus_ceil_z[intidx] * eps33[floor_x[intidx], ceil_y[intidx], floor_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_ceil_y[intidx] * z_minus_ceil_z[intidx] * eps33[floor_x[intidx], floor_y[intidx], floor_z[intidx]]
  except Exception as err:
    print('Caught Exception:', err)
    print('intidx:',intidx)
    print(intidx[0].shape)
    print(torch.sum(x_minus_floor_x[intidx]))
    print(torch.sum(eps11[ceil_x[intidx], ceil_y[intidx], ceil_z[intidx]]))
    raise

  # Find index where no interpolation is needed and just copy the values
  nointidx = torch.where((x_minus_floor_x + x_minus_ceil_x
               + y_minus_floor_y + y_minus_ceil_y
               + z_minus_floor_z + z_minus_ceil_z) < 1e-14)

  try:
    if len(nointidx[0]) > 0:
      # Since x == floor_x, y == floor_y and z == floor_z in this case, use them to index to avoid type error
      # IndexError: tensors used as indices must be long, byte or bool tensors
      tens[nointidx[0][:],0,0] = eps11[floor_x[nointidx],floor_y[nointidx],floor_z[nointidx]]
      tens[nointidx[0][:],0,1] = eps12[floor_x[nointidx],floor_y[nointidx],floor_z[nointidx]]
      tens[nointidx[0][:],1,0] = tens[nointidx[0][:],0,1]
      tens[nointidx[0][:],0,2] = eps13[floor_x[nointidx],floor_y[nointidx],floor_z[nointidx]]
      tens[nointidx[0][:],2,0] = tens[nointidx[0][:],0,2]
      tens[nointidx[0][:],1,1] = eps22[floor_x[nointidx],floor_y[nointidx],floor_z[nointidx]]
      tens[nointidx[0][:],1,2] = eps23[floor_x[nointidx],floor_y[nointidx],floor_z[nointidx]]
      tens[nointidx[0][:],2,1] = tens[nointidx[0][:],1,2]    
      tens[nointidx[0][:],2,2] = eps33[floor_x[nointidx],floor_y[nointidx],floor_z[nointidx]]
  except Exception as err:
    print('Caught Exception:', err)
    print('nointidx:',nointidx)
    print(nointidx[0].shape)
    print(torch.sum(x_minus_floor_x[nointidx]))
    print(torch.sum(eps11[ceil_x[nointidx], ceil_y[nointidx], ceil_z[nointidx]]))
    raise

  return (tens)
# end batch_tens_interp_3d_torch


# compute eigenvectors according to A Method for Fast Diagonalization of a 2x2 or 3x3 Real Symmetric Matrix
# M.J. Kronenburg
# https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwj6zeiLut3qAhUPac0KHcyjDn4QFjAGegQIAxAB&url=https%3A%2F%2Farxiv.org%2Fpdf%2F1306.6291&usg=AOvVaw0BbaDECw-ghHGxek-LaB33
def eigv(tens):
    # TODO check dimensions, for now assuming 2D
    # The hope is that this implementation fixes the sign issue
    phi = 0.5 * np.arctan2(2 * tens[:,:,0,1] , (tens[:,:,0,0] - tens[:,:,1,1]))
    vs = np.zeros_like(tens)
    # want vs [:,:,0] to contain the eigenvector corresponding to the smallest
    # eigenvalue, which is \lambda_2 from the link above ie [-sin(\phi) cos(\phi)]
    vs[:,:,1,0] = np.cos(phi)
    vs[:,:,1,1] = np.sin(phi)
    vs[:,:,0,1] = vs[:,:,1,0] # cos(phi)
    vs[:,:,0,0] = -vs[:,:,1,1] # -sin(phi)
    return (vs)

def eigv_up(tens):
    # Compute eigenvectors for 2D tensors stored in upper triangular format
    # TODO check dimensions, for now assuming 2D
    # The hope is that this implementation fixes the sign issue
    phi = 0.5 * np.arctan2(2 * tens[:,:,1] , (tens[:,:,0] - tens[:,:,2]))
    vs = np.zeros_like(tens)
    # want vs [:,:,0] to contain the eigenvector corresponding to the smallest
    # eigenvalue, which is \lambda_2 from the link above ie [-sin(\phi) cos(\phi)]
    vs[:,:,1,0] = np.cos(phi)
    vs[:,:,1,1] = np.sin(phi)
    vs[:,:,0,1] = vs[:,:,1,0] # cos(phi)
    vs[:,:,0,0] = -vs[:,:,1,1] # -sin(phi)
    return (vs)

def eigv_3d(tens):
  # Find principal eigenvectors of 3d tensor field.
  fw, fw_name = get_framework(tens)
  # if fw_name == 'numpy':
  #   eigenvals, eigenvecs = np.linalg.eigh(tens)
  # else:
  #   eigenvals, eigenvecs = torch.symeig(tens,eigenvectors=True)
  eigenvals, eigenvecs = fw.linalg.eigh(tens)
  return (eigenvecs)

def eigv_sign_deambig(eigenvecs):
  # deambiguate eigenvector sign in each direction independently
  # fw is framework.  Defaults to numpy, torch works as well
  # want center pixel eigenvector to have same sign as both neighbors, when both neighbors sign matches
  # lr_dot, bt_dot, rf_dot > 0 ==> neighbors sign matches
  # lp_dot, bp_dot, rp_dot < 0 ==> pixel sign does not match neighbor
  fw, fw_name = get_framework(eigenvecs)
  if fw_name == "numpy":
    vecsx = fw.copy(eigenvecs)
    vecsy = fw.copy(eigenvecs)
    vecsz = fw.copy(eigenvecs)
  else:
    vecsx = fw.clone(eigenvecs)
    vecsy = fw.clone(eigenvecs)
    vecsz = fw.clone(eigenvecs)

  #lp_dot = np.einsum('...j,...j',eigenvecs[:-1,:,:,:,2],eigenvecs[1:,:,:,:,2])
  #lr_dot = np.einsum('...j,...j',eigenvecs[:-2,:,:,:,2],eigenvecs[2:,:,:,:,2])
  #bp_dot = np.einsum('...j,...j',eigenvecs[:,:-1,:,:,2],eigenvecs[:,1:,:,:,2])
  #bt_dot = np.einsum('...j,...j',eigenvecs[:,:-2,:,:,2],eigenvecs[:,2:,:,:,2])
  #rp_dot = np.einsum('...j,...j',eigenvecs[:,:,:-1,:,2],eigenvecs[:,:,1:,:,2])
  #rf_dot = np.einsum('...j,...j',eigenvecs[:,:,:-2,:,2],eigenvecs[:,:,2:,:,2])
  lp_dot = fw.einsum('...j,...j',eigenvecs[:-1,:,:,:],eigenvecs[1:,:,:,:])
  lr_dot = fw.einsum('...j,...j',eigenvecs[:-2,:,:,:],eigenvecs[2:,:,:,:])
  bp_dot = fw.einsum('...j,...j',eigenvecs[:,:-1,:,:],eigenvecs[:,1:,:,:])
  bt_dot = fw.einsum('...j,...j',eigenvecs[:,:-2,:,:],eigenvecs[:,2:,:,:])
  rp_dot = fw.einsum('...j,...j',eigenvecs[:,:,:-1,:],eigenvecs[:,:,1:,:])
  rf_dot = fw.einsum('...j,...j',eigenvecs[:,:,:-2,:],eigenvecs[:,:,2:,:])
  for xx in range(eigenvecs.shape[0]):
    for yy in range(eigenvecs.shape[1]):
      for zz in range(eigenvecs.shape[2]):
        if xx < lr_dot.shape[0]:
          if lr_dot[xx,yy,zz] > 0 and lp_dot[xx,yy,zz] < 0:
            vecsx[xx+1,yy,zz,:] = -vecsx[xx+1,yy,zz,:]
        if yy < bt_dot.shape[1]:
          if bt_dot[xx,yy,zz] > 0 and bp_dot[xx,yy,zz] < 0:
            vecsy[xx,yy+1,zz,:] = -vecsy[xx,yy+1,zz,:]
        if zz < rf_dot.shape[2]:
          if rf_dot[xx,yy,zz] > 0 and rp_dot[xx,yy,zz] < 0:
            vecsz[xx,yy,zz+1,:] = -vecsz[xx,yy,zz+1,:]
            
  return (vecsx, vecsy, vecsz)

# def eigv_up_3d(tens):
#     # Compute eigenvectors for 3D tensors stored in upper triangular format
#     # TODO check dimensions, for now assuming 3D
#     # The hope is that this implementation fixes the sign issue
# NOT IMPLEMENTED YET
#     phi = 0.5 * np.arctan2(2 * tens[:,:,:,1] , (tens[:,:,:,0] - tens[:,:,:,2]))
#     vs = np.zeros_like(tens)
#     # want vs [:,:,0] to contain the eigenvector corresponding to the smallest
#     # eigenvalue, which is \lambda_2 from the link above ie [-sin(\phi) cos(\phi)]
#     vs[:,:,:,1,0] = np.cos(phi)
#     vs[:,:,:,1,1] = np.sin(phi)
#     vs[:,:,:,0,1] = vs[:,:,:,1,0] # cos(phi)
#     vs[:,:,:,0,0] = -vs[:,:,:,1,1] # -sin(phi)
#     return (vs)

def make_pos_def(tens, mask, small_eval = 0.00005, skip_small_eval=False):
  # make any small or negative eigenvalues slightly positive and then reconstruct tensors
  #print('entering tensors.make_pos_def, max(tens)', torch.max(tens))
  det_threshold=1e-11
  tens[torch.det(tens)<=det_threshold] = torch.eye((3)).double().to(device=tens.device)

  fw, fw_name = get_framework(tens)
  if fw_name == 'numpy':
    #sym_tens = (tens + tens.transpose(0,1,2,4,3))/2
    sym_tens = (tens + tens.transpose(len(tens.shape)-2,len(tens.shape)-1))/2
    evals, evecs = np.linalg.eig(sym_tens)
  else:
    #sym_tens = (tens + torch.transpose(tens,3,4))/2
    sym_tens = (tens + torch.transpose(tens,len(tens.shape)-2,len(tens.shape)-1))/2
    #evals, evecs = torch.symeig(sym_tens,eigenvectors=True)
    evals, evecs = se.apply(sym_tens.reshape((-1,3,3)))
    evals = evals.reshape((*tens.shape[:-2],3))
    evecs = evecs.reshape((*tens.shape[:-2],3,3))
  #cmplx_evals, cmplx_evecs = fw.linalg.eig(sym_tens)
  #evals = fw.real(cmplx_evals)
  #evecs = fw.real(cmplx_evecs)
  #np.abs(evals, out=evals)
  idx = fw.where(evals < small_eval)
  #idx = np.where(evals < 0)
  num_found = 0
  #print(len(idx[0]), 'tensors found with eigenvalues <', small_eval)
  for ee in range(len(idx[0])):
    eeidx = [idx[ii][ee] for ii in range(len(idx))]
    if mask is None or mask[eeidx[:-1]]:
      num_found += 1
      # If largest eigenvalue is negative, replace with identity
      #eval_2 = (idx[3][ee]+1) % 3
      #eval_3 = (idx[3][ee]+2) % 3
      eval_2 = (eeidx[-1]+1) % 3
      eval_3 = (eeidx[-1]+2) % 3
      #if ((evals[idx[0][ee], idx[1][ee], idx[2][ee], eval_2] < 0) and 
      #   (evals[idx[0][ee], idx[1][ee], idx[2][ee], eval_3] < 0)):
      #  evecs[idx[0][ee], idx[1][ee], idx[2][ee]] = fw.eye(3, dtype=tens.dtype)
      #  evals[idx[0][ee], idx[1][ee], idx[2][ee], idx[3][ee]] = small_eval
      #else:
      #  # otherwise just set this eigenvalue to small_eval
      #  evals[idx[0][ee], idx[1][ee], idx[2][ee], idx[3][ee]] = small_eval
      if ((evals[(*eeidx[:-1], eval_2)] < 0) and 
         (evals[(*eeidx[:-1], eval_3)] < 0)):
        evecs[eeidx[:-1]] = fw.eye(3, dtype=tens.dtype).to(device=tens.device)
        #evals[(*eeidx[:-1], eeidx[-1])] = small_eval
        evals[eeidx] = small_eval
      else:
        # otherwise just set this eigenvalue to small_eval
        #evals[(*eeidx[:-1], eeidx[-1])] = small_eval
        evals[eeidx] = small_eval

  #print(num_found, 'tensors found with eigenvalues <', small_eval)
  #print(num_found, 'tensors found with eigenvalues < 0')
  mod_tens = fw.einsum('...ij,...jk,...k,...lk->...il',
                       evecs, fw.eye(3, dtype=tens.dtype).to(device=tens.device), evals, evecs)
  #mod_tens = fw.einsum('...ij,...j,...jk->...ik',
  #                     evecs, evals, evecs)

  # Need small_eval fix for sure for inv_RieExp calculation, hence why this version is different than that in BrainAtlasBuilding3DUkfCudaImg
  if skip_small_eval:
    print("WARNING!!!! Ignoring small_eval fix in tensors.make_pos_def")
    mod_tens = tens.clone()
  chol = batch_cholesky(mod_tens)
  idx = fw.where(fw.isnan(chol))
  iso_tens = small_eval * fw.eye(3, dtype=tens.dtype).to(device=tens.device)
  for pt in range(len(idx[0])):
    #mod_tens[idx[0][pt],idx[1][pt],idx[2][pt]] = iso_tens
    mod_tens[idx[0][pt]] = iso_tens

  if fw_name == 'numpy':
    #mod_sym_tens = (mod_tens + mod_tens.transpose(0,1,2,4,3))/2
    mod_sym_tens = (tens + mod_tens.transpose(len(tens.shape)-2,len(tens.shape)-1))/2
  else:
    mod_sym_tens = (mod_tens + torch.transpose(mod_tens,len(mod_tens.shape)-2,len(mod_tens.shape)-1))/2
  mod_sym_tens[torch.det(mod_sym_tens)<=det_threshold] = fw.eye(3, dtype=tens.dtype).to(device=tens.device)

  #print('leaving tensors.make_pos_def, max(mod_sym_tens)', torch.max(mod_sym_tens))

  return(mod_sym_tens)
    
 
def scale_by_alpha(tensors, alpha):
  # This scaling function assumes that the input provided for scaling are diffusion tensors
  # and hence scales by 1/e^{\alpha}.
  # If the inverse-tensor metric is provided instead, we would need to scale by e^\alpha
  fw, fw_name = get_framework(tensors)
  if fw_name == "numpy":
    out_tensors = fw.copy(tensors)
  else:
    out_tensors = fw.clone(tensors)

  if tensors.shape[2] == 3:
    for kk in range(3): 
      out_tensors[:,:,kk] /= fw.exp(alpha)
  elif tensors.shape[2:] == (2, 2):
    for jj in range(2):
      for kk in range(2):
        out_tensors[:,:,jj,kk] /= fw.exp(alpha)
  elif tensors.shape[3] == 6:
    for kk in range(6): 
      out_tensors[:,:,:,kk] /= fw.exp(alpha)
  elif tensors.shape[3:] == (3, 3):
    for jj in range(3):
      for kk in range(3):
        out_tensors[:,:,:,jj,kk] /= fw.exp(alpha)
  else:
    print(tensors.shape, "unexpected tensor shape")
  return(out_tensors)

def get_norm(tens, axis):
  fw, fw_name = get_framework(tens)
  if fw_name == 'numpy':
    tens_norm = np.linalg.norm(tens, axis=axis)
  else:
    tens_norm = torch.linalg.norm(tens, dim=axis)
  return(tens_norm)

def threshold_to_input(tens_to_thresh, input_tens, mask, ratio=1.0):
  # scale the tens_to_thresh by the ratio * norm^2 of the largest tensor in input_tens
  # assumes input tens are full 2x2 tensors
  # TODO confirm that ratio is between 0 and 1
  fw, fw_name = get_framework(tens_to_thresh)
  if input_tens.shape[2] == 3:
    norm_in_tens = get_norm(input_tens,axis=(2))
  elif input_tens.shape[2:] == (2,2):
    norm_in_tens = get_norm(input_tens,axis=(2,3))
  elif input_tens.shape[3] == 6:
    norm_in_tens = get_norm(input_tens,axis=(3))
  elif input_tens.shape[3:] == (3,3):
    norm_in_tens = get_norm(input_tens,axis=(3,4))
  else:
    print(input_tens.shape, "unexpected tensor shape")
  if tens_to_thresh.shape[2] == 3:
    norm_sq = get_norm(tens_to_thresh,axis=(2))
  elif tens_to_thresh.shape[2:] == (2,2):
    norm_sq = get_norm(tens_to_thresh,axis=(2,3))
  elif tens_to_thresh.shape[3] == 6:
    norm_sq = get_norm(tens_to_thresh,axis=(3))    
  elif tens_to_thresh.shape[3:] == (3,3):
    norm_sq = get_norm(tens_to_thresh,axis=(3,4))    
  else:
    print(tens_to_thresh.shape, "unexpected tensor shape")  
  norm_sq = norm_sq * norm_sq # norm squared of each tensor in tens_to_thresh

  # just square the threshold, no need to element-wise square the entire norm_in_tens matrix
  thresh = fw.max(norm_in_tens)
  thresh = ratio * thresh * thresh
  
  if fw_name == "numpy":
    thresh_tens = fw.copy(tens_to_thresh)
  else:
    thresh_tens = fw.clone(tens_to_thresh)

  scale_factor = fw.ones_like(norm_sq)
  scale_factor[norm_sq > thresh] = thresh / norm_sq[norm_sq > thresh]
  scale_factor[mask == 0] = 1

  if tens_to_thresh.shape[2] == 3:
    for kk in range(3): 
      thresh_tens[:,:,kk] *= scale_factor
  elif tens_to_thresh.shape[2:] == (2, 2):
    for jj in range(2):
      for kk in range(2):
        thresh_tens[:,:,jj,kk] *= scale_factor
  elif tens_to_thresh.shape[3] == 6:
    for kk in range(6): 
      thresh_tens[:,:,:,kk] *= scale_factor
  elif tens_to_thresh.shape[3:] == (3, 3):
    for jj in range(3):
      for kk in range(3):
        thresh_tens[:,:,:,jj,kk] *= scale_factor
  else:
    print(tens_to_thresh.shape, "unexpected tensor shape")

  return(thresh_tens)
# end threshold_to_input

def tensor_cleaning(g, scale_factor):
    abnormal_map = torch.where(torch.det(g)>4,1.,0.)
    background = torch.einsum("mno,ij->mnoij", torch.ones(*tensor_met_zeros.shape[:3]), torch.eye(3, dtype=g.dtype))*scale_factor
#     return torch.einsum('ijk...,lijk->ijk...', g, 1.-abnormal_map.unsqueeze(0))+\
#             torch.einsum('ijk...,lijk->ijk...', background, abnormal_map.unsqueeze(0))
    return torch.einsum('ijk...,lijk->ijk...', g, 1.-abnormal_map.unsqueeze(0))+\
            torch.einsum('ijk...,lijk->ijk...', g, (abnormal_map/torch.det(g)).unsqueeze(0))

  
