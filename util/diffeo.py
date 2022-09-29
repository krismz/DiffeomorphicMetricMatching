# Some useful operations for diffeomorphisms

import math
from lazy_imports import np
from lazy_imports import sio
from lazy_imports import torch
from util.tensors import direction, direction_3d


def coord_register(point_x, point_y, diffeo):
  # TODO work out which is y and which is x, maintain consistency.
  # For now, pass in y for point_x, x for point_y
  new_point_x, new_point_y = [], []
  for i in range(len(point_x)):
    D = point_y[i] - math.floor(point_y[i])
    C = point_x[i] - math.floor(point_x[i])
    new_point_x.append((1. - D) * (1. - C) * diffeo[
            0, math.floor(point_y[i]) % diffeo.shape[1], math.floor(point_x[i]) % diffeo.shape[2]]
                           + C * (1. - D) * diffeo[
                               0, math.floor(point_y[i]) % diffeo.shape[1], math.ceil(point_x[i]) % diffeo.shape[2]]
                           + D * (1. - C) * diffeo[
                               0, math.ceil(point_y[i]) % diffeo.shape[1], math.floor(point_x[i]) % diffeo.shape[2]]
                           + C * D * diffeo[
                               0, math.ceil(point_y[i]) % diffeo.shape[1], math.ceil(point_x[i]) % diffeo.shape[2]])
 
    new_point_y.append((1. - D) * (1. - C) * diffeo[
            1, math.floor(point_y[i]) % diffeo.shape[1], math.floor(point_x[i]) % diffeo.shape[2]]
                           + C * (1. - D) * diffeo[
                               1, math.floor(point_y[i]) % diffeo.shape[1], math.ceil(point_x[i]) % diffeo.shape[2]]
                           + D * (1. - C) * diffeo[
                               1, math.ceil(point_y[i]) % diffeo.shape[1], math.floor(point_x[i]) % diffeo.shape[2]]
                           + C * D * diffeo[
                               1, math.ceil(point_y[i]) % diffeo.shape[1], math.ceil(point_x[i]) % diffeo.shape[2]])
 
  return(new_point_x, new_point_y)

def coord_register_3d(point_x, point_y, point_z, diffeo):
  # x,y,z order is correct and as expected
  height, width, depth=diffeo.shape[-3:]
  new_point_x, new_point_y, new_point_z = [], [], []
  for i in range(len(point_x)):
    C = point_x[i] - math.floor(point_x[i])
    D = point_y[i] - math.floor(point_y[i])
    E = point_z[i] - math.floor(point_z[i])
    new_point_x.append(\
      (1.-C)*(1.-D)*(1.-E)*diffeo[0, math.floor(point_x[i])%height, math.floor(point_y[i])%width, math.floor(point_z[i])%depth]\
    + (1.-C)*D*(1.-E)*diffeo[0, math.floor(point_x[i])%height, math.ceil(point_y[i])%width, math.floor(point_z[i])%depth]\
    + C*(1.-D)*(1.-E)*diffeo[0, math.ceil(point_x[i])%height, math.floor(point_y[i])%width, math.floor(point_z[i])%depth]\
    + C*D*(1.-E)*diffeo[0, math.ceil(point_x[i])%height, math.ceil(point_y[i])%width, math.floor(point_z[i])%depth]\
    + (1.-C)*(1.-D)*E*diffeo[0, math.floor(point_x[i])%height, math.floor(point_y[i])%width, math.ceil(point_z[i])%depth]\
    + (1.-C)*D*E*diffeo[0, math.floor(point_x[i])%height, math.ceil(point_y[i])%width, math.ceil(point_z[i])%depth]\
    + C*(1.-D)*E*diffeo[0, math.ceil(point_x[i])%height, math.floor(point_y[i])%width, math.ceil(point_z[i])%depth]\
    + C*D*E*diffeo[0, math.ceil(point_x[i])%height, math.ceil(point_y[i])%width, math.ceil(point_z[i])%depth])
 
    new_point_y.append(\
      (1.-C)*(1.-D)*(1.-E)*diffeo[1, math.floor(point_x[i])%height, math.floor(point_y[i])%width, math.floor(point_z[i])%depth]\
    + (1.-C)*D*(1.-E)*diffeo[1, math.floor(point_x[i])%height, math.ceil(point_y[i])%width, math.floor(point_z[i])%depth]\
    + C*(1.-D)*(1.-E)*diffeo[1, math.ceil(point_x[i])%height, math.floor(point_y[i])%width, math.floor(point_z[i])%depth]\
    + C*D*(1.-E)*diffeo[1, math.ceil(point_x[i])%height, math.ceil(point_y[i])%width, math.floor(point_z[i])%depth]\
    + (1.-C)*(1.-D)*E*diffeo[1, math.floor(point_x[i])%height, math.floor(point_y[i])%width, math.ceil(point_z[i])%depth]\
    + (1.-C)*D*E*diffeo[1, math.floor(point_x[i])%height, math.ceil(point_y[i])%width, math.ceil(point_z[i])%depth]\
    + C*(1.-D)*E*diffeo[1, math.ceil(point_x[i])%height, math.floor(point_y[i])%width, math.ceil(point_z[i])%depth]\
    + C*D*E*diffeo[1, math.ceil(point_x[i])%height, math.ceil(point_y[i])%width, math.ceil(point_z[i])%depth])
 
    new_point_z.append(\
      (1.-C)*(1.-D)*(1.-E)*diffeo[2, math.floor(point_x[i])%height, math.floor(point_y[i])%width, math.floor(point_z[i])%depth]\
    + (1.-C)*D*(1.-E)*diffeo[2, math.floor(point_x[i])%height, math.ceil(point_y[i])%width, math.floor(point_z[i])%depth]\
    + C*(1.-D)*(1.-E)*diffeo[2, math.ceil(point_x[i])%height, math.floor(point_y[i])%width, math.floor(point_z[i])%depth]\
    + C*D*(1.-E)*diffeo[2, math.ceil(point_x[i])%height, math.ceil(point_y[i])%width, math.floor(point_z[i])%depth]\
    + (1.-C)*(1.-D)*E*diffeo[2, math.floor(point_x[i])%height, math.floor(point_y[i])%width, math.ceil(point_z[i])%depth]\
    + (1.-C)*D*E*diffeo[2, math.floor(point_x[i])%height, math.ceil(point_y[i])%width, math.ceil(point_z[i])%depth]\
    + C*(1.-D)*E*diffeo[2, math.ceil(point_x[i])%height, math.floor(point_y[i])%width, math.ceil(point_z[i])%depth]\
    + C*D*E*diffeo[2, math.ceil(point_x[i])%height, math.ceil(point_y[i])%width, math.ceil(point_z[i])%depth])
  return (new_point_x, new_point_y, new_point_z)

def coord_register_batch_3d(point_x, point_y, point_z, diffeo):
  # x,y,z order is correct and as expected
  height, width, depth=diffeo.shape[-3:]
  num_points = len(point_x.flatten())
  new_point_x = np.zeros_like(point_x)
  new_point_y = np.zeros_like(point_y)
  new_point_z = np.zeros_like(point_z)

  ceil_x = np.mod(np.ceil(point_x),height).astype(np.int_)
  floor_x = np.mod(np.floor(point_x),height).astype(np.int_)
  ceil_y = np.mod(np.ceil(point_y),width).astype(np.int_)
  floor_y = np.mod(np.floor(point_y),width).astype(np.int_)
  ceil_z = np.mod(np.ceil(point_z),depth).astype(np.int_)
  floor_z = np.mod(np.floor(point_z),depth).astype(np.int_)
  #ceil_x = np.minimum(np.ceil(point_x),height-1).astype(np.int_)
  #floor_x = np.maximum(np.floor(point_x),0).astype(np.int_)
  #ceil_y = np.minimum(np.ceil(point_y),width-1).astype(np.int_)
  #floor_y = np.maximum(np.floor(point_y),0).astype(np.int_)
  #ceil_z = np.minimum(np.ceil(point_z),depth-1).astype(np.int_)
  #floor_z = np.maximum(np.floor(point_z),0).astype(np.int_)
  
  C = np.abs(point_x - floor_x)
  D = np.abs(point_y - floor_y)
  E = np.abs(point_z - floor_z)

  for p in range(num_points):
    new_point_x[p] = (1.-C[p])*(1.-D[p])*(1.-E[p])*diffeo[0, floor_x[p], floor_y[p], floor_z[p]] \
                   + (1.-C[p])*D[p]*(1.-E[p])*diffeo[0, floor_x[p], ceil_y[p], floor_z[p]]\
                   + C[p]*(1.-D[p])*(1.-E[p])*diffeo[0, ceil_x[p], floor_y[p], floor_z[p]]\
                   + C[p]*D[p]*(1.-E[p])*diffeo[0, ceil_x[p], ceil_y[p], floor_z[p]]\
                   + (1.-C[p])*(1.-D[p])*E[p]*diffeo[0, floor_x[p], floor_y[p], ceil_z[p]]\
                   + (1.-C[p])*D[p]*E[p]*diffeo[0, floor_x[p], ceil_y[p], ceil_z[p]]\
                   + C[p]*(1.-D[p])*E[p]*diffeo[0, ceil_x[p], floor_y[p], ceil_z[p]]\
                   + C[p]*D[p]*E[p]*diffeo[0, ceil_x[p], ceil_y[p], ceil_z[p]]
 
    new_point_y[p] = (1.-C[p])*(1.-D[p])*(1.-E[p])*diffeo[1, floor_x[p], floor_y[p], floor_z[p]]\
                   + (1.-C[p])*D[p]*(1.-E[p])*diffeo[1, floor_x[p], ceil_y[p], floor_z[p]]\
                   + C[p]*(1.-D[p])*(1.-E[p])*diffeo[1, ceil_x[p], floor_y[p], floor_z[p]]\
                   + C[p]*D[p]*(1.-E[p])*diffeo[1, ceil_x[p], ceil_y[p], floor_z[p]]\
                   + (1.-C[p])*(1.-D[p])*E[p]*diffeo[1, floor_x[p], floor_y[p], ceil_z[p]]\
                   + (1.-C[p])*D[p]*E[p]*diffeo[1, floor_x[p], ceil_y[p], ceil_z[p]]\
                   + C[p]*(1.-D[p])*E[p]*diffeo[1, ceil_x[p], floor_y[p], ceil_z[p]]\
                   + C[p]*D[p]*E[p]*diffeo[1, ceil_x[p], ceil_y[p], ceil_z[p]]
 
    new_point_z[p] = (1.-C[p])*(1.-D[p])*(1.-E[p])*diffeo[2, floor_x[p], floor_y[p], floor_z[p]]\
                   + (1.-C[p])*D[p]*(1.-E[p])*diffeo[2, floor_x[p], ceil_y[p], floor_z[p]]\
                   + C[p]*(1.-D[p])*(1.-E[p])*diffeo[2, ceil_x[p], floor_y[p], floor_z[p]]\
                   + C[p]*D[p]*(1.-E[p])*diffeo[2, ceil_x[p], ceil_y[p], floor_z[p]]\
                   + (1.-C[p])*(1.-D[p])*E[p]*diffeo[2, floor_x[p], floor_y[p], ceil_z[p]]\
                   + (1.-C[p])*D[p]*E[p]*diffeo[2, floor_x[p], ceil_y[p], ceil_z[p]]\
                   + C[p]*(1.-D[p])*E[p]*diffeo[2, ceil_x[p], floor_y[p], ceil_z[p]]\
                   + C[p]*D[p]*E[p]*diffeo[2, ceil_x[p], ceil_y[p], ceil_z[p]]

    if (ceil_x[p] < point_x[p]) or (ceil_y[p] < point_y[p]) or (ceil_z[p] < point_z[p]):
      print('Strange things at point',p,':',point_x[p],point_y[p],point_z[p],ceil_x[p],ceil_y[p],ceil_z[p])

  return (new_point_x, new_point_y, new_point_z)


def coord_velocity_register(point_x, point_y, tensor_field, delta_t, diffeo):
  # TODO work out which is y and which is x, maintain consistency.
  # For now, pass in y for point_x, x for point_y
  # returns new y, new x as new_point_x, new_point_y
  # keeping y-x convention from coord_register for now for points,
  # but fixing it to x-y for velocity
  # tensor_field is in [tensor, x, y] order 
  # velocity will be returned as [new_vel_x, new_vel_y]
  new_point_x, new_point_y = coord_register(point_x, point_y, diffeo)
  new_velocity = []

  print("WARNING WARNING WARNING!!! Treat the following velocity code as highly suspect!!!")
  
  for i in range(len(point_x)):
    v = direction([new_point_y[i], new_point_x[i]], tensor_field)
    end_x = point_x[i] + v[1] * delta_t
    end_y = point_y[i] + v[0] * delta_t
    new_end_x, new_end_y = coord_register([end_x], [end_y], diffeo)

    new_velocity.append([(new_end_y[0] - new_point_y[i]) / delta_t, (new_end_x[0] - new_point_x[i]) / delta_t])

  return(new_point_x, new_point_y, new_velocity)

def coord_velocity_register_3d(point_x, point_y, point_z, tensor_field, delta_t, diffeo):
  # returns new x, new y, new z as new_point_x, new_point_y, new_point_z
  # tensor_field is in [tensor, x, y, z] order 
  # velocity will be returned as [new_vel_x, new_vel_y, new_vel_z]
  new_point_x, new_point_y, new_point_z = coord_register_3d(point_x, point_y, point_z, diffeo)
  new_velocity = []

  for i in range(len(point_x)):
    v = direction_3d([new_point_x[i], new_point_y[i], new_point_z[i]], tensor_field)
    end_x = point_x[i] + v[0] * delta_t
    end_y = point_y[i] + v[1] * delta_t
    end_z = point_z[i] + v[2] * delta_t
    new_end_x, new_end_y, new_end_z = coord_register_3d([end_x], [end_y], [end_z], diffeo)

    new_velocity.append([(new_end_x[0] - new_point_x[i]) / delta_t,
                         (new_end_y[0] - new_point_y[i]) / delta_t,
                         (new_end_z[0] - new_point_z[i]) / delta_t])

  return(new_point_x, new_point_y, new_point_z, new_velocity)


# define the pullback action of phi
def phi_pullback(phi, g):
    idty = get_idty(*g.shape[-2:]).to(phi.device)
#     four layers of scalar field, of all 1, all 0, all 1, all 0, where the shape of each layer is g.shape[-2:]?
    d_phi = get_jacobian_matrix(phi - idty) + torch.einsum("ij,mn->ijmn", [torch.eye(2,dtype=torch.double, device=phi.device),
                                                                           torch.ones(g.shape[-2:],dtype=torch.double, device=phi.device)])
    g_phi = compose_function(g, phi)
#     matrix multiplication
# the last two dimension stays the same means point-wise multiplication, ijmn instead of jimn means the first d_phi need to be transposed
    return torch.einsum("ijmn,ikmn,klmn->jlmn",[d_phi, g_phi, d_phi])

def phi_pullback_3d(phi, g):
#     input: phi.shape = [3, h, w, d]; g.shape = [h, w, d, 3, 3]
#     output: shape = [h, w, d, 3, 3]
#     torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    g = g.permute(3, 4, 0, 1, 2)
    idty = get_idty_3d(*g.shape[-3:]).to(g.device)
    #     four layers of scalar field, of all 1, all 0, all 1, all 0, where the shape of each layer is g.shape[-2:]?
    eye = torch.eye(3, device=phi.device)
    ones = torch.ones(*g.shape[-3:], device=g.device)
    d_phi = get_jacobian_matrix_3d(phi - idty) + torch.einsum("ij,mno->ijmno", eye, ones)
    g_phi = compose_function_3d(g, phi)
    return torch.einsum("ij...,ik...,kl...->...jl", d_phi, g_phi, d_phi)

def get_jacobian_matrix(diffeo): # diffeo: 2 x size_h x size_w
#     return torch.stack((get_gradient(diffeo[1]), get_gradient(diffeo[0])))
    return torch.stack((get_gradient(diffeo[0]), get_gradient(diffeo[1])))

def get_jacobian_matrix_3d(diffeo):  # diffeo: 3 x size_h x size_w x size_d
    return torch.stack((get_gradient_3d(diffeo[0]), get_gradient_3d(diffeo[1]), get_gradient_3d(diffeo[2]))).type(torch.DoubleTensor).to(diffeo.device)#.to(device=torch.device('cuda'))

def get_gradient(F):  # 2D F: size_h x size_w
    F_padded = torch.zeros((F.shape[0]+2,F.shape[1]+2), device=F.device)#, dtype=torch.double)
    F_padded[1:-1,1:-1] = F
    F_padded[0,:] = F_padded[1,:]
    F_padded[-1,:] = F_padded[-2,:]
    F_padded[:,0] = F_padded[:,1]
    F_padded[:,-1] = F_padded[:,-2]
    F_x = (torch.roll(F_padded, shifts=(0, -1), dims=(0, 1)) - torch.roll(F_padded, shifts=(0, 1), dims=(0, 1)))/2
    F_y = (torch.roll(F_padded, shifts=(-1, 0), dims=(0, 1)) - torch.roll(F_padded, shifts=(1, 0), dims=(0, 1)))/2
    return torch.stack((F_x[1:-1,1:-1], F_y[1:-1,1:-1]))
#     F_x = (torch.roll(F, shifts=(0, -1), dims=(0, 1)) - torch.roll(F, shifts=(0, 1), dims=(0, 1)))/2
#     F_y = (torch.roll(F, shifts=(-1, 0), dims=(0, 1)) - torch.roll(F, shifts=(1, 0), dims=(0, 1)))/2
#     return torch.stack((F_x, F_y))

def get_gradient_3d(F):  # 3D F: size_h x size_w x size_d
    F_padded = torch.zeros((F.shape[0] + 2, F.shape[1] + 2, F.shape[2] + 2), device=F.device)#, dtype=torch.double)
    F_padded[1:-1, 1:-1, 1:-1] = F
    F_padded[0, :, :] = F_padded[1, :, :]
    F_padded[-1, :, :] = F_padded[-2, :, :]
    F_padded[:, 0, :] = F_padded[:, 1, :]
    F_padded[:, -1, :] = F_padded[:, -2, :]
    F_padded[:, :, 0] = F_padded[:, :, 1]
    F_padded[:, :, -1] = F_padded[:, :, -2]
#     original
#     F_x = (torch.roll(F_padded, shifts=(0, 0, -1), dims=(0, 1, 2))
#            - torch.roll(F_padded, shifts=(0, 0, 1), dims=(0, 1, 2))) / 2
#     F_y = (torch.roll(F_padded, shifts=(0, -1, 0), dims=(0, 1, 2))
#            - torch.roll(F_padded, shifts=(0, 1, 0), dims=(0, 1, 2))) / 2
#     F_z = (torch.roll(F_padded, shifts=(-1, 0, 0), dims=(0, 1, 2))
#            - torch.roll(F_padded, shifts=(1, 0, 0), dims=(0, 1, 2))) / 2
# 4.3 version
    F_x = (torch.roll(F_padded, shifts=(-1, 0, 0), dims=(0, 1, 2))
           - torch.roll(F_padded, shifts=(1, 0, 0), dims=(0, 1, 2))) / 2
    F_y = (torch.roll(F_padded, shifts=(0, -1, 0), dims=(0, 1, 2))
           - torch.roll(F_padded, shifts=(0, 1, 0), dims=(0, 1, 2))) / 2
    F_z = (torch.roll(F_padded, shifts=(0, 0, -1), dims=(0, 1, 2))
           - torch.roll(F_padded, shifts=(0, 0, 1), dims=(0, 1, 2))) / 2
    return torch.stack((F_x[1:-1, 1:-1, 1:-1],
                        F_y[1:-1, 1:-1, 1:-1],
                        F_z[1:-1, 1:-1, 1:-1])).type(torch.DoubleTensor).to(F.device)


# get the identity mapping
def get_idty(size_h, size_w): 
    HH, WW = torch.meshgrid([torch.arange(size_h, dtype=torch.double), torch.arange(size_w, dtype=torch.double)])
#     return torch.stack((HH, WW))
    return torch.stack((WW, HH))

# get the identity mapping
def get_idty_3d(size_h, size_w, size_d):
    HH, WW, DD = torch.meshgrid([torch.arange(size_h),#, dtype=torch.double
                                 torch.arange(size_w),#, dtype=torch.double
                                 torch.arange(size_d)])#, dtype=torch.double
# original and 4.3
    return torch.stack((HH, WW, DD)).float()#.double() #.half()
# 4.7
#     return torch.stack((DD, WW, HH)).double() #.half()
  

# my interpolation function
def compose_function(f, diffeo, mode='periodic'):  # f: N x m x n  diffeo: 2 x m x n
    
    f = f.permute(f.dim()-2, f.dim()-1, *range(f.dim()-2))  # change the size of f to m x n x ...
    
    size_h, size_w = f.shape[:2]
#     Ind_diffeo = torch.stack((torch.floor(diffeo[0]).long()%size_h, torch.floor(diffeo[1]).long()%size_w))
    Ind_diffeo = torch.stack((torch.floor(diffeo[1]).long()%size_h, torch.floor(diffeo[0]).long()%size_w))

    F = torch.zeros(size_h+1, size_w+1, *f.shape[2:], dtype=torch.double, device=f.device)
    
    if mode=='border':
        F[:size_h,:size_w] = f
        F[-1, :size_w] = f[-1]
        F[:size_h, -1] = f[:, -1]
        F[-1, -1] = f[-1,-1]
    elif mode =='periodic':
        # extend the function values periodically (1234 1)
        F[:size_h,:size_w] = f
        F[-1, :size_w] = f[0]
        F[:size_h, -1] = f[:, 0]
        F[-1, -1] = f[0,0]
    
    # use the bilinear interpolation method
    F00 = F[Ind_diffeo[0], Ind_diffeo[1]].permute(*range(2, f.dim()), 0, 1)  # change the size to ...*m*n
    F01 = F[Ind_diffeo[0], Ind_diffeo[1]+1].permute(*range(2, f.dim()), 0, 1)
    F10 = F[Ind_diffeo[0]+1, Ind_diffeo[1]].permute(*range(2, f.dim()), 0, 1)
    F11 = F[Ind_diffeo[0]+1, Ind_diffeo[1]+1].permute(*range(2, f.dim()), 0, 1)

#     C = diffeo[0] - Ind_diffeo[0].type(torch.DoubleTensor)
#     D = diffeo[1] - Ind_diffeo[1].type(torch.DoubleTensor)
    C = diffeo[0] - Ind_diffeo[1].type(torch.DoubleTensor)
    D = diffeo[1] - Ind_diffeo[0].type(torch.DoubleTensor)

    F0 = F00 + (F01 - F00)*C
    F1 = F10 + (F11 - F10)*C
    return F0 + (F1 - F0)*D
#     return (1-D)*(1-C)*F00+C*(1-D)*F01+D*(1-C)*F10+C*D*F11

# my interpolation function
def compose_function_orig_3d(f, diffeo, mode='periodic'):  # f: N x h x w x d  diffeo: 3 x h x w x d
 #   print('f.shape',f.shape,'\nf:\n',f)
    #f = f.permute(f.dim() - 3, f.dim() - 2, f.dim() - 1, *range(f.dim() - 3))  # change the size of f to m x n x ...
    #size_h, size_w, size_d = f.shape[:3]
    size_h, size_w, size_d = f.shape[-3:]

#     original and 4.3
    if mode == 'id' or mode == 'zero':
#      print('diffeo:\n',diffeo)
      #Ind_diffeo_bdry = torch.stack((torch.floor(diffeo[0]).long(),
      #                          torch.floor(diffeo[1]).long(),
      #                          torch.floor(diffeo[2]).long()))#.to(device=torch.device('cuda'))
      Ind_diffeo = torch.stack((torch.floor(diffeo[0]).long(),
                                torch.floor(diffeo[1]).long(),
                                torch.floor(diffeo[2]).long()))#.to(device=torch.device('cuda'))
      #Ind_diffeo = torch.stack((diffeo[0].long(),
      #                          diffeo[1].long(),
      #                          diffeo[2].long()))#.to(device=torch.device('cuda'))

      #for i in range(Ind_diffeo.shape[0]):
      #  idx = torch.where((diffeo[i] < 0) & (diffeo[i] != Ind_diffeo[i]))
      #  Ind_diffeo[i][idx] = diffeo[i,idx[0],idx[1],idx[2]].long() - 1
      
    else:
      Ind_diffeo = torch.stack((torch.floor(diffeo[0]).long() % size_h,
                                torch.floor(diffeo[1]).long() % size_w,
                                torch.floor(diffeo[2]).long() % size_d))#.to(device=torch.device('cuda'))
#     4.7
#     Ind_diffeo = torch.stack((torch.floor(diffeo[2]).long() % size_h,
#                               torch.floor(diffeo[1]).long() % size_w,
#                               torch.floor(diffeo[0]).long() % size_d))#.to(device=torch.device('cuda'))

    #F = torch.zeros(size_h + 1, size_w + 1, size_d + 1, *f.shape[3:], device=f.device)#, dtype=torch.double
    #F = torch.zeros(*f.shape[:-3],size_h + 1, size_w + 1, size_d + 1, device=f.device)#, dtype=torch.double
    F = torch.zeros(*f.shape[:-3],size_h + 2, size_w + 2, size_d + 2, device=f.device)#, dtype=torch.double

    if mode == 'border':
        # F[:size_h, :size_w, :size_d] = f
        # F[-1, :size_w, :size_d] = f[-1]
        # F[:size_h, -1, :size_d] = f[:, -1]
        # F[:size_h, :size_w, -1] = f[:, :, -1]
        # F[-1, -1, :size_d] = f[-1, -1, :]
        # F[-1, :size_w, -1] = f[-1, :, -1]
        # F[:size_h, -1, -1] = f[:, -1, -1]
        # F[-1, -1, -1] = f[-1, -1, -1]
        # F[..., :size_h, :size_w, :size_d] = f
        # F[..., -1, :size_w, :size_d] = f[..., -1]
        # F[..., :size_h, -1, :size_d] = f[..., :, -1]
        # F[..., :size_h, :size_w, -1] = f[..., :, :, -1]
        # F[..., -1, -1, :size_d] = f[..., -1, -1, :]
        # F[..., -1, :size_w, -1] = f[..., -1, :, -1]
        # F[..., :size_h, -1, -1] = f[..., :, -1, -1]
        # F[..., -1, -1, -1] = f[..., -1, -1, -1]

        F[..., 1:size_h+1, 1:size_w+1, 1:size_d+1] = f
        F[..., 0, 1:size_w+1, 1:size_d+1] = f[..., 0, :, :]
        F[..., -1, 1:size_w+1, 1:size_d+1] = f[..., -1, :, :]
        F[..., 1:size_h+1, 0, 1:size_d+1] = f[..., :, 0, :]
        F[..., 1:size_h+1, -1, 1:size_d+1] = f[..., :, -1, :]
        F[..., 1:size_h+1, 1:size_w+1, 0] = f[..., :, :, 0]
        F[..., 1:size_h+1, 1:size_w+1, -1] = f[..., :, :, -1]
        F[..., 0, 0, 1:size_d+1] = f[..., 0, 0, :]
        F[..., -1, 0, 1:size_d+1] = f[..., -1, 0, :]
        F[..., 0, -1, 1:size_d+1] = f[..., 0, -1, :]
        F[..., -1, -1, 1:size_d+1] = f[..., -1, -1, :]
        F[..., 0, 1:size_w+1, 0] = f[..., 0, :, 0]
        F[..., -1, 1:size_w+1, 0] = f[..., -1, :, 0]
        F[..., 0, 1:size_w+1, -1] = f[..., 0, :, -1]
        F[..., -1, 1:size_w+1, -1] = f[..., -1, :, -1]
        F[..., 1:size_h+1, 0, 0] = f[..., :, 0,  0]
        F[..., 1:size_h+1, -1, 0] = f[..., :, -1,  0]
        F[..., 1:size_h+1, 0, -1] = f[..., :, 0, -1]
        F[..., 1:size_h+1, -1, -1] = f[..., :, -1, -1]
        F[..., 0, 0, 0] = f[..., 0, 0, 0]
        F[..., -1, -1, -1] = f[..., -1, -1, -1]
    elif mode == 'periodic':
        # extend the function values periodically (1234 1)
        # F[:size_h, :size_w, :size_d] = f
        # F[-1, :size_w, :size_d] = f[0]
        # F[:size_h, -1, :size_d] = f[:, 0]
        # F[:size_h, :size_w, -1] = f[:, :, 0]
        # F[-1, -1, :size_d] = f[0, 0, :]
        # F[-1, :size_w, -1] = f[0, :, 0]
        # F[:size_h, -1, -1] = f[:, 0, 0]
        # F[-1, -1, -1] = f[0, 0, 0]
        # F[..., :size_h, :size_w, :size_d] = f
        # F[..., -1, :size_w, :size_d] = f[..., 0]
        # F[..., :size_h, -1, :size_d] = f[..., :, 0]
        # F[..., :size_h, :size_w, -1] = f[..., :, :, 0]
        # F[..., -1, -1, :size_d] = f[..., 0, 0, :]
        # F[..., -1, :size_w, -1] = f[..., 0, :, 0]
        # F[..., :size_h, -1, -1] = f[..., :, 0, 0]
        # F[..., -1, -1, -1] = f[..., 0, 0, 0]
        
        F[..., 1:size_h+1, 1:size_w+1, 1:size_d+1] = f
        F[..., 0, 1:size_w+1, 1:size_d+1] = f[..., -1, :, :]
        F[..., -1, 1:size_w+1, 1:size_d+1] = f[..., 0, :, :]
        F[..., 1:size_h+1, 0, 1:size_d+1] = f[..., :, -1, :]
        F[..., 1:size_h+1, -1, 1:size_d+1] = f[..., :, 0, :]
        F[..., 1:size_h+1, 1:size_w+1, 0] = f[..., :, :, -1]
        F[..., 1:size_h+1, 1:size_w+1, -1] = f[..., :, :, 0]
        F[..., 0, 0, 1:size_d+1] = f[..., -1, -1, :]
        F[..., -1, 0, 1:size_d+1] = f[..., 0, -1, :]
        F[..., 0, -1, 1:size_d+1] = f[..., -1, 0, :]
        F[..., -1, -1, 1:size_d+1] = f[..., 0, 0, :]
        F[..., 0, 1:size_w+1, 0] = f[..., -1, :, -1]
        F[..., -1, 1:size_w+1, 0] = f[..., 0, :, -1]
        F[..., 0, 1:size_w+1, -1] = f[..., -1, :, 0]
        F[..., -1, 1:size_w+1, -1] = f[..., 0, :, 0]
        F[..., 1:size_h+1, 0, 0] = f[..., :, -1, -1]
        F[..., 1:size_h+1, -1, 0] = f[..., :, 0, -1]
        F[..., 1:size_h+1, 0, -1] = f[..., :, -1, 0]
        F[..., 1:size_h+1, -1, -1] = f[..., :, 0, 0]
        F[..., 0, 0, 0] = f[..., -1, -1, -1]
        F[..., -1, -1, -1] = f[..., 0, 0, 0]
    elif mode == 'id':
        #print("mode == 'id'")
        # F[:size_h, :size_w, :size_d] = f
        # F[-1, :size_w, :size_d] = Ind_diffeo[0] % size_h
        # F[:size_h, -1, :size_d] = Ind_diffeo[1] % size_w
        # F[:size_h, :size_w, -1] = Ind_diffeo[2] % size_d
        # F[-1, -1, :size_d] = f[0, 0, :]
        # F[-1, :size_w, -1] = f[0, :, 0]
        # F[:size_h, -1, -1] = f[:, 0, 0]
        # F[-1, -1, -1] = Ind_diffeo[:][-1, -1, -1]
        # F[..., :size_h, :size_w, :size_d] = f
        #F[..., -1, :size_w, :size_d] = Ind_diffeo[0] % size_h
        #F[..., :size_h, -1, :size_d] = Ind_diffeo[1] % size_w
        #F[..., :size_h, :size_w, -1] = Ind_diffeo[2] % size_d
        # F[..., -1, :size_w, :size_d] = Ind_diffeo[:][-1, :, :]
        # F[..., :size_h, -1, :size_d] = Ind_diffeo[:][:, -1, :]
        # F[..., :size_h, :size_w, -1] = Ind_diffeo[:][:, :, -1]
        # F[..., -1, -1, :size_d] = Ind_diffeo[:][-1, -1, :]
        # F[..., -1, :size_w, -1] = Ind_diffeo[:][-1, :, -1]
        # F[..., :size_h, -1, -1] = Ind_diffeo[:][:, -1, -1]
        # F[..., -1, -1, -1] = Ind_diffeo[:][-1, -1, -1]
        F[..., 1:size_h+1, 1:size_w+1, 1:size_d+1] = f
        F[..., 0, 1:size_w+1, 1:size_d+1] = Ind_diffeo[:, 0, :size_w, :size_d]
        F[..., -1, 1:size_w+1, 1:size_d+1] = Ind_diffeo[:, -1, :size_w, :size_d]
        F[..., 1:size_h+1, 0, 1:size_d+1] = Ind_diffeo[:, :size_h, 0, :size_d]
        F[..., 1:size_h+1, -1, 1:size_d+1] = Ind_diffeo[:, :size_h, -1, :size_d]
        F[..., 1:size_h+1, 1:size_w+1, 0] = Ind_diffeo[:, :size_h, :size_w, 0]
        F[..., 1:size_h+1, 1:size_w+1, -1] = Ind_diffeo[:, :size_h, :size_w, -1]
        F[..., 0, 0, 1:size_d+1] = Ind_diffeo[:, 0, 0, :size_d]
        F[..., -1, 0, 1:size_d+1] = Ind_diffeo[:, -1, 0, :size_d]
        F[..., 0, -1, 1:size_d+1] = Ind_diffeo[:, 0, -1, :size_d]
        F[..., -1, -1, 1:size_d+1] = Ind_diffeo[:, -1, -1, :size_d]
        F[..., 0, 1:size_w+1, 0] = Ind_diffeo[:, 0, :size_w, 0]
        F[..., -1, 1:size_w+1, 0] = Ind_diffeo[:, -1, :size_w, 0]
        F[..., 0, 1:size_w+1, -1] = Ind_diffeo[:, 0, :size_w, -1]
        F[..., -1, 1:size_w+1, -1] = Ind_diffeo[:, -1, :size_w, -1]
        F[..., 1:size_h+1, 0, 0] = Ind_diffeo[:, :size_h, 0, 0]
        F[..., 1:size_h+1, -1, 0] = Ind_diffeo[:, :size_h, -1, 0]
        F[..., 1:size_h+1, 0, -1] = Ind_diffeo[:, :size_h, 0, -1]
        F[..., 1:size_h+1, -1, -1] = Ind_diffeo[:, :size_h, -1, -1]
        F[..., 0, 0, 0] = Ind_diffeo[:, 0, 0, 0]
        F[..., -1, -1, -1] = Ind_diffeo[:, -1, -1, -1]


    elif mode == 'zero':
    #    #print("mode == 'id'")
        #F[:size_h, :size_w, :size_d] = f
        #F[-1, :size_w, :size_d] = Ind_diffeo[0] % size_h
        #F[:size_h, -1, :size_d] = Ind_diffeo[1] % size_w
        #F[:size_h, :size_w, -1] = Ind_diffeo[2] % size_d
        #F[-1, -1, -1] = 0
        # F[..., :size_h, :size_w, :size_d] = f
        # F[..., -1, :size_w, :size_d] = 0
        # F[..., :size_h, -1, :size_d] = 0
        # F[..., :size_h, :size_w, -1] = 0
        # F[..., -1, -1, :size_d] = 0
        # F[..., -1, :size_w, -1] = 0
        # F[..., :size_h, -1, -1] = 0
        # F[..., -1, -1, -1] = 0
        F[..., 1:size_h+1, 1:size_w+1, 1:size_d+1] = f
        F[..., 0, 1:size_w+1, 1:size_d+1] = 0
        F[..., -1, 1:size_w+1, 1:size_d+1] = 0
        F[..., 1:size_h+1, 0, 1:size_d+1] = 0
        F[..., 1:size_h+1, -1, 1:size_d+1] = 0
        F[..., 1:size_h+1, 1:size_w+1, 0] = 0
        F[..., 1:size_h+1, 1:size_w+1, -1] = 0
        F[..., 0, 0, 1:size_d+1] = 0
        F[..., -1, 0, 1:size_d+1] = 0
        F[..., 0, -1, 1:size_d+1] = 0
        F[..., -1, -1, 1:size_d+1] = 0
        F[..., 0, 1:size_w+1, 0] = 0
        F[..., -1, 1:size_w+1, 0] = 0
        F[..., 0, 1:size_w+1, -1] = 0
        F[..., -1, 1:size_w+1, -1] = 0
        F[..., 1:size_h+1, 0, 0] = 0
        F[..., 1:size_h+1, -1, 0] = 0
        F[..., 1:size_h+1, 0, -1] = 0
        F[..., 1:size_h+1, -1, -1] = 0
        F[..., 0, 0, 0] = 0
        F[..., -1, -1, -1] = 0
        

    #print('Ind_diffeo[0]:\n', Ind_diffeo[0])    
    #print('Ind_diffeo[1]:\n', Ind_diffeo[1])    
    #print('Ind_diffeo[2]:\n', Ind_diffeo[2])
    #print('F:\n', F)    

    # # use the trilinear interpolation method
    #F000 = F[Ind_diffeo[0], Ind_diffeo[1], Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)
    #F010 = F[Ind_diffeo[0], Ind_diffeo[1] + 1, Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)
    #F100 = F[Ind_diffeo[0] + 1, Ind_diffeo[1], Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)
    #F110 = F[Ind_diffeo[0] + 1, Ind_diffeo[1] + 1, Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)
    #F001 = F[Ind_diffeo[0], Ind_diffeo[1], Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)
    #F011 = F[Ind_diffeo[0], Ind_diffeo[1] + 1, Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)
    #F101 = F[Ind_diffeo[0] + 1, Ind_diffeo[1], Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)
    #F111 = F[Ind_diffeo[0] + 1, Ind_diffeo[1] + 1, Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)
    if mode == 'id' or mode == 'zero':
      #F000 = torch.zeros(*f.shape[3:], size_h, size_w, size_d, device=f.device)
      F000 = torch.zeros(*f.shape[:-3], size_h, size_w, size_d, device=f.device)
      F010 = torch.zeros_like(F000)
      F100 = torch.zeros_like(F000)
      F110 = torch.zeros_like(F000)
      F001 = torch.zeros_like(F000)
      F011 = torch.zeros_like(F000)
      F101 = torch.zeros_like(F000)
      F111 = torch.zeros_like(F000)
      idx = torch.where((Ind_diffeo[0] >= 0) & (Ind_diffeo[0] < size_h) &
                        (Ind_diffeo[1] >= 0) & (Ind_diffeo[1] < size_w) &
                        (Ind_diffeo[2] >= 0) & (Ind_diffeo[2] < size_d))
      #F000[:,idx[0],idx[1],idx[2]] = F[Ind_diffeo[0][idx], Ind_diffeo[1][idx], Ind_diffeo[2][idx]].transpose(0,1)
      #F010[:,idx[0],idx[1],idx[2]] = F[Ind_diffeo[0][idx], Ind_diffeo[1][idx] + 1, Ind_diffeo[2][idx]].transpose(0,1)
      #F100[:,idx[0],idx[1],idx[2]] = F[Ind_diffeo[0][idx] + 1, Ind_diffeo[1][idx], Ind_diffeo[2][idx]].transpose(0,1)
      #F110[:,idx[0],idx[1],idx[2]] = F[Ind_diffeo[0][idx] + 1, Ind_diffeo[1][idx] + 1, Ind_diffeo[2][idx]].transpose(0,1)
      #F001[:,idx[0],idx[1],idx[2]] = F[Ind_diffeo[0][idx], Ind_diffeo[1][idx], Ind_diffeo[2][idx] + 1].transpose(0,1)
      #F011[:,idx[0],idx[1],idx[2]] = F[Ind_diffeo[0][idx], Ind_diffeo[1][idx] + 1, Ind_diffeo[2][idx] + 1].transpose(0,1)
      #F101[:,idx[0],idx[1],idx[2]] = F[Ind_diffeo[0][idx] + 1, Ind_diffeo[1][idx], Ind_diffeo[2][idx] + 1].transpose(0,1)
      #F111[:,idx[0],idx[1],idx[2]] = F[Ind_diffeo[0][idx] + 1, Ind_diffeo[1][idx] + 1, Ind_diffeo[2][idx] + 1].transpose(0,1)
      # F000[...,idx[0],idx[1],idx[2]] = F[..., Ind_diffeo[0][idx], Ind_diffeo[1][idx], Ind_diffeo[2][idx]]
      # F010[...,idx[0],idx[1],idx[2]] = F[..., Ind_diffeo[0][idx], Ind_diffeo[1][idx] + 1, Ind_diffeo[2][idx]]
      # F100[...,idx[0],idx[1],idx[2]] = F[..., Ind_diffeo[0][idx] + 1, Ind_diffeo[1][idx], Ind_diffeo[2][idx]]
      # F110[...,idx[0],idx[1],idx[2]] = F[..., Ind_diffeo[0][idx] + 1, Ind_diffeo[1][idx] + 1, Ind_diffeo[2][idx]]
      # F001[...,idx[0],idx[1],idx[2]] = F[..., Ind_diffeo[0][idx], Ind_diffeo[1][idx], Ind_diffeo[2][idx] + 1]
      # F011[...,idx[0],idx[1],idx[2]] = F[..., Ind_diffeo[0][idx], Ind_diffeo[1][idx] + 1, Ind_diffeo[2][idx] + 1]
      # F101[...,idx[0],idx[1],idx[2]] = F[..., Ind_diffeo[0][idx] + 1, Ind_diffeo[1][idx], Ind_diffeo[2][idx] + 1]
      # F111[...,idx[0],idx[1],idx[2]] = F[..., Ind_diffeo[0][idx] + 1, Ind_diffeo[1][idx] + 1, Ind_diffeo[2][idx] + 1]
      # Add 1 to every index since F is padded by 1 compared to diffeo and f
      # F000[...,idx[0],idx[1],idx[2]] = F[..., Ind_diffeo[0][idx]+1, Ind_diffeo[1][idx]+1, Ind_diffeo[2][idx]+1]
      # F010[...,idx[0],idx[1],idx[2]] = F[..., Ind_diffeo[0][idx]+1, Ind_diffeo[1][idx] + 2, Ind_diffeo[2][idx]+1]
      # F100[...,idx[0],idx[1],idx[2]] = F[..., Ind_diffeo[0][idx] + 2, Ind_diffeo[1][idx]+1, Ind_diffeo[2][idx]+1]
      # F110[...,idx[0],idx[1],idx[2]] = F[..., Ind_diffeo[0][idx] + 2, Ind_diffeo[1][idx] + 2, Ind_diffeo[2][idx]+1]
      # F001[...,idx[0],idx[1],idx[2]] = F[..., Ind_diffeo[0][idx]+1, Ind_diffeo[1][idx]+1, Ind_diffeo[2][idx] + 2]
      # F011[...,idx[0],idx[1],idx[2]] = F[..., Ind_diffeo[0][idx]+1, Ind_diffeo[1][idx] + 2, Ind_diffeo[2][idx] + 2]
      # F101[...,idx[0],idx[1],idx[2]] = F[..., Ind_diffeo[0][idx] + 2, Ind_diffeo[1][idx]+1, Ind_diffeo[2][idx] + 2]
      # F111[...,idx[0],idx[1],idx[2]] = F[..., Ind_diffeo[0][idx] + 2, Ind_diffeo[1][idx] + 2, Ind_diffeo[2][idx] + 2]
      F000[...,idx[0],idx[1],idx[2]] = F[..., Ind_diffeo[0,idx[0],idx[1],idx[2]]+1, Ind_diffeo[1,idx[0],idx[1],idx[2]]+1, Ind_diffeo[2,idx[0],idx[1],idx[2]]+1]
      idx = torch.where((Ind_diffeo[0] >= 0) & (Ind_diffeo[0] < size_h) &
                        (Ind_diffeo[1]+1 >= 0) & (Ind_diffeo[1]+1 < size_w) &
                        (Ind_diffeo[2] >= 0) & (Ind_diffeo[2] < size_d))
      F010[...,idx[0],idx[1],idx[2]] = F[..., Ind_diffeo[0,idx[0],idx[1],idx[2]]+1, Ind_diffeo[1,idx[0],idx[1],idx[2]]+2, Ind_diffeo[2,idx[0],idx[1],idx[2]]+1]
      idx = torch.where((Ind_diffeo[0]+1 >= 0) & (Ind_diffeo[0]+1 < size_h) &
                        (Ind_diffeo[1] >= 0) & (Ind_diffeo[1] < size_w) &
                        (Ind_diffeo[2] >= 0) & (Ind_diffeo[2] < size_d))
      
      F100[...,idx[0],idx[1],idx[2]] = F[..., Ind_diffeo[0,idx[0],idx[1],idx[2]]+2, Ind_diffeo[1,idx[0],idx[1],idx[2]]+1, Ind_diffeo[2,idx[0],idx[1],idx[2]]+1]
      idx = torch.where((Ind_diffeo[0]+1 >= 0) & (Ind_diffeo[0]+1 < size_h) &
                        (Ind_diffeo[1]+1 >= 0) & (Ind_diffeo[1]+1 < size_w) &
                        (Ind_diffeo[2] >= 0) & (Ind_diffeo[2] < size_d))
      
      F110[...,idx[0],idx[1],idx[2]] = F[..., Ind_diffeo[0,idx[0],idx[1],idx[2]]+2, Ind_diffeo[1,idx[0],idx[1],idx[2]]+2, Ind_diffeo[2,idx[0],idx[1],idx[2]]+1]
      idx = torch.where((Ind_diffeo[0] >= 0) & (Ind_diffeo[0] < size_h) &
                        (Ind_diffeo[1] >= 0) & (Ind_diffeo[1] < size_w) &
                        (Ind_diffeo[2]+1 >= 0) & (Ind_diffeo[2]+1 < size_d))
      
      F001[...,idx[0],idx[1],idx[2]] = F[..., Ind_diffeo[0,idx[0],idx[1],idx[2]]+1, Ind_diffeo[1,idx[0],idx[1],idx[2]]+1, Ind_diffeo[2,idx[0],idx[1],idx[2]]+2]
      idx = torch.where((Ind_diffeo[0] >= 0) & (Ind_diffeo[0] < size_h) &
                        (Ind_diffeo[1]+1 >= 0) & (Ind_diffeo[1]+1 < size_w) &
                        (Ind_diffeo[2]+1 >= 0) & (Ind_diffeo[2]+1 < size_d))
      
      F011[...,idx[0],idx[1],idx[2]] = F[..., Ind_diffeo[0,idx[0],idx[1],idx[2]]+1, Ind_diffeo[1,idx[0],idx[1],idx[2]]+2, Ind_diffeo[2,idx[0],idx[1],idx[2]]+2]
      idx = torch.where((Ind_diffeo[0]+1 >= 0) & (Ind_diffeo[0]+1 < size_h) &
                        (Ind_diffeo[1] >= 0) & (Ind_diffeo[1] < size_w) &
                        (Ind_diffeo[2]+1 >= 0) & (Ind_diffeo[2]+1 < size_d))
      
      F101[...,idx[0],idx[1],idx[2]] = F[..., Ind_diffeo[0,idx[0],idx[1],idx[2]]+2, Ind_diffeo[1,idx[0],idx[1],idx[2]]+1, Ind_diffeo[2,idx[0],idx[1],idx[2]]+2]
      idx = torch.where((Ind_diffeo[0]+1 >= 0) & (Ind_diffeo[0]+1 < size_h) &
                        (Ind_diffeo[1]+1 >= 0) & (Ind_diffeo[1]+1 < size_w) &
                        (Ind_diffeo[2]+1 >= 0) & (Ind_diffeo[2]+1 < size_d))
      
      F111[...,idx[0],idx[1],idx[2]] = F[..., Ind_diffeo[0,idx[0],idx[1],idx[2]]+2, Ind_diffeo[1,idx[0],idx[1],idx[2]]+2, Ind_diffeo[2,idx[0],idx[1],idx[2]]+2]

    else:
      F000 = F[..., Ind_diffeo[0], Ind_diffeo[1], Ind_diffeo[2]]
      F010 = F[..., Ind_diffeo[0], Ind_diffeo[1] + 1, Ind_diffeo[2]]
      F100 = F[..., Ind_diffeo[0] + 1, Ind_diffeo[1], Ind_diffeo[2]]
      F110 = F[..., Ind_diffeo[0] + 1, Ind_diffeo[1] + 1, Ind_diffeo[2]]
      F001 = F[..., Ind_diffeo[0], Ind_diffeo[1], Ind_diffeo[2] + 1]
      F011 = F[..., Ind_diffeo[0], Ind_diffeo[1] + 1, Ind_diffeo[2] + 1]
      F101 = F[..., Ind_diffeo[0] + 1, Ind_diffeo[1], Ind_diffeo[2] + 1]
      F111 = F[..., Ind_diffeo[0] + 1, Ind_diffeo[1] + 1, Ind_diffeo[2] + 1]
      
   
    if mode == 'id' or mode == 'zero':
#      print('mode:', mode)
      #indices = []
      #indices.append(torch.where((Ind_diffeo[0] < 0) | (Ind_diffeo[0] >= size_h)))
      #indices.append(torch.where((Ind_diffeo[1] < 0) | (Ind_diffeo[1] >= size_w)))
      #indices.append(torch.where((Ind_diffeo[2] < 0) | (Ind_diffeo[2] >= size_d)))
#      if mode == 'id':
        #for idx in indices:
          #F000[:,idx[0],idx[1],idx[2]] = torch.stack([Ind_diffeo[0][idx], Ind_diffeo[1][idx], Ind_diffeo[2][idx]]).double()
          #F010[:,idx[0],idx[1],idx[2]] = torch.stack([Ind_diffeo[0][idx], Ind_diffeo[1][idx] + 1, Ind_diffeo[2][idx]]).double()
          #F100[:,idx[0],idx[1],idx[2]] = torch.stack([Ind_diffeo[0][idx] + 1, Ind_diffeo[1][idx], Ind_diffeo[2][idx]]).double()
          #F110[:,idx[0],idx[1],idx[2]] = torch.stack([Ind_diffeo[0][idx] + 1, Ind_diffeo[1][idx] + 1, Ind_diffeo[2][idx]]).double()
          #F001[:,idx[0],idx[1],idx[2]] = torch.stack([Ind_diffeo[0][idx], Ind_diffeo[1][idx], Ind_diffeo[2][idx] + 1]).double()
          #F011[:,idx[0],idx[1],idx[2]] = torch.stack([Ind_diffeo[0][idx], Ind_diffeo[1][idx] + 1, Ind_diffeo[2][idx] + 1]).double()
          #F101[:,idx[0],idx[1],idx[2]] = torch.stack([Ind_diffeo[0][idx] + 1, Ind_diffeo[1][idx], Ind_diffeo[2][idx] + 1]).double()
          #F111[:,idx[0],idx[1],idx[2]] = torch.stack([Ind_diffeo[0][idx] + 1, Ind_diffeo[1][idx] + 1, Ind_diffeo[2][idx] + 1]).double()
        idx = torch.where((Ind_diffeo[0] < 0) | (Ind_diffeo[0] >= size_h) |
                          (Ind_diffeo[1] < 0) | (Ind_diffeo[1] >= size_w) |
                          (Ind_diffeo[2] < 0) | (Ind_diffeo[2] >= size_d))

        if mode == 'id':
          F000[...,idx[0],idx[1],idx[2]] = torch.stack([Ind_diffeo[0,idx[0],idx[1],idx[2]], Ind_diffeo[1,idx[0],idx[1],idx[2]], Ind_diffeo[2,idx[0],idx[1],idx[2]]]).double()
        elif mode == 'zero':
          F000[...,idx[0],idx[1],idx[2]] = 0
        
        idx = torch.where((Ind_diffeo[0] < 0) | (Ind_diffeo[0] >= size_h) |
                          (Ind_diffeo[1]+1 < 0) | (Ind_diffeo[1]+1 >= size_w) |
                          (Ind_diffeo[2] < 0) | (Ind_diffeo[2] >= size_d))
        if mode == 'id':
          F010[...,idx[0],idx[1],idx[2]] = torch.stack([Ind_diffeo[0,idx[0],idx[1],idx[2]], Ind_diffeo[1,idx[0],idx[1],idx[2]]+1, Ind_diffeo[2,idx[0],idx[1],idx[2]]]).double()
        elif mode == 'zero':
          F010[...,idx[0],idx[1],idx[2]] = 0

        idx = torch.where((Ind_diffeo[0]+1 < 0) | (Ind_diffeo[0]+1 >= size_h) |
                          (Ind_diffeo[1] < 0) | (Ind_diffeo[1] >= size_w) |
                          (Ind_diffeo[2] < 0) | (Ind_diffeo[2] >= size_d))
        if mode == 'id':
          F100[...,idx[0],idx[1],idx[2]] = torch.stack([Ind_diffeo[0,idx[0],idx[1],idx[2]]+1, Ind_diffeo[1,idx[0],idx[1],idx[2]], Ind_diffeo[2,idx[0],idx[1],idx[2]]]).double()
        elif mode == 'zero':
          F100[...,idx[0],idx[1],idx[2]] = 0

        idx = torch.where((Ind_diffeo[0]+1 < 0) | (Ind_diffeo[0]+1 >= size_h) |
                          (Ind_diffeo[1]+1 < 0) | (Ind_diffeo[1]+1 >= size_w) |
                          (Ind_diffeo[2] < 0) | (Ind_diffeo[2] >= size_d))
        if mode == 'id':
          F110[...,idx[0],idx[1],idx[2]] = torch.stack([Ind_diffeo[0,idx[0],idx[1],idx[2]]+1, Ind_diffeo[1,idx[0],idx[1],idx[2]]+1, Ind_diffeo[2,idx[0],idx[1],idx[2]]]).double()
        elif mode == 'zero':
          F110[...,idx[0],idx[1],idx[2]] = 0

        idx = torch.where((Ind_diffeo[0] < 0) | (Ind_diffeo[0] >= size_h) |
                          (Ind_diffeo[1] < 0) | (Ind_diffeo[1] >= size_w) |
                          (Ind_diffeo[2]+1 < 0) | (Ind_diffeo[2]+1 >= size_d))
        if mode == 'id':
          F001[...,idx[0],idx[1],idx[2]] = torch.stack([Ind_diffeo[0,idx[0],idx[1],idx[2]], Ind_diffeo[1,idx[0],idx[1],idx[2]], Ind_diffeo[2,idx[0],idx[1],idx[2]]+1]).double()
        elif mode == 'zero':
          F001[...,idx[0],idx[1],idx[2]] = 0

        idx = torch.where((Ind_diffeo[0] < 0) | (Ind_diffeo[0] >= size_h) |
                          (Ind_diffeo[1]+1 < 0) | (Ind_diffeo[1]+1 >= size_w) |
                          (Ind_diffeo[2]+1 < 0) | (Ind_diffeo[2]+1 >= size_d))
        if mode == 'id':
          F011[...,idx[0],idx[1],idx[2]] = torch.stack([Ind_diffeo[0,idx[0],idx[1],idx[2]], Ind_diffeo[1,idx[0],idx[1],idx[2]]+1, Ind_diffeo[2,idx[0],idx[1],idx[2]]+1]).double()
        elif mode == 'zero':
          F011[...,idx[0],idx[1],idx[2]] = 0

        idx = torch.where((Ind_diffeo[0]+1 < 0) | (Ind_diffeo[0]+1 >= size_h) |
                          (Ind_diffeo[1] < 0) | (Ind_diffeo[1] >= size_w) |
                          (Ind_diffeo[2]+1 < 0) | (Ind_diffeo[2]+1 >= size_d))
        if mode == 'id':
          F101[...,idx[0],idx[1],idx[2]] = torch.stack([Ind_diffeo[0,idx[0],idx[1],idx[2]]+1, Ind_diffeo[1,idx[0],idx[1],idx[2]], Ind_diffeo[2,idx[0],idx[1],idx[2]]+1]).double()
        elif mode == 'zero':
          F101[...,idx[0],idx[1],idx[2]] = 0

        idx = torch.where((Ind_diffeo[0]+1 < 0) | (Ind_diffeo[0]+1 >= size_h) |
                          (Ind_diffeo[1]+1 < 0) | (Ind_diffeo[1]+1 >= size_w) |
                          (Ind_diffeo[2]+1 < 0) | (Ind_diffeo[2]+1 >= size_d))
        if mode == 'id':
          F111[...,idx[0],idx[1],idx[2]] = torch.stack([Ind_diffeo[0,idx[0],idx[1],idx[2]]+1, Ind_diffeo[1,idx[0],idx[1],idx[2]]+1, Ind_diffeo[2,idx[0],idx[1],idx[2]]+1]).double()
        elif mode == 'zero':
          F111[...,idx[0],idx[1],idx[2]] = 0

        # for idx in indices:
        #   zeros = torch.zeros_like(Ind_diffeo[0][idx])
        #   zerostack = torch.stack([zeros, zeros, zeros]).double()
        #   #F000[:,idx[0],idx[1],idx[2]] = zerostack
        #   #F010[:,idx[0],idx[1],idx[2]] = zerostack
        #   #F100[:,idx[0],idx[1],idx[2]] = zerostack
        #   #F110[:,idx[0],idx[1],idx[2]] = zerostack
        #   #F001[:,idx[0],idx[1],idx[2]] = zerostack
        #   #F011[:,idx[0],idx[1],idx[2]] = zerostack
        #   #F101[:,idx[0],idx[1],idx[2]] = zerostack
        #   #F111[:,idx[0],idx[1],idx[2]] = zerostack
        #   F000[...,idx[0],idx[1],idx[2]] = zerostack
        #   F010[...,idx[0],idx[1],idx[2]] = zerostack
        #   F100[...,idx[0],idx[1],idx[2]] = zerostack
        #   F110[...,idx[0],idx[1],idx[2]] = zerostack
        #   F001[...,idx[0],idx[1],idx[2]] = zerostack
        #   F011[...,idx[0],idx[1],idx[2]] = zerostack
        #   F101[...,idx[0],idx[1],idx[2]] = zerostack
        #   F111[...,idx[0],idx[1],idx[2]] = zerostack

      # update this part for both id and zero modes (corresponds to partial_id and partial_zero in PyCA)
      # print('Updating', len(idx[0]), 'indices')
      # for i in range(len(idx[0])):
      #   #floorx=Ind_diffeo[0][idx[0][i],idx[1][i],idx[2][i]]
      #   #floory=Ind_diffeo[1][idx[0][i],idx[1][i],idx[2][i]]
      #   #floorz=Ind_diffeo[2][idx[0][i],idx[1][i],idx[2][i]]
      #   floorx=Ind_diffeo[0,idx[0][i],idx[1][i],idx[2][i]]
      #   floory=Ind_diffeo[1,idx[0][i],idx[1][i],idx[2][i]]
      #   floorz=Ind_diffeo[2,idx[0][i],idx[1][i],idx[2][i]]
      #   if ((floorx >= 0) and (floorx < size_h) and 
      #       (floory >= 0) and (floory < size_w) and 
      #       (floorz >= 0) and (floorz < size_d)):
      #     #F000[...,idx[0][i],idx[1][i],idx[2][i]] = F[..., floorx, floory, floorz]
      #     F000[...,idx[0][i],idx[1][i],idx[2][i]] = f[..., floorx, floory, floorz]
            
      #   if ((floorx >= 0) and (floorx < size_h) and 
      #       (floory+1 >= 0) and (floory+1 < size_w) and 
      #       (floorz >= 0) and (floorz < size_d)):
      #     #F010[...,idx[0][i],idx[1][i],idx[2][i]] = F[..., floorx, floory+1, floorz]
      #     F010[...,idx[0][i],idx[1][i],idx[2][i]] = f[..., floorx, floory+1, floorz]
          
      #   if ((floorx+1 >= 0) and (floorx+1 < size_h) and 
      #       (floory >= 0) and (floory < size_w) and 
      #       (floorz >= 0) and (floorz < size_d)):
      #     #F100[...,idx[0][i],idx[1][i],idx[2][i]] = F[..., floorx+1, floory, floorz]
      #     F100[...,idx[0][i],idx[1][i],idx[2][i]] = f[..., floorx+1, floory, floorz]

      #   if ((floorx+1 >= 0) and (floorx+1 < size_h) and 
      #       (floory+1 >= 0) and (floory+1 < size_w) and 
      #       (floorz >= 0) and (floorz < size_d)):
      #     #F110[...,idx[0][i],idx[1][i],idx[2][i]] = F[..., floorx+1, floory+1, floorz]
      #     F110[...,idx[0][i],idx[1][i],idx[2][i]] = f[..., floorx+1, floory+1, floorz]

      #   if ((floorx >= 0) and (floorx < size_h) and 
      #       (floory >= 0) and (floory < size_w) and 
      #       (floorz+1 >= 0) and (floorz+1 < size_d)):
      #     #F001[...,idx[0][i],idx[1][i],idx[2][i]] = F[..., floorx, floory, floorz+1]
      #     F001[...,idx[0][i],idx[1][i],idx[2][i]] = f[..., floorx, floory, floorz+1]
            
      #   if ((floorx >= 0) and (floorx < size_h) and 
      #       (floory+1 >= 0) and (floory+1 < size_w) and 
      #       (floorz+1 >= 0) and (floorz+1 < size_d)):
      #     #F011[...,idx[0][i],idx[1][i],idx[2][i]] = F[..., floorx, floory+1, floorz+1]
      #     F011[...,idx[0][i],idx[1][i],idx[2][i]] = f[..., floorx, floory+1, floorz+1]

      #   if ((floorx+1 >= 0) and (floorx+1 < size_h) and 
      #       (floory >= 0) and (floory < size_w) and 
      #       (floorz+1 >= 0) and (floorz+1 < size_d)):
      #     #F101[...,idx[0][i],idx[1][i],idx[2][i]] = F[..., floorx+1, floory, floorz+1]
      #     F101[...,idx[0][i],idx[1][i],idx[2][i]] = f[..., floorx+1, floory, floorz+1]

      #   if ((floorx+1 >= 0) and (floorx+1 < size_h) and 
      #       (floory+1 >= 0) and (floory+1 < size_w) and 
      #       (floorz+1 >= 0) and (floorz+1 < size_d)):
      #     #F111[...,idx[0][i],idx[1][i],idx[2][i]] = F[..., floorx+1, floory+1, floorz+1]
      #     F111[...,idx[0][i],idx[1][i],idx[2][i]] = f[..., floorx+1, floory+1, floorz+1]
            

#     original and 4.3
    C = diffeo[0] - Ind_diffeo[0]#.type(torch.DoubleTensor)
    D = diffeo[1] - Ind_diffeo[1]#.type(torch.DoubleTensor)
    E = diffeo[2] - Ind_diffeo[2]#.type(torch.DoubleTensor)

# # 4.7
# #     C = diffeo[0] - Ind_diffeo[2]#.type(torch.DoubleTensor)
# #     D = diffeo[1] - Ind_diffeo[1]#.type(torch.DoubleTensor)
# #     E = diffeo[2] - Ind_diffeo[0]#.type(torch.DoubleTensor)

    #interp_f = (1 - C) * (1 - D) * (1 - E) * F000 \
    #           + (1 - C) * D * (1 - E) * F010 \
    #           + C * (1 - D) * (1 - E) * F100 \
    #           + C * D * (1 - E) * F110 \
    #           + (1 - C) * (1 - D) * E * F001 \
    #           + (1 - C) * D * E * F011 \
    #           + C * (1 - D) * E * F101 \
    #           + C * D * E * F111
    interp_f = ((1 - C) * ((1 - D) * ((1 - E) * F000 + E * F001) +
                           D       * ((1 - E) * F010 + E * F011)) +
                C       * ((1 - D) * ((1 - E) * F100 + E * F101) +
                           D       * ((1 - E) * F110 + E * F111)))

#     del F000, F010, F100, F110, F001, F011, F101, F111, C, D, E
#     torch.cuda.empty_cache()
#    print('interp_f:\n', interp_f)
    return interp_f
# end compose_function_orig_3d

def compose_function_3d(f, diffeo, mode='periodic'):  # f: N x h x w x d  diffeo: 3 x h x w x d
    f = f.permute(f.dim() - 3, f.dim() - 2, f.dim() - 1, *range(f.dim() - 3))  # change the size of f to m x n x ...
    size_h, size_w, size_d = f.shape[:3]

#     original and 4.3
    Ind_diffeo = torch.stack((torch.floor(diffeo[0]).long() % size_h,
                              torch.floor(diffeo[1]).long() % size_w,
                              torch.floor(diffeo[2]).long() % size_d))#.to(device=torch.device('cuda'))
#     4.7
#     Ind_diffeo = torch.stack((torch.floor(diffeo[2]).long() % size_h,
#                               torch.floor(diffeo[1]).long() % size_w,
#                               torch.floor(diffeo[0]).long() % size_d))#.to(device=torch.device('cuda'))

    F = torch.zeros(size_h + 1, size_w + 1, size_d + 1, *f.shape[3:], device=f.device)#, dtype=torch.double

    if mode == 'border':
        F[:size_h, :size_w, :size_d] = f
        F[-1, :size_w, :size_d] = f[-1]
        F[:size_h, -1, :size_d] = f[:, -1]
        F[:size_h, :size_w, -1] = f[:, :, -1]
        F[-1, -1, -1] = f[-1, -1, -1]
    elif mode == 'periodic':
        # extend the function values periodically (1234 1)
        F[:size_h, :size_w, :size_d] = f
        F[-1, :size_w, :size_d] = f[0]
        F[:size_h, -1, :size_d] = f[:, 0]
        F[:size_h, :size_w, -1] = f[:, :, 0]
        F[-1, -1, -1] = f[0, 0, 0]

    # Break up following into pieces to reduce memory usage:
    # But do so in a way that allows back-propagation to work...
    # # use the bilinear interpolation method
#     F000 = F[Ind_diffeo[0], Ind_diffeo[1], Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)
#     F010 = F[Ind_diffeo[0], Ind_diffeo[1] + 1, Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)
#     F100 = F[Ind_diffeo[0] + 1, Ind_diffeo[1], Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)
#     F110 = F[Ind_diffeo[0] + 1, Ind_diffeo[1] + 1, Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)
#     F001 = F[Ind_diffeo[0], Ind_diffeo[1], Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)
#     F011 = F[Ind_diffeo[0], Ind_diffeo[1] + 1, Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)
#     F101 = F[Ind_diffeo[0] + 1, Ind_diffeo[1], Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)
#     F111 = F[Ind_diffeo[0] + 1, Ind_diffeo[1] + 1, Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)

# #     original and 4.3
#     C = diffeo[0] - Ind_diffeo[0]#.type(torch.DoubleTensor)
#     D = diffeo[1] - Ind_diffeo[1]#.type(torch.DoubleTensor)
#     E = diffeo[2] - Ind_diffeo[2]#.type(torch.DoubleTensor)
# # 4.7
# #     C = diffeo[0] - Ind_diffeo[2]#.type(torch.DoubleTensor)
# #     D = diffeo[1] - Ind_diffeo[1]#.type(torch.DoubleTensor)
# #     E = diffeo[2] - Ind_diffeo[0]#.type(torch.DoubleTensor)

#     interp_f = (1 - C) * (1 - D) * (1 - E) * F000 \
#                + (1 - C) * D * (1 - E) * F010 \
#                + C * (1 - D) * (1 - E) * F100 \
#                + C * D * (1 - E) * F110 \
#                + (1 - C) * (1 - D) * E * F001 \
#                + (1 - C) * D * E * F011 \
#                + C * (1 - D) * E * F101 \
#                + C * D * E * F111

    # Reduced memory usage version below.  Issues with back propagation...

#     original and 4.3
    C = diffeo[0] - Ind_diffeo[0]#.type(torch.DoubleTensor)
    D = diffeo[1] - Ind_diffeo[1]#.type(torch.DoubleTensor)
    E = diffeo[2] - Ind_diffeo[2]#.type(torch.DoubleTensor)

# 4.7
#     C = diffeo[0] - Ind_diffeo[2]#.type(torch.DoubleTensor)
#     D = diffeo[1] - Ind_diffeo[1]#.type(torch.DoubleTensor)
#     E = diffeo[2] - Ind_diffeo[0]#.type(torch.DoubleTensor)

   # use the bilinear interpolation method
    #F000 = F[Ind_diffeo[0], Ind_diffeo[1], Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)
    interp_f = (1 - C) * (1 - D) * (1 - E) * F[Ind_diffeo[0], Ind_diffeo[1], Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)

    #F010 = F[Ind_diffeo[0], Ind_diffeo[1] + 1, Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)
    interp_f += (1 - C) * D * (1 - E) * F[Ind_diffeo[0], Ind_diffeo[1] + 1, Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)
    
    #F100 = F[Ind_diffeo[0] + 1, Ind_diffeo[1], Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)
    interp_f += C * (1 - D) * (1 - E) * F[Ind_diffeo[0] + 1, Ind_diffeo[1], Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2) 
    
    #F110 = F[Ind_diffeo[0] + 1, Ind_diffeo[1] + 1, Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)
    interp_f += C * D * (1 - E) * F[Ind_diffeo[0] + 1, Ind_diffeo[1] + 1, Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)
    
    #F001 = F[Ind_diffeo[0], Ind_diffeo[1], Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)
    interp_f += (1 - C) * (1 - D) * E * F[Ind_diffeo[0], Ind_diffeo[1], Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)
    
    #F011 = F[Ind_diffeo[0], Ind_diffeo[1] + 1, Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)
    interp_f += (1 - C) * D * E * F[Ind_diffeo[0], Ind_diffeo[1] + 1, Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)
    
    #F101 = F[Ind_diffeo[0] + 1, Ind_diffeo[1], Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)
    interp_f += C * (1 - D) * E * F[Ind_diffeo[0] + 1, Ind_diffeo[1], Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)
    
    #F111 = F[Ind_diffeo[0] + 1, Ind_diffeo[1] + 1, Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)
    interp_f += C * D * E * F[Ind_diffeo[0] + 1, Ind_diffeo[1] + 1, Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)

#     del F000, F010, F100, F110, F001, F011, F101, F111, C, D, E
#     torch.cuda.empty_cache()
    return interp_f
# end compose_function_3d

# my interpolation function
def compose_function_in_place_3d(f, diffeo, mode='periodic'):  # f: N x h x w x d  diffeo: 3 x h x w x d
    f = f.permute(f.dim() - 3, f.dim() - 2, f.dim() - 1, *range(f.dim() - 3))  # change the size of f to m x n x ...
    size_h, size_w, size_d = f.shape[:3]
#     original and 4.3
    Ind_diffeo = torch.stack((torch.floor(diffeo[0]).long() % size_h,
                              torch.floor(diffeo[1]).long() % size_w,
                              torch.floor(diffeo[2]).long() % size_d))#.to(device=torch.device('cuda'))
#     4.7
#     Ind_diffeo = torch.stack((torch.floor(diffeo[2]).long() % size_h,
#                               torch.floor(diffeo[1]).long() % size_w,
#                               torch.floor(diffeo[0]).long() % size_d))#.to(device=torch.device('cuda'))

    F = torch.zeros(size_h + 1, size_w + 1, size_d + 1, *f.shape[3:], device=f.device)#, dtype=torch.double

    if mode == 'border':
        F[:size_h, :size_w, :size_d] = f
        F[-1, :size_w, :size_d] = f[-1]
        F[:size_h, -1, :size_d] = f[:, -1]
        F[:size_h, :size_w, -1] = f[:, :, -1]
        F[-1, -1, -1] = f[-1, -1, -1]
    elif mode == 'periodic':
        # extend the function values periodically (1234 1)
        F[:size_h, :size_w, :size_d] = f
        F[-1, :size_w, :size_d] = f[0]
        F[:size_h, -1, :size_d] = f[:, 0]
        F[:size_h, :size_w, -1] = f[:, :, 0]
        F[-1, -1, -1] = f[0, 0, 0]

    # Break up following into pieces to reduce memory usage:
    # But do so in a way that allows back-propagation to work...
    # # use the bilinear interpolation method
#     F000 = F[Ind_diffeo[0], Ind_diffeo[1], Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)
#     F010 = F[Ind_diffeo[0], Ind_diffeo[1] + 1, Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)
#     F100 = F[Ind_diffeo[0] + 1, Ind_diffeo[1], Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)
#     F110 = F[Ind_diffeo[0] + 1, Ind_diffeo[1] + 1, Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)
#     F001 = F[Ind_diffeo[0], Ind_diffeo[1], Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)
#     F011 = F[Ind_diffeo[0], Ind_diffeo[1] + 1, Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)
#     F101 = F[Ind_diffeo[0] + 1, Ind_diffeo[1], Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)
#     F111 = F[Ind_diffeo[0] + 1, Ind_diffeo[1] + 1, Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)

# #     original and 4.3
#     C = diffeo[0] - Ind_diffeo[0]#.type(torch.DoubleTensor)
#     D = diffeo[1] - Ind_diffeo[1]#.type(torch.DoubleTensor)
#     E = diffeo[2] - Ind_diffeo[2]#.type(torch.DoubleTensor)
# # 4.7
# #     C = diffeo[0] - Ind_diffeo[2]#.type(torch.DoubleTensor)
# #     D = diffeo[1] - Ind_diffeo[1]#.type(torch.DoubleTensor)
# #     E = diffeo[2] - Ind_diffeo[0]#.type(torch.DoubleTensor)

#     f = f.permute(*range(3, f.dim()), 0, 1, 2)  # change the size of f to N x m x n x ...
#     f[:] = (1 - C) * (1 - D) * (1 - E) * F000 \
#                + (1 - C) * D * (1 - E) * F010 \
#                + C * (1 - D) * (1 - E) * F100 \
#                + C * D * (1 - E) * F110 \
#                + (1 - C) * (1 - D) * E * F001 \
#                + (1 - C) * D * E * F011 \
#                + C * (1 - D) * E * F101 \
#                + C * D * E * F111

    # Reduced memory usage version below.  Issues with back propagation...

#     original and 4.3
    C = diffeo[0] - Ind_diffeo[0]#.type(torch.DoubleTensor)
    D = diffeo[1] - Ind_diffeo[1]#.type(torch.DoubleTensor)
    E = diffeo[2] - Ind_diffeo[2]#.type(torch.DoubleTensor)

# 4.7
#     C = diffeo[0] - Ind_diffeo[2]#.type(torch.DoubleTensor)
#     D = diffeo[1] - Ind_diffeo[1]#.type(torch.DoubleTensor)
#     E = diffeo[2] - Ind_diffeo[0]#.type(torch.DoubleTensor)

   # use the bilinear interpolation method
    #F000 = F[Ind_diffeo[0], Ind_diffeo[1], Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)
    f = f.permute(*range(3, f.dim()), 0, 1, 2)  # change the size of f to N x m x n x ...
    f[:] = (1 - C) * (1 - D) * (1 - E) * F[Ind_diffeo[0], Ind_diffeo[1], Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)

    #F010 = F[Ind_diffeo[0], Ind_diffeo[1] + 1, Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)
    f += (1 - C) * D * (1 - E) * F[Ind_diffeo[0], Ind_diffeo[1] + 1, Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)
    
    #F100 = F[Ind_diffeo[0] + 1, Ind_diffeo[1], Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)
    f += C * (1 - D) * (1 - E) * F[Ind_diffeo[0] + 1, Ind_diffeo[1], Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2) 
    
    #F110 = F[Ind_diffeo[0] + 1, Ind_diffeo[1] + 1, Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)
    f += C * D * (1 - E) * F[Ind_diffeo[0] + 1, Ind_diffeo[1] + 1, Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)
    
    #F001 = F[Ind_diffeo[0], Ind_diffeo[1], Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)
    f += (1 - C) * (1 - D) * E * F[Ind_diffeo[0], Ind_diffeo[1], Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)
    
    #F011 = F[Ind_diffeo[0], Ind_diffeo[1] + 1, Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)
    f += (1 - C) * D * E * F[Ind_diffeo[0], Ind_diffeo[1] + 1, Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)
    
    #F101 = F[Ind_diffeo[0] + 1, Ind_diffeo[1], Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)
    f += C * (1 - D) * E * F[Ind_diffeo[0] + 1, Ind_diffeo[1], Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)
    
    #F111 = F[Ind_diffeo[0] + 1, Ind_diffeo[1] + 1, Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)
    f += C * D * E * F[Ind_diffeo[0] + 1, Ind_diffeo[1] + 1, Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)

#     del F000, F010, F100, F110, F001, F011, F101, F111, C, D, E
#     torch.cuda.empty_cache()
    #return interp_f  


def get_div(v):
#     original
#     v_x = (torch.roll(v[0], shifts=(0, 0, -1), dims=(0, 1, 2))
#            - torch.roll(v[0], shifts=(0, 0, -1), dims=(0, 1, 2))) / 2
#     v_y = (torch.roll(v[1], shifts=(0, -1, 0), dims=(0, 1, 2))
#            - torch.roll(v[1], shifts=(0, -1, 0), dims=(0, 1, 2))) / 2
#     v_z = (torch.roll(v[2], shifts=(-1, 0, 0), dims=(0, 1, 2))
#            - torch.roll(v[2], shifts=(-1, 0, 0), dims=(0, 1, 2))) / 2
# 4.3 version
#     v_z = (torch.roll(v[0], shifts=(0, 0, -1), dims=(0, 1, 2))
#            - torch.roll(v[0], shifts=(0, 0, -1), dims=(0, 1, 2))) / 2
#     v_y = (torch.roll(v[1], shifts=(0, -1, 0), dims=(0, 1, 2))
#            - torch.roll(v[1], shifts=(0, -1, 0), dims=(0, 1, 2))) / 2
#     v_x = (torch.roll(v[2], shifts=(-1, 0, 0), dims=(0, 1, 2))
#            - torch.roll(v[2], shifts=(-1, 0, 0), dims=(0, 1, 2))) / 2
# 4.7 version
#     print('div')
    v_x = (torch.roll(v[0], shifts=(-1, 0, 0), dims=(0, 1, 2))
           - torch.roll(v[0], shifts=(-1, 0, 0), dims=(0, 1, 2))) / 2
    v_y = (torch.roll(v[1], shifts=(0, -1, 0), dims=(0, 1, 2))
           - torch.roll(v[1], shifts=(0, -1, 0), dims=(0, 1, 2))) / 2
    v_z = (torch.roll(v[2], shifts=(0, 0, -1), dims=(0, 1, 2))
           - torch.roll(v[2], shifts=(0, 0, -1), dims=(0, 1, 2))) / 2
    return v_x + v_y + v_z
# end get_div

# ad_transpose
# spatial domain: K(D(v^T) Lw + D(Lw) v + div(L*w x v))
def ad_transpose(v, w):
#     input: v/w.shape = [3, h, w, d]
#     output: shape = [3, h, w, d]
    '''
    this function applies the ad transpose operator to two vector fields, v and w, of size 3 x size_h x size_w x size_d
    '''
    print("WARNING!!! ad_transpose has not been tested yet!!!")

    Lw = applyL(w)

    Jac_v = get_jacobian_matrix_3d(v)
    Jac_Lw = get_jacobian_matrix_3d(Lw)

    div_Lw = get_div(Lw)

    rhs = torch.einsum("ji...,i...->j...",Jac_v, Lw) + \
          torch.einsum("i...,ij...->j...",v,Jac_Lw) + div_Lw * v

    ad_T = applyL(rhs, True)
    return ad_T
# end ad_transpose

# coAd
# dual of Ad operator
# coAd_(phiinv)(m0) = (D phiinv)^T m0 \circ phiinv |D phiinv|
def coAd(m0, phiinv):
#     input: m0/phiinv.shape = [3, h, w, d]
#     output: shape = [3, h, w, d]
    '''
    this function applies the dual of the Adjoint operator w.r.t. phiinv to initial momentum field, m0, of size 3 x size_h x size_w x size_d
    '''
    size_h, size_w, size_d = m0.shape[-3:]
    idty = get_idty_3d(size_h, size_w, size_d)

    J = get_jacobian_matrix_3d(phiinv)

    det = J[0, 0] * (J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1]) - \
          J[0, 1] * (J[1, 0] * J[2, 2] - J[1, 2] * J[2, 0]) + \
          J[0, 2] * (J[1, 0] * J[2, 1] - J[1, 1] * J[2, 0])

    mg = compose_function_orig_3d(m0, phiinv)

    coAd_x = det * (J[0,0] * mg[0] + J[1,0] * mg[1] + J[2,0] * mg[2])
    coAd_y = det * (J[0,1] * mg[0] + J[1,1] * mg[1] + J[2,1] * mg[2])
    coAd_z = det * (J[0,2] * mg[0] + J[1,2] * mg[1] + J[2,2] * mg[2])
    
    return torch.stack((coAd_x, coAd_y, coAd_z)).to(m0.device)
# end coAd


def update_inverse(phiinv, v, num_iters = 1):
  # Input phi_inv and epsilon * v(x,t)
  # Return first order approx d(phiinv_t) = phiinv_t(x-eps v(x,t), t)

  # commented out code is from vectormomentum UpdateInverse 
  idty = get_idty_3d(*v.shape[-3:]).to(v.device)
  prev_phiinv_t1 = torch.clone(phiinv).double()
  prev_v = v
  dt = 1. / num_iters
  #for ii in range(num_iters):
    #v_phiinv_t1 = compose_function_orig_3d(v, phiinv_t1, mode='zero')
    #id_m_v_phiinv_t1 = idty - v_phiinv_t1
    #phiinv_t1 = compose_function_orig_3d(phiinv, id_m_v_phiinv_t1, mode='id')
    #cur_phiinv_t1 = prev_phiinv_t1 - v
    #id_m_v_t = idty - dt * v
    #phiinv_id_m_v_t = compose_function_orig_3d(cur_phiinv_t1, id_m_v_t, mode='id')
    #cur_phiinv_t1 = phiinv_id_m_v_t
    #prev_phiinv_t1 = cur_phiinv_t1
    
  id_m_v_t = idty - v
  #phiinv_id_m_v_t = compose_function_orig_3d(phiinv, id_m_v_t, mode='id')
  phiinv_id_m_v_t = compose_function_orig_3d(phiinv, id_m_v_t, mode='id')
  cur_phiinv_t1 = phiinv_id_m_v_t
    
  return(cur_phiinv_t1)
# end update_inverse

# apply L operator
# m = Lv = applyL(v, False)
# v = Km = applyL(m, True)
def applyL(v, applyInverse=False):

  size_h, size_w, size_d = v.shape[-3:]
  idty = get_idty_3d(size_h, size_w, size_d)
  alpha = 1.0 # orig
  gamma = 0.0 # orig
  lpow = 1.0 # orig
  alpha = 0.01 # match CAvmGeodesicShooting
  gamma = 0.001 # match CAvmGeodesicShooting
  lpow= 1.0
  coeffs = gamma + alpha * (6. - 2. * (torch.cos(2. * np.pi * idty[0] / size_h) +
                                       torch.cos(2. * np.pi * idty[1] / size_w) +
                                       torch.cos(2. * np.pi * idty[2] / size_d)))
  if lpow == 1.0:
    lap = coeffs
  else:
    lap = torch.pow(coeffs, lpow)
  
  #print('lap:\n',lap)
  if applyInverse:
    #lap[0, 0, 0] = 1.
    lapinv = torch.nan_to_num(1. / lap, nan=1.0)
    #lap[0, 0, 0] = 0.
    #lapinv[0, 0, 0] = 1.
    fvx = torch.fft.fftn(v[0])
    fvy = torch.fft.fftn(v[1])
    fvz = torch.fft.fftn(v[2])
    lx = fvx * lapinv
    ly = fvy * lapinv
    lz = fvz * lapinv
    Kv_x = torch.real(torch.fft.ifftn(lx))
    Kv_y = torch.real(torch.fft.ifftn(ly))
    Kv_z = torch.real(torch.fft.ifftn(lz))
    return torch.stack((Kv_x, Kv_y, Kv_z)).to(v.device)
  else:
    fvx = torch.fft.fftn(v[0])
    fvy = torch.fft.fftn(v[1])
    fvz = torch.fft.fftn(v[2])
    lx = fvx * lap
    ly = fvy * lap
    lz = fvz * lap
    Lv_x = torch.real(torch.fft.ifftn(lx))
    Lv_y = torch.real(torch.fft.ifftn(ly))
    Lv_z = torch.real(torch.fft.ifftn(lz))
    return torch.stack((Lv_x, Lv_y, Lv_z)).to(v.device)
# end applyL
    
# shoot a geodesic from initial velocity, v0, to get diffeomorphism, phi and phiinv, at time t=1
def shoot_geodesic_velocity_formulation(v0, num_time_steps, do_RK4=True):
  print("WARNING!!! shoot_geodesic_velocity_formulation has not been tested yet!!!")
  dt = 1.0 / num_time_steps

  # This implementation derived from FLASH and uses FLASH convention for
  # phi and phiinv which is opposite of VectorMomentum convention
  idty = get_idty_3d(*v0.shape[-3:]).to(v0.device)
  prev_phi = get_idty_3d(*v0.shape[-3:]).to(v0.device)
  prev_phiinv = get_idty_3d(*v0.shape[-3:]).to(v0.device)
  prev_v = v0

  for ii in range(num_time_steps):
    # generate phi^{-1} and phi under left invariant metric

    # Integrate v0
    if do_RK4:
      # v1 = v0 - (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
      # k1
      v_k1 = ad_transpose(prev_v, prev_v)

      # k2
      d_k1 = prev_v - (dt / 2.0) * v_k1
      v_k2 = ad_transpose(d_k1, d_k1)
     
      # k3
      d_k2 = prev_v - (dt / 2.0) * v_k2
      v_k3 = ad_transpose(d_k2, d_k2)
     
      # k4 
      d_k3 = prev_v - dt * v_k3
      v_k4 = ad_transpose(d_k4, d_k4)

      # update v1 = v0 - (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
      cur_v = prev_v - (dt / 6.0) * (v_k1 + 2*v_k2 + 2*v_k3 + v_k4)
      prev_v = cur_v
    else:
      # v0 = v0 - dt * adTranspose(v0, v0)
      advv = ad_transpose(prev_v, prev_v)
      cur_v = prev_v - dt * advv
      prev_v = cur_v

    # Update phi, phiinv (flash and vectormomentum have phi and phiinv swapped)
    # TODO Determine which fits the convention of metric matching.
    eye = torch.eye(3, device=v0.device)
    ones = torch.ones(*v0.shape[-3:], device=v0.device)
    d_phiinv = get_jacobian_matrix_3d(prev_phiinv - idty) + torch.einsum("ij,mno->ijmno", eye, ones)
    dphiinv_vt = torch.einsum("ji...,i...->j...",d_phiinv, cur_v)
    print(d_phiinv.shape, cur_v.shape, dphiinv_vt.shape, prev_phiinv.shape)
    cur_phiinv = prev_phiinv - dt * dphiinv_vt
    cur_phi = update_inverse(prev_phi, cur_v)
    prev_phiinv = cur_phiinv
    prev_phi = cur_phi
  
  return(cur_phiinv, cur_phi, cur_v)
# end shoot_geodesic_velocity_formulation

# shoot a geodesic from initial velocity, v0, to get diffeomorphism, phi and phiinv, at time t=1
def shoot_geodesic_momenta_formulation(v0, num_time_steps, do_RK4=True):
  dt = 1.0 / num_time_steps

  # This implementation derived from VectorMomentum and uses VectorMomentum
  # convention for phi and phiinv which is opposite of FLASH convention
  idty = get_idty_3d(*v0.shape[-3:]).to(v0.device)
  prev_phi = get_idty_3d(*v0.shape[-3:]).to(v0.device)
  prev_phiinv = get_idty_3d(*v0.shape[-3:]).to(v0.device)
  m0 = applyL(v0)
  #prev_v = torch.clone(v0)
  prev_v = applyL(m0, True)
  

  for ii in range(num_time_steps):
    # generate phi^{-1} and phi under left invariant metric
    print(f'shoot_geodesic_momenta_formulation, iteration {ii}, current GPU mem usage (MB):', torch.cuda.memory_allocated() * 1e-6)

    # Integrate v0
    if do_RK4:
      # v1 = v0 + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
      # k1
      phi_k1 = prev_phi
      phiinv_k1 = prev_phiinv
      v_k1 = prev_v
      k1 = dt * compose_function_orig_3d(v_k1, phi_k1)
      
      # k2
      #phi_k2 = phi_k1 + 0.5 * k1
      phi_k2 = prev_phi + 0.5 * k1
      #phiinv_k2 = update_inverse(phiinv_k1, k1) 
      phiinv_k2 = update_inverse(prev_phiinv, k1) 
      v_k2 = applyL(coAd(m0, phiinv_k2), True)
      k2 = dt * compose_function_orig_3d(v_k2, phi_k2)
     
      # k3
      #phi_k3 = phi_k2 + 0.5 * k2
      phi_k3 = prev_phi + 0.5 * k2
      #phiinv_k3 = update_inverse(phiinv_k2, k2) 
      phiinv_k3 = update_inverse(prev_phiinv, k2) 
      v_k3 = applyL(coAd(m0, phiinv_k3), True)
      k3 = dt * compose_function_orig_3d(v_k3, phi_k3)
     
      # k4 
      #phi_k4 = phi_k3 + k3
      phi_k4 = prev_phi + k3
      #phiinv_k4 = update_inverse(phiinv_k3, k3) 
      phiinv_k4 = update_inverse(prev_phiinv, k3) 
      v_k4 = applyL(coAd(m0, phiinv_k4), True)
      k4 = dt * compose_function_orig_3d(v_k4, phi_k4)

      update = (1 / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
      cur_phi = prev_phi + update
      cur_phiinv = update_inverse(prev_phiinv, update)
      cur_v = applyL(coAd(m0, cur_phiinv), True)
      prev_v = cur_v
      prev_phi = cur_phi
      prev_phiinv = cur_phiinv
    else:
      # vt+1 = vt + dt * K(coAd(m0, phiinv_t)) \circ phi_t
      update = dt * compose_function_orig_3d(prev_v, prev_phi)
      cur_phi = prev_phi + update
      cur_phiinv = update_inverse(prev_phiinv, update)
      cur_v = applyL(coAd(m0, cur_phiinv), True)
      prev_v = cur_v
      prev_phi = cur_phi
      prev_phiinv = cur_phiinv

  return(cur_phi, cur_phiinv, cur_v)
# end shoot_geodesic_momenta_formulation


if __name__ == "__main__":
  phi = sio.loadmat('103818toTemp_phi.mat')
  phi_inv = sio.loadmat('103818toTemp_phi_inv.mat')
  phi = phi['diffeo']
  phi_inv = phi_inv['diffeo']
  new_points_x, new_points_y = coord_register(points_x, points_y, phi)
