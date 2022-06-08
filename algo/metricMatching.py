from lazy_imports import torch
from util.diffeo import get_idty_3d, phi_pullback_3d, compose_function_3d, compose_function_in_place_3d
from util.SplitEbinMetric3D import energy_ebin, energy_ebin_no_phi
from util.vectorfield import laplace_inverse

def is_symmetric(g, name=''):
    # assumes g.shape = [h,w,d,3,3]
    symm = True
    idx = torch.where(torch.abs(g[:,:,:,0,1] - g[:,:,:,1,0]) > 1e-7)
    if len(idx[0]) > 0:
      print(len(idx[0]), 'elements of', name, 'are not symmetric')
      print(g[idx])
      symm = False
    idx = torch.where(torch.abs(g[:,:,:,0,2] - g[:,:,:,2,0]) > 1e-7)
    if len(idx[0]) > 0:
      print(len(idx[0]), 'elements of', name, 'are not symmetric')
      print(g[idx])
      symm = False
    idx = torch.where(torch.abs(g[:,:,:,2,1] - g[:,:,:,1,2]) > 1e-7)
    if len(idx[0]) > 0:
      print(len(idx[0]), 'elements of', name, 'are not symmetric')
      print(g[idx])
      symm = False
    return(symm)

def make_symmetric(g):
    # assumes g.shape = [h,w,d,3,3]
    g[:,:,:,0,1] = g[:,:,:,1,0]
    g[:,:,:,0,2] = g[:,:,:,2,0]
    g[:,:,:,2,1] = g[:,:,:,1,2]

def metric_matching(gi, gm, height, width, depth, mask, iter_num, epsilon, sigma, dim, use_idty=True):
    if not is_symmetric(gi, 'gi'):
      print('making gi symmetric')
      make_symmetric(gi)
    if not is_symmetric(gm, 'gm'):
      print('making gm symmetric')
      make_symmetric(gm)
    #phi_inv = get_idty_3d(height, width, depth).to(gi.device)
    phi_inv = get_idty_3d(height, width, depth).cpu()
    phi = get_idty_3d(height, width, depth).cpu()
    #idty = get_idty_3d(height, width, depth).to(gi.device)
    idty = get_idty_3d(height, width, depth).cpu()
    if use_idty:
      idty.requires_grad_()
    else:
      idty.requires_grad_()
    #f0 = torch.eye(int(dim), dtype=torch.FloatTensor).repeat(height, width, depth, 1, 1)
    phi_actsf0 = torch.eye(int(dim), device=gi.device).repeat(height, width, depth, 1, 1)
    f1 = torch.eye(int(dim), device=gi.device).repeat(height, width, depth, 1, 1)
    phi_actsg0 = gi
    
    for j in range(iter_num):
        #phi_actsg0 = phi_pullback_3d(phi_inv, gi)
        #phi_actsf0 = phi_pullback_3d(phi_inv, f0)
        # need one linked to idty for gradient
        phi_actsg0 = phi_pullback_3d(idty.cpu(), phi_actsg0.cpu()).to(gi.device)
        #phi_star_gm = phi_pullback_3d(idty, gm)    

        if use_idty:
          E = energy_ebin(idty, phi_actsg0, gm, phi_actsf0, f1, sigma, dim, mask)
        else:
          #E = energy_ebin_no_phi(phi_actsg0, phi_star_gm, phi_actsf0, f1, sigma, dim, mask) 
          E = energy_ebin_no_phi(phi_actsg0, gm, phi_actsf0, f1, sigma, dim, mask) 
        print(j, E.item())
        E.backward()
        if use_idty:
          v = - laplace_inverse(idty.grad).float().to(gi.device)
        else:
          v = - laplace_inverse(idty.grad).float().cpu()#.to(gi.device)

        with torch.no_grad():
            psi =  idty.cpu() + epsilon*v.cpu()
            psi[0][psi[0] > height - 1] = height - 1
            psi[1][psi[1] > width - 1] = width - 1
            psi[2][psi[2] > depth - 1] = depth - 1
            psi[psi < 0] = 0
            psi_inv =  idty - epsilon*v.cpu()
            psi_inv[0][psi_inv[0] > height - 1] = height - 1
            psi_inv[1][psi_inv[1] > width - 1] = width - 1
            psi_inv[2][psi_inv[2] > depth - 1] = depth - 1
            psi_inv[psi_inv < 0] = 0
            phi[:] = compose_function_3d(psi, phi)
            phi_inv[:] = compose_function_3d(phi_inv, psi_inv.cpu())
            phi_actsg0 = phi_pullback_3d(psi_inv, phi_actsg0.cpu()).to(gi.device)
            phi_actsf0 = phi_pullback_3d(psi_inv, phi_actsf0.cpu()).to(gi.device)
            if not is_symmetric(phi_actsg0, 'phi_actsg0'):
              print('making phi_actsg0 symmetric')
              make_symmetric(phi_actsg0)
            if not is_symmetric(phi_actsf0, 'phi_actsf0'):
              print('making phi_actsf0 symmetric')
              make_symmetric(phi_actsf0)

            #compose_function_in_place_3d(phi_actsg0, psi_inv)
            #compose_function_in_place_3d(phi_actsf0, psi_inv)
            #compose_function_in_place_3d(psi, phi)
            #compose_function_in_place_3d(phi_inv, psi_inv)
            if use_idty:
              idty.grad.data.zero_()
            else:
              #phi_inv.requires_grad_()
              idty.grad.data.zero_()
            del v
            
    gi = phi_pullback_3d(phi_inv, gi.cpu()).to(gi.device)
    return gi, phi, phi_inv
