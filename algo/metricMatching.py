from lazy_imports import torch
from util.diffeo import get_idty_3d, phi_pullback_3d, compose_function_3d
from util.SplitEbinMetric3D import energy_ebin
from util.vectorfield import laplace_inverse


def metric_matching(gi, gm, height, width, depth, mask, iter_num, epsilon, sigma, dim):
    phi_inv = get_idty_3d(height, width, depth).to(gi.device)
    phi = get_idty_3d(height, width, depth).to(gi.device)
    idty = get_idty_3d(height, width, depth).to(gi.device)
    idty.requires_grad_()
    f0 = torch.eye(int(dim), device=gi.device).repeat(height, width, depth, 1, 1)
    f1 = torch.eye(int(dim), device=gi.device).repeat(height, width, depth, 1, 1)
    
    for j in range(iter_num):
        phi_actsg0 = phi_pullback_3d(phi_inv, gi)
        phi_actsf0 = phi_pullback_3d(phi_inv, f0)
        E = energy_ebin(idty, phi_actsg0, gm, phi_actsf0, f1, sigma, dim, mask) 
        print(j, E.item())
        E.backward()
        v = - laplace_inverse(idty.grad)
        with torch.no_grad():
            psi =  idty + epsilon*v  
            psi[0][psi[0] > height - 1] = height - 1
            psi[1][psi[1] > width - 1] = width - 1
            psi[2][psi[2] > depth - 1] = depth - 1
            psi[psi < 0] = 0
            psi_inv =  idty - epsilon*v
            psi_inv[0][psi_inv[0] > height - 1] = height - 1
            psi_inv[1][psi_inv[1] > width - 1] = width - 1
            psi_inv[2][psi_inv[2] > depth - 1] = depth - 1
            psi_inv[psi_inv < 0] = 0
            phi = compose_function_3d(psi, phi)
            phi_inv = compose_function_3d(phi_inv, psi_inv)
            idty.grad.data.zero_()
            
    gi = phi_pullback_3d(phi_inv, gi)
    return gi, phi, phi_inv
