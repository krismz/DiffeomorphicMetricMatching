from lazy_imports import np
from lazy_imports import torch
#from scipy.linalg import expm, logm
import warnings
#from lazy_imports import plt
from IPython.core.debugger import set_trace
from util.diffeo import phi_pullback_3d
'''
SplitEbinMetric.py stays the same from Atlas2D to Atlas3D
'''

def trKsquare(B, A):
    #G = torch.cholesky(B)
    G = torch.linalg.cholesky(B)
    inv_G = torch.inverse(G)
    W = torch.einsum("...ij,...jk,...lk->...il", inv_G, A, inv_G)
    #lamda = torch.symeig(W, eigenvectors=True)[0]
    lamda = torch.linalg.eigh(W, UPLO='U')[0]
    result = torch.sum(torch.log(lamda) ** 2, (-1))
    return result

def Squared_distance_Ebin_field(g0, g1, a, mask):
#     inputs: g0.shape, g1.shape = [h, w, d, 3, 3]
#     output: scalar
#     3.3.4 https://www.cs.utah.edu/~haocheng/notes/NoteonMatching.pdf
    inv_g0_g1 = torch.einsum("...ik,...kj->...ij", torch.inverse(g0), g1)
    trK0square = trKsquare(g0, g1) - torch.log(torch.det(inv_g0_g1)) ** 2 *a  # torch.log(torch.det(inv_g0_g1) + 1e-25)
    theta = torch.min((trK0square / a + 1e-40).sqrt() / 4., torch.tensor(np.pi, dtype=torch.double, device=g0.device))  
    alpha, beta = torch.det(g0).pow(1. / 4.), torch.det(g1).pow(1. / 4.)
    E = 16 * a * (alpha ** 2 - 2 * alpha * beta * torch.cos(theta) + beta ** 2)
    return (E*mask)


def Squared_distance_Ebin(g0, g1, a, mask):
#     inputs: g0.shape, g1.shape = [h, w, d, 3, 3]
#     output: scalar
#     3.3.4 https://www.cs.utah.edu/~haocheng/notes/NoteonMatching.pdf
    inv_g0_g1 = torch.einsum("...ik,...kj->...ij", torch.inverse(g0), g1)
    trK0square = trKsquare(g0, g1) - torch.log(torch.det(inv_g0_g1)) ** 2 *a  # torch.log(torch.det(inv_g0_g1) + 1e-25)
    theta = torch.min((trK0square / a + 1e-40).sqrt() / 4., torch.tensor(np.pi, dtype=torch.double, device=g0.device))  
    alpha, beta = torch.det(g0).pow(1. / 4.), torch.det(g1).pow(1. / 4.)
    E = 16 * a * (alpha ** 2 - 2 * alpha * beta * torch.cos(theta) + beta ** 2)
    return torch.einsum("hwd,hwd->", E, mask)

def energy_ebin(phi, g0, g1, f0, f1, sigma, dim, mask): 
#     input: phi.shape = [3, h, w, d]; g0/g1/f0/f1.shape = [h, w, d, 3, 3]; sigma/dim = scalar; mask.shape = [1, h, w, d]
#     output: scalar
# the phi here is identity
    phi_star_g1 = phi_pullback_3d(phi, g1)
    phi_star_f1 = phi_pullback_3d(phi, f1)# the compose operation in this step uses a couple of thousands MB of memory
    E1 = sigma * Squared_distance_Ebin(f0, phi_star_f1, 1./dim, mask)
    E2 = Squared_distance_Ebin(g0, phi_star_g1, 1./dim, mask)
    return E1 + E2

def energy_L2(phi, g0, g1, f0, f1, sigma, mask): 
#     input: phi.shape = [3, h, w, d]; g0/g1/f0/f1.shape = [h, w, d, 3, 3]; sigma = scalar; mask.shape = [1, h, w, d]
#     output: scalar
    phi_star_g1 = phi_pullback_3d(phi, g1)
    phi_star_f1 = phi_pullback_3d(phi, f1)
    E1 = sigma * torch.einsum("ijk...,lijk->", (f0 - phi_star_f1) ** 2, mask.unsqueeze(0))
    E2 = torch.einsum("ijk...,lijk->", (g0 - phi_star_g1) ** 2, mask.unsqueeze(0))
    # E = E1 + E2
#     del phi_star_g1, phi_star_f1
#     torch.cuda.empty_cache()
    return E1 + E2

def logm_invB_A(B, A):
#     inputs: A/B.shape = [h, w, d, 3, 3]
#     output: shape = [h, w, d, 3, 3]
    G = torch.linalg.cholesky(B)
    #torch.cholesky(A) # KMC commented out, seems like unneccessary computation
    inv_G = torch.inverse(G)
    W = torch.einsum("...ij,...jk,...lk->...il", inv_G, A, inv_G)
    #lamda, Q = torch.symeig(W, eigenvectors=True)
    lamda, Q = torch.linalg.eigh(W, UPLO='U')
    log_lamda = torch.zeros((*lamda.shape, lamda.shape[-1]),dtype=torch.double,device=B.device)
    # for i in range(lamda.shape[-1]):
    #     log_lamda[:, i, i] = torch.log(lamda[:, i])
    log_lamda = torch.diag_embed(torch.log(lamda))
    V = torch.einsum('...ji,...jk->...ik', inv_G, Q)
    inv_V = torch.inverse(V)
    return torch.einsum('...ij,...jk,...kl->...il', V, log_lamda, inv_V)


# 2 without for loops using Kyle's method
def inv_RieExp(g0, g1, a):  # g0,g1: two tensors of size (s,t,...,3,3), where g0\neq 0
    '''this function is to calculate the inverse Riemannian exponential of g1 in the image of the maximal domain of the Riemannian exponential at g0
    '''
    n = g1.size(-1)
    #     matrix multiplication
    inv_g0_g1 = torch.einsum("...ik,...kj->...ij", torch.inverse(g0), g1)  # (s,t,...,3,3)

    def get_u_g0direction(g0, inv_g0_g1):  # (-1,3,3) first reshape g0,g1,inv_g..
        #         permute
        inv_g0_g1 = torch.einsum("...ij->ij...", inv_g0_g1)  # (3,3,-1)
        s = inv_g0_g1[0, 0]  # (-1)
        u = 4 / n * (s ** (n / 4) - 1) * torch.einsum("...ij->ij...", g0)  # (-1)@(3,3,-1) -> (3,3,-1)
        return u.permute(2, 0, 1)  # (-1,3,3)

    def get_u_ng0direction(g0, g1, inv_g0_g1, a):  # (-1,3,3) first reshape g0,g1,inv_g..
        K = logm_invB_A(g0, g1)
        KTrless = K - torch.einsum("...ii,kl->...kl", K, torch.eye(n, dtype=torch.double, device=g0.device)) / n  # (-1,3,3)
        #         AA^T
        theta = (1 / a * torch.einsum("...ik,...ki->...", KTrless, KTrless)).sqrt() / 4  # (-1)
        gamma = torch.det(g1).pow(1 / 4) / (torch.det(g0).pow(1 / 4))  # (-1)

        A = 4 / n * (gamma * torch.cos(theta) - 1)  # (-1)
        B = 1 / theta * gamma * torch.sin(theta)
        u = A * torch.einsum("...ij->ij...", g0) + B * torch.einsum("...ik,...kj->ij...", g0, KTrless)  # (-1)@(3,3,-1) -> (3,3,-1)
        return u.permute(2, 0, 1)  # (-1,3,3)

    inv_g0_g1_trless = inv_g0_g1 - torch.einsum("...ii,kl->...kl", inv_g0_g1, torch.eye(n, dtype=torch.double, device=g0.device)) / n  # (s,t,...,2,2)
    norm0 = torch.einsum("...ij,...ij->...", inv_g0_g1_trless, inv_g0_g1_trless).reshape(-1)  # (-1)

    # find the indices for which the entries are 0s and non0s
    Ind0 = (norm0 <= 1e-12).nonzero().reshape(-1)  # using squeeze results in [1,1]->[]
    Indn0 = (norm0 > 1e-12).nonzero().reshape(-1)

    u = torch.zeros(g0.reshape(-1, n, n).size(), dtype=torch.double, device=g0.device)  # (-1,3,3)
    if len(Indn0) == 0:
        u = get_u_g0direction(g0.reshape(-1, n, n), inv_g0_g1.reshape(-1, n, n))
    elif len(Ind0) == 0:
        u = get_u_ng0direction(g0.reshape(-1, n, n), g1.reshape(-1, n, n), inv_g0_g1.reshape(-1, n, n), a)
    else:
        u[Ind0] = get_u_g0direction(g0.reshape(-1, n, n)[Ind0], inv_g0_g1.reshape(-1, n, n)[Ind0])
        u[Indn0] = get_u_ng0direction(g0.reshape(-1, n, n)[Indn0], g1.reshape(-1, n, n)[Indn0], inv_g0_g1.reshape(-1, n, n)[Indn0], a)

    return u.reshape(g1.size())

# KMC added function
def Direct(g0, g1, t, a):
    print('in Direct')
    n = g0.size(-1)
    K = logm_invB_A(g0, g1)
    KTrless = K - torch.einsum("...ii,kl->...kl", K, torch.eye(n, dtype=torch.double, device=g0.device)) / n  # (-1,3,3)
    theta = (1 / a * torch.einsum("...ik,...ki->...", KTrless, KTrless)).sqrt() / 4  # (-1)
    gamma = torch.det(g1).pow(1 / 4) / (torch.det(g0).pow(1 / 4))  # (-1)

    q = 1 + t * (gamma * torch.cos(theta) - 1)
    r = t * gamma * torch.sin(theta)

    def get_g1_g0direction(g0, q):  # first reshape g0 (-1,3,3) and trU (-1)
        u = (q).pow(4 / n) * torch.einsum("...ij->ij...", g0)  # (3,3,-1)
        print(u.shape)
        return u.permute(2, 0, 1)  # (-1,3,3)

    #     not in g0 direction SplitEbinMetric.pdf Theorem 1 :K_0\not=0
    def get_g1_ng0direction(g0, q, r, KTrless, theta, t):  # first reshape g0,UTrless (-1,3,3) and trU (-1)
        if len((q < -4).nonzero().reshape(-1)) != 0:
            warnings.warn('The tangent vector u is out of the maximal domain of the Riemannian exponential.', DeprecationWarning)

        ArctanKtrless = torch.atan2(r*t, q) * torch.einsum("...ij->ij...", KTrless) / theta  # use (2,2,-1) for computation
        ExpArctanKtrless = torch.matrix_exp(ArctanKtrless.permute(2, 0, 1)).permute(1, 2, 0)
        u = (q ** 2 + r ** 2).pow(2 / n) * torch.einsum("...ik,kj...->ij...", g0, ExpArctanKtrless)  # (2,2,-1)
        print(u.shape)
        return u.permute(2, 0, 1)  # (-1,2,2)

    #     pointwise multiplication Tr(U^TU)
    norm0 = torch.einsum("...ij,...ij->...", KTrless, KTrless).reshape(-1)  # (-1)

    # find the indices for which the entries are 0s and non0s
    #     k_0=0 or \not=0
    Ind0 = (norm0 <= 1e-12).nonzero().reshape(-1)
    Indn0 = (norm0 > 1e-12).nonzero().reshape(-1)

    u = torch.zeros(g0.reshape(-1, n, n).size(), dtype=torch.double, device=g0.device)  # (-1,2,2)
    if len(Indn0) == 0:
        u = get_g1_g0direction(g0.reshape(-1, n, n), q.reshape(-1))
    elif len(Ind0) == 0:
        u = get_g1_ng0direction(g0.reshape(-1, n, n), q.reshape(-1), r.reshape(-1),
                                KTrless.reshape(-1, n, n), theta.reshape(-1), t)
    else:
        u[Ind0] = get_g1_g0direction(g0.reshape(-1, n, n)[Ind0], q.reshape(-1)[Ind0])
        u[Indn0] = get_g1_ng0direction(g0.reshape(-1, n, n)[Indn0], q.reshape(-1)[Indn0], r.reshape(-1)[Indn0],
                                       KTrless.reshape(-1, n, n)[Indn0], theta.reshape(-1)[Indn0], t)

    return u.reshape(g0.size())

# KMC added function
def Direct_extended(g0, g1, t, a):
    size = g0.size()
    g0, g1 = g0.reshape(-1, *size[-2:]), g1.reshape(-1, *size[-2:])  # (-1,3,3)
    detg0 = torch.det(g0)

    Ind_g0_is0 = (detg0 == 0).nonzero().reshape(-1)
    Ind_g0_isnot0 = (detg0 != 0).nonzero().reshape(-1)
    Ind_g1_is0 = (detg1 == 0).nonzero().reshape(-1)
    Ind_g1_isnot0 = (detg1 != 0).nonzero().reshape(-1)

    if len(Ind_g0_isnot0) == 0:  # g0x are 0s for all x
        #u = g1 * g0.size(-1) / 4
        u = torch.zeros(g0.size(), dtype=torch.double, device=g0.device)
        u[Ind_g1_isnot0] = t.pow(4/n) * g1[Ind_g1_isnot0]
        print(u.shape)
    elif len(Ind_g0_is0) == 0:  # g0x are PD for all x
        # KMC edit
        u = torch.zeros(g0.size(), dtype=torch.double, device=g0.device)
        u[Ind_g1_isnot0] = Direct(g0[Ind_g1_isnot0], g1[Ind_g1_isnot0], t, a)
        u[Ind_g1_is0] = (1-t).pow(4/n) * g0[Ind_g1_is0]
    else:
        u = torch.zeros(g0.size(), dtype=torch.double, device=g0.device)
        u[Ind_g0_isnot0] = Direct(g0[Ind_g0_isnot0], g1[Ind_g0_isnot0], t, a)
        u[Ind_g1_isnot0] = Direct(g0[Ind_g1_isnot0], g1[Ind_g1_isnot0], t, a)
        u[Ind_g0_is0] = t.pow(4/n) * g1[Ind_g0_is0]
        u[Ind_g1_is0] = (1-t).pow(4/n) * g0[Ind_g1_is0]
    return u.reshape(size)

# KMC added function
def Combined(g0, g1, t, a):
    n = g0.size(-1)
    detg0 = torch.det(g0).pow(1/4)
    detg1 = torch.det(g1).pow(1/4)
    # split at detg0/(detg0+detg1)
    Ind_g0_g1_isnot0 = ((detg0 + detg1) > 1e-12).nonzero().reshape(-1)
    Ind_split_lt_t = (detg0[Ind_g0_g1_isnot0] / (detg0[Ind_g0_g1_isnot0] + detg1[Ind_g0_g1_isnot0]) < t).nonzero().reshape(-1)
    Ind_split_gt_t = (detg0[Ind_g0_g1_isnot0] / (detg0[Ind_g0_g1_isnot0] + detg1[Ind_g0_g1_isnot0]) > t).nonzero().reshape(-1)

    u = torch.zeros(g0.size(), dtype=torch.double, device=g0.device)
    u[Ind_split_lt_t] = (1 - t - t * (detg1[Ind_split_lt_t] / detg0[Ind_split_lt_t])).pow(4/n) * g0[Ind_split_lt_t]
    u[Ind_split_gt_t] = ((t-1) * (detg0[Ind_split_gt_t] / detg1[Ind_split_gt_t]) + t).pow(4/n) * g1[Ind_split_gt_t]
    return(u)
    
# KMC added function
def First_Half(g0, g1, t, a):
    n = g0.size(-1)
    detg0 = torch.det(g0).pow(1/4)
    detg1 = torch.det(g1).pow(1/4)
    Ind_g0_isnot0 = (detg0 != 0).nonzero().reshape(-1)

    u = torch.zeros(g0.size(), dtype=torch.double, device=g0.device)
    u[Ind_g0_isnot0] = (1 - t - t * (detg1[Ind_g0_is0] / detg0[Ind_g0_is0])).pow(4/n) * g0[Ind_g0_is0]
    return u

# KMC added function
def Second_Half(g0, g1, t, a):
    n = g1.size(-1)
    detg0 = torch.det(g0).pow(1/4)
    detg1 = torch.det(g1).pow(1/4)
    Ind_g1_isnot0 = (detg1 != 0).nonzero().reshape(-1)

    u = torch.zeros(g1.size(), dtype=torch.double, device=g0.device)
    u[Ind_g1_isnot0] = ((t-1) * (detg0[Ind_g1_is0] / detg1[Ind_g1_is0]) + t).pow(4/n) * g1[Ind_g1_is0]
    return u

# KMC added t param
def Rie_Exp(g0, u, a, t=1):  # here g0 is of size (s,t,...,3,3) and u is of size (s,t,...,3,3), where g0\neq 0
    '''this function is to calculate the Riemannian exponential of u in the the maximal domain of the Riemannian exponential at g0
    '''
    n = g0.size(-1)
    U = torch.einsum("...ik,...kj->...ij", torch.inverse(g0), u)  # (s,t,...,3,3)
    trU = torch.einsum("...ii->...", U)  # (s,t,...)
    UTrless = U - torch.einsum("...,ij->...ij", trU, torch.eye(n, n, dtype=torch.double, device=g0.device)) / n  # (s,t,...,3,3)

    def get_g1_g0direction(g0, trU):  # first reshape g0 (-1,3,3) and trU (-1)
        g1 = (trU / 4 + 1).pow(4 / n) * torch.einsum("...ij->ij...", g0)  # (3,3,-1)
        #print(g1.shape)
        return g1.permute(2, 0, 1)  # (-1,3,3)

    #     not in g0 direction SplitEbinMetric.pdf Theorem 1 :K_0\not=0
    def get_g1_ng0direction(g0, trU, UTrless, a):  # first reshape g0,UTrless (-1,3,3) and trU (-1)
        if len((trU < -4).nonzero().reshape(-1)) != 0:
            warnings.warn('The tangent vector u is out of the maximal domain of the Riemannian exponential.', DeprecationWarning)

        q = trU / 4 + 1  # (-1)
        r = (1 / a * torch.einsum("...ik,...ki->...", UTrless, UTrless)).sqrt() / 4  # (-1)

        # KMC need this edit?
        ArctanUtrless = torch.atan2(r, q) * torch.einsum("...ij->ij...", UTrless) / r  # use (2,2,-1) for computation
        ExpArctanUtrless = torch.matrix_exp(ArctanUtrless.permute(2, 0, 1)).permute(1, 2, 0)
        #ArctanUtrless = torch.atan2(r*t, q) * torch.einsum("...ij->ij...", UTrless) / r  # use (2,2,-1) for computation
        #ExpArctanUtrless = torch.matrix_exp(ArctanUtrless.permute(2, 0, 1)*t).permute(1, 2, 0)

        g1 = (q ** 2 + r ** 2).pow(2 / n) * torch.einsum("...ik,kj...->ij...", g0, ExpArctanUtrless)  # (2,2,-1)
        #print(g1.shape)
        return g1.permute(2, 0, 1)  # (-1,2,2)

    #     pointwise multiplication Tr(U^TU)
    norm0 = torch.einsum("...ij,...ij->...", UTrless, UTrless).reshape(-1)  # (-1)

    # find the indices for which the entries are 0s and non0s
    #     k_0=0 or \not=0
    Ind0 = (norm0 <= 1e-12).nonzero().reshape(-1)
    Indn0 = (norm0 > 1e-12).nonzero().reshape(-1)

    g1 = torch.zeros(g0.reshape(-1, n, n).size(), dtype=torch.double, device=g0.device)  # (-1,2,2)
    if len(Indn0) == 0:
        g1 = get_g1_g0direction(g0.reshape(-1, n, n), trU.reshape(-1))
    elif len(Ind0) == 0:
        g1 = get_g1_ng0direction(g0.reshape(-1, n, n), trU.reshape(-1), UTrless.reshape(-1, n, n), a)
    else:
        g1[Ind0] = get_g1_g0direction(g0.reshape(-1, n, n)[Ind0], trU.reshape(-1)[Ind0])
        g1[Indn0] = get_g1_ng0direction(g0.reshape(-1, n, n)[Indn0], trU.reshape(-1)[Indn0], UTrless.reshape(-1, n, n)[Indn0], a)

    return g1.reshape(g0.size())


''' 
The following Riemannian exponential and inverse Riemannian exponential are extended to the case g0=0 
'''
# KMC added option t
def Rie_Exp_extended(g0, u, a, t=1):  # here g0 is of size (s,t,...,3,3) and u is of size (s,t,...,3,3)
    size = g0.size()
    g0, u = g0.reshape(-1, *size[-2:]), u.reshape(-1, *size[-2:])  # (-1,3,3)
    detg0 = torch.det(g0)

    Ind_g0_is0 = (detg0 == 0).nonzero().reshape(-1)
    Ind_g0_isnot0 = (detg0 != 0).nonzero().reshape(-1)

    if len(Ind_g0_isnot0) == 0:  # g0x are 0s for all x
        g1 = u * g0.size(-1) / 4
        #print(g1.shape)
    elif len(Ind_g0_is0) == 0:  # g0x are PD for all x
        # KMC edit
        #g1 = Rie_Exp(g0, u, a)
        g1 = Rie_Exp(g0, u, a, t)
    else:
        g1 = torch.zeros(g0.size(), dtype=torch.double, device=g0.device)
        g1[Ind_g0_is0] = u[Ind_g0_is0] * g0.size(-1) / 4
        # KMC edit
        #g1[Ind_g0_isnot0] = Rie_Exp(g0[Ind_g0_isnot0], u[Ind_g0_isnot0], a)
        g1[Ind_g0_isnot0] = Rie_Exp(g0[Ind_g0_isnot0], u[Ind_g0_isnot0], a, t)
    return g1.reshape(size)


def inv_RieExp_extended(g0, g1, a):  # g0, g1: (s,t,...,3,3)
    size = g0.size()
    g0, g1 = g0.reshape(-1, *size[-2:]), g1.reshape(-1, *size[-2:])  # (-1,3,3)
    detg0 = torch.det(g0)

    Ind_g0_is0 = (detg0 == 0).nonzero().reshape(-1)
    Ind_g0_isnot0 = (detg0 != 0).nonzero().reshape(-1)

    if len(Ind_g0_isnot0) == 0:  # g0x are 0s for all x
        u = g1 * 4 / g0.size(-1)
    elif len(Ind_g0_is0) == 0:  # g0x are PD for all x
        u = inv_RieExp(g0, g1, a)
    else:
        u = torch.zeros(g0.size(), dtype=torch.double, device=g0.device)
        u[Ind_g0_is0] = g1[Ind_g0_is0] * 4 / g0.size(-1)
        u[Ind_g0_isnot0] = inv_RieExp(g0[Ind_g0_isnot0], g1[Ind_g0_isnot0], a)
    return u.reshape(size)


def get_geo(g0, g1, a, Tpts):  # (s,t,...,,3,3)
    '''
    use odd number Tpts of time points since the geodesic may go
    though the zero matrix which will give the middle point of the geodesic
    '''
    size = g0.size()

    g0, g1 = g0.reshape(-1, *size[-2:]), g1.reshape(-1, *size[-2:])  # (-1,3,3)

    Time = torch.arange(Tpts, out=torch.DoubleTensor()) / (Tpts - 1)  # (Tpts)

    U = logm_invB_A(g0, g1)
    UTrless = U - torch.einsum("...ii,kl->...kl", U, torch.eye(g1.size(-1), dtype=torch.double, device=g0.device)) / g1.size(-1)  # (...,3,3)
    theta = ((1 / a * torch.einsum("...ik,...ki->...", UTrless, UTrless)).sqrt() / 4 - np.pi)
    Ind_inRange = (theta < 0).nonzero().reshape(-1)
    Ind_notInRange = (theta >= 0).nonzero().reshape(-1)

    def geo_in_range(g0, g1, a, Tpts):
        n = g0.size(-1)
        detg0 = torch.det(g0)
        detg1 = torch.det(g1)
        Ind_g0_is0 = (detg0 < 1e-12).nonzero().reshape(-1)
        Ind_g1_is0 = (detg1 < 1e-12).nonzero().reshape(-1)
        u = inv_RieExp_extended(g0, g1, a)  # (-1,3,3)
        geo = torch.zeros(Tpts, *g0.size(), dtype=torch.double, device=g0.device)  # (Tpts,-1,3,3)
        geo[0], geo[-1] = g0, g1
        for i in range(1, Tpts - 1):
            # KMC edit
            geo[i] = Rie_Exp_extended(g0, u * Time[i], a)
            #geo[i] = Direct(g0, g1, Time[i], a)
            #geo[i][Ind_g0_is0] = (Time[i]).pow(4/n) * g1[Ind_g0_is0]
            #geo[i][Ind_g1_is0] = (1-Time[i]).pow(4/n) * g0[Ind_g1_is0]
        return geo  # (Tpts,-1,2,2)

    def geo_not_in_range(g0, g1, a, Tpts):  # (-1,3,3)
        m0 = torch.zeros(g0.size(), dtype=torch.double, device=g0.device)
        u0 = inv_RieExp_extended(g0, m0, a)
        u1 = inv_RieExp_extended(g1, m0, a)

        geo = torch.zeros(Tpts, *g0.size(), dtype=torch.double, device=g0.device)  # (Tpts,-1,3,3)
        geo[0], geo[-1] = g0, g1
        
        for i in range(1, int((Tpts - 1) / 2)):
            # KMC edit
            #geo[i] = Rie_Exp_extended(g0, u0 * Time[i], a)
            geo[i] = Rie_Exp_extended(g0, u0 * Time[i], a, 1)
            #geo[i] = First_Half(g0, g1, Time[i], a)
            #geo[i] = Combined(g0, g1, Time[i], a)
        # KMC edit
        #for j in range(-int((Tpts - 1) / 2), -1):
        #    # KMC edit
        #    #geo[j] = Rie_Exp_extended(g1, u1 * (1 - Time[j]), a)
        #    geo[j] = Rie_Exp_extended(g1, u1 * (1 - Time[j]), a, (1-Time[j]))
        for j in range(int((Tpts - 1) / 2), Tpts):
            # KMC edit
            geo[j] = Rie_Exp_extended(g1, u1 * (1 - Time[j]), a)
            #geo[j] = Rie_Exp_extended(m0, u1 * Time[j], a,1)
            #geo[j] = Second_Half(g0, g1, Time[j], a)
            #geo[j] = Combined(g0, g1, Time[j], a)
        return geo  # (Tpts,-1,2,2)

    # If g1 = 0, len(Ind_notInRange) and len(Ind_inRange) are both zero. In this case we say that g1 is in the range
    if len(Ind_notInRange) == 0:  # all in the range
        #print('all in range')
        geo = geo_in_range(g0, g1, a, Tpts)
    elif len(Ind_inRange) == 0:  # all not in range
        #print('all not in range')
        geo = geo_not_in_range(g0, g1, a, Tpts)
    else:
        #print('some in range, some not')
        geo = torch.zeros(Tpts, *g0.size(), dtype=torch.double, device=g0.device)  # (Tpts,-1,3,3)
        geo[:, Ind_inRange] = geo_in_range(g0[Ind_inRange], g1[Ind_inRange], a, Tpts)
        geo[:, Ind_notInRange] = geo_not_in_range(g0[Ind_notInRange], g1[Ind_notInRange], a, Tpts)
    return geo.reshape(Tpts, *size)


def ptPick_notInRange(g0, g1, i):  # (-1,3,3)
    alpha = torch.det(g1).pow(1 / 4) / torch.det(g0).pow(1 / 4)  # (-1)
    Ind_close_to_g0 = (alpha <= i).nonzero().reshape(-1)
    Ind_close_to_g1 = (alpha > i).nonzero().reshape(-1)

    def get_gm_inLine_0g0(alpha, g0, i):
        kn_over4 = -(1 + alpha) / (i + 1)  # (-1)
        gm = (1 + kn_over4) ** (4 / g0.size(-1)) * torch.einsum("...ij->ij...", g0)  # (3,3,-1)
        return gm.permute(2, 0, 1)  # (-1,3,3)

    def get_gm_inLine_0g1(alpha, g1, i):
        kn_over4 = -i * (1 + 1 / alpha) / (i + 1)  # (-1)
        gm = (1 + kn_over4) ** (4 / g1.size(-1)) * torch.einsum("...ij->ij...", g1)  # (3,3,-1)
        return gm.permute(2, 0, 1)

    if len(Ind_close_to_g1) == 0:  # all are close to g0
        gm = get_gm_inLine_0g0(alpha, g0, i)
    elif len(Ind_close_to_g0) == 0:
        gm = get_gm_inLine_0g1(alpha, g1, i)
    else:
        gm = torch.zeros(g0.size(), dtype=torch.double, device=g0.device)
        gm[Ind_close_to_g0] = get_gm_inLine_0g0(alpha[Ind_close_to_g0], g0[Ind_close_to_g0], i)
        gm[Ind_close_to_g1] = get_gm_inLine_0g1(alpha[Ind_close_to_g1], g1[Ind_close_to_g1], i)
    return gm



def get_karcher_mean(G, a):
    size = G.size()
    G = G.reshape(size[0], -1, *size[-2:])  # (T,-1,3,3)
    gm = G[0]
    for i in range(1, G.size(0)):
        U = logm_invB_A(gm, G[i])
        UTrless = U - torch.einsum("...ii,kl->...kl", U, torch.eye(size[-1], dtype=torch.double, device=U.device)) / size[
            -1]  # (...,2,2)
        theta = ((torch.einsum("...ik,...ki->...", UTrless, UTrless) / a).sqrt() / 4 - np.pi)
        Ind_inRange = (theta < 0).nonzero().reshape(-1)  ## G[i] is in the range of the exponential map at gm
        Ind_notInRange = (theta >= 0).nonzero().reshape(-1)  ## G[i] is not in the range

        # when g1 = 0, len(Ind_notInRange) and len(Ind_inRange) are both zero. So check len(Ind_notInRange) first
        if len(Ind_notInRange) == 0:  # all in the range
            gm = Rie_Exp_extended(gm, inv_RieExp_extended(gm, G[i], a) / (i + 1), a)
        elif len(Ind_inRange) == 0:  # all not in range
            gm = ptPick_notInRange(gm, G[i], i)
        else:
            gm[Ind_inRange] = Rie_Exp_extended(gm[Ind_inRange],
                                               inv_RieExp_extended(gm[Ind_inRange], G[i, Ind_inRange], a) / (i + 1),
                                               a)  # stop here
            gm[Ind_notInRange] = ptPick_notInRange(gm[Ind_notInRange], G[i, Ind_notInRange], i)

    return gm.reshape(*size[1:])