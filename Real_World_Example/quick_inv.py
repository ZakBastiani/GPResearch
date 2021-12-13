import torch


def quick_inv(space_k, time_k, alpha, noise_sd):
    U_s, S_s, V_s = torch.linalg.svd(space_k, full_matrices=True)
    U_t, S_t, V_t = torch.linalg.svd(time_k, full_matrices=True)
    S_st = torch.kron(S_t, S_s)
    U_st = torch.kron(U_t, U_s)
    V_st = torch.kron(V_t, V_s)
    S_st = 1. / (alpha**2 * S_st + noise_sd**2 * torch.ones(len(S_st)))
    S_st = torch.diag(S_st)
    sigma_inv = U_st @ S_st @ V_st
    return sigma_inv
