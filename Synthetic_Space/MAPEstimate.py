import numpy as np
import torch
import math

jitter = 0.0001

def map_estimate_torch(X, Y, Xt, Yt, bias, alpha, noise, Sigma, space_kernel, time_kernel, kernel, alpha_mean,
                       alpha_variance, bias_variance, bias_kernel, N_sensors, N_time, theta_not):
    Sigma_hat = Sigma + noise**2*torch.eye(N_sensors*N_time)
    bias_sigma = torch.tensor(np.kron(np.eye(N_sensors), bias_kernel(N_time, N_time))).float()

    chunk1 = -(1/2) * (torch.logdet(alpha**2 * Sigma_hat)
                       + (Y - bias).T @ torch.cholesky_inverse(alpha**2 * Sigma_hat) @ (Y - bias)
                       + N_sensors * math.log(2 * math.pi))

    prob_a = -(1/2) * (((alpha - alpha_mean) ** 2 / (alpha_variance)) + math.log((alpha_variance) * 2 * math.pi))
    prob_b = -(1/2) * (torch.logdet(bias_sigma)
                       + bias.T @ torch.cholesky_inverse(bias_sigma) @ bias
                       + len(bias) * math.log(2 * math.pi))
    chunk2 = prob_a + prob_b

    def v(x):
        k = kernel([x.detach().numpy()], X.detach().numpy()).T
        k = torch.from_numpy(k)
        output = theta_not - k.T @ torch.cholesky_inverse(Sigma_hat) @ k
        if output < 0:
            print('Error')
        return output

    def mu(x):
        k = kernel([x.detach().numpy()], X.detach().numpy()).T
        k = torch.from_numpy(k)
        return (k.T @ torch.linalg.inv(Sigma_hat) @ (Y - bias))/alpha

    chunk3 = 0
    for i in range(0, len(Xt)):
        holder = mu(Xt[i])
        var = v(Xt[i])
        chunk3 += -(1/2) * (torch.log(var) + ((Yt[i] - holder)**2)/var + math.log(2 * math.pi))

    return chunk1 + chunk2 + chunk3


def map_estimate_numpy(X, Y, Xt, Yt, bias, alpha, noise, Sigma, space_kernel, time_kernel, kernel, alpha_mean,
                       alpha_variance, bias_sigma, N_sensors, N_time, theta_not):
    Sigma_hat = Sigma + noise**2*np.eye(N_sensors*N_time)
    bias_sigma = bias_sigma + jitter*np.eye(len(bias_sigma))
    chunk1 = -(1/2) * (np.log(np.linalg.det(alpha**2 * Sigma_hat))
                       + (Y - bias).T @ np.linalg.cholesky(alpha**2 * Sigma_hat) @ (Y - bias)
                       + N_sensors * math.log(2 * math.pi))

    prob_a = -(1/2) * (((alpha - alpha_mean) ** 2 / (alpha_variance)) + math.log((alpha_variance) * 2 * math.pi))
    prob_b = -(1/2) * (np.log(np.linalg.det(bias_sigma))
                       + bias.T @ np.linalg.cholesky(bias_sigma) @ bias
                       + len(bias) * math.log(2 * math.pi))
    chunk2 = prob_a + prob_b

    def v(x):
        k = kernel([x], X).T
        output = theta_not - k.T @ np.linalg.inv(Sigma_hat) @ k
        if output < 0:
            print('Error')
        return output

    def mu(x):
        k = kernel([x], X).T
        return (k.T @ np.linalg.inv(Sigma_hat) @ (Y - bias))/alpha

    chunk3 = 0
    for i in range(0, len(Xt)):
        holder = mu(Xt[i])
        var = v(Xt[i])
        chunk3 += -(1/2) * (np.log(var) + ((Yt[i] - holder)**2)/var + math.log(2 * math.pi))

    return chunk1 + chunk2 + chunk3
