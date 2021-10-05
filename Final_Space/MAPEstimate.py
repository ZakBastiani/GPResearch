import numpy as np
import torch
import math

jitter = 0.000001


def map_estimate_torch(X, Y, Xt, Yt, bias, alpha, noise, Sigma, space_kernel, time_kernel, kernel, alpha_mean,
                       alpha_sd, bias_sigma, N_sensors, N_time, theta_not):
    torch.set_default_dtype(torch.float64)
    Sigma_hat = alpha**2 * Sigma + (noise**2) * torch.eye(N_sensors*N_time)
    bias_sigma = bias_sigma

    chunk1 = -(1/2) * (torch.logdet(Sigma_hat)  # currently giving -inf
                       + (Y - bias).T @ torch.inverse(Sigma_hat) @ (Y - bias)
                       + N_sensors * math.log(2 * math.pi))

    prob_a = -(1/2) * (((alpha - alpha_mean) ** 2 / (alpha_sd ** 2)) + math.log((alpha_sd ** 2) * 2 * math.pi))
    prob_b = -(1/2) * (torch.logdet(bias_sigma)
                       + bias.T @ torch.inverse(bias_sigma) @ bias
                       + len(bias) * math.log(2 * math.pi))
    chunk2 = prob_a + prob_b

    def v(x):
        k = kernel(x.reshape(1, -1), X)
        output = theta_not - (alpha * k.T) @ torch.inverse(Sigma_hat) @ (alpha * k)
        if output < 0:
            print('Error')
        return output

    def mu(x):
        k = kernel(x.reshape(1, -1), X)
        return (alpha * k.T) @ torch.inverse(Sigma_hat) @ (Y - bias)

    chunk3 = 0
    for i in range(0, len(Xt)):
        holder = mu(Xt[i])
        var = v(Xt[i])
        chunk3 += -(1/2) * (torch.log(var) + ((Yt[i] - holder)**2)/var + math.log(2 * math.pi))

    # print("torch chunk1: " + str(chunk1))
    # print("torch chunk2: " + str(chunk2))
    # print("torch chunk3: " + str(chunk3))

    return chunk1 + chunk2 + chunk3


def map_estimate_torch_chi2(X, Y, Xt, Yt, bias, alpha, noise, Sigma, space_kernel, time_kernel, kernel, v,
                            t2, bias_sigma, N_sensors, N_time, theta_not):
    torch.set_default_dtype(torch.float64)
    Sigma_hat = alpha**2 * Sigma + (noise**2) * torch.eye(N_sensors*N_time)
    bias_sigma = bias_sigma

    chunk1 = -(1/2) * (torch.logdet(Sigma_hat)  # currently giving -inf
                       + (Y - bias).T @ torch.inverse(Sigma_hat) @ (Y - bias)
                       + N_sensors * math.log(2 * math.pi))

    chi2 = torch.distributions.gamma.Gamma(v/2, v*t2/2)
    prob_a = chi2.log_prob(1/alpha)
    prob_b = -(1/2) * (torch.logdet(bias_sigma)
                       + bias.T @ torch.inverse(bias_sigma) @ bias
                       + len(bias) * math.log(2 * math.pi))
    chunk2 = prob_a + prob_b

    def v(x):
        k = kernel(x.reshape(1, -1), X)
        output = theta_not - (alpha * k.T) @ torch.inverse(Sigma_hat) @ (alpha * k)
        if output < 0:
            print('Error')
        return output

    def mu(x):
        k = kernel(x.reshape(1, -1), X)
        return (alpha * k.T) @ torch.inverse(Sigma_hat) @ (Y - bias)

    chunk3 = 0
    for i in range(0, len(Xt)):
        holder = mu(Xt[i])
        var = v(Xt[i])
        chunk3 += -(1/2) * (torch.log(var) + ((Yt[i] - holder)**2)/var + math.log(2 * math.pi))

    # print("torch chunk1: " + str(chunk1))
    # print("torch chunk2: " + str(chunk2))
    # print("torch chunk3: " + str(chunk3))

    return chunk1 + chunk2 + chunk3