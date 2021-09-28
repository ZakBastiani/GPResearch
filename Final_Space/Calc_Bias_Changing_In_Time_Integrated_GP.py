import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from Final_Space import Gaussian_Process
from Final_Space import MAPEstimate


class ChangingBiasIntGP(Gaussian_Process.GaussianProcess):
    def __init__(self, space_X, time_X, _Y, space_Xt, time_Xt, _Yt, space_kernel, time_kernel, kernel, noise_sd, theta_not,
                 bias_variance, bias_mean, bias_kernel, alpha, alpha_variance, alpha_mean):
        sigma = np.kron(space_kernel(space_X, space_X), time_kernel(time_X, time_X))
        sigma_hat_inv = np.linalg.inv(sigma + (noise_sd ** 2) * np.eye(len(sigma)))
        bias_sigma = np.kron(np.eye(len(space_X)), bias_kernel(time_X, time_X))

        N_sensors = len(space_X)
        N_time = len(time_X)
        X = np.concatenate((np.repeat(space_X, len(time_X), axis=0),
                            np.tile(time_X, len(space_X)).reshape((-1, 1))), axis=1)
        true_X = np.concatenate((np.repeat(space_Xt, len(time_Xt), axis=0),
                                 np.tile(time_Xt, len(space_Xt)).reshape((-1, 1))), axis=1)
        Y = _Y.flatten()
        true_Y = _Yt.flatten()

        # Build and calc A and C
        A = np.zeros(shape=(N_sensors * N_time, N_sensors * N_time))
        C = np.zeros(shape=(1, N_sensors * N_time))
        current_C = 0
        for n in range(len(true_X)):
            k_star = kernel([true_X[n]], X).T
            holder = (k_star.T @ sigma_hat_inv @ k_star)[0][0]
            holder2 = (k_star.T @ sigma_hat_inv).T @ (k_star.T @ sigma_hat_inv)
            A += holder2 / (theta_not - holder)
            current_C += ((k_star.T @ sigma_hat_inv @ Y) * (k_star.T @ sigma_hat_inv)
                          - alpha * true_Y[n] * (k_star.T @ sigma_hat_inv)) / (theta_not - holder)
        A += (sigma_hat_inv).T

        A += (alpha ** 2) * np.linalg.inv(bias_sigma)
        C[0] = Y.T @ sigma_hat_inv + current_C

        # Inverse A and multiply it by C
        A_inverse = np.linalg.inv(A)
        b = C @ A_inverse

        N_sensors = int(math.sqrt(N_sensors))

        self.type = "Gaussian Process Regression with a calculated changing bias using an integrated GP and a provided alpha"
        self.space_X = space_X # np.concatenate((space_X, space_Xt))
        self.time_X = time_X
        self.alpha = alpha
        self.bias = np.reshape(b, (N_sensors, N_sensors, N_time))
        self.Y = (_Y - self.bias)/self.alpha # np.concatenate(((_Y - self.bias)/self.alpha, _Yt))
        self.noise = noise_sd
        self.space_kernel = space_kernel
        self.time_kernel = time_kernel
        self.Sigma = np.kron(self.space_kernel(self.space_X, self.space_X), self.time_kernel(self.time_X, self.time_X))
        self.L = np.linalg.cholesky((self.Sigma + noise_sd ** 2 * np.eye(len(self.Sigma))))
        self.loss = MAPEstimate.map_estimate_numpy(X, Y, true_X, true_Y, self.bias.flatten(), alpha, noise_sd,
                                                   self.Sigma, space_kernel, time_kernel, kernel, alpha_mean,
                                                   alpha_variance,
                                                   np.kron(np.eye(len(space_X)), bias_kernel(time_X, time_X)),
                                                   len(space_X), len(time_X), theta_not)





