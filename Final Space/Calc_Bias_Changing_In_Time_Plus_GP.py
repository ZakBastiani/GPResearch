import numpy as np
import matplotlib.pyplot as plt
import torch
from Synthetic_Space_2D import Gaussian_Process


class ChangingBiasPlusGP(Gaussian_Process.GaussianProcess):
    def __init__(self, space_X, time_X, _Y, space_Xt, time_Xt, _Yt, space_kernel, time_kernel, kernel, noise, theta_not,
                 bias_variance, bias_mean, bias_kernel, alpha):
        sigma = np.kron(space_kernel(space_X, space_X), time_kernel(time_X, time_X))
        sigma_hat_inv = np.linalg.inv((sigma + (noise**2) * np.eye(len(sigma))))

        N_sensors = len(space_X)
        N_time = len(time_X)
        X = np.array((np.outer(space_X, np.ones(N_time)).flatten(),
                                        np.outer(time_X, np.ones(N_sensors)).T.flatten()))
        true_X = np.array((np.outer(space_Xt, np.ones(N_time)).flatten(),
                                      np.outer(time_Xt, np.ones(len(space_Xt))).T.flatten()))
        Y = _Y.flatten()
        true_Y = _Yt.flatten()

        # Build and calc A and C
        A = np.zeros(shape=(N_sensors*N_time, N_sensors*N_time))
        C = np.zeros(shape=(1, N_sensors*N_time))
        for k in range(0, N_sensors*N_time):
            extend_bias = np.zeros(shape=(N_sensors * N_time, 1))
            extend_bias[k][0] = 1
            current_A = np.zeros(shape=(1, N_sensors * N_time))
            current_C = 0
            for n in range(len(true_X.T)):
                k_star = kernel([true_X.T[n]], X.T).T
                holder = (k_star.T @ sigma_hat_inv @ k_star)[0][0]
                holder2 = (k_star.T @ sigma_hat_inv @ extend_bias) * (k_star.T @ sigma_hat_inv)
                current_A += holder2 / (theta_not - holder)
                current_C += ((k_star.T @ sigma_hat_inv @ Y) * (k_star.T @ sigma_hat_inv @ extend_bias)
                              - alpha * true_Y[n] * (k_star.T @ sigma_hat_inv @ extend_bias)) / (theta_not - holder)
            current_A += (sigma_hat_inv @ extend_bias).T
            A[k] = current_A
            A[k][k] += (alpha**2) / (bias_variance**2)
            C[0][k] = Y.T @ sigma_hat_inv @ extend_bias + current_C + (bias_mean / (bias_variance**2)) * (alpha**2)

        # Inverse A and multiply it by C
        A_inverse = np.linalg.inv(A)
        b = C @ A_inverse

        # Fix sigma to be more mailable later
        Sigma = np.kron(np.eye(N_sensors), bias_kernel(time_X, time_X))
        L = np.linalg.cholesky(Sigma + noise * np.eye(len(Sigma)))

        Lk = np.linalg.solve(L, Sigma)
        b = np.dot(Lk.T, np.linalg.solve(L, b.T))

        self.type = "Gaussian Process Regression with a calculated changing bias smoothed using a GP and a provided alpha"
        self.space_X = space_X # np.concatenate((space_X, space_Xt))
        self.time_X = time_X
        self.alpha = alpha
        self.bias = np.reshape(b, (-1, N_time))
        self.Y = (_Y - self.bias)/self.alpha  # np.concatenate(((_Y - self.bias)/self.alpha, _Yt))
        self.noise = noise
        self.space_kernel = space_kernel
        self.time_kernel = time_kernel
        self.Sigma = np.kron(self.space_kernel(self.space_X, self.space_X), self.time_kernel(self.time_X, self.time_X))
        self.L = np.linalg.cholesky((self.Sigma + noise * np.eye(len(self.Sigma))))
