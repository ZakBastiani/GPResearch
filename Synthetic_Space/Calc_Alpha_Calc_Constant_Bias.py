import numpy as np
import matplotlib.pyplot as plt
import torch
from Synthetic_Space import Gaussian_Process


class CalcBoth(Gaussian_Process.GaussianProcess):
    def __init__(self, space_X, time_X, _Y, space_Xt, time_Xt, _Yt,
                 space_kernel, time_kernel, kernel, noise, theta_not, bias_variance, bias_mean, alpha_mean, alpha_variance):
        # Let guess with alpha = mean
        alpha = alpha_mean
        b = bias_mean*np.ones(len(space_X))
        sigma = np.kron(space_kernel(space_X, space_X), time_kernel(time_X, time_X))
        sigma_inv = np.linalg.inv(sigma + noise * np.eye(len(sigma)))

        N_sensors = len(space_X)
        N_time = len(time_X)

        X = np.array((np.outer(space_X, np.ones(N_time)).flatten(),
                      np.outer(time_X, np.ones(N_sensors)).T.flatten())).T
        Xt = np.array((np.outer(space_Xt, np.ones(N_time)).flatten(),
                           np.outer(time_Xt, np.ones(len(space_Xt))).T.flatten())).T
        Y = _Y.flatten()
        Yt = _Yt.flatten()

        for counter in range(5):
            sigma_hat_inv = np.linalg.inv(sigma + noise * np.eye(len(sigma)))
            # Build and calc A and C
            A = np.zeros(shape=(N_sensors, N_sensors))
            C = np.zeros(shape=(1, N_sensors))
            for k in range(0, N_sensors):
                extend_bias = np.zeros(shape=(N_sensors * N_time, 1))
                for j in range(0, N_time):
                    extend_bias[k * N_time + j][0] = 1
                current_A = np.zeros(shape=(1, N_sensors * N_time))
                current_C = 0
                for n in range(len(Xt)):
                    k_star = kernel([Xt[n]], X).T
                    holder = (k_star.T @ sigma_hat_inv @ k_star)[0][0]
                    holder2 = (k_star.T @ sigma_hat_inv @ extend_bias) * (k_star.T @ sigma_hat_inv)
                    current_A += holder2 / (theta_not - holder)
                    current_C += ((k_star.T @ sigma_hat_inv @ Y) * (k_star.T @ sigma_hat_inv @ extend_bias)
                                  - alpha * Yt[n] * (k_star.T @ sigma_hat_inv @ extend_bias)) / (
                                         theta_not - holder)
                current_A += (sigma_hat_inv @ extend_bias).T
                # Need to condense current_A into b_i variables
                for i in range(0, N_sensors):
                    sum = 0
                    for j in range(0, N_time):
                        sum += current_A[0][i * N_time + j]
                    A[k][i] = sum
                A[k][k] += (alpha ** 2) / (bias_variance ** 2)

                C[0][k] = Y.T @ sigma_hat_inv @ extend_bias + current_C + (alpha ** 2) * bias_mean / (
                            bias_variance ** 2)

            # Inverse A and multiply it by C
            A_inverse = np.linalg.inv(A)
            b = C @ A_inverse

            alpha_poly = np.zeros(5)
            y_min_bias = (Y - np.outer(b.T, np.ones(N_time)).flatten()).T
            alpha_poly[4] = y_min_bias.T @ sigma_inv @ y_min_bias
            # alpha_poly[2] = len(space_X) * len(time_X)
            alpha_poly[1] = alpha_mean / (alpha_variance ** 2)
            alpha_poly[0] = -1 / (alpha_variance ** 2)
            for i in range(len(Xt)):
                k_star = kernel([Xt[i]], X).T
                divisor = (theta_not - k_star.T @ sigma_inv @ k_star)
                alpha_poly[4] += (k_star.T @ sigma_inv @ y_min_bias) ** 2 / divisor
                alpha_poly[3] -= (Yt[i] * k_star.T @ sigma_inv @ y_min_bias) / divisor

            roots = np.roots(alpha_poly)
            # print(roots)
            real_roots = []
            for root in roots:
                if root.imag == 0:
                    real_roots.append(root.real)

            if len(real_roots) != 0:
                closest = real_roots[0]
                for r in real_roots:
                    if abs(closest - alpha_mean) > abs(r - alpha_mean):
                        closest = r
                alpha = (closest + alpha)/2
            print(alpha)

        self.type = "Gaussian Process Regression calculating both bias and alpha"
        self.space_X = space_X  # np.concatenate((space_X, space_Xt))
        self.time_X = time_X
        self.alpha = alpha
        self.bias = np.outer(b.T, np.ones(N_time))
        self.Y = (_Y - self.bias)/self.alpha  # np.concatenate(((_Y - self.bias)/self.alpha, _Yt))
        self.noise = noise
        self.space_kernel = space_kernel
        self.time_kernel = time_kernel
        self.Sigma = np.kron(self.space_kernel(self.space_X, self.space_X), self.time_kernel(self.time_X, self.time_X))
        self.L = np.linalg.cholesky(self.Sigma + noise * np.eye(len(self.Sigma)))



