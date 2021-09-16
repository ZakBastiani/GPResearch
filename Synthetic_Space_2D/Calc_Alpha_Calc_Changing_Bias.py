import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from Synthetic_Space_2D import Gaussian_Process
from Synthetic_Space_2D import MAPEstimate


class CalcBothChangingBias(Gaussian_Process.GaussianProcess):
    def __init__(self, space_X, time_X, _Y, space_Xt, time_Xt, _Yt,
                 space_kernel, time_kernel, kernel, noise, theta_not, bias_kernel, alpha_mean, alpha_variance):
        # Let guess with alpha = mean
        alpha = alpha_mean
        sigma = np.kron(space_kernel(space_X, space_X), time_kernel(time_X, time_X))
        sigma_inv = np.linalg.inv(sigma + (noise**2) * np.eye(len(sigma)))
        bias_sigma = np.kron(np.eye(len(space_X)), bias_kernel(time_X, time_X))

        N_sensors = len(space_X)
        N_time = len(time_X)

        X = np.concatenate((np.repeat(space_X, len(time_X), axis=0),
                            np.tile(time_X, len(space_X)).reshape((-1, 1))), axis=1)
        Xt = np.concatenate((np.repeat(space_Xt, len(time_Xt), axis=0),
                                 np.tile(time_Xt, len(space_Xt)).reshape((-1, 1))), axis=1)
        Y = _Y.flatten()
        Yt = _Yt.flatten()
        sigma_hat_inv = np.linalg.inv(sigma + (noise ** 2) * np.eye(len(sigma)))

        for counter in range(1):
            # Build and calc A and C
            A = np.zeros(shape=(N_sensors * N_time, N_sensors * N_time))
            C = np.zeros(shape=(1, N_sensors * N_time))
            current_C = 0
            for n in range(len(Xt)):
                k_star = kernel([Xt[n]], X).T
                holder = (k_star.T @ sigma_hat_inv @ k_star)[0][0]
                holder2 = (k_star.T @ sigma_hat_inv).T @ (k_star.T @ sigma_hat_inv)
                A += holder2 / (theta_not - holder)
                current_C += ((k_star.T @ sigma_hat_inv @ Y) * (k_star.T @ sigma_hat_inv)
                              - alpha * Yt[n] * (k_star.T @ sigma_hat_inv)) / (theta_not - holder)
            A += (sigma_hat_inv).T

            A += (alpha ** 2) * np.linalg.inv(bias_sigma)
            C[0] = Y.T @ sigma_hat_inv + current_C

            # Inverse A and multiply it by C
            A_inverse = np.linalg.inv(A)
            b = C @ A_inverse

            alpha_poly = np.zeros(5)
            y_min_bias = (Y - b).T
            alpha_poly[4] = y_min_bias.T @ sigma_inv @ y_min_bias
            alpha_poly[2] = -len(space_X) * len(time_X)
            alpha_poly[1] = alpha_mean / (alpha_variance ** 2)
            alpha_poly[0] = -1 / (alpha_variance ** 2)
            for i in range(len(Xt)):
                k_star = kernel([Xt[i]], X).T
                divisor = (theta_not - k_star.T @ sigma_inv @ k_star)
                alpha_poly[4] += (k_star.T @ sigma_inv @ y_min_bias) ** 2 / divisor
                alpha_poly[3] -= (Yt[i] * k_star.T @ sigma_inv @ y_min_bias) / divisor

            roots = np.roots(alpha_poly)
            print(roots)
            real_roots = []
            for root in roots:
                if root.imag == 0:
                    real_roots.append(root.real)

            if len(real_roots) != 0:
                closest = real_roots[0]
                for r in real_roots:
                    if abs(closest - alpha_mean) > abs(r - alpha_mean):
                        closest = r
                alpha = closest

        N_sensors = int(math.sqrt((N_sensors)))

        self.type = "Gaussian Process Regression calculating both a changing bias and alpha"
        self.space_X = space_X  # np.concatenate((space_X, space_Xt))
        self.time_X = time_X
        self.alpha = alpha
        self.bias =  np.reshape(b, (N_sensors, N_sensors, N_time))
        self.Y = (_Y - self.bias)/self.alpha  # np.concatenate(((_Y - self.bias)/self.alpha, _Yt))
        self.noise = noise
        self.space_kernel = space_kernel
        self.time_kernel = time_kernel
        self.Sigma = np.kron(self.space_kernel(self.space_X, self.space_X), self.time_kernel(self.time_X, self.time_X))
        self.L = np.linalg.cholesky(self.Sigma + noise * np.eye(len(self.Sigma)))
        self.loss = MAPEstimate.map_estimate_numpy(X, Y, Xt, Yt, self.bias.flatten(), alpha, noise,
                                                   self.Sigma, space_kernel, time_kernel, kernel, alpha_mean,
                                                   alpha_variance,
                                                   np.kron(np.eye(len(space_X)), bias_kernel(time_X, time_X)),
                                                   len(space_X), len(time_X), theta_not)



