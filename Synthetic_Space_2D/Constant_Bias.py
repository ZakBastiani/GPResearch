import math
import numpy as np
import matplotlib.pyplot as plt
from Synthetic_Space_2D import Gaussian_Process
from Synthetic_Space_2D import MAPEstimate


class ConstantBias(Gaussian_Process.GaussianProcess):
    def __init__(self, space_X, time_X, _Y, space_Xt, time_Xt, _Yt,
                 space_kernel, time_kernel, kernel, noise, theta_not, bias_variance, bias_mean, alpha,
                 alpha_mean, alpha_variance, bias_kernel):
        sigma = np.kron(space_kernel(space_X, space_X), time_kernel(time_X, time_X))
        sigma_hat_inv = np.linalg.inv(sigma + (noise**2) * np.eye(len(sigma)))

        N_sensors = len(space_X)
        N_time = len(time_X)
        X = np.concatenate((np.repeat(space_X, len(time_X), axis=0),
                            np.tile(time_X, len(space_X)).reshape((-1, 1))), axis=1)
        true_X = np.concatenate((np.repeat(space_Xt, len(time_Xt), axis=0),
                                 np.tile(time_Xt, len(space_Xt)).reshape((-1, 1))), axis=1)
        Y = _Y.flatten()
        true_Y = _Yt.flatten()

        # Build and calc A and C
        A = np.zeros(shape=(N_sensors, N_sensors))
        C = np.zeros(shape=(1, N_sensors))
        for k in range(0, N_sensors):
            extend_bias = np.zeros(shape=(N_sensors * N_time, 1))
            for j in range(0, N_time):
                extend_bias[k * N_time + j][0] = 1
            current_A = np.zeros(shape=(1, N_sensors * N_time))
            current_C = 0
            for n in range(len(true_X)):
                k_star = kernel([true_X[n]], X).T
                holder = (k_star.T @ sigma_hat_inv @ k_star)[0][0]
                holder2 = (k_star.T @ sigma_hat_inv @ extend_bias) * (k_star.T @ sigma_hat_inv)
                current_A += holder2 / (theta_not - holder)
                current_C += ((k_star.T @ sigma_hat_inv @ Y) * (k_star.T @ sigma_hat_inv @ extend_bias)
                              - alpha * true_Y[n] * (k_star.T @ sigma_hat_inv @ extend_bias)) / (
                                     theta_not - holder)
            current_A += (sigma_hat_inv @ extend_bias).T
            # Need to condense current_A into b_i variables
            for i in range(0, N_sensors):
                sum = 0
                for j in range(0, N_time):
                    sum += current_A[0][i * N_time + j]
                A[k][i] = sum
            A[k][k] += (alpha ** 2) / (bias_variance ** 2)

            C[0][k] = Y.T @ sigma_hat_inv @ extend_bias + current_C + (alpha ** 2) * bias_mean / (bias_variance ** 2)

        # Inverse A and multiply it by C
        A_inverse = np.linalg.inv(A)
        b = C @ A_inverse

        N_sensors = int(math.sqrt(N_sensors))
        self.type = "Gaussian Process Regression with a calculated constant bias and a provided alpha"
        self.space_X = space_X  # np.concatenate((space_X, space_Xt))
        self.time_X = time_X
        self.alpha = alpha
        self.bias = np.outer(b.T, np.ones(N_time)).reshape(N_sensors, N_sensors, N_time)
        self.Y = (_Y - self.bias) / self.alpha  # np.concatenate(((_Y - self.bias) / self.alpha, _Yt))
        self.noise = noise
        self.space_kernel = space_kernel
        self.time_kernel = time_kernel
        self.Sigma = np.kron(self.space_kernel(self.space_X, self.space_X), self.time_kernel(self.time_X, self.time_X))
        self.L = np.linalg.cholesky((self.Sigma + noise**2 * np.eye(len(self.Sigma))))
        self.loss = MAPEstimate.map_estimate_numpy(X, Y, true_X, true_Y, self.bias.flatten(), alpha, noise,
                                                   self.Sigma, space_kernel, time_kernel, kernel, alpha_mean,
                                                   alpha_variance,
                                                   np.kron(np.eye(len(space_X)), bias_kernel(time_X, time_X)),
                                                   len(space_X), len(time_X), theta_not)


