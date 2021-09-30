import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from Final_Space import Gaussian_Process
from Final_Space import MAPEstimate


class CalcBothChangingBias(Gaussian_Process.GaussianProcess):
    def __init__(self, space_X, time_X, _Y, space_Xt, time_Xt, _Yt,
                 space_kernel, time_kernel, kernel, noise_sd, theta_not, bias_kernel, alpha_mean, alpha_variance):

        self.points = torch.cat((space_X.repeat(len(time_X), 1), time_X.repeat_interleave(len(space_X)).repeat(1, 1).T), 1)

        # Let guess with alpha = mean
        alpha = alpha_mean
        b = torch.zeros((len(space_X) * len(time_X)))
        sigma = kernel(self.points, self.points)

        bias_sigma = torch.kron(torch.eye(len(space_X)), bias_kernel(time_X, time_X))

        N_sensors = len(space_X)
        N_time = len(time_X)

        self.points = torch.cat((space_X.repeat(len(time_X), 1), time_X.repeat_interleave(len(space_X)).repeat(1, 1).T), 1)

        # Need to alter the sensor matrix and the data matrix
        X = torch.cat((space_X.repeat(len(time_Xt), 1),
                       time_Xt.repeat_interleave(len(space_X)).repeat(1, 1).T), 1)
        Y = _Y.flatten()

        Xt = torch.cat((space_Xt.repeat(len(time_X), 1),
                        time_X.repeat_interleave(len(space_Xt)).repeat(1, 1).T), 1)
        Yt = _Yt.flatten()

        for counter in range(5):
            noise_lag = noise_sd/alpha
            sigma_inv = torch.linalg.inv(sigma + (noise_lag ** 2) * torch.eye(len(sigma)))
            # Build and calc A and C
            A = torch.zeros((N_sensors * N_time, N_sensors * N_time))
            C = torch.zeros((1, N_sensors * N_time))
            current_C = 0
            for n in range(len(Xt)):
                k_star = kernel(Xt[n].unsqueeze(0), X)
                holder = (k_star.T @ sigma_inv @ k_star)[0][0]
                holder2 = (k_star.T @ sigma_inv).T @ (k_star.T @ sigma_inv)
                A += holder2 / (theta_not - holder)
                current_C += ((k_star.T @ sigma_inv @ Y) * (k_star.T @ sigma_inv)
                              - alpha * Yt[n] * (k_star.T @ sigma_inv)) / (theta_not - holder)
            A += (sigma_inv).T

            A += (alpha ** 2) * torch.linalg.inv(bias_sigma)
            C[0] = Y.T @ sigma_inv + current_C

            # Inverse A and multiply it by C
            b = C @  torch.linalg.inv(A)

            alpha_poly = torch.zeros(5)
            y_min_bias = (Y - b).T
            alpha_poly[4] = y_min_bias.T @ sigma_inv @ y_min_bias
            alpha_poly[2] = -len(space_X) * len(time_X)
            alpha_poly[1] = alpha_mean / (alpha_variance ** 2)
            alpha_poly[0] = -1 / (alpha_variance ** 2)
            for i in range(len(Xt)):
                k_star = kernel(Xt[i].unsqueeze(0), X)
                divisor = (theta_not - k_star.T @ sigma_inv @ k_star)
                alpha_poly[4] += ((k_star.T @ sigma_inv @ y_min_bias) ** 2 / divisor).item()
                alpha_poly[3] -= ((Yt[i] * k_star.T @ sigma_inv @ y_min_bias) / divisor).item()

            roots = np.roots(alpha_poly.detach().numpy())
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
                alpha = closest
            alpha = torch.tensor(alpha)

        self.type = "Gaussian Process Regression calculating both a changing bias and alpha"
        self.space_X = space_X  # np.concatenate((space_X, space_Xt))
        self.time_X = time_X
        self.alpha = alpha
        self.bias = b
        self.Y = (_Y - self.bias)/self.alpha  # np.concatenate(((_Y - self.bias)/self.alpha, _Yt))
        self.noise_sd = noise_sd
        self.space_kernel = space_kernel
        self.time_kernel = time_kernel
        self.kernel = kernel
        self.Sigma = self.kernel(self.points, self.points) + noise_sd * torch.eye(len(self.points))
        self.L = torch.linalg.cholesky(self.Sigma)
        self.loss = MAPEstimate.map_estimate_torch(X, Y, Xt, Yt, self.bias.flatten(), alpha, noise_sd,
                                                   self.Sigma, space_kernel, time_kernel, kernel, alpha_mean,
                                                   alpha_variance,
                                                   torch.kron(torch.eye(len(space_X)), bias_kernel(time_X, time_X)),
                                                   len(space_X), len(time_X), theta_not)



