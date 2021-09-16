import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from Synthetic_Space_2D import Gaussian_Process
from Synthetic_Space_2D import Calc_Alpha_Calc_Changing_Bias
from Synthetic_Space_2D import MAPEstimate


class OptTheta(Gaussian_Process.GaussianProcess):
    def __init__(self, space_X, time_X, _Y, space_Xt, time_Xt, _Yt,
                 noise, theta_not, bias_kernel, alpha_mean, alpha_variance):
        torch.set_default_dtype(torch.float64)
        N_sensors = len(space_X)
        N_time = len(time_X)
        space_X = torch.tensor(space_X)
        time_X = torch.tensor(time_X)

        class theta_opt(nn.Module):
            def __init__(self, X, Y, N_sensors, N_time):
                super(theta_opt, self).__init__()
                self.X = X
                self.N_sensors = N_sensors
                self.Y = Y
                self.N_sensors = N_sensors
                self.N_time = N_time
                self.bias = torch.zeros((N_time * N_sensors, 1))
                self.alpha = torch.eye(1) * alpha_mean
                self.theta_space = nn.Parameter(torch.tensor(2.0))
                self.theta_time = nn.Parameter(torch.tensor(1.0))

            def space_kernel(self, X, Y):
                kernel = theta_not * torch.exp(
                    -((X.T[0].repeat(len(Y), 1) - Y.T[0].repeat(len(X), 1).T) ** 2) / (2 * self.theta_space ** 2)
                    - ((X.T[1].repeat(len(Y), 1) - Y.T[1].repeat(len(X), 1).T) ** 2) / (2 * self.theta_space ** 2))
                return kernel

            def time_kernel(self, X, Y):
                kernel = torch.exp(-((X.repeat(len(Y), 1) - Y.repeat(len(X), 1).T) ** 2) / (2 * self.theta_time ** 2))
                return kernel

            def kernel(self, X, Y):
                kern = theta_not * torch.exp(
                    - ((X.T[0].repeat(len(Y), 1) - Y.T[0].repeat(len(X), 1).T) ** 2) / (2 * self.theta_space ** 2)
                    - ((X.T[1].repeat(len(Y), 1) - Y.T[1].repeat(len(X), 1).T) ** 2) / (2 * self.theta_space ** 2)
                    - ((X.T[2].repeat(len(Y), 1) - Y.T[2].repeat(len(X), 1).T) ** 2) / (2 * self.theta_time ** 2))
                return kern.T

            def calcboth(self, Xt, Yt):
                alpha = self.alpha
                b = self.bias
                sigma = torch.kron(self.space_kernel(space_X, space_X), self.time_kernel(time_X, time_X))
                bias_sigma = torch.kron(torch.eye(len(space_X)), torch.tensor(bias_kernel(time_X, time_X)))
                sigma_hat_inv = torch.inverse(sigma + (noise ** 2) * torch.eye(len(sigma)))
                for counter in range(5):
                    # Build and calc A and C
                    A = torch.zeros((N_sensors * N_time, N_sensors * N_time))
                    C = torch.zeros((1, N_sensors * N_time))
                    current_C = 0
                    for n in range(len(Xt)):
                        k_star = self.kernel(Xt[n].reshape(1, -1), X).T
                        holder = (k_star.T @ sigma_hat_inv @ k_star)
                        holder2 = (k_star.T @ sigma_hat_inv).T @ (k_star.T @ sigma_hat_inv)
                        A += holder2 / (theta_not - holder)
                        current_C += ((k_star.T @ sigma_hat_inv @ self.Y) * (k_star.T @ sigma_hat_inv)
                                      - alpha * Yt[n] * (k_star.T @ sigma_hat_inv)) / (theta_not - holder)
                    A += (sigma_hat_inv).T

                    A += (alpha ** 2) * torch.inverse(bias_sigma)
                    C[0] = self.Y.T @ sigma_hat_inv + current_C

                    # Inverse A and multiply it by C
                    b = C @ torch.inverse(A)

                    alpha_poly = torch.zeros(5)
                    y_min_bias = (self.Y - b.T)
                    alpha_poly[4] = y_min_bias.T @ sigma_hat_inv @ y_min_bias
                    alpha_poly[2] = -len(space_X) * len(time_X)
                    alpha_poly[1] = alpha_mean / (alpha_variance ** 2)
                    alpha_poly[0] = -1 / (alpha_variance ** 2)
                    for i in range(len(Xt)):
                        k_star = self.kernel(Xt[i].reshape(1, -1), self.X).T
                        divisor = (theta_not - k_star.T @ sigma_hat_inv @ k_star)
                        holder = k_star.T @ sigma_hat_inv @ y_min_bias
                        alpha_poly[4] += (holder ** 2 / divisor).item()
                        alpha_poly[3] -= ((Yt[i] * holder) / divisor).item()

                    roots = np.roots(alpha_poly.detach().numpy())  # maybe?
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
                        alpha = (closest + alpha) / 2
                self.alpha = torch.tensor(alpha)
                self.bias = b

            # This is the MAP Estimate of the GP
            def forward(self, Xt, Yt):
                self.calcboth(Xt, Yt)
                sigma = torch.kron(self.space_kernel(space_X, space_X), self.time_kernel(time_X, time_X))
                bias_sigma = torch.kron(torch.eye(len(space_X)), torch.tensor(bias_kernel(time_X, time_X)))
                return MAPEstimate.map_estimate_torch(self.X, self.Y, Xt, Yt, self.bias.T, self.alpha, noise, sigma,
                                                      self.space_kernel, self.time_kernel, self.kernel, alpha_mean,
                                                      alpha_variance, bias_sigma.float(), len(space_X), len(time_X),
                                                      theta_not)

        X = torch.tensor(np.concatenate((np.repeat(space_X, len(time_X), axis=0),
                                         np.tile(time_X, len(space_X)).reshape((-1, 1))), axis=1))
        Xt = torch.tensor(np.concatenate((np.repeat(space_Xt, len(time_Xt), axis=0),
                                          np.tile(time_Xt, len(space_Xt)).reshape((-1, 1))), axis=1))
        Y = torch.reshape(torch.tensor(_Y.flatten()), (1, -1))
        Yt = torch.reshape(torch.tensor(_Yt.flatten()), (1, -1))

        # setting the model and then using torch to optimize
        theta_model = theta_opt(X, Y.T, len(space_X), len(time_X))
        optimizer = torch.optim.Adagrad(theta_model.parameters(), lr=0.01)
        smallest_loss = 1000
        for i in range(300):
            optimizer.zero_grad()
            loss = -theta_model.forward(Xt, Yt.T)
            if loss < smallest_loss:
                smallest_loss = loss
            if i % 100 == 0:
                print(loss)
            loss.backward()
            optimizer.step()
        # print("Smallest Loss:" + str(smallest_loss))
        with torch.no_grad():
            holder = theta_model(Xt, Yt.T)

        N_sensors = int(math.sqrt((N_sensors)))

        self.type = "Gaussian Process Regression calculating both a changing bias and alpha"
        self.space_theta = theta_model.theta_space
        self.time_theta = theta_model.theta_time
        self.space_X = space_X  # np.concatenate((space_X, space_Xt))
        self.time_X = time_X
        self.alpha = theta_model.alpha.item()
        self.bias = np.reshape(theta_model.bias.detach().numpy(), (N_sensors, N_sensors, N_time))
        print(self.alpha)
        print(self.bias)
        print(self.time_theta)
        print(self.space_theta)
        self.Y = (_Y - self.bias) / self.alpha  # np.concatenate(((_Y - self.bias)/self.alpha, _Yt))
        self.noise = noise
        self.space_kernel = theta_model.space_kernel
        self.time_kernel = theta_model.time_kernel
        self.Sigma = torch.kron(self.space_kernel(self.space_X, self.space_X), self.time_kernel(self.time_X, self.time_X))
        self.L = np.linalg.cholesky(self.Sigma.detach().numpy() + self.noise ** 2 * np.eye(len(self.Sigma)))
        self.loss = -smallest_loss
