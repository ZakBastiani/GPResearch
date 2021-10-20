import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from Final_Space import Gaussian_Process
from Final_Space import MAPEstimate


class OptAll(Gaussian_Process.GaussianProcess):
    def __init__(self, space_X, time_X, _Y, space_Xt, time_Xt, _Yt,
                 noise_sd, theta_not, bias_kernel, alpha_mean, alpha_sd, alpha, bias):
        N_sensors = len(space_X)
        N_time = len(time_X)

        class theta_opt(nn.Module):
            def __init__(self, X, Y, N_sensors, N_time):
                super(theta_opt, self).__init__()
                self.X = X
                self.N_sensors = N_sensors
                self.Y = Y
                self.N_sensors = N_sensors
                self.N_time = N_time
                self.bias = np.zeros(len(space_X) * len(time_X))
                self.alpha = nn.Parameter(torch.tensor(1.0))
                self.theta_space = nn.Parameter(torch.tensor(1.0))
                self.theta_time = nn.Parameter(torch.tensor(1.0))
                self.theta_not = theta_not

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
                return kern

            # This is the MAP Estimate of the GP
            def forward(self, Xt, Yt):
                sigma = self.kernel(self.X, self.X)
                noise_lag = noise_sd/self.alpha
                sigma_inv = torch.linalg.inv(sigma + (noise_lag ** 2) * torch.eye(len(sigma)))
                bias_sigma = torch.kron(torch.eye(len(space_X)), bias_kernel(time_X, time_X))
                # Build and calc A and C
                A = torch.zeros((N_sensors * N_time, N_sensors * N_time))
                C = torch.zeros((1, N_sensors * N_time))
                current_C = 0
                for n in range(len(Xt)):
                    k_star = self.kernel(Xt[n].unsqueeze(0), X)
                    holder = (k_star.T @ sigma_inv @ k_star)[0][0]
                    holder2 = (k_star.T @ sigma_inv).T @ (k_star.T @ sigma_inv)
                    A += holder2 / (self.theta_not - holder)
                    current_C += ((k_star.T @ sigma_inv @ Y) * (k_star.T @ sigma_inv)
                                  - alpha * Yt[n] * (k_star.T @ sigma_inv)) / (self.theta_not - holder)
                A += (sigma_inv).T

                A += (alpha ** 2) * torch.linalg.inv(bias_sigma)
                C[0] = Y.T @ sigma_inv + current_C

                # Inverse A and multiply it by C
                A_inverse = torch.linalg.inv(A)
                b = C @ A_inverse
                self.bias = b.flatten()

                return MAPEstimate.map_estimate_torch_chi2(self.X, self.Y, Xt, Yt, self.bias, self.alpha, noise_sd,
                                                          sigma, self.space_kernel, self.time_kernel, self.kernel,
                                                          alpha_mean, alpha_sd, bias_sigma,
                                                            len(space_X), len(time_X), self.theta_not)


        X = torch.cat((space_X.repeat(len(time_Xt), 1),
                       time_Xt.repeat_interleave(len(space_X)).repeat(1, 1).T), 1)
        Y = _Y.flatten()

        Xt = torch.cat((space_Xt.repeat(len(time_X), 1),
                        time_X.repeat_interleave(len(space_Xt)).repeat(1, 1).T), 1)
        Yt = _Yt.flatten()

        # setting the model and then using torch to optimize
        theta_model = theta_opt(X, Y, len(space_X), len(time_X))
        optimizer = torch.optim.Adagrad(theta_model.parameters(), lr=0.02)
        smallest_loss = 5000
        for i in range(1000):
            optimizer.zero_grad()
            loss = -theta_model.forward(Xt, Yt)
            if loss < smallest_loss:
                smallest_loss = loss
            if i % 100 == 0:
                print(loss)
            loss.backward()
            optimizer.step()
        # print("Smallest Loss:" + str(smallest_loss))
        with torch.no_grad():
            holder = theta_model(Xt, Yt.T)

        self.type = "Gaussian Process Regression Optimizing alpha, bias and Theta"
        self.space_theta = theta_model.theta_space.detach().clone()
        self.time_theta = theta_model.theta_time.detach().clone()
        self.space_X = space_X  # np.concatenate((space_X, space_Xt))
        self.time_X = time_X
        self.alpha = theta_model.alpha.item()
        self.bias = theta_model.bias.detach().clone()
        self.Y = (_Y - self.bias) / self.alpha  # np.concatenate(((_Y - self.bias)/self.alpha, _Yt))
        self.noise_sd = noise_sd
        self.space_kernel = theta_model.space_kernel
        self.time_kernel = theta_model.time_kernel
        self.kernel = theta_model.kernel
        self.points = torch.cat((space_X.repeat(len(time_X), 1),
                                 time_X.repeat_interleave(len(space_X)).repeat(1, 1).T), 1)
        self.Sigma = self.kernel(self.points, self.points)
        self.L = torch.linalg.cholesky(self.Sigma + self.noise_sd ** 2 * torch.eye(len(self.Sigma)))
        self.loss = -smallest_loss
