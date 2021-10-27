import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from Final_Space import Gaussian_Process
from Final_Space import MAPEstimate


class OptBias(Gaussian_Process.GaussianProcess):
    def __init__(self, space_X, time_X, _Y, space_Xt, time_Xt, _Yt,
                 noise_sd, theta_not, bias_kernel, alpha_mean, alpha_sd):
        torch.set_default_dtype(torch.float64)
        N_sensors = len(space_X)
        N_time = len(time_X)

        class opt_bias(nn.Module):
            def __init__(self, X, Y, N_sensors, N_time, Xt, Yt, bias, alpha):
                super(opt_bias, self).__init__()
                self.X = X
                self.N_sensors = N_sensors
                self.Y = Y
                self.theta_not = theta_not
                self.N_sensors = N_sensors
                self.N_time = N_time
                self.theta_space = torch.tensor(2.0)
                self.theta_time = torch.tensor(1.0)
                self.points = torch.cat((space_X.repeat(len(time_X), 1),
                                         time_X.repeat_interleave(len(space_X)).repeat(1, 1).T), 1)
                self.bias = nn.Parameter(bias)
                self.alpha = alpha

            def space_kernel(self, X, Y):
                kernel = self.theta_not * torch.exp(
                    -((X.T[0].repeat(len(Y), 1) - Y.T[0].repeat(len(X), 1).T) ** 2) / (2 * self.theta_space ** 2)
                    - ((X.T[1].repeat(len(Y), 1) - Y.T[1].repeat(len(X), 1).T) ** 2) / (2 * self.theta_space ** 2))
                return kernel

            def time_kernel(self, X, Y):
                kernel = torch.exp(-((X.repeat(len(Y), 1) - Y.repeat(len(X), 1).T) ** 2) / (2 * self.theta_time ** 2))
                return kernel

            def kernel(self, X, Y):
                kern = self.theta_not * torch.exp(
                    - ((X.T[0].repeat(len(Y), 1) - Y.T[0].repeat(len(X), 1).T) ** 2) / (2 * self.theta_space ** 2)
                    - ((X.T[1].repeat(len(Y), 1) - Y.T[1].repeat(len(X), 1).T) ** 2) / (2 * self.theta_space ** 2)
                    - ((X.T[2].repeat(len(Y), 1) - Y.T[2].repeat(len(X), 1).T) ** 2) / (2 * self.theta_time ** 2))
                return kern

            # This is the MAP Estimate of the GP
            def forward(self, Xt, Yt):
                sigma = self.kernel(self.points, self.points)
                bias_sigma = torch.kron(torch.eye(len(space_X)), bias_kernel(time_X, time_X))
                return MAPEstimate.map_estimate_torch(self.X, self.Y, Xt, Yt, self.bias.flatten(), self.alpha, noise_sd, sigma,
                                                      self.space_kernel, self.time_kernel, self.kernel, alpha_mean,
                                                      alpha_sd, bias_sigma, len(space_X), len(time_X),
                                                      theta_not)

        class opt_alpha(nn.Module):
            def __init__(self, X, Y, N_sensors, N_time, Xt, Yt, bias, alpha):
                super(opt_bias, self).__init__()
                self.X = X
                self.N_sensors = N_sensors
                self.Y = Y
                self.theta_not = theta_not
                self.N_sensors = N_sensors
                self.N_time = N_time
                self.theta_space = torch.tensor(2.0)
                self.theta_time = torch.tensor(1.0)
                self.points = torch.cat((space_X.repeat(len(time_X), 1),
                                         time_X.repeat_interleave(len(space_X)).repeat(1, 1).T), 1)
                self.bias = bias
                self.alpha = nn.Parameter(alpha)

            def space_kernel(self, X, Y):
                kernel = self.theta_not * torch.exp(
                    -((X.T[0].repeat(len(Y), 1) - Y.T[0].repeat(len(X), 1).T) ** 2) / (2 * self.theta_space ** 2)
                    - ((X.T[1].repeat(len(Y), 1) - Y.T[1].repeat(len(X), 1).T) ** 2) / (2 * self.theta_space ** 2))
                return kernel

            def time_kernel(self, X, Y):
                kernel = torch.exp(-((X.repeat(len(Y), 1) - Y.repeat(len(X), 1).T) ** 2) / (2 * self.theta_time ** 2))
                return kernel

            def kernel(self, X, Y):
                kern = self.theta_not * torch.exp(
                    - ((X.T[0].repeat(len(Y), 1) - Y.T[0].repeat(len(X), 1).T) ** 2) / (2 * self.theta_space ** 2)
                    - ((X.T[1].repeat(len(Y), 1) - Y.T[1].repeat(len(X), 1).T) ** 2) / (2 * self.theta_space ** 2)
                    - ((X.T[2].repeat(len(Y), 1) - Y.T[2].repeat(len(X), 1).T) ** 2) / (2 * self.theta_time ** 2))
                return kern

            # This is the MAP Estimate of the GP
            def forward(self, Xt, Yt):
                sigma = self.kernel(self.points, self.points)
                bias_sigma = torch.kron(torch.eye(len(space_X)), bias_kernel(time_X, time_X))
                return MAPEstimate.map_estimate_torch(self.X, self.Y, Xt, Yt, self.bias.flatten(), self.alpha, noise_sd, sigma,
                                                      self.space_kernel, self.time_kernel, self.kernel, alpha_mean,
                                                      alpha_sd, bias_sigma, len(space_X), len(time_X),
                                                      theta_not)

        X = torch.cat((space_X.repeat(len(time_X), 1),
                       time_X.repeat_interleave(len(space_X)).repeat(1, 1).T), 1)
        Y = _Y.flatten()

        Xt = torch.cat((space_Xt.repeat(len(time_Xt), 1),
                        time_Xt.repeat_interleave(len(space_Xt)).repeat(1, 1).T), 1)
        Yt = _Yt.flatten()

        bias = torch.zeros(N_time*N_sensors)
        alpha = torch.tensor(1.0)

        smallest_loss = 1000
        for j in range(10):
            bias_model = opt_bias(X, Y, len(space_X), len(time_X), Xt, Yt, bias, alpha)
            optimizer = torch.optim.Adam(bias_model.parameters(), lr=0.00001)
            for i in range(50):
                optimizer.zero_grad()
                loss = -bias_model.forward(Xt, Yt)
                if loss < smallest_loss:
                    smallest_loss = loss
                if i % 10 == 0:
                    print(loss)
                loss.backward()
                optimizer.step()

            alpha_model = opt_alpha(X, Y, len(space_X), len(time_X), Xt, Yt, bias, alpha)
            optimizer = torch.optim.Adam(alpha_model.parameters(), lr=0.01)
            for i in range(50):
                optimizer.zero_grad()
                loss = -alpha_model.forward(Xt, Yt)
                if loss < smallest_loss:
                    smallest_loss = loss
                if i % 10 == 0:
                    print(loss)
                loss.backward()
                optimizer.step()

        # print("Smallest Loss:" + str(smallest_loss))
        with torch.no_grad():
            holder = alpha_model(Xt, Yt)

        self.type = "Gaussian Process Regression Optimizing Theta Calc Alpha Bias"
        self.space_theta = alpha_model.theta_space
        self.time_theta = alpha_model.theta_time
        self.space_X = space_X  # np.concatenate((space_X, space_Xt))
        self.time_X = time_X
        self.alpha = alpha_model.alpha.item()
        self.bias = alpha_model.bias.detach().clone()
        self.Y = (_Y - self.bias) / self.alpha  # np.concatenate(((_Y - self.bias)/self.alpha, _Yt))
        self.noise = noise_sd
        self.space_kernel = alpha_model.space_kernel
        self.time_kernel = alpha_model.time_kernel
        self.kernel = alpha_model.kernel
        self.points = torch.cat((space_X.repeat(len(time_X), 1),
                                 time_X.repeat_interleave(len(space_X)).repeat(1, 1).T), 1)
        self.Sigma = self.kernel(self.points, self.points)
        self.L = torch.linalg.cholesky(self.Sigma + self.noise ** 2 * torch.eye(len(self.Sigma)))
        self.loss = -smallest_loss
