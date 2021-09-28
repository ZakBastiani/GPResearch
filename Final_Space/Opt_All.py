import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from Synthetic_Space_2D import Gaussian_Process
from Synthetic_Space_2D import Calc_Alpha_Calc_Changing_Bias
from Synthetic_Space_2D import MAPEstimate


class OptAll(Gaussian_Process.GaussianProcess):
    def __init__(self, space_X, time_X, _Y, space_Xt, time_Xt, _Yt,
                 noise, theta_not, bias_kernel, alpha_mean, alpha_variance):
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
                self.bias = nn.Parameter(torch.zeros((N_time * N_sensors, 1)))
                self.alpha = nn.Parameter(torch.eye(1))
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


            # This is the MAP Estimate of the GP
            def forward(self, Xt, Yt):
                sigma = torch.kron(self.space_kernel(space_X, space_X), self.time_kernel(time_X, time_X))
                bias_sigma = torch.kron(torch.eye(len(space_X)), torch.tensor(bias_kernel(time_X, time_X)))
                return MAPEstimate.map_estimate_torch(self.X, self.Y, Xt, Yt, self.bias, self.alpha, noise,
                                                      sigma, self.space_kernel, self.time_kernel, self.kernel,
                                                      alpha_mean, alpha_variance,
                                                      bias_sigma.float(),
                                                      len(space_X), len(time_X), theta_not)

        X = torch.tensor(np.concatenate((np.repeat(space_X, len(time_X), axis=0),
                                         np.tile(time_X, len(space_X)).reshape((-1, 1))), axis=1))
        Xt = torch.tensor(np.concatenate((np.repeat(space_Xt, len(time_Xt), axis=0),
                                          np.tile(time_Xt, len(space_Xt)).reshape((-1, 1))), axis=1))
        Y = torch.reshape(torch.tensor(_Y.flatten()), (1, -1))
        Yt = torch.reshape(torch.tensor(_Yt.flatten()), (1, -1))

        # setting the model and then using torch to optimize
        theta_model = theta_opt(X, Y.T, len(space_X), len(time_X))
        optimizer = torch.optim.Adagrad(theta_model.parameters(), lr=0.03)
        smallest_loss = 5000
        for i in range(2000):
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

        self.type = "Gaussian Process Regression Optimizing alpha, bias and Theta"
        self.space_theta = theta_model.theta_space.detach().numpy()
        self.time_theta = theta_model.theta_time.detach().numpy()
        self.space_X = space_X  # np.concatenate((space_X, space_Xt))
        self.time_X = time_X
        self.alpha = theta_model.alpha.item()
        self.bias = np.reshape(theta_model.bias.detach().numpy(), (N_sensors, N_sensors, N_time))
        self.Y = (_Y - self.bias) / self.alpha  # np.concatenate(((_Y - self.bias)/self.alpha, _Yt))
        self.noise = noise
        self.space_kernel = theta_model.space_kernel
        self.time_kernel = theta_model.time_kernel
        self.Sigma = torch.kron(self.space_kernel(self.space_X, self.space_X), self.time_kernel(self.time_X, self.time_X))
        self.L = torch.linalg.cholesky(self.Sigma + self.noise ** 2 * torch.eye(len(self.Sigma)))
        self.loss = -smallest_loss

    def build(self, space_points, time_points, N_space):
        l_inv = torch.inverse(self.L)
        Lk = l_inv @ torch.kron(self.space_kernel(self.space_X, torch.tensor(space_points)),
                                self.time_kernel(self.time_X, torch.tensor(time_points))).T
        mu = Lk.T @ l_inv @ torch.tensor(self.Y.flatten())

        # Should just be able to use reshape fix later
        # mu = np.reshape([mu], (test_space, test_time))
        holder = np.ndarray(shape=(N_space, N_space, len(time_points)))
        for i in range(0, N_space):
            for k in range(0, N_space):
                for j in range(0, len(time_points)):
                    holder[i][k][j] = mu[i * N_space * len(time_points) + k * len(time_points) + j].item()
        return holder