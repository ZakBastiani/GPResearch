import math
from scipy.sparse import linalg
import matplotlib.pyplot as plt
import torch
from torch import nn
from Final_Space import Gaussian_Process
from Real_World_Example import MAPEstimate
from Real_World_Example import fast_functions

class OptAlphaCalcBias(Gaussian_Process.GaussianProcess):
    def __init__(self, space_X, time_X, _Y, Xt, Yt,
                 noise_sd, theta_not, bias_kernel, v, t2):
        torch.set_default_dtype(torch.float64)
        N_sensors = len(space_X)
        N_time = len(time_X)

        class theta_opt(nn.Module):
            def __init__(self, space_X, time_X, X, Y, N_sensors, N_time):
                super(theta_opt, self).__init__()
                self.space_X = space_X
                self.time_X = time_X
                self.X = X
                self.Y = Y
                self.N_sensors = N_sensors
                self.N_time = N_time
                self.bias = torch.zeros(len(space_X) * len(time_X))
                self.alpha = nn.Parameter(torch.tensor(1.0))
                self.theta_space = torch.tensor(4000.0)
                self.theta_time = torch.tensor(0.25)
                self.theta_alt = torch.tensor(100.0)
                self.theta_not = theta_not

            def space_kernel(self, X, Y):
                kernel = theta_not * torch.exp(
                    - ((X.T[0].repeat(len(Y), 1) - Y.T[0].repeat(len(X), 1).T) ** 2) / (2 * self.theta_space ** 2)
                    - ((X.T[1].repeat(len(Y), 1) - Y.T[1].repeat(len(X), 1).T) ** 2) / (2 * self.theta_space ** 2)
                    - ((X.T[2].repeat(len(Y), 1) - Y.T[2].repeat(len(X), 1).T) ** 2) / (2 * self.theta_alt ** 2))
                return kernel

            def time_kernel(self, X, Y):
                kernel = torch.exp(-((X.T[0].repeat(len(Y), 1) - Y.T[0].repeat(len(X), 1).T) ** 2) / (2 * self.theta_time ** 2))
                return kernel

            def kernel(self, X, Y):
                kern = theta_not * torch.exp(
                    - ((X.T[0].repeat(len(Y), 1) - Y.T[0].repeat(len(X), 1).T) ** 2) / (2 * self.theta_space ** 2)
                    - ((X.T[1].repeat(len(Y), 1) - Y.T[1].repeat(len(X), 1).T) ** 2) / (2 * self.theta_space ** 2)
                    - ((X.T[2].repeat(len(Y), 1) - Y.T[2].repeat(len(X), 1).T) ** 2) / (2 * self.theta_alt ** 2)
                    - ((X.T[3].repeat(len(Y), 1) - Y.T[3].repeat(len(X), 1).T) ** 2) / (2 * self.theta_time ** 2))
                return kern

            # This is the MAP Estimate of the GP
            def forward(self, Xt, Yt):
                space_k = self.space_kernel(self.space_X, self.space_X)
                time_k = self.time_kernel(self.time_X, self.time_X)
                sigma = torch.kron(time_k, space_k)
                sigma_inv = fast_functions.quick_inv(space_k, time_k, self.alpha, noise_sd)
                bias_sigma = torch.kron(torch.eye(len(self.space_X)), bias_kernel(self.time_X, self.time_X))

                # Build and calc A and C
                A = torch.zeros((N_sensors * N_time, N_sensors * N_time))
                C = torch.zeros((1, N_sensors * N_time))
                current_C = 0
                for n in range(len(Xt)):
                    k_star = self.alpha * self.kernel(Xt[n].unsqueeze(0), X)
                    holder = (k_star.T @ sigma_inv @ k_star)[0][0]
                    holder2 = (k_star.T @ sigma_inv).T @ (k_star.T @ sigma_inv)
                    A += holder2 / (self.theta_not - holder)
                    current_C += ((k_star.T @ sigma_inv @ Y) * (k_star.T @ sigma_inv)
                                  - Yt[n] * (k_star.T @ sigma_inv)) / (self.theta_not - holder)
                A += sigma_inv.T

                A += torch.linalg.inv(bias_sigma)
                C[0] = Y.T @ sigma_inv + current_C
                C = C.T

                # Inverse A and multiply it by C
                A_inverse = torch.linalg.inv(A)
                b = A_inverse @ C

                self.bias = b.flatten()

                return MAPEstimate.map_estimate_torch_chi2(self.X, self.Y, Xt, Yt, self.bias, self.alpha, noise_sd,
                                                           sigma, sigma_inv, self.space_kernel, self.time_kernel,
                                                           self.kernel, v, t2, bias_sigma,
                                                           len(space_X), len(time_X), self.theta_not)

        X = torch.cat((space_X.repeat(len(time_X), 1),
                       time_X.repeat_interleave(len(space_X)).repeat(1, 1).T), 1)
        Y = _Y.flatten()

        # setting the model and then using torch to optimize
        theta_model = theta_opt(space_X, time_X, X, Y, len(space_X), len(time_X))
        optimizer = torch.optim.Adam(theta_model.parameters(), lr=0.01)
        smallest_loss = 5000
        for i in range(1000):
            optimizer.zero_grad()
            loss = -theta_model.forward(Xt, Yt)
            if loss < smallest_loss:
                smallest_loss = loss
            if i % 100 == 0:
                print('Loss: ' + str(loss))
                print('Alpha: ' + str(theta_model.alpha))
            loss.backward()
            optimizer.step()
        # print("Smallest Loss:" + str(smallest_loss))
        with torch.no_grad():
            holder = theta_model(Xt, Yt.T)

        self.type = "Gaussian Process Regression Optimizing alpha, bias and Theta"
        self.space_theta = theta_model.theta_space.detach().clone()
        self.time_theta = theta_model.theta_time.detach().clone()
        self.alt_theta = theta_model.theta_alt.detach().clone()
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
        # self.Sigma = self.kernel(self.points, self.points)
        # self.L = torch.linalg.cholesky(self.Sigma + self.noise_sd ** 2 * torch.eye(len(self.Sigma)))
        self.loss = -smallest_loss
