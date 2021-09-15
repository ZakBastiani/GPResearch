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
                self.bias = torch.zeros((N_time*N_sensors, 1))
                self.alpha = torch.eye(1)
                self.theta_space = nn.Parameter(torch.tensor(2.0), requires_grad=True)
                self.theta_time = nn.Parameter(torch.tensor(1.0), requires_grad=True)

            def space_kernel(self, X, Y):
                kernel = np.ndarray(shape=(len(X), len(Y)))
                for x in range(0, len(X)):
                    for y in range(0, len(Y)):
                        kernel[x][y] = theta_not * np.exp(- ((X[x][0] - Y[y][0]) ** 2) / (2 * self.theta_space.item() ** 2)
                                                          - ((X[x][1] - Y[y][1]) ** 2) / (2 * self.theta_space.item() ** 2))
                return kernel

            def time_kernel(self, X, Y):
                kernel = np.ndarray(shape=(len(X), len(Y)))
                for x in range(0, len(X)):
                    for y in range(0, len(Y)):
                        kernel[x][y] = np.exp(-((X[x] - Y[y]) ** 2) / (2 * self.theta_time.item() ** 2))
                return kernel

            def kernel(self, X, Y):
                kern = np.ndarray(shape=(len(X), len(Y)))
                for x in range(0, len(X)):
                    for y in range(0, len(Y)):
                        kern[x][y] = theta_not * np.exp(- ((X[x][0] - Y[y][0]) ** 2) / (2 * self.theta_space.item() ** 2)
                                                        - ((X[x][1] - Y[y][1]) ** 2) / (2 * self.theta_space.item() ** 2)
                                                        - ((X[x][2] - Y[y][2]) ** 2) / (2 * self.theta_time.item() ** 2))
                return kern

            def calcboth(self, Xt, Yt):
                alpha = alpha_mean
                b = torch.tensor((N_time*N_sensors, 1))
                sigma = np.kron(self.space_kernel(space_X, space_X), self.time_kernel(time_X, time_X))
                bias_sigma = np.kron(np.eye(len(space_X)), bias_kernel(time_X, time_X))
                sigma_hat_inv = np.linalg.inv(sigma + (noise ** 2) * np.eye(len(sigma)))
                X = self.X.detach().numpy()
                Y = self.Y.detach().numpy()
                for counter in range(10):
                    # Build and calc A and C
                    A = np.zeros(shape=(N_sensors * N_time, N_sensors * N_time))
                    C = np.zeros(shape=(1, N_sensors * N_time))
                    current_C = 0
                    for n in range(len(Xt)):
                        k_star = self.kernel([Xt[n]], X).T
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
                    y_min_bias = (Y - b.T)
                    alpha_poly[4] = y_min_bias.T @ sigma_hat_inv @ y_min_bias
                    alpha_poly[2] = -len(space_X) * len(time_X)
                    alpha_poly[1] = alpha_mean / (alpha_variance ** 2)
                    alpha_poly[0] = -1 / (alpha_variance ** 2)
                    for i in range(len(Xt)):
                        k_star = self.kernel([Xt[i]], X).T
                        divisor = (theta_not - k_star.T @ sigma_hat_inv @ k_star)
                        alpha_poly[4] += (k_star.T @ sigma_hat_inv @ y_min_bias) ** 2 / divisor
                        alpha_poly[3] -= (Yt[i] * k_star.T @ sigma_hat_inv @ y_min_bias) / divisor

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
                        alpha = (closest + alpha) / 2
                self.alpha = torch.tensor(alpha)
                self.bias = torch.tensor(b)

            # This is the MAP Estimate of the GP
            def forward(self, Xt, Yt):
                self.calcboth(Xt.detach().numpy(), Yt.detach().numpy())
                sigma = np.kron(self.space_kernel(space_X, space_X), self.time_kernel(time_X, time_X))
                bias_sigma = np.kron(np.eye(len(space_X)), bias_kernel(time_X, time_X))
                return MAPEstimate.map_estimate_torch(self.X, self.Y, Xt, Yt, self.bias.T, self.alpha, noise,
                                                   torch.tensor(sigma), self.space_kernel, self.time_kernel, self.kernel,
                                                    alpha_mean, alpha_variance,
                                                   torch.tensor(bias_sigma),
                                                   len(space_X), len(time_X), theta_not)

        X = torch.tensor(np.concatenate((np.repeat(space_X, len(time_X), axis=0),
                            np.tile(time_X, len(space_X)).reshape((-1, 1))), axis=1))
        Xt = torch.tensor(np.concatenate((np.repeat(space_Xt, len(time_Xt), axis=0),
                             np.tile(time_Xt, len(space_Xt)).reshape((-1, 1))), axis=1))
        Y = torch.reshape(torch.tensor(_Y.flatten()), (1, -1))
        Yt = torch.reshape(torch.tensor(_Yt.flatten()), (1, -1))


        # setting the model and then using torch to optimize
        theta_model = theta_opt(X, Y.T, len(space_X), len(time_X))
        optimizer = torch.optim.Adagrad(theta_model.parameters(), lr=0.02)  # lr is very important, lr>0.1 lead to failure
        smallest_loss = 1000
        for i in range(1000):
            optimizer.zero_grad()
            loss = -theta_model.forward(Xt, Yt.T)
            if loss < smallest_loss:
                smallest_loss = loss
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
        self.Y = (_Y - self.bias)/self.alpha  # np.concatenate(((_Y - self.bias)/self.alpha, _Yt))
        self.noise = noise
        self.space_kernel = theta_model.space_kernel
        self.time_kernel = theta_model.time_kernel
        self.Sigma = np.kron(self.space_kernel(self.space_X, self.space_X), self.time_kernel(self.time_X, self.time_X))
        self.L = np.linalg.cholesky(self.Sigma + self.noise**2 * np.eye(len(self.Sigma)))
        self.loss = -smallest_loss



