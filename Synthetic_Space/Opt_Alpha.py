import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
from Synthetic_Space import Gaussian_Process


class OptAlphaCalcBias(Gaussian_Process.GaussianProcess):
    def __init__(self, space_X, time_X, _Y, space_Xt, time_Xt, _Yt, space_kernel, time_kernel,
                 kernel, noise, theta_not, bias, alpha_variance, alpha_mean):

        # Opimization module used to maximaxize Wei's equation
        # should get similar results to the above model
        class zak_gpr(nn.Module):
            def __init__(self, X, Y, K, N_sensors, N_time):
                super(zak_gpr, self).__init__()
                self.X = X
                self.Y = Y
                self.Sigma = torch.tensor(K)
                self.N_sensors = N_sensors
                self.N_time = N_time
                self.bias = torch.tensor(bias.reshape(-1, 1))
                # self.alpha = nn.Parameter(torch.tensor(alpha))
                self.alpha = nn.Parameter(torch.eye(1))

            def forward(self, Xt, Yt):
                Sigma_hat = self.Sigma + torch.eye(len(self.Sigma)) * noise

                chunk1 = -(1 / 2) * torch.log(torch.det(self.alpha**2 * Sigma_hat))  # currently giving -inf or inf
                # print("chunk1: " + str(chunk1))
                chunk2 = -(1 / 2) * (self.Y - self.bias).T @ torch.cholesky_inverse(self.alpha**2 * Sigma_hat) @ (
                        self.Y - self.bias)
                # print("chunk2: " + str(chunk2))
                prob_a = -(1 / 2) * (((self.alpha - alpha_mean) ** 2 / alpha_variance) + math.log(alpha_variance * 2 * math.pi))
                chunk3 = -(self.N_sensors / 2) * math.log(2 * math.pi) + prob_a # fix later
                # print("chunk3: " + str(chunk3))

                chunk4 = 0

                def v(x):
                    k = np.kron(space_kernel(np.array([[x[0]]]),
                                             np.array(torch.unique(self.X.T[0]))),
                                time_kernel(np.array([[x[1]]]),
                                            np.array(torch.unique(self.X.T[1]))))
                    k = torch.tensor(k)
                    output = theta_not - k @ torch.cholesky_inverse(Sigma_hat) @ k.T
                    if 0 > output:
                        print("Negative variance of " + str(output))
                        return abs(output)
                    return output

                def mu(x):
                    k = np.kron(space_kernel(np.array([[x[0]]]),
                                             np.array(torch.unique(self.X.T[0]))),
                                time_kernel(np.array([[x[1]]]),
                                            np.array(torch.unique(self.X.T[1]))))
                    k = torch.tensor(k)
                    return (k @ torch.cholesky_inverse(Sigma_hat) @ (self.Y - self.bias))/self.alpha

                for i in range(0, len(Xt)):
                    chunk4 += (1 / 2) * (
                            -torch.log(v(Xt[i])) - ((Yt[i] - mu(Xt[i])) ** 2) / v(Xt[i]) - math.log(2 * math.pi))
                # print("chunk4: " + str(chunk4))

                return chunk1 + chunk2 + chunk3 + chunk4  # Add back chunk1

        # Need to alter the sensor matrix and the data matrix
        X = torch.tensor([np.outer(space_X, np.ones(len(time_X))).flatten(),
                          np.outer(time_X, np.ones(len(space_X))).T.flatten()]).T
        Y = torch.reshape(torch.tensor(_Y), (1, -1))

        Xt = torch.tensor([np.outer(space_Xt, np.ones(len(time_Xt))).flatten(),
                           np.outer(time_Xt, np.ones(len(space_Xt))).T.flatten()]).T
        Yt = torch.reshape(torch.tensor(_Yt), (1, -1))

        K = np.kron(space_kernel(space_X, space_X), time_kernel(time_X, time_X))

        # setting the model and then using torch to optimize
        zaks_model = zak_gpr(X, Y.T, K, len(space_X), len(time_X))
        optimizer = torch.optim.Adam(zaks_model.parameters(), lr=0.01)  # lr is very important, lr>0.1 lead to failure
        smallest_loss = 1000
        best_alpha = 0
        for i in range(100):
            optimizer.zero_grad()
            loss = -zaks_model.forward(Xt, Yt.T)
            loss.backward()
            optimizer.step()
            # print("i: " + str(i) + ", loss: " + str(loss[0][0]))
            # print("alpha: " + str(zaks_model.alpha))
            if smallest_loss > loss:
                smallest_loss = loss
                best_alpha = zaks_model.alpha.clone()
            #     print("New Best")

        # print("Best Alpha Found: " + str(best_alpha))

        with torch.no_grad():
            zaks_model.alpha = nn.Parameter(best_alpha.clone().detach())
            holder = zaks_model(Xt, Yt.T)

        self.type = "Gaussian Process Regression optimizing for alpha and calculating bias"
        self.space_X = space_X  # np.concatenate((space_X, space_Xt))
        self.time_X = time_X
        self.alpha = zaks_model.alpha.detach().numpy()
        self.bias = zaks_model.bias.detach().numpy().reshape(len(space_X), len(time_X))
        self.Y = (_Y - self.bias) / self.alpha  # np.concatenate(((_Y - self.bias) / self.alpha, _Yt))
        self.noise = noise
        self.space_kernel = space_kernel
        self.time_kernel = time_kernel
        self.Sigma = np.kron(self.space_kernel(self.space_X, self.space_X), self.time_kernel(self.time_X, self.time_X))
        self.L = np.linalg.cholesky(self.Sigma + noise * np.eye(len(self.Sigma)))


