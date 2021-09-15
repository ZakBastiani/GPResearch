import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
from Synthetic_Space import Gaussian_Process
import MAPEstimate

class OptAlphaCalcBias(Gaussian_Process.GaussianProcess):
    def __init__(self, space_X, time_X, _Y, space_Xt, time_Xt, _Yt, space_kernel, time_kernel,
                 kernel, noise, theta_not, bias, alpha_variance, alpha_mean, bias_kernel):

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
                self.alpha = nn.Parameter(torch.tensor(1.0))


            def v(self, x, Sigma_hat):
                k = kernel([x.detach().numpy()], self.X.detach().numpy()).T
                k = torch.from_numpy(k)
                output = theta_not - k.T @ torch.cholesky_inverse(Sigma_hat) @ k
                if output < 0:
                    print('Error')
                return output

            def mu(self, x, Sigma_hat):
                k = kernel([x.detach().numpy()], self.X.detach().numpy()).T
                k = torch.from_numpy(k)
                return (k.T @ torch.linalg.inv(Sigma_hat) @ (self.Y - self.bias))/self.alpha

            # Maximizing the predicted bias based on direct function. This is the MAP Estimate of the GP
            def forward(self, Xt, Yt):
                return MAPEstimate.map_estimate_torch(X, Y.T, Xt, Yt, self.bias, self.alpha, noise,
                                                      torch.tensor(self.Sigma), space_kernel, time_kernel, kernel, alpha_mean,
                                                      alpha_variance, torch.tensor(np.kron(np.eye(len(space_X)), bias_kernel(time_X, time_X))),
                                                      len(space_X), len(time_X), theta_not)

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
        optimizer = torch.optim.Adagrad(zaks_model.parameters(), lr=0.01)  # lr is very important, lr>0.1 lead to failure
        smallest_loss = 1000
        best_alpha = 0
        for i in range(1500):
            optimizer.zero_grad()
            loss = -zaks_model.forward(Xt, Yt.T)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(loss)
            if smallest_loss > loss:
                smallest_loss = loss
                best_alpha = zaks_model.alpha.clone()
            #     print("New Best")

        # print("Best Alpha Found: " + str(best_alpha))

        with torch.no_grad():
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
        self.loss = MAPEstimate.map_estimate_torch(X, Y.T, Xt, Yt.T, zaks_model.bias, zaks_model.alpha, noise,
                                                   torch.tensor(self.Sigma), space_kernel, time_kernel, kernel, alpha_mean,
                                                   alpha_variance, torch.tensor(np.kron(np.eye(len(space_X)), bias_kernel(time_X, time_X))),
                                                   len(space_X), len(time_X), theta_not)
