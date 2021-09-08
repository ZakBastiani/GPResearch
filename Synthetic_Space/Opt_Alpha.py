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
                Sigma_hat = self.Sigma + torch.eye(len(self.Sigma)) * noise**2
                chunk1 = -(1/2) * (len(Sigma_hat)*torch.log(self.alpha**2) + torch.log(torch.det(Sigma_hat))
                                   + ((self.Y - self.bias).T @ torch.inverse(Sigma_hat) @ (self.Y - self.bias))/(self.alpha**2)
                                   + len(Sigma_hat) * math.log(2 * math.pi))
                # print("chunk1: " + str(chunk1))
                chunk2 = -(1 / 2) * (((self.alpha - alpha_mean) ** 2 / alpha_variance) + math.log(alpha_variance * 2 * math.pi))
                # print("chunk2: " + str(chunk2))

                chunk3 = 0
                for i in range(0, len(Xt)):
                    holder = self.mu(Xt[i], Sigma_hat)
                    var = self.v(Xt[i], Sigma_hat)
                    chunk3 += -(1/2) * (torch.log(var) + ((Yt[i] - holder)**2)/var + math.log(2 * math.pi))
                # print("chunk3: " + str(chunk3))

                return chunk1 + chunk2 + chunk3

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
        for i in range(200):
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


