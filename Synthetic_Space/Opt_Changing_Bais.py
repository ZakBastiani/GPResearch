import torch
import torch.nn as nn
import numpy as np
from Synthetic_Space import Gaussian_Process
import math

class OptChangingBias(Gaussian_Process.GaussianProcess):
    def __init__(self, space_X, time_X, _Y, space_Xt, time_Xt, _Yt, space_kernel, time_kernel, kernel, noise, theta_not,
                 bias_variance, bias_mean, bias_kernel, alpha):

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
                # self.bias = nn.Parameter(torch.zeros((N_time*N_sensors, 1)))
                self.bias = nn.Parameter(torch.tensor([bias_kernel.flatten()]).T)
                self.alpha = torch.tensor(alpha)

            def v(self, x, Sigma_hat):
                k = kernel([x.detach().numpy()], self.X.detach().numpy()).T
                k = torch.from_numpy(k)
                output = theta_not - k.T @ torch.cholesky_inverse(Sigma_hat) @ k
                return output

            def mu(self, x, Sigma_hat):
                k = kernel([x.detach().numpy()], self.X.detach().numpy()).T
                k = torch.from_numpy(k)
                holder = ((self.Y - self.bias) / self.alpha)
                return k.T @ torch.linalg.inv(Sigma_hat) @ holder

            # Maximizing the predicted bias based on direct function
            def forward(self, Xt, Yt):
                Sigma_hat = self.Sigma + noise*torch.eye(self.N_sensors*self.N_time)

                chunk1 = -(1/2) * torch.logdet(Sigma_hat)  # currently giving -inf or inf
                # print("chunk1: " + str(chunk1))
                chunk2 = -(1/2) * ((self.Y - self.bias)/self.alpha).T @ torch.cholesky_inverse(Sigma_hat) @ ((self.Y - self.bias)/self.alpha)
                # print("chunk2: " + str(chunk2))
                prob_b = -(1/2) * ((((self.bias - bias_mean) ** 2) / bias_variance**2)
                                   + math.log(bias_variance**2))
                chunk3 = torch.sum(prob_b)/len(prob_b)
                # print("chunk3: " + str(chunk3))

                chunk4 = 0
                for i in range(0, len(Xt)):
                    holder = self.mu(Xt[i], Sigma_hat)
                    var = self.v(Xt[i], Sigma_hat)
                    chunk4 += -(1/2) * (torch.log(var) + ((Yt[i] - holder)**2)/var)
                # print("chunk4: " + str(chunk4))

                return chunk2 + chunk3 + chunk4  # Add back chunk1


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
        optimizer = torch.optim.Adam(zaks_model.parameters(),
                                     lr=0.01)  # lr is very important, lr>0.1 lead to failure
        smallest_loss = 1000
        for i in range(100):
            optimizer.zero_grad()
            loss = -zaks_model.forward(Xt, Yt.T)
            if loss < smallest_loss:
                smallest_loss = loss
                # print(loss)
            loss.backward()
            optimizer.step()
        # print("Smallest Loss:" + str(smallest_loss))
        with torch.no_grad():
            holder = zaks_model(Xt, Yt.T)

        self.type = "Gaussian Process Regression optimizing bias given alpha"
        self.space_X = space_X  # np.concatenate((space_X, space_Xt))
        self.time_X = time_X
        self.alpha = zaks_model.alpha.detach().numpy()
        self.bias = np.reshape(zaks_model.bias.detach().numpy(), (-1, len(time_X)))
        self.Y = (_Y - self.bias) / self.alpha  # np.concatenate(((_Y - self.bias) / self.alpha, _Yt))
        self.noise = noise
        self.space_kernel = space_kernel
        self.time_kernel = time_kernel
        self.Sigma = np.kron(self.space_kernel(self.space_X, self.space_X), self.time_kernel(self.time_X, self.time_X))
        self.L = np.linalg.cholesky(self.Sigma + noise * np.eye(len(self.Sigma)))

