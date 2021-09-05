import torch
import torch.nn as nn
import numpy as np
from Synthetic_Space import Gaussian_Process
import math

class OptChangingBiasAndAlpha(Gaussian_Process.GaussianProcess):
    def __init__(self, space_X, time_X, _Y, space_Xt, time_Xt, _Yt, space_kernel, time_kernel, kernel, noise, theta_not,
                 bias_variance, bias_mean, bias_kernel, alpha_mean, alpha_variance):

        # Opimization module used to maximaxize Wei's equation
        # should get similar results to the above model
        class zak_gpr(nn.Module):
            def __init__(self, X, Y, K, N_sensors, N_time):
                super(zak_gpr, self).__init__()
                self.X = X
                self.N_sensors = N_sensors
                self.Y = Y
                self.Sigma = torch.tensor(K)
                self.N_sensors = N_sensors
                self.N_time = N_time
                self.bias = nn.Parameter(torch.zeros((N_time*N_sensors, 1)))
                # self.bias = nn.Parameter(torch.tensor([actual_bias.flatten()]).T)
                self.alpha = nn.Parameter(torch.eye(1))

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

            # Maximizing the predicted bias based on direct function
            def forward(self, Xt, Yt):
                Sigma_hat = self.Sigma + noise*torch.eye(self.N_sensors*self.N_time)
                bias_sigma = torch.tensor(np.kron(np.eye(len(space_X)), bias_kernel(time_X, time_X))).float()

                chunk1 = -(1/2) * (torch.logdet(self.alpha**2 * Sigma_hat)
                                   + (self.Y - self.bias).T @ torch.cholesky_inverse(self.alpha**2 * Sigma_hat) @ (self.Y - self.bias)
                                   + self.N_sensors * math.log(2 * math.pi))
                # print("chunk2: " + str(chunk2))
                prob_a = -(1 / 2) * (((self.alpha - alpha_mean) ** 2 / alpha_variance) + math.log(alpha_variance * 2 * math.pi))
                chunk2 = -(1/2) * (torch.logdet(bias_sigma)
                                   + self.bias.T @ torch.cholesky_inverse(bias_sigma) @ self.bias
                                   + len(self.bias) * math.log(2 * math.pi)) + prob_a
                # chunk2 = -(1/2) * (math.log(bias_variance**2) + ((self.bias - bias_mean)**2)/(bias_variance**2)+ math.log(2 * math.pi))
                # print("chunk3: " + str(chunk3))

                chunk3 = 0
                for i in range(0, len(Xt)):
                    holder = self.mu(Xt[i], Sigma_hat)
                    var = self.v(Xt[i], Sigma_hat)
                    chunk3 += - (1/2) * (torch.log(var) + ((Yt[i] - holder)**2)/var + math.log(2 * math.pi))
                # print("chunk4: " + str(chunk4))

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
        optimizer = torch.optim.Adam(zaks_model.parameters(),
                                     lr=0.001)  # lr is very important, lr>0.1 lead to failure
        smallest_loss = 1000
        guess_bias = []
        for i in range(500):
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

        self.type = "Gaussian Process Regression optimizing bias and alpha"
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

