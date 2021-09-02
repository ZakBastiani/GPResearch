import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from Synthetic_Space_2D import Gaussian_Process


class OptBoth(Gaussian_Process.GaussianProcess):
    def __init__(self, space_X, time_X, _Y, space_Xt, time_Xt, _Yt,
                 space_kernel, time_kernel, kernel, noise, theta_not, bias_variance, bias_mean, alpha_mean, alpha_variance):
        N_sensors = len(space_X)
        N_time = len(time_X)
        N_true_sensors = len(space_Xt)
        # Need to alter the sensor matrix and the data matrix
        X = torch.tensor([np.outer(space_X, np.ones(N_time)).flatten(),
                          np.outer(time_X, np.ones(N_sensors)).T.flatten()]).T
        Y = torch.reshape(torch.tensor(_Y), (-1, 1))

        Xt = torch.tensor([np.outer(space_Xt, np.ones(N_time)).flatten(),
                           np.outer(time_Xt, np.ones(N_true_sensors)).T.flatten()]).T
        Yt = torch.reshape(torch.tensor(_Yt), (-1, 1))

        jitter = noise

        class gpr(nn.Module):
            def __init__(self, X, Y):  # Basic constructor
                super(gpr, self).__init__()
                self.X = X
                self.Y = Y
                self.log_beta = nn.Parameter(torch.zeros(1))
                self.log_length_scale = nn.Parameter(torch.zeros(X.size(1)))
                self.log_scale = nn.Parameter(torch.zeros(1))
                self.bias = nn.Parameter(torch.zeros(N_sensors))
                self.gain = nn.Parameter(torch.ones(1))

            def K_cross(self, X, X2):  # Building K, which is used to calculate sigma
                length_scale = torch.exp(self.log_length_scale).view(1, -1)

                X = X / length_scale.expand(X.size(0), -1)
                X2 = X2 / length_scale.expand(X2.size(0), -1)

                X_norm2 = torch.sum(X * X, dim=1).view(-1, 1)
                X2_norm2 = torch.sum(X2 * X2, dim=1).view(-1, 1)

                K = -2.0 * X @ X2.t() + X_norm2.expand(
                    X.size(0), X2.size(0)) + X2_norm2.t().expand(
                    X.size(0), X2.size(0))
                K = self.log_scale.exp() * torch.exp(-K)
                return K

            def forward(self, Xte):  # Moving forward one step
                # with torch.no_grad():
                n_test = Xte.size(0)
                Sigma = self.K_cross(
                    self.X, self.X) + torch.exp(self.log_beta).pow(-1) * torch.eye(
                    self.X.size(0)) + jitter * torch.eye(self.X.size(0))
                kx = self.K_cross(Xte, self.X)

                y_bias = self.bias.view(-1, 1).repeat(N_time, 1)
                Y = self.Y * self.gain - y_bias
                # via cholesky decompositon
                L = torch.cholesky(Sigma)
                mean = kx @ torch.cholesky_solve(Y, L)
                alpha = L.inverse() @ kx.t()
                var_diag = self.log_scale.exp().expand(
                    n_test, 1) - (alpha.t() @ alpha).diag().view(-1, 1)

                return mean, var_diag

            def neg_log_likelihood(self):
                Sigma = self.K_cross(
                    self.X, self.X) + torch.exp(self.log_beta).pow(-1) * torch.eye(
                    self.X.size(0)) + jitter * torch.eye(self.X.size(0))
                y_bias = self.bias.view(-1, 1).repeat(N_time, 1)
                Y = self.Y * self.gain - y_bias
                prob = torch.distributions.multivariate_normal.MultivariateNormal(
                    torch.zeros(self.X.size(0)), Sigma)
                return -prob.log_prob(Y.t())

        # setting the model and then using torch to optimize
        model = gpr(X, Y)
        # optimizer = torch.optim.LBFGS(model.parameters(), lr=0.001)  #lr is very important, lr>0.1 lead to failure
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        lossFunc = nn.MSELoss()
        for i in range(100):
            # optimizer.zero_grad()
            # LBFGS
            def closure():
                optimizer.zero_grad()
                loss = model.neg_log_likelihood()
                print('nll:', loss.item())
                loss.backward()
                return loss

            # optimizer.step(closure)

            # adam
            # check loss functions as they are returning arrays
            optimizer.zero_grad()
            # loss1 = model.neg_log_likelihood()/3      %average nll
            loss1 = model.neg_log_likelihood()

            ypred, yvar = model(Xt)
            prob = torch.distributions.multivariate_normal.MultivariateNormal(
                ypred.t().squeeze(),
                yvar.squeeze().diag_embed())
            loss2 = -prob.log_prob(Yt.t().squeeze())

            if loss2 < 0:
                loss = loss1 + loss2
            else:
                loss = loss2

            # loss =  loss1 + loss2
            loss.backward()
            optimizer.step()
            print(
                'loss1:',
                loss.item(),
                'loss2:',
                loss2.item(),
            )

        self.type = "Gaussian Process Regression optimizing both bias and alpha"
        self.space_X = np.concatenate((space_X, space_Xt))
        self.time_X = time_X
        self.alpha = model.gain.detach().numpy()
        self.bias = model.bias.detach().numpy()
        self.Y = np.concatenate(((_Y - np.outer(self.bias.T, np.ones(N_time))) / self.alpha, _Yt))
        self.noise = noise
        self.space_kernel = space_kernel
        self.time_kernel = time_kernel
        self.Sigma = np.kron(self.space_kernel(self.space_X, self.space_X), self.time_kernel(self.time_X, self.time_X))
        self.L = np.linalg.cholesky(self.Sigma + noise * np.eye(len(self.Sigma)))