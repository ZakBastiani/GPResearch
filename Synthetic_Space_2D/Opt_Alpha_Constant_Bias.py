import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
from Synthetic_Space_2D import Gaussian_Process


class OptAlphaCalcBias(Gaussian_Process.GaussianProcess):
    def __init__(self, space_X, time_X, _Y, space_Xt, time_Xt, _Yt, space_kernel, time_kernel,
                 kernel, noise, theta_not, bias_variance, bias_mean, alpha_variance, alpha_mean):

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
                self.bias = torch.zeros(N_sensors)
                # self.alpha = nn.Parameter(torch.tensor(alpha))
                self.alpha = nn.Parameter(torch.eye(1))

                # Maximizing the predicted bias based on direct function

            def calc_bias_matrix(self, sigma, X, Y, true_X, true_Y, alpha):
                sigma = sigma.detach().numpy()
                X = X.detach().numpy()
                Y = Y.detach().numpy()
                N_sensors = len(space_X)
                N_time = len(time_X)
                true_X = true_X.detach().numpy()
                true_Y = true_Y.detach().numpy()
                alpha = alpha.detach().numpy()
                sigma_inv = np.linalg.inv(sigma + noise * np.eye(len(sigma)))

                Y = _Y.flatten()
                true_Y = _Yt.flatten()

                def k_star_calc(x):
                    k = kernel(x, X.T)
                    return k

                # Build and calc A and C
                A = np.zeros(shape=(N_sensors, N_sensors))
                C = np.zeros(shape=(1, N_sensors))
                for k in range(0, N_sensors):
                    extend_bias = np.zeros(shape=(N_sensors * N_time, 1))
                    for j in range(0, N_time):
                        extend_bias[k * N_time + j][0] = 1
                    current_A = np.zeros(shape=(1, N_sensors * 10))
                    current_C = 0
                    for n in range(len(true_X.T)):
                        k_star = kernel([true_X.T[n]], X.T).T
                        holder = (k_star.T @ sigma_inv @ k_star)[0][0]
                        holder2 = (k_star.T @ sigma_inv @ extend_bias) * (k_star.T @ sigma_inv)
                        current_A += holder2 / (theta_not - holder)
                        current_C += ((k_star.T @ sigma_inv @ Y) * (k_star.T @ sigma_inv @ extend_bias)
                                      - alpha * true_Y[n] * (k_star.T @ sigma_inv @ extend_bias)) / (
                                             theta_not - holder)
                    current_A += (sigma_inv @ extend_bias).T
                    # Need to condense current_A into b_i variables
                    for i in range(0, N_sensors):
                        sum = 0
                        for j in range(0, N_time):
                            sum += current_A[0][i * N_time + j]
                        A[k][i] = sum
                    A[k][k] += 1 / bias_variance

                    C[0][k] = Y.T @ sigma_inv @ extend_bias + current_C + bias_mean / bias_variance

                # Inverse A and multiply it by C
                A_inverse = np.linalg.inv(A)
                b = C @ A_inverse
                return torch.from_numpy(b)

            def forward(self, Xt, Yt):
                Sigma_hat = (self.alpha ** 2) * self.Sigma + torch.eye(len(self.Sigma)) * noise
                self.bias = self.calc_bias_matrix(self.Sigma, self.X.T, self.Y, Xt.T, Yt, self.alpha)
                extend_bias = torch.reshape(self.bias.repeat(self.N_time, 1).T, (-1, 1))

                chunk1 = (1 / 2) * torch.log(torch.det(Sigma_hat))  # currently giving -inf or inf
                # print("chunk1: " + str(chunk1))
                chunk2 = -(1 / 2) * (self.Y - extend_bias).T @ torch.cholesky_inverse(Sigma_hat) @ (
                        self.Y - extend_bias)
                # print("chunk2: " + str(chunk2))
                prob_a = -(1 / 2) * ((self.alpha - alpha_mean) ** 2 / alpha_variance) + math.log(
                    (alpha_variance * math.sqrt(2 * math.pi)))
                prob_b = -(1 / 2) * ((self.bias - bias_mean) ** 2 / bias_variance) + math.log(
                    (bias_variance * math.sqrt(2 * math.pi)))
                chunk3 = -(self.N_sensors / 2) * math.log(2 * math.pi) + prob_a + torch.sum(prob_b) / len(
                    prob_b)  # fix later
                # print("chunk3: " + str(chunk3))

                chunk4 = 0

                def v(x):
                    k = np.kron(space_kernel(np.array([[x[0]]]),
                                             np.array(torch.unique(self.X.T[0]))),
                                time_kernel(np.array([[x[1]]]),
                                            np.array(torch.unique(self.X.T[1]))))
                    k = torch.tensor(k)
                    output = theta_not / self.alpha ** 2 - k @ torch.cholesky_inverse(Sigma_hat) @ k.T
                    if 0 > output:
                        # print("Negative variance of " + str(output))
                        return abs(output)
                    return output

                def mu(x):
                    k = np.kron(space_kernel(np.array([[x[0]]]),
                                             np.array(torch.unique(self.X.T[0]))),
                                time_kernel(np.array([[x[1]]]),
                                            np.array(torch.unique(self.X.T[1]))))
                    k = torch.tensor(k)
                    return k @ torch.cholesky_inverse(Sigma_hat) @ (self.Y - extend_bias)

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
        self.space_X = np.concatenate((space_X, space_Xt))
        self.time_X = time_X
        self.alpha = zaks_model.alpha.detach().numpy()
        self.bias = zaks_model.bias.detach().numpy()
        self.Y = np.concatenate(((_Y - np.outer(self.bias.T, np.ones(len(time_X)))) / self.alpha, _Yt))
        self.noise = noise
        self.space_kernel = space_kernel
        self.time_kernel = time_kernel
        self.Sigma = np.kron(self.space_kernel(self.space_X, self.space_X), self.time_kernel(self.time_X, self.time_X))
        self.L = np.linalg.cholesky(self.Sigma * self.alpha ** 2 + noise * np.eye(len(self.Sigma)))


