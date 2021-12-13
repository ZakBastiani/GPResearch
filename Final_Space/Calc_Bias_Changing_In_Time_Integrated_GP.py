import math
import matplotlib.pyplot as plt
import torch
from Final_Space import Gaussian_Process
from Final_Space import MAPEstimate


class ChangingBiasIntGP(Gaussian_Process.GaussianProcess):
    def __init__(self, space_X, time_X, _Y, space_Xt, time_Xt, _Yt, space_kernel, time_kernel, kernel, noise_sd, theta_not,
                bias_kernel, alpha, alpha_mean, alpha_sd, true_bias):
        torch.set_default_dtype(torch.float64)
        self.points = torch.cat((space_X.repeat(len(time_X), 1),
                                 time_X.repeat_interleave(len(space_X)).repeat(1, 1).T), 1)
        sigma = kernel(self.points, self.points)
        sigma_inv = torch.linalg.inv(alpha**2 * sigma + (noise_sd ** 2) * torch.eye(len(sigma)))

        bias_sigma = torch.kron(torch.eye(len(space_X)), bias_kernel(time_X, time_X))

        N_sensors = len(space_X)
        N_time = len(time_X)

        # Need to alter the sensor matrix and the data matrix
        X = torch.cat((space_X.repeat(len(time_X), 1),
                       time_X.repeat_interleave(len(space_X)).repeat(1, 1).T), 1)
        Y = _Y.flatten()

        Xt = torch.cat((space_Xt.repeat(len(time_Xt), 1),
                        time_Xt.repeat_interleave(len(space_Xt)).repeat(1, 1).T), 1)
        Yt = _Yt.flatten()

        # Build and calc A and C MOVE THE ALPHAS INTO THE KERNEL
        A = torch.zeros((N_sensors * N_time, N_sensors * N_time))
        C = torch.zeros((1, N_sensors * N_time))
        current_C = 0
        for n in range(len(Xt)):
            k_star = alpha * kernel(Xt[n].unsqueeze(0), X)
            holder = (k_star.T @ sigma_inv @ k_star)[0][0]
            holder2 = (k_star.T @ sigma_inv).T @ (k_star.T @ sigma_inv)
            A += holder2 / (theta_not - holder)
            current_C += ((k_star.T @ sigma_inv @ Y) * (k_star.T @ sigma_inv)
                          - Yt[n] * (k_star.T @ sigma_inv)) / (theta_not - holder)
        A += sigma_inv.T

        A += torch.linalg.inv(bias_sigma)
        C[0] = Y.T @ sigma_inv + current_C

        # Inverse A and multiply it by C
        A_inverse = torch.linalg.inv(A)
        b = C @ A_inverse

        print(torch.norm(b))
        print("Error: " + str(torch.norm(C - (b @ A))))
        print('True Bias Error: ' + str(torch.norm(C - (true_bias.T @ A))))

        self.type = "Gaussian Process Regression with a calculated changing bias using an integrated GP and a provided alpha"
        self.space_X = space_X # np.concatenate((space_X, space_Xt))
        self.time_X = time_X
        self.alpha = alpha
        self.bias = b
        self.Y = (_Y - self.bias)/self.alpha # np.concatenate(((_Y - self.bias)/self.alpha, _Yt))
        self.noise_sd = noise_sd
        self.space_kernel = space_kernel
        self.time_kernel = time_kernel
        self.kernel = kernel
        self.Sigma = self.kernel(self.points, self.points)
        self.L = torch.linalg.cholesky(self.Sigma + noise_sd ** 2 * torch.eye(len(self.points)))
        self.loss = MAPEstimate.map_estimate_torch(X, Y, Xt, Yt, self.bias.flatten(), alpha, noise_sd,
                                                   self.Sigma, space_kernel, time_kernel, kernel, alpha_mean, alpha_sd,
                                                   torch.kron(torch.eye(len(space_X)), bias_kernel(time_X, time_X)),
                                                   len(space_X), len(time_X), theta_not)

        # # Building a graph showing the loss function for values of alpha to see how good our calc_both is
        # alpha_range = torch.linspace(0.9, 1.1, 100)
        # y = []
        # ll = 0
        # for a in alpha_range:
        #     noise_lag = noise_sd / a
        #     sigma_inv = torch.linalg.inv(sigma + (noise_lag ** 2) * torch.eye(len(sigma)))
        #     # Build and calc A and C
        #     A = torch.zeros((N_sensors * N_time, N_sensors * N_time))
        #     C = torch.zeros((1, N_sensors * N_time))
        #     current_C = 0
        #     for n in range(len(Xt)):
        #         k_star = kernel(Xt[n].unsqueeze(0), X)
        #         holder = (k_star.T @ sigma_inv @ k_star)[0][0]
        #         holder2 = (k_star.T @ sigma_inv).T @ (k_star.T @ sigma_inv)
        #         A += holder2 / (theta_not - holder)
        #         current_C += ((k_star.T @ sigma_inv @ Y) * (k_star.T @ sigma_inv)
        #                       - a * Yt[n] * (k_star.T @ sigma_inv)) / (theta_not - holder)
        #     A += (sigma_inv).T
        #
        #     A += (a ** 2) * torch.linalg.inv(bias_sigma)
        #     C[0] = Y.T @ sigma_inv + current_C
        #
        #     # Inverse A and multiply it by C
        #     A_inverse = torch.linalg.inv(A)
        #     b = C @ A_inverse
        #
        #     l = MAPEstimate.map_estimate_torch(X, Y, Xt, Yt, b.flatten(), a, noise_sd,
        #                                        sigma, space_kernel,
        #                                        time_kernel, kernel, alpha_mean, alpha_sd,
        #                                        torch.kron(torch.eye(len(space_X)), bias_kernel(time_X, time_X)),
        #                                        len(space_X), len(time_X), theta_not)
        #     y.append(l)
        #     if l > ll:
        #         ll = l
        # fig, ax = plt.subplots(figsize=(12, 6))
        # ax.plot(alpha_range, y, 'b-')
        # ax.plot(self.alpha, self.loss, marker='o')
        # # plt.ylim([1000, 2500])
        # plt.title("Calc loss based on a normal alpha in calc both")
        # plt.show()
        # print(ll)



