import numpy as np
import matplotlib.pyplot as plt
import imageio
import torch
import math
from Final_Space import MAPEstimate


class GaussianProcess:
    def __init__(self, space_X, time_X, _Y, space_Xt, time_Xt, _Yt, space_kernel, time_kernel, kernel, noise_sd, N_space,
               alpha, alpha_mean, alpha_sd, bias_kernel, theta_not):
        torch.set_default_dtype(torch.float64)
        self.type = "Basic Gaussian Process Regression"
        self.space_X = space_X  # np.concatenate((space_X, space_Xt))
        self.time_X = time_X
        self.Y = _Y  # np.concatenate((_Y, _Yt))
        self.alpha = torch.tensor(alpha_mean)
        self.bias = torch.zeros((N_space * len(time_X)))
        self.noise_sd = noise_sd
        self.space_kernel = space_kernel
        self.time_kernel = time_kernel
        self.kernel = kernel
        self.points = torch.cat((self.space_X.repeat(len(self.time_X), 1), self.time_X.repeat_interleave(N_space).repeat(1, 1).T), 1)
        self.Sigma = (alpha**2)*kernel(self.points, self.points) + noise_sd ** 2 * torch.eye(len(space_X)*len(time_X))
        self.L = torch.linalg.cholesky(self.Sigma)

        # Need to alter the sensor matrix and the data matrix
        X = torch.cat((space_X.repeat(len(time_Xt), 1),
                       time_Xt.repeat_interleave(len(space_X)).repeat(1, 1).T), 1)
        Y = _Y.flatten()

        Xt = torch.cat((space_Xt.repeat(len(time_X), 1),
                        time_X.repeat_interleave(len(space_Xt)).repeat(1, 1).T), 1)
        Yt = _Yt.flatten()

        self.loss = MAPEstimate.map_estimate_torch(X, Y, Xt, Yt, self.bias.flatten(), self.alpha, noise_sd,
                                                   self.Sigma, space_kernel, time_kernel, kernel, alpha_mean, alpha_sd,
                                                   torch.kron(torch.eye(len(space_X)), bias_kernel(time_X, time_X)),
                                                   len(space_X), len(time_X), theta_not)

    def build(self, space_points, time_points):
        points = torch.cat((space_points.repeat(len(time_points), 1), time_points.repeat_interleave(len(space_points)).repeat(1, 1).T), 1)
        Lk = torch.linalg.solve(self.L, self.kernel(points, self.points))
        mu = Lk.T @ torch.linalg.solve(self.L, self.Y.flatten())

        return mu.detach().numpy()

    @staticmethod
    def display(space_points, time_points, data, N_space, N_time, title):
        plt.ion()
        fig = plt.figure(figsize=(8, 6), dpi=80)
        data = data.reshape(N_space, N_space, N_time)
        ax = fig.add_subplot(111, projection='3d')
        images = []
        for k in range(0, len(data)):
            ax.cla()
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Data')
            ax.set_title(title + ": " + str("{:.2f}".format(time_points[k])))
            ax.plot_surface(space_points.T[0].reshape(N_space, -1),
                            space_points.T[1].reshape(N_space, -1),
                            data[k],
                            cmap='viridis', edgecolor='none')
            ax.set_zlim([-4, 4])
            plt.draw()
            plt.tight_layout()
            plt.savefig("graphs/" + title + str(k))
            plt.pause(0.5)
            images.append(imageio.imread("graphs/" + title + str(k) + '.png'))
        imageio.mimsave("graphs/" + str(title) + '.gif', images[1:], duration=10/len(data))

    def print_error(self, true_alpha, true_bias, true_y, guess_y, gt_data, gt_guess):
        print("-------" + self.type + "-------")
        print("True Alpha: " + str(true_alpha))
        print("Guess Alpha: " + str(self.alpha))
        alpha_error = abs(true_alpha - self.alpha)
        # print("Actual Bias: " + str(true_bias))
        # print("Guess Bias: " + str(self.bias))
        bias_error = sum(((self.bias - true_bias) ** 2).flatten()) / len(true_bias.flatten())
        print("Avg L2 Bias Error: " + str(bias_error))
        gt_error = sum(((gt_data - gt_guess) ** 2).flatten()) / (len(gt_guess.flatten()))
        print("Avg L2 error at Ground Truth Points: " + str(gt_error))
        error = sum(((true_y - guess_y) ** 2).flatten()) / (len(guess_y.flatten()))
        print("Avg L2 error at Test Points: " + str(error))
        print("Loss: " + str(self.loss))
        print("")

        return np.array([alpha_error, bias_error, gt_error, error, float(self.loss)], dtype=float)

