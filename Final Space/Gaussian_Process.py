import numpy as np
import matplotlib.pyplot as plt
import imageio
import torch
import math
from Synthetic_Space_2D import MAPEstimate


class GaussianProcess:
    def __init__(self, space_X, time_X, _Y, space_Xt, time_Xt, _Yt, space_kernel, time_kernel, kernel, noise, N_space,
                 alpha_variance, alpha_mean, bias_kernel, theta_not):
        self.type = "Basic Gaussian Process Regression"
        self.space_X = space_X  # np.concatenate((space_X, space_Xt))
        self.time_X = time_X
        self.Y = _Y  # np.concatenate((_Y, _Yt))
        self.alpha = 1
        self.bias = np.zeros(shape=(N_space, N_space, len(time_X)))
        self.noise = noise
        self.space_kernel = space_kernel
        self.time_kernel = time_kernel
        self.Sigma = np.kron(self.space_kernel(self.space_X, self.space_X), self.time_kernel(self.time_X, self.time_X))
        self.L = np.linalg.cholesky(self.Sigma + noise * np.eye(len(self.Sigma)))

        # Need to alter the sensor matrix and the data matrix
        X = np.concatenate((np.repeat(space_X, len(time_X), axis=0),
                            np.tile(time_X, len(space_X)).reshape((-1, 1))), axis=1)
        Y = _Y.flatten()

        Xt = np.concatenate((np.repeat(space_Xt, len(time_Xt), axis=0),
                             np.tile(time_Xt, len(space_Xt)).reshape((-1, 1))), axis=1)
        Yt = _Yt.flatten()

        self.loss = self.loss = MAPEstimate.map_estimate_numpy(X, Y, Xt, Yt, self.bias.flatten(), self.alpha, noise,
                                                               self.Sigma, space_kernel, time_kernel, kernel, alpha_mean,
                                                               alpha_variance,
                                                               np.kron(np.eye(len(space_X)), bias_kernel(time_X, time_X)),
                                                               len(space_X), len(time_X), theta_not)

    def build(self, space_points, time_points, N_space):

        Lk = np.linalg.solve(self.L, np.kron(self.space_kernel(self.space_X, space_points),
                                             self.time_kernel(self.time_X, time_points)))
        mu = np.dot(Lk.T, np.linalg.solve(self.L, self.Y.flatten()))

        # Should just be able to use reshape fix later
        # mu = np.reshape([mu], (test_space, test_time))
        holder = np.ndarray(shape=(N_space, N_space, len(time_points)))
        for i in range(0, N_space):
            for k in range(0, N_space):
                for j in range(0, len(time_points)):
                    holder[i][k][j] = mu[i * N_space * len(time_points) + k * len(time_points) + j]
        return holder

    @staticmethod
    def display(space_points, N_space, time_points, data, title):
        plt.ion()
        fig = plt.figure(figsize=(8, 6), dpi=80)
        ax = fig.add_subplot(111, projection='3d')
        images = []
        data = np.transpose(data, (2, 0, 1))
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

