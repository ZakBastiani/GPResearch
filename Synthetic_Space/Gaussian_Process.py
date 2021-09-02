import numpy as np
import matplotlib.pyplot as plt


class GaussianProcess:
    def __init__(self, space_X, time_X, _Y, space_Xt, time_Xt, _Yt, space_kernel, time_kernel, noise):
        self.type = "Basic Gaussian Process Regression"
        self.space_X = space_X  # np.concatenate((space_X, space_Xt))
        self.time_X = time_X
        self.Y = _Y  # np.concatenate((_Y, _Yt))
        self.alpha = 1
        self.bias = np.zeros(shape=(len(space_X), len(time_X)))
        self.noise = noise
        self.space_kernel = space_kernel
        self.time_kernel = time_kernel
        self.Sigma = np.kron(self.space_kernel(self.space_X, self.space_X), self.time_kernel(self.time_X, self.time_X))
        self.L = np.linalg.cholesky(self.Sigma + noise * np.eye(len(self.Sigma)))

    def build(self, space_points, time_points):

        Lk = np.linalg.solve(self.L, np.kron(self.space_kernel(self.space_X, space_points),
                                             self.time_kernel(self.time_X, time_points)))
        mu = np.dot(Lk.T, np.linalg.solve(self.L, self.Y.flatten()))

        # Should just be able to use reshape fix later
        # mu = np.reshape([mu], (test_space, test_time))
        holder = np.ndarray(shape=(len(space_points), len(time_points)))
        for i in range(0, len(space_points)):
            for j in range(0, len(time_points)):
                holder[i][j] = mu[i * len(time_points) + j]
        return holder

    @staticmethod
    def display(space_points, time_points, data, title):
        # PLOTS:
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(np.repeat([space_points], len(time_points), axis=0).T,
                        np.repeat([time_points], len(space_points), axis=0),
                        data,
                        cmap='viridis', edgecolor='none')
        ax.set_xlabel('Space')
        ax.set_ylabel('Time')
        ax.set_zlabel('Data')
        ax.set_title(title)

    def print_error(self, true_alpha, true_bias, true_y, guess_y, gt_data, gt_guess):
        print("-------" + self.type + "-------")
        print("True Alpha: " + str(true_alpha))
        print("Guess Alpha: " + str(self.alpha))
        # print("Actual Bias: " + str(true_bias))
        # print("Guess Bias: " + str(self.bias))
        print("Avg L2 Bias Error: " + str(
            sum(((self.bias - true_bias)**2).flatten()) / len(true_bias.flatten())))
        gt_error = sum(((gt_data - gt_guess)**2).flatten()) / (len(gt_guess.flatten()))
        print("Avg L2 error at Ground Truth Points: " + str(gt_error))
        error = sum(((true_y - guess_y)**2).flatten()) / (len(guess_y.flatten()))
        print("Avg L2 error at Test Points: " + str(error))
        print("")

