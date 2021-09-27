import numpy as np
import matplotlib.pyplot as plt
import imageio
from mpl_toolkits.mplot3d import Axes3D


class RandomGaussian:
    def __init__(self, space_range, time_range, N_space, N_time, space_kernel, time_kernel, mean, noise):
        # Displaying the function that we are trying to model.
        self.space_kernel = space_kernel
        self.time_kernel = time_kernel
        self.N_space = N_space
        self.N_time = N_time
        self.space = np.array(([np.linspace(0, space_range, N_space).repeat(N_space),
                                np.outer(np.linspace(0, space_range, N_space), np.ones(N_space)).T.flatten()])).T
        self.time = np.linspace(0, time_range, N_time)
        self.points = np.ndarray(shape=(N_space * N_space * N_time, 3))
        for i in range(N_space):
            for j in range(N_time):
                self.points[i * N_time + j][0] = self.space[i][0]
                self.points[i * N_time + j][1] = self.space[i][1]
                self.points[i * N_time + j][2] = self.time[j]

        kern = np.kron(self.space_kernel(self.space, self.space), self.time_kernel(self.time, self.time))

        self.ret_matrix = np.random.multivariate_normal(mean, kern, 1)

        self.matrix2d = np.ndarray(shape=(N_space, N_space, N_time))

        for i in range(0, N_space):
            for k in range(0, N_space):
                for j in range(0, N_time):
                    self.matrix2d[i][k][j] = self.ret_matrix[0][i * N_space * N_time + k * N_time + j]

        self.L_function = np.linalg.cholesky(kern + noise ** 2 * np.eye(len(kern)))

    def function(self, space_points, time_points, N_space):
        Lk_function = np.linalg.solve(self.L_function, np.kron(self.space_kernel(self.space, space_points),
                                                               self.time_kernel(self.time, time_points)))
        mu_function = np.dot(Lk_function.T, np.linalg.solve(self.L_function, self.ret_matrix[0]))

        # Should just be able to use reshape fix later
        # mu = np.reshape([mu], (test_space, test_time))
        output = np.ndarray(shape=(N_space, N_space, len(time_points)))
        for i in range(0, N_space):
            for k in range(0, N_space):
                for j in range(0, len(time_points)):
                    output[i][k][j] = mu_function[i * N_space * len(time_points) + k * len(time_points) + j]
        return output

    def display(self, title):
        plt.ion()
        fig = plt.figure(figsize=(8, 6), dpi=80)
        ax = fig.add_subplot(111, projection='3d')
        data = np.transpose(self.matrix2d, (2, 0, 1))
        images = []
        for k in range(0, len(self.matrix2d[0][0])):
            ax.cla()
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Data')
            ax.set_title(
                'Basic random Gaussian function surface changing in time: ' + str("{:.2f}".format(self.time[k])))
            ax.plot_surface(self.space.T[0].reshape(self.N_space, -1),
                            self.space.T[1].reshape(self.N_space, -1),
                            data[k],
                            cmap='viridis', edgecolor='none')
            ax.set_zlim([-4, 4])
            plt.draw()
            plt.tight_layout()
            plt.savefig("graphs/" + title + str(k))
            plt.pause(0.5)
            images.append(imageio.imread("graphs/" + title + str(k) + '.png'))
        imageio.mimsave("graphs/" + str(title) + '.gif', images[1:], duration=10/len(self.matrix2d[0][0]))
