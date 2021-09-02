import numpy as np
import matplotlib.pyplot as plt


class RandomGaussian:
    def __init__(self, space_range, time_range, N_space, N_time, space_kernel, time_kernel, mean, noise):
        # Displaying the function that we are trying to model.
        self.space_kernel = space_kernel
        self.time_kernel = time_kernel
        self.space = np.linspace(0, space_range, N_space)
        self.time = np.linspace(0, time_range, N_time)
        self.points = np.ndarray(shape=(N_space*N_time, 2))
        for i in range(N_space):
            for j in range(N_time):
                self.points[i*N_time + j][0] = self.space[i]
                self.points[i*N_time + j][1] = self.time[j]

        kern = np.kron(self.space_kernel(self.space, self.space), self.time_kernel(self.time, self.time))

        self.ret_matrix = np.random.multivariate_normal(mean, kern, 1)

        self.matrix2d = np.ndarray(shape=(N_space, N_time))
        for i in range(0, N_space):
            for j in range(0, N_time):
                self.matrix2d[i][j] = self.ret_matrix[0][i * N_time + j]

        self.L_function = np.linalg.cholesky(kern + noise**2 * np.eye(len(kern)))

    def function(self, space_points, time_points):
        Lk_function = np.linalg.solve(self.L_function, np.kron(self.space_kernel(self.space, space_points),
                                                               self.time_kernel(self.time, time_points)))
        mu_function = np.dot(Lk_function.T, np.linalg.solve(self.L_function, self.ret_matrix[0]))

        # Should just be able to use reshape fix later
        # mu = np.reshape([mu], (test_space, test_time))
        output = np.ndarray(shape=(len(space_points), len(time_points)))
        for i in range(0, len(space_points)):
            for j in range(0, len(time_points)):
                output[i][j] = mu_function[i * len(time_points) + j]
        return output

    def display(self, title):
        plt.figure(1)
        ax = plt.axes(projection='3d')
        ax.plot_surface(np.repeat([self.space], len(self.time), axis=0).T, np.repeat([self.time], len(self.space), axis=0),
                        self.matrix2d, cmap='viridis', edgecolor='none')
        ax.set_xlabel('Space')
        ax.set_ylabel('Time')
        ax.set_zlabel('Data')
        ax.set_title('Basic random Gaussian function surface')
