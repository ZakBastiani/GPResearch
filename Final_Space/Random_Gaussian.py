import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio
from mpl_toolkits.mplot3d import Axes3D

jitter = 0.00000001


class RandomGaussian:
    def __init__(self, space_range, time_range, N_space, N_time, space_kernel, time_kernel, kernel, sensor_points,
                 gt_sensor_points):

        # Displaying the function that we are trying to model.
        self.space_kernel = space_kernel
        self.time_kernel = time_kernel
        self.N_space = N_space
        self.N_time = N_time
        self.space = torch.stack((torch.linspace(0, space_range, N_space).repeat(N_space),
                                  torch.linspace(0, space_range, N_space).repeat(N_space, 1).T.flatten())).T
        self.time = torch.linspace(0, time_range, N_time)
        self.points = torch.cat((self.space.repeat(N_time, 1), self.time.repeat_interleave(N_space*N_space).repeat(1, 1).T), 1)
        all_points = torch.cat((self.points, sensor_points, gt_sensor_points), 0)

        kern = kernel(all_points, all_points) + jitter * torch.eye(len(all_points))

        self.dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(len(kern)), kern)
        self.ret_matrix = self.dist.sample()

        self.underlying_data = self.ret_matrix[:(N_time*N_space*N_space)].detach().numpy()
        self.sensor_data = self.ret_matrix[(N_time*N_space*N_space):(N_time*N_space*N_space + len(sensor_points))]
        self.gt_sensor_data = self.ret_matrix[(N_time*N_space*N_space + len(sensor_points)):]

    def display(self, title):
        plt.ion()
        fig = plt.figure(figsize=(8, 6), dpi=80)
        ax = fig.add_subplot(111, projection='3d')
        data = self.underlying_data.reshape((self.N_time, self.N_space, self.N_space))
        images = []
        time = self.time.detach().numpy()
        for k in range(0, self.N_time):
            ax.cla()
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Data')
            ax.set_title(
                'Basic random Gaussian function surface changing in time: ' + str("{:.2f}".format(time[k])))
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
        imageio.mimsave("graphs/" + str(title) + '.gif', images[1:], duration=10/self.N_time)
