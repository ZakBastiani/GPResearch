import numpy as np
import numpy.polynomial.legendre as lg
import matplotlib.pyplot as plt
import numpy.random as random

import scipy.spatial.distance as distance
import scipy.spatial as spatial
import math as math


class LegendreRandom():
    def __init__(self, kernel, std_devs):
        self._kernel = kernel
        self._std_devs = std_devs

    def samples(self, X, num_samples):
        num_bases = self._std_devs.shape[0]
        num_points = X.shape[0]
        normalizations = np.zeros(num_bases)
        for i in range(num_bases):
            normalizations[i] = math.sqrt((2.0 * i + 1.0) / 2.0)
        mean = np.zeros(num_points)
        ret_matrix = np.random.multivariate_normal(mean, self._kernel, num_samples)
        for i in range(num_samples):
            weights = math.sqrt(2.0 / num_points) * random.randn(num_bases)
            weights = np.multiply(std_devs, weights)
            weights = np.multiply(normalizations, weights)
            ret_matrix[i, :] += lg.legval(X, weights)
        return ret_matrix

    def bases(self, X):
        # only display the bases that have nonzero coefficients
        num_bases = int(np.sum((std_devs > 0.0).astype('float32')))
        num_points = X.shape[0]
        curves_return = np.zeros((num_points, num_bases))
        this_curve_idx = 0
        for i in range(self._std_devs.shape[0]):
            curves_weights = np.zeros(self._std_devs.shape[0])
            normalization = math.sqrt((2.0 * i + 1.0) / 2.0)
            if (self._std_devs[i] != 0.0):
                print("got basis at location " + str(i))
                curves_weights[i] = (2.0 * normalization / num_points)
                curves_return[:, this_curve_idx] = lg.legval(X, curves_weights)
                this_curve_idx += 1
        return curves_return


# This produces samples where noise is orthongal to signal
# def samples_ind(self, X, num_samples):
#     num_points = X.shape[0]
#     num_bases = self._std_devs.shape[0]
#     for i in range(num_bases):
#         if self._std_devs[i] > 0.0


#     mean = np.zeros(num_points)
#     ret_matrix = np.random.multivariate_normal(mean, self._kernel, num_samples)
#     for i in range(num_samples):
#         weights = random.randn(num_bases)
#         weights = np.multiply(std_devs,weights)
#         ret_matrix[i,:] += lg.legval(X, weights)
#     return ret_matrix


def distanceSqMatrix(X, Y=[]):
    data_size = X.shape[0]
    if len(Y) == 0:
        # print(X.reshape(-1, 1))
        new_matrix = distance.squareform(np.power(distance.pdist(X.reshape(-1, 1)), 2))
        # matrix = distance.pdist(X)
        # new_matrix = np.zeros((data_size, data_size))
        # for i in range(data_size):
        #     for j in range(data_size):
        #         if (i == j):
        #             new_matrix[i, j] = 0.0
        #         elif (i > j):
        #             new_matrix[i, j] = (matrix[(data_size*j - int(j*(j+1)/2)) + i - (j+1)])**2
        #         else:
        #             new_matrix[i, j] = (matrix[(data_size*i - int(i*(i+1)/2)) + j - (i+1)])**2
    else:
        new_matrix = np.power(distance.cdist(X, Y), 2)

    return new_matrix


def GP_Kernel(X, band_width, sigma_smooth, sigma):
    band_var = band_width * band_width
    return ((sigma_smooth ** 2) * np.exp(distanceSqMatrix(X) / (-2.0 * band_var)) + (sigma ** 2) * np.identity(
        X.shape[0]))


class GP_PCA_EM_solver:
    def __init__(self, kernel, Y_data):
        self._kernel_inv = np.linalg.inv(kernel)
        # this needs to be fixed -- the data seems to be given wrongly.
        self._Y_data = Y_data.transpose()
        self._cov = (1.0 / Y_data.shape[1]) * (Y_data @ Y_data.transpose())
        self._d = (self._Y_data).shape[0]

    def latent_expectation(self, current_W, current_sigma_sq):
        W = current_W
        num_samples = self._Y_data.shape[1]
        num_bases = current_W.shape[1]
        d = current_W.shape[0]
        M_K_sigma = W.transpose() @ self._kernel_inv @ W + current_sigma_sq * np.identity(num_bases)
        M_K_sigma_inv = np.linalg.inv(M_K_sigma)
        #     print(M_K_sigma)
        #    print(M_K_sigma_inv)
        #     print(self._Y_data.shape)
        x_mean = M_K_sigma_inv @ W.transpose() @ self._kernel_inv @ self._Y_data
        # print((1.0/num_samples)*np.sum(x_mean, 1))
        x_cov = np.zeros((num_bases, num_bases, num_samples))
        for i in range(num_samples):
            this_mean = (x_mean[:, i]).reshape(-1, 1)
            x_cov[:, :, i] = current_sigma_sq * M_K_sigma_inv + this_mean @ this_mean.transpose()
        return x_mean, x_cov

    def update_solution(self, current_W, current_sigma_sq):
        num_samples = self._Y_data.shape[1]
        d = self._Y_data.shape[0]
        x_mean, x_cov = self.latent_expectation(current_W, current_sigma_sq)
        x_cov_inv = np.linalg.inv(np.sum(x_cov, 2))
        #       print("x_cov")
        #       print(np.sum(x_cov, 2))
        #        for q in range(5):
        #            print(x_cov[:,:,q])
        new_W = (self._Y_data @ x_mean.transpose()) @ x_cov_inv
        new_sigma_sq = (1.0 / (num_samples * d)) * (
                    np.trace(self._Y_data.transpose() @ self._kernel_inv @ (self._Y_data)) - 2.0 * np.trace(
                self._Y_data.transpose() @ self._kernel_inv @ current_W @ x_mean) + np.trace(
                np.sum(np.einsum('ij,jkl->ikl', (current_W.transpose() @ self._kernel_inv @ current_W), x_cov), 2)))
        # for i in range(num_samples):
        #     this_y = self._Y_data[:, i]
        #     new_sigma_sq += self._Y_data[:, i
        return new_W, new_sigma_sq


def plot_curves(independent, dependent, time=0):
    num_curves = dependent.shape[1]
    for i in range(num_curves):
        plt.plot(independent, dependent[:, i])
    if time == 0:
        plt.show()
    else:
        plt.pause(time)
        plt.clf()


num_points = 500
num_samples = 400
# std_devs = np.array((10.0, 0.0, 0.0, 2.0, 20.0, 6.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 20.0))
std_devs = np.zeros((40))
std_devs[3] = 25.0
std_devs[4] = 30.0
std_devs[30] = 20.0
std_devs[39] = 20.0
# std_devs[39] = 8.0
num_bases = int(np.sum((std_devs > 0.0).astype('float32')))
print("num bases is " + str(num_bases))
sigma_noise = 0.1
sigma_smooth_noise = 1.0
kernel_sigma = 5.0

X = np.arange(-1.0, 1.0, 2.0 / num_points)
kernel = GP_Kernel(X, 20.0 * 1.0 / num_points, sigma_smooth_noise, sigma_noise)
#  Identity KERNEL
# kernel = np.identity(num_points)
# std_devs = np.array((0.0, 1.0, 2.0, 3.0, 4.0, 5.0))
LG = LegendreRandom((kernel_sigma ** 2) * kernel, std_devs)
samples = LG.samples(X, num_samples)

for i in range(10):
    c = np.zeros(10)
    c[i] = 1.0
    basis = math.sqrt((2.0*i + 1.0)/2.0)*lg.legval(X, c)
    print("basis norm " + str(basis.transpose()@basis))
    plt.plot(X,  lg.legval(X, c), 0.3)

plt.show()


for i in range(20):
    plt.plot(X, samples[i, :])
plt.show()

basis_curves = LG.bases(X)
plot_curves(X, basis_curves)

# U_k, S_k, V_k = np.linalg.svd(kernel)
# print("kernel spectrum")
# print(S_k[0:num_samples])

# regular PCA
samples_mean = np.mean(samples, axis=0)
samples_c = samples - samples_mean
data_cov = (1.0 / num_samples) * np.matmul(samples_c.transpose(), samples_c)
# data_cov = (2.0/num_points)* (1.0/num_samples)*np.matmul(samples_c.transpose(), samples_c)
U, S, V = np.linalg.svd(data_cov)
plot_curves(X, U[:, 0:num_bases])

gp_solver = GP_PCA_EM_solver(kernel, samples_c)
total_data_variance = np.trace(data_cov)
current_sigma_sq = (1.0 / (num_points - num_bases)) * (total_data_variance - np.sum(S[0:num_bases]))
print("init sigma " + str(current_sigma_sq))
# from PCA above
current_W = U[:, 0:num_bases] @ np.diagflat(np.sqrt(S[0:num_bases]))
for i in range(50):
    plot_curves(X, current_W, 0.2)
    # estimate the data noise level
    #    data_noise = kernel_sigma**2
    print("data noise level is " + str(current_sigma_sq))
    current_W, current_sigma_sq = gp_solver.update_solution(current_W, current_sigma_sq)

plot_curves(X, current_W)
exit()

print("now find the Gaussian in that space based on data")
# new_bases = bases
# # print(new_bases.shape)
# # print(X.shape)
# loadings = np.matmul(samples_c, new_bases)
# # print(loadings.shape)
# new_cov = np.matmul(loadings.transpose(), loadings)
# # print(new_cov.shape)
# U_n, S_n, V_n = np.linalg.svd(new_cov)
# new_bases = np.matmul(new_bases, U_n)
# # print(new_bases.shape)
# plot_curves(X, bases)

exit()





