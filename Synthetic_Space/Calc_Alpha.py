import numpy as np
import matplotlib.pyplot as plt
from Synthetic_Space import Gaussian_Process
from Synthetic_Space import MAPEstimate


class CalcAlpha(Gaussian_Process.GaussianProcess):
    def __init__(self, space_X, time_X, _Y, space_Xt, time_Xt, _Yt, space_kernel, time_kernel, kernel, noise, theta_not,
                 alpha_mean, alpha_variance, bias, bias_kernel, bias_variance):
        sigma_inv = np.linalg.inv(np.kron(space_kernel(space_X, space_X), time_kernel(time_X, time_X)) + noise**2 * np.eye(
            len(space_X) * len(time_X)))
        # Need to alter the sensor matrix and the data matrix
        X = np.array([np.outer(space_X, np.ones(len(time_X))).flatten(),
                      np.outer(time_X, np.ones(len(space_X))).T.flatten()]).T
        Y = _Y.flatten()

        Xt = np.array([np.outer(space_Xt, np.ones(len(time_Xt))).flatten(),
                       np.outer(time_Xt, np.ones(len(space_Xt))).T.flatten()]).T
        Yt = _Yt.flatten()

        alpha_poly = np.zeros(5)
        y_min_bias = (Y - bias.flatten()).T
        alpha_poly[4] = y_min_bias.T @ sigma_inv @ y_min_bias
        alpha_poly[2] = -len(space_X) * len(time_X)
        alpha_poly[1] = alpha_mean / (alpha_variance**2)
        alpha_poly[0] = -1 / (alpha_variance**2)
        for i in range(len(Xt)):
            k_star = kernel([Xt[i]], X).T
            divisor = (theta_not - k_star.T @ sigma_inv @ k_star)
            alpha_poly[4] += (k_star.T @ sigma_inv @ y_min_bias)**2 / divisor
            alpha_poly[3] -= (Yt[i]*k_star.T @ sigma_inv @ y_min_bias) / divisor

        roots = np.roots(alpha_poly)  # The algorithm relies on computing the eigenvalues of the companion matrix
        # print(roots)
        real_roots = []
        alpha = 1
        for root in roots:
            if root.imag == 0:
                real_roots.append(root.real)

        if len(real_roots) != 0:
            closest = real_roots[0]
            for r in real_roots:
                if abs(closest - alpha_mean) > abs(r - alpha_mean):
                    closest = r
            alpha = closest

        self.type = "Gaussian Process Regression calculating alpha with a provided bias"
        self.space_X = space_X # np.concatenate((space_X, space_Xt))
        self.time_X = time_X
        self.alpha = alpha
        self.bias = bias
        self.Y = (_Y - bias) / self.alpha  # np.concatenate(((_Y - bias) / self.alpha, _Yt))
        self.noise = noise
        self.space_kernel = space_kernel
        self.time_kernel = time_kernel
        self.Sigma = np.kron(self.space_kernel(self.space_X, self.space_X), self.time_kernel(self.time_X, self.time_X))
        self.L = np.linalg.cholesky((self.Sigma + noise * np.eye(len(self.Sigma))))
        self.loss = MAPEstimate.map_estimate_numpy(X, Y, Xt, Yt, bias.flatten(), alpha, noise, self.Sigma, space_kernel, time_kernel, kernel, alpha_mean,
                                                   alpha_variance, np.kron(np.eye(len(space_X)), bias_kernel(time_X, time_X)), len(space_X), len(time_X), theta_not)
        print(self.loss)



