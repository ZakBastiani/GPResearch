import numpy as np
import torch
import matplotlib.pyplot as plt
from Final_Space import Gaussian_Process
from Final_Space import MAPEstimate


class CalcAlpha(Gaussian_Process.GaussianProcess):
    def __init__(self, space_X, time_X, _Y, space_Xt, time_Xt, _Yt,
                 space_kernel, time_kernel, kernel, noise_sd, theta_not, v, t2, bias, bias_kernel):

        self.points = torch.cat((space_X.repeat(len(time_X), 1), time_X.repeat_interleave(len(space_X)).repeat(1, 1).T), 1)

        # Need to alter the sensor matrix and the data matrix
        X = torch.cat((space_X.repeat(len(time_X), 1),
                       time_X.repeat_interleave(len(space_X)).repeat(1, 1).T), 1)
        Y = _Y.flatten()

        Xt = torch.cat((space_Xt.repeat(len(time_Xt), 1),
                        time_Xt.repeat_interleave(len(space_Xt)).repeat(1, 1).T), 1)
        Yt = _Yt.flatten()

        alpha = 1.0
        for i in range(5):
            noise_lag = noise_sd / alpha
            sigma_inv = torch.linalg.inv(kernel(self.points, self.points) + (noise_lag ** 2) * np.eye(len(space_X) * len(time_X)))
            alpha_poly = torch.zeros(3)
            y_min_bias = (Y - bias.flatten()).T
            alpha_poly[2] = y_min_bias.T @ sigma_inv @ y_min_bias
            alpha_poly[0] = -len(space_X) * len(time_X)
            alpha_poly[1] += v * t2 / 2
            alpha_poly[0] += -(1 + v/2)
            for i in range(len(Xt)):
                k_star = kernel(Xt[i].unsqueeze(0), X)
                divisor = (theta_not - k_star.T @ sigma_inv @ k_star)
                alpha_poly[2] += ((k_star.T @ sigma_inv @ y_min_bias) ** 2 / divisor).item()
                alpha_poly[1] -= ((Yt[i] * k_star.T @ sigma_inv @ y_min_bias) / divisor).item()

            roots = np.roots(alpha_poly.detach().numpy())  # The algorithm relies on computing the eigenvalues of the companion matrix
            # print(roots)
            real_roots = []

            for root in roots:
                if root.imag == 0:
                    real_roots.append(root.real)

            if len(real_roots) != 0:
                closest = real_roots[0]
                for r in real_roots:
                    if abs(closest - 1) > abs(r - 1):
                        closest = r
                alpha = closest
            alpha = torch.tensor(alpha)

        self.type = "Gaussian Process Regression calculating chi alpha with a provided bias"
        self.space_X = space_X  # np.concatenate((space_X, space_Xt))
        self.time_X = time_X
        self.alpha = alpha
        self.bias = bias
        self.Y = (_Y - bias) / self.alpha  # np.concatenate(((_Y - bias) / self.alpha, _Yt))
        self.noise_sd = noise_sd
        self.space_kernel = space_kernel
        self.time_kernel = time_kernel
        self.kernel = kernel
        self.Sigma = self.kernel(self.points, self.points)
        self.L = torch.linalg.cholesky(self.Sigma + noise_sd ** 2 * torch.eye(len(self.points)))
        self.loss = MAPEstimate.map_estimate_torch(X, Y, Xt, Yt, bias.flatten(), alpha, noise_sd, self.Sigma, space_kernel,
                                                   time_kernel, kernel, 1, 0.25,
                                                   torch.kron(torch.eye(len(space_X)), bias_kernel(time_X, time_X)),
                                                   len(space_X), len(time_X), theta_not)

        # Building a graph showing the loss function for values of alpha to see how good our calc_both is
        alpha_range = torch.linspace(0.95, 1.15, 100)
        y = []
        ll = 0
        for a in alpha_range:
            noise_lag = noise_sd / a
            l = MAPEstimate.map_estimate_torch_chi2(X, Y, Xt, Yt, self.bias.flatten(), a, noise_sd,
                                                    self.Sigma, space_kernel, time_kernel, kernel, v, t2,
                                                    torch.kron(torch.eye(len(space_X)), bias_kernel(time_X, time_X)),
                                                    len(space_X), len(time_X), theta_not)
            y.append(l)
            if l > ll:
                ll = l
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(alpha_range, y, 'b-')
        chi2loss =  MAPEstimate.map_estimate_torch_chi2(X, Y, Xt, Yt, self.bias.flatten(), self.alpha, noise_sd,
                                                        self.Sigma, space_kernel, time_kernel, kernel, v, t2,
                                                        torch.kron(torch.eye(len(space_X)), bias_kernel(time_X, time_X)),
                                                        len(space_X), len(time_X), theta_not)
        ax.plot(self.alpha, chi2loss, marker='o')
        # plt.ylim([1000, 2500])
        plt.title("Calc loss based on chi2 alpha with true bias")
        plt.show()
        print(ll)



