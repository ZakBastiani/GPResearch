import numpy as np
import math
import torch
import matplotlib.pyplot as plt
from Final_Space import Gaussian_Process
from Final_Space import Random_Gaussian
from Final_Space import Calc_Chi_Alpha_Calc_Bias
from Final_Space import Opt_alpha_calc_bias_chi2
from Final_Space import Opt_all_but_bias_chi2

# Variables that control the space
N_sensors = 50  # Number of sensors
N_true_sensors = 2  # Number of ground truth sensor points
N_time = 10  # Number of time samples
N_true_time = 10  # Number of gt time samples
noise_sd = 0.01  # random noise in the system Standard Deviation

space_range = 10
time_range = 10
space_points = 25
time_points = 25

# Hyper-parameters
theta_space = 2
theta_time = 1
theta_not = 1

# Model Parameters
bias_sd = 1
bias_mean = 0
theta_sensor_time_bias = 8
alpha_mean = 1
alpha_sd = 0.25
v = N_sensors
t2 = 1.0

torch.set_default_dtype(torch.float64)

# setting the seed for the program
# seed = torch.seed()
seed = torch.manual_seed(185545686495700)
print("Seed: " + str(seed))


def space_kernel(X, Y):
    kernel = theta_not * torch.exp(
        -((X.T[0].repeat(len(Y), 1) - Y.T[0].repeat(len(X), 1).T) ** 2) / (2 * theta_space ** 2)
        - ((X.T[1].repeat(len(Y), 1) - Y.T[1].repeat(len(X), 1).T) ** 2) / (2 * theta_space ** 2))
    return kernel


def time_kernel(X, Y):
    kernel = torch.exp(-((X.repeat(len(Y), 1) - Y.repeat(len(X), 1).T) ** 2) / (2 * theta_time ** 2))
    return kernel


def kernel(X, Y):
    kern = theta_not * torch.exp(
        - ((X.T[0].repeat(len(Y), 1) - Y.T[0].repeat(len(X), 1).T) ** 2) / (2 * theta_space ** 2)
        - ((X.T[1].repeat(len(Y), 1) - Y.T[1].repeat(len(X), 1).T) ** 2) / (2 * theta_space ** 2)
        - ((X.T[2].repeat(len(Y), 1) - Y.T[2].repeat(len(X), 1).T) ** 2) / (2 * theta_time ** 2))
    return kern


def bias_kernel(X, Y):
    kern = (bias_sd ** 2) * torch.exp(-((X.repeat(len(Y), 1) - Y.repeat(len(X), 1).T) ** 2) / (2 * theta_sensor_time_bias ** 2))
    return kern


N_trials = 30
gp_error = np.zeros(5)
calc_both_chi_alpha_error = np.zeros(5)
opt_alpha_chi2_error = np.zeros(5)
opt_all_chi2_error = np.zeros(5)
theta_errors = torch.zeros(2)

for i in range(0, N_trials):
    # Select the location of the sensors, and extend them through time as they are constant
    sensors = torch.rand(N_sensors, 2) * space_range

    # Set the time interval and extend through space
    sensor_time = torch.linspace(0, space_range, N_time)
    all_sensor_points = torch.cat((sensors.repeat(N_time, 1),
                                   sensor_time.repeat_interleave(N_sensors).repeat(1, 1).T), 1)

    # Selecting the location of the ground truth points
    true_sensors = torch.rand(N_true_sensors, 2) * space_range
    # Setting the time matrix for the true sensors
    true_sensor_time = torch.linspace(0, space_range, N_true_time)
    all_true_sensor_points = torch.cat((true_sensors.repeat(N_true_time, 1),
                                        true_sensor_time.repeat_interleave(N_true_sensors).repeat(1, 1).T), 1)

    # Building the function
    gaussian = Random_Gaussian.RandomGaussian(space_range, time_range, space_points, time_points, space_kernel,
                                              time_kernel, kernel, all_sensor_points, all_true_sensor_points)
    # gaussian.display('Displaying function')

    # SELECTING THE BIAS AND GAIN FOR THE SYSTEM

    # Normal Distribution
    # alpha = torch.normal(alpha_mean, torch.tensor(alpha_sd))

    # Scaled inverse chi squared distribution
    scaled_inv_chi2 = torch.distributions.gamma.Gamma(v/2, v*t2/2)
    alpha = 1/scaled_inv_chi2.sample()

    # Smooth sensor bias in time
    sensor_time = torch.linspace(0, space_range, N_time)
    sensor_dist = torch.distributions.multivariate_normal.MultivariateNormal(bias_mean * torch.ones(N_time),
                                                                             bias_kernel(sensor_time, sensor_time))
    sensor_bias = sensor_dist.sample((N_sensors,)).flatten()

    # Data received with sensor bias
    # data = gaussian.sensor_data
    noise = noise_sd * torch.randn(N_sensors * N_time)
    data = alpha * gaussian.sensor_data + sensor_bias + noise
    true_data = gaussian.gt_sensor_data

    # Building a basic GP
    gp = Gaussian_Process.GaussianProcess(sensors, sensor_time, data, true_sensors, true_sensor_time, true_data,
                                          space_kernel, time_kernel, kernel, noise_sd, N_sensors, alpha, alpha_mean, alpha_sd, bias_kernel, theta_not)
    estimate = gp.build(gaussian.space, gaussian.time)
    gt_estimate = gp.build(true_sensors, true_sensor_time)
    gp_error += gp.print_error(alpha, sensor_bias, gaussian.underlying_data, estimate, gaussian.gt_sensor_data, gt_estimate)

    # Building a GP that predicts both bias and alpha using lagging variables
    calc_both_chi_alpha_gp = Calc_Chi_Alpha_Calc_Bias.CalcBothChiAlpha(sensors, sensor_time, data, true_sensors, sensor_time, true_data,
                                                                       space_kernel, time_kernel, kernel, noise_sd, theta_not, bias_kernel, v, t2)
    calc_both_chi_alpha_estimate = calc_both_chi_alpha_gp.build(gaussian.space, gaussian.time)
    calc_both_chi_alpha_gt_estimate = calc_both_chi_alpha_gp.build(true_sensors, true_sensor_time)
    calc_both_chi_alpha_error += calc_both_chi_alpha_gp.print_error(alpha, sensor_bias, gaussian.underlying_data, calc_both_chi_alpha_estimate, true_data, calc_both_chi_alpha_gt_estimate)

    # Letting an optimizer calc alpha
    opt_alpha_chi2 = Opt_alpha_calc_bias_chi2.OptAlphaCalcBias(sensors, sensor_time, data, true_sensors, sensor_time, true_data,
                                                noise_sd, theta_not, bias_kernel, v, t2)
    opt_alpha_chi2_estimate = opt_alpha_chi2.build(gaussian.space, gaussian.time)
    opt_alpha_chi2_gt_estimate = opt_alpha_chi2.build(true_sensors, true_sensor_time)
    opt_alpha_chi2_error += opt_alpha_chi2.print_error(alpha, sensor_bias, gaussian.underlying_data, opt_alpha_chi2_estimate, true_data, opt_alpha_chi2_gt_estimate)

    # Letting an optimizer do all the work
    opt_all_chi2 = Opt_all_but_bias_chi2.OptAll(sensors, sensor_time, data, true_sensors, sensor_time, true_data,
                                                noise_sd, theta_not, bias_kernel, v, t2)
    opt_all_chi2_estimate = opt_all_chi2.build(gaussian.space, gaussian.time)
    opt_all_chi2_gt_estimate = opt_all_chi2.build(true_sensors, true_sensor_time)
    opt_all_chi2_error += opt_all_chi2.print_error(alpha, sensor_bias, gaussian.underlying_data, opt_all_chi2_estimate, true_data, opt_all_chi2_gt_estimate)
    theta_errors[0] += abs(theta_time - opt_all_chi2.time_theta)
    theta_errors[1] += abs(theta_time - opt_all_chi2.time_theta)

    print(i)

gp_error = gp_error / N_trials
calc_both_chi_alpha_error = calc_both_chi_alpha_error / N_trials
opt_alpha_chi2_error = opt_alpha_chi2_error / N_trials
opt_all_chi2_error = opt_all_chi2_error / N_trials
theta_errors = theta_errors / N_trials

print("Number of trials: " + str(N_trials))
print("Number of sensors: " + str(N_sensors))
print("Number of GT sensors: " + str(N_true_sensors))
print(gp_error)
print(calc_both_chi_alpha_error)
print(opt_alpha_chi2_error)
print(opt_all_chi2_error)
print(theta_errors / N_trials)
