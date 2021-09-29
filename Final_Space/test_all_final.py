import numpy as np
import math
import torch
import matplotlib.pyplot as plt
from Final_Space import Gaussian_Process
from Final_Space import Random_Gaussian
from Final_Space import Calc_Alpha
from Final_Space import Calc_Bias_Changing_In_Time
from Final_Space import Calc_Bias_Changing_In_Time_Integrated_GP
from Final_Space import Calc_Alpha_Calc_Changing_Bias
from Final_Space import Opt_Theta
from Final_Space import Opt_All

# Variables that control the space
N_sensors = 50  # Number of sensors
N_true_sensors = 4  # Number of ground truth sensor points
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

torch.set_default_dtype(torch.float64)

# setting the seed for the program
# seed = torch.tensor(torch.rand(100000))
# seed = 36096893
seed = torch.seed()
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
    return kern.T


def bias_kernel(X, Y):
    kern = (bias_sd ** 2) * torch.exp(-((X.repeat(len(Y), 1) - Y.repeat(len(X), 1).T) ** 2) / (2 * theta_sensor_time_bias ** 2))
    return kern


N_trials = 1
gp_error = np.zeros(5)
calc_alpha_errors = np.zeros(5)
calc_constant_bias_errors = np.zeros(5)
calc_changing_bias_error = np.zeros(5)
calc_changing_int_bias_error = np.zeros(5)
calc_both_error = np.zeros(5)
opt_theta_error = np.zeros(5)
opt_all_error = np.zeros(5)
theta_errors = torch.zeros(4)

for i in range(0, N_trials):
    # Select the location of the sensors, and extend them through time as they are constant
    sensors = torch.rand(N_sensors, 2) * space_range
    plt.scatter(sensors.T[0], sensors.T[1])
    plt.title("Sensor Locations")
    plt.show()
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
    gaussian = Random_Gaussian.RandomGaussian(space_range,
                                              time_range,
                                              space_points,
                                              time_points,
                                              space_kernel,
                                              time_kernel,
                                              kernel,
                                              all_sensor_points,
                                              all_true_sensor_points)
    gaussian.display('Displaying function')

    # SELECTING THE BIAS AND GAIN FOR THE SYSTEM
    # Constant alpha for the whole system
    alpha = torch.normal(alpha_mean, torch.tensor(alpha_sd))
    # Smooth sensor bias in time
    sensor_time = torch.linspace(0, space_range, N_time)
    sensor_dist = torch.distributions.multivariate_normal.MultivariateNormal(bias_mean * torch.ones(N_time),
                                                                             bias_kernel(sensor_time, sensor_time))
    sensor_bias = sensor_dist.sample((N_sensors,)).flatten()

    # Data received with sensor bias
    # data = gaussian.sensor_data
    data = alpha * gaussian.sensor_data + sensor_bias + noise_sd * torch.randn(N_sensors * N_time)
    true_data = gaussian.gt_sensor_data

    # print(sensors)
    # print(sensor_time)
    # print(sensor_bias)
    # print(true_sensors)
    # print(true_data)

    # Building a basic GP
    gp = Gaussian_Process.GaussianProcess(sensors, sensor_time, data, true_sensors, true_sensor_time, true_data,
                                          space_kernel, time_kernel, kernel, noise_sd, N_sensors, alpha_sd,
                                          alpha_mean, bias_kernel, theta_not)
    estimate = gp.build(gaussian.space, gaussian.time)
    gt_estimate = gp.build(true_sensors, true_sensor_time)
    gp_error = gp.print_error(alpha, sensor_bias, gaussian.underlying_data, estimate, gaussian.gt_sensor_data, gt_estimate)
    gp.display(gaussian.space, gaussian.time, estimate, space_points, time_points, "Basic GP on the received data")

    # Building a GP that predicts alpha given bias
    calc_alpha = Calc_Alpha.CalcAlpha(sensors, sensor_time, data, true_sensors, sensor_time, true_data,
                                      space_kernel, time_kernel, kernel, noise_sd, theta_not, alpha_mean, alpha_sd,
                                      sensor_bias, bias_kernel)
    calc_alpha_estimate = calc_alpha.build(gaussian.space, gaussian.time)
    calc_alpha_gt_estimate = calc_alpha.build(true_sensors, true_sensor_time)
    calc_alpha_errors += calc_alpha.print_error(alpha, sensor_bias, gaussian.underlying_data, calc_alpha_estimate, true_data, calc_alpha_gt_estimate)

#     # Building a GP that predicts the bias but is given alpha
#     changing_bias_gp = Calc_Bias_Changing_In_Time.ChangingBias(sensors, sensor_time, data, true_sensors, sensor_time,
#                                                                true_data, space_kernel, time_kernel, kernel, noise_sd,
#                                                                theta_not, bias_sd, bias_mean, bias_kernel, alpha,
#                                                                alpha_mean, alpha_sd)
#     changing_bias_estimate = changing_bias_gp.build(gaussian.space, gaussian.time, space_points)
#     changing_bias_gt_estimate = changing_bias_gp.build(true_sensors, true_sensor_time, N_true_sensors)
#     calc_changing_bias_error += changing_bias_gp.print_error(alpha, sensor_bias, gaussian.underlying_data, changing_bias_estimate, true_data, changing_bias_gt_estimate)
#     # changing_bias_gp.display(gaussian.space, space_points, gaussian.time, changing_bias_estimate,
#     #                          "GP with given alpha assuming the bias is changing in time")
#
#     # Building a GP that predicts the bias using an integrated GP but is given alpha
#     changing_bias_int_gp = Calc_Bias_Changing_In_Time_Integrated_GP.ChangingBiasIntGP(sensors, sensor_time, data, true_sensors, sensor_time,
#                                                                                       true_data, space_kernel, time_kernel, kernel, noise_sd,
#                                                                                       theta_not, bias_sd, bias_mean, bias_kernel, alpha, alpha_mean, alpha_sd)
#     changing_bias_int_estimate = changing_bias_int_gp.build(gaussian.space, gaussian.time, space_points)
#     changing_bias_int_gt_estimate = changing_bias_int_gp.build(true_sensors, true_sensor_time, N_true_sensors)
#     calc_changing_int_bias_error += changing_bias_int_gp.print_error(alpha, sensor_bias, gaussian.underlying_data, changing_bias_int_estimate, true_data, changing_bias_int_gt_estimate)
#     # changing_bias_int_gp.display(gaussian.space, space_points, gaussian.time, changing_bias_int_estimate,
#     #                              "GP with given alpha assuming the bias is changing in time with integrated GP")
#
#     # Building a GP that predicts both bias and alpha using lagging variables
#     calc_both_changing_bias_gp = Calc_Alpha_Calc_Changing_Bias.CalcBothChangingBias(sensors, sensor_time, data, true_sensors, sensor_time, true_data,
#                                                                                     space_kernel, time_kernel, kernel, noise_sd, theta_not, bias_kernel, alpha_mean, alpha_sd)
#     calc_both_changing_bias_estimate = calc_both_changing_bias_gp.build(gaussian.space, gaussian.time, space_points)
#     calc_both_changing_bias_gt_estimate = calc_both_changing_bias_gp.build(true_sensors, true_sensor_time, N_true_sensors)
#     calc_both_error += calc_both_changing_bias_gp.print_error(alpha, sensor_bias, gaussian.underlying_data, calc_both_changing_bias_estimate, true_data, calc_both_changing_bias_gt_estimate)
#     # changing_bias_int_gp.display(gaussian.space, space_points, gaussian.time, calc_both_changing_bias_estimate,
#     #                              "GP calculating both a changing bias and alpha with int gp")
#     plt.show()
#
#     # Using an optimizer to find theta_time and theta_space
#     opt_theta = Opt_Theta.OptTheta(sensors, sensor_time, data, true_sensors, sensor_time, true_data,
#                                    noise_sd, theta_not, bias_kernel, alpha_mean, alpha_sd)
#     opt_theta_estimate = opt_theta.build(gaussian.space, gaussian.time, space_points)
#     opt_theta_gt_estimate = opt_theta.build(true_sensors, true_sensor_time, N_true_sensors)
#     opt_theta_error += opt_theta.print_error(alpha, sensor_bias, gaussian.underlying_data, opt_theta_estimate, true_data, opt_theta_gt_estimate)
#     theta_errors[0] += theta_space - opt_theta.space_theta
#     theta_errors[1] += theta_time - opt_theta.time_theta
#
#     # Letting an optimizer do all the work
#     opt_all = Opt_All.OptAll(sensors, sensor_time, data, true_sensors, sensor_time, true_data,
#                              noise_sd, theta_not, bias_kernel, alpha_mean, alpha_sd)
#     opt_all_estimate = opt_all.build(gaussian.space, gaussian.time, space_points)
#     opt_all_gt_estimate = opt_all.build(true_sensors, true_sensor_time, N_true_sensors)
#     opt_all_error += opt_all.print_error(alpha, sensor_bias, gaussian.underlying_data, opt_all_estimate, true_data, opt_all_gt_estimate)
#     theta_errors[2] += theta_time - opt_all.time_theta
#     theta_errors[3] += theta_time - opt_all.time_theta
#
#     print(i)
#
#
# calc_alpha_errors = calc_alpha_errors/N_trials
# calc_constant_bias_errors = calc_constant_bias_errors/N_trials
# calc_changing_bias_error = calc_changing_bias_error/N_trials
# calc_changing_int_bias_error = calc_changing_int_bias_error/N_trials
# calc_both_error = calc_both_error/N_trials
# opt_theta_error = theta_errors / N_trials
# opt_all_error = opt_all_error/N_trials
#
# print("Number of trails: " + str(N_trials))
# print("Number of sensors: " + str(N_sensors**2))
# print("Number of GT sensors: " + str(N_true_sensors**2))
# print(gp_error)
# print(calc_alpha_errors)
# print(calc_constant_bias_errors)
# print(calc_changing_bias_error)
# print(calc_changing_int_bias_error)
# print(calc_both_error)
# print(opt_theta_error)
# print(opt_all_error)
# print(theta_errors / N_trials)
