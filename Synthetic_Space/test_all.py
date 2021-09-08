import numpy as np
import matplotlib.pyplot as plt
from Synthetic_Space import Gaussian_Process
from Synthetic_Space import Random_Gaussian
from Synthetic_Space import Constant_Bias
from Synthetic_Space import Opt_Alpha
from Synthetic_Space import Calc_Alpha
from Synthetic_Space import Calc_Alpha_Calc_Constant_Bias
from Synthetic_Space import Wei_Model
from Synthetic_Space import Calc_Bias_Changing_In_Time
from Synthetic_Space import Calc_Bias_Changing_In_Time_Plus_GP
from Synthetic_Space import Calc_Bias_Changing_In_Time_Integrated_GP
from Synthetic_Space import Opt_Changing_Bais
from Synthetic_Space import Calc_Alpha_Calc_Changing_Bias
from Synthetic_Space import Opt_Changing_Bias_and_Alpha

# Variables that control the space
N_sensors = 10  # Number of sensors
N_true_sensors = 2  # Number of ground truth sensor points
N_time = 10  # Number of time samples
noise = 0.01  # random noise in the system

space_range = 10
time_range = 10
space_points = 50
time_points = 50

# Hyper-parameters Bandwidths
theta_space = 2
theta_time = 1
theta_not = 1

# Model Parameters
bias_variance = 1
bias_mean = 0
theta_sensor_time_bias = 8  # bandwidths
alpha_mean = 1
alpha_variance = 0.25

N_space_test = 50
N_time_test = 50

# setting the seed for the program
# seed = np.random.randint(100000000)
seed = 55169561
# seed = 16804736
np.random.seed(seed)
print("Seed: " + str(seed))

def space_kernel(X, Y):
    kernel = np.ndarray(shape=(len(X), len(Y)))
    for x in range(0, len(X)):
        for y in range(0, len(Y)):
            kernel[x][y] = theta_not * np.exp(- ((X[x] - Y[y]) ** 2) / (2*theta_space**2))
    return kernel


def time_kernel(X, Y):
    kernel = np.ndarray(shape=(len(X), len(Y)))
    for x in range(0, len(X)):
        for y in range(0, len(Y)):
            kernel[x][y] = np.exp(-((X[x] - Y[y]) ** 2) / (2*theta_time**2))
    return kernel


def kernel(X, Y):
    kern = np.ndarray(shape=(len(X), len(Y)))
    for x in range(0, len(X)):
        for y in range(0, len(Y)):
            kern[x][y] = theta_not * np.exp(- ((X[x][0] - Y[y][0]) ** 2)/(2*theta_space**2)
                                            - ((X[x][1] - Y[y][1]) ** 2)/(2*theta_time**2))
    return kern


def bias_kernel(X, Y):
    kernel = np.ndarray(shape=(len(X), len(Y)))
    for x in range(0, len(X)):
        for y in range(0, len(Y)):
            kernel[x][y] = (bias_variance**2)*np.exp(-((X[x] - Y[y]) ** 2) / (2*theta_sensor_time_bias**2))
    return kernel


# Building the function
gaussian = Random_Gaussian.RandomGaussian(space_range,
                                          time_range,
                                          space_points,
                                          time_points,
                                          space_kernel,
                                          time_kernel,
                                          np.zeros(time_points*space_points),
                                          noise)
gaussian.display('Displaying function')

# Select the location of the sensors, and extend them through time as they are constant
sensors = np.linspace(0, space_range, N_sensors)
# Set the time interval and extend through space
sensor_time = np.linspace(0, space_range, N_time)

# SELECTING THE BIAS AND GAIN FOR THE SYSTEM
# Constant alpha for the whole system
alpha = np.random.normal(alpha_mean, alpha_variance, size=1)
# Constant sensor bias in time
# sensor_bias = np.outer(np.random.normal(bias_mean, bias_variance, size=N_sensors), np.ones(N_time))
# Smooth sensor bias in time
sensor_bias = np.random.multivariate_normal(bias_mean * np.ones(N_time), bias_kernel(sensor_time, sensor_time), N_sensors)
first_data = gaussian.function(sensors, sensor_time)

# Data received with sensor bias
# data = f(sensors, sensor_time) + noise*np.random.randn(N_sensors, N_time)
data = alpha * (first_data) + sensor_bias

# Selecting the location of the ground truth points
true_sensors = np.linspace(0, space_range, N_true_sensors+2)[1:-1]
# Setting the time matrix for the true sensors
true_sensor_time = np.linspace(0, space_range, N_time)

true_data = gaussian.function(true_sensors, true_sensor_time)

# print(sensors)
# print(sensor_time)
# print(sensor_bias)
# print(true_sensors)
# print(true_data)

# Building a basic GP
gp = Gaussian_Process.GaussianProcess(sensors, sensor_time, data, true_sensors, true_sensor_time, true_data,
                                      space_kernel, time_kernel, noise)
estimate = gp.build(gaussian.space, gaussian.time)
gt_estimate = gp.build(true_sensors, true_sensor_time)
gp.print_error(alpha, sensor_bias, gaussian.matrix2d, estimate, true_data, gt_estimate)
gp.display(gaussian.space, gaussian.time, estimate, "Basic GP on the received data")

# Building a GP that predicts alpha given bias
calc_alpha = Calc_Alpha.CalcAlpha(sensors, sensor_time, data, true_sensors, sensor_time, true_data,
                 space_kernel, time_kernel, kernel, noise, theta_not, alpha_mean, alpha_variance, sensor_bias)
calc_alpha_estimate = calc_alpha.build(gaussian.space, gaussian.time)
calc_alpha_gt_estimate = calc_alpha.build(true_sensors, true_sensor_time)
calc_alpha.print_error(alpha, sensor_bias, gaussian.matrix2d, calc_alpha_estimate, true_data, calc_alpha_gt_estimate)

# # Building a GP that predicts the bias but is given alpha
# bias_gp = Constant_Bias.ConstantBias(sensors, sensor_time, data, true_sensors, sensor_time, true_data,
#                                      space_kernel, time_kernel, kernel, noise, theta_not, bias_variance, bias_mean, alpha)
# constant_bias_estimate = bias_gp.build(gaussian.space, gaussian.time)
# constant_bias_gt_estimate = bias_gp.build(true_sensors, true_sensor_time)
# bias_gp.print_error(alpha, sensor_bias, gaussian.matrix2d, constant_bias_estimate, true_data, constant_bias_gt_estimate)

# # Building a GP that predicts both bias and alpha using lagging variables
# calc_both = Calc_Alpha_Calc_Constant_Bias.CalcBoth(sensors, sensor_time, data, true_sensors, sensor_time, true_data,
#                                      space_kernel, time_kernel, kernel, noise, theta_not, bias_variance, bias_mean, alpha_mean, alpha_variance)
# calc_both_estimate = calc_both.build(gaussian.space, gaussian.time)
# calc_both_gt_estimate = calc_both.build(true_sensors, true_sensor_time)
# calc_both.print_error(alpha, sensor_bias, gaussian.matrix2d, calc_both_estimate, true_data, calc_both_gt_estimate )

# Building a Gp that predicts bias and optimizes alpha
opt_alpha = Opt_Alpha.OptAlphaCalcBias(sensors, sensor_time, data, true_sensors, sensor_time, true_data, space_kernel,
                                       time_kernel, kernel, noise, theta_not, sensor_bias, alpha_variance, alpha_mean)
opt_alpha_estimate = opt_alpha.build(gaussian.space, gaussian.time)
opt_alpha_gt_estimate = opt_alpha.build(true_sensors, true_sensor_time)
opt_alpha.print_error(alpha, sensor_bias, gaussian.matrix2d, opt_alpha_estimate, true_data, opt_alpha_gt_estimate)

# # # Building a Gp that optimizes both alpha and bias
# # opt_both = Wei_Model.OptBoth(sensors, sensor_time, data, true_sensors, sensor_time, true_data,
# #                              space_kernel, time_kernel, kernel, noise, theta_not, bias_variance,
# #                              bias_mean, alpha_variance, alpha_mean)
# # opt_both_estimate = opt_both.build(gaussian.space, gaussian.time)
# # opt_both_gt_estimate = opt_both.build(true_sensors, true_sensor_time)
# # opt_both.print_error(alpha, sensor_bias.T[0], gaussian.matrix2d, opt_both_estimate, true_data, opt_both_gt_estimate)

# # Building a GP that predicts the bias but is given alpha
# changing_bias_gp = Calc_Bias_Changing_In_Time.ChangingBias(sensors, sensor_time, data, true_sensors, sensor_time,
#                                                            true_data, space_kernel, time_kernel, kernel, noise,
#                                                            theta_not, bias_variance, bias_mean, bias_kernel, alpha)
# changing_bias_estimate = changing_bias_gp.build(gaussian.space, gaussian.time)
# changing_bias_gt_estimate = changing_bias_gp.build(true_sensors, true_sensor_time)
# changing_bias_gp.print_error(alpha, sensor_bias, gaussian.matrix2d, changing_bias_estimate, true_data, changing_bias_gt_estimate)
# # changing_bias_gp.display(gaussian.space, gaussian.time, changing_bias_estimate,
# #                          "GP with given alpha assuming the bias is changing in time")
#
# # # Building a GP that predicts the bias and applies a GP but is given alpha
# # changing_bias_plus_gp = Calc_Bias_Changing_In_Time_Plus_GP.ChangingBiasPlusGP(sensors, sensor_time, data, true_sensors, sensor_time,
# #                                                            true_data, space_kernel, time_kernel, kernel, noise,
# #                                                            theta_not, bias_variance, bias_mean, bias_kernel, alpha)
# # changing_bias_plus_estimate = changing_bias_plus_gp.build(gaussian.space, gaussian.time)
# # changing_bias_plus_gt_estimate = changing_bias_plus_gp.build(true_sensors, true_sensor_time)
# # changing_bias_plus_gp.print_error(alpha, sensor_bias, gaussian.matrix2d, changing_bias_plus_estimate, true_data, changing_bias_plus_gt_estimate)
# # # changing_bias_plus_gp.display(gaussian.space, gaussian.time, changing_bias_plus_estimate,
# # #                               "GP with given alpha assuming the bias is changing in time plus applied GP")
#
# Building a GP that predicts the bias using an integrated GP but is given alpha
changing_bias_int_gp = Calc_Bias_Changing_In_Time_Integrated_GP.ChangingBiasIntGP(sensors, sensor_time, data, true_sensors, sensor_time,
                                                           true_data, space_kernel, time_kernel, kernel, noise,
                                                           theta_not, bias_variance, bias_mean, bias_kernel, alpha)
changing_bias_int_estimate = changing_bias_int_gp.build(gaussian.space, gaussian.time)
changing_bias_int_gt_estimate = changing_bias_int_gp.build(true_sensors, true_sensor_time)
changing_bias_int_gp.print_error(alpha, sensor_bias, gaussian.matrix2d, changing_bias_int_estimate, true_data, changing_bias_int_gt_estimate)
# changing_bias_int_gp.display(gaussian.space, gaussian.time, changing_bias_int_estimate,
#                              "GP with given alpha assuming the bias is changing in time with integrated GP")
#
# Building a GP that predicts the bias but is given alpha
opt_changing_bias_gp = Opt_Changing_Bais.OptChangingBias(sensors, sensor_time, data, true_sensors, sensor_time,
                                                        true_data, space_kernel, time_kernel, kernel, noise,
                                                        theta_not, bias_variance, bias_mean, bias_kernel, alpha)
opt_changing_bias_estimate = opt_changing_bias_gp.build(gaussian.space, gaussian.time)
opt_changing_bias_gt_estimate = opt_changing_bias_gp.build(true_sensors, true_sensor_time)
opt_changing_bias_gp.print_error(alpha, sensor_bias, gaussian.matrix2d, opt_changing_bias_estimate, true_data, opt_changing_bias_gt_estimate)
opt_changing_bias_gp.display(gaussian.space, gaussian.time, opt_changing_bias_estimate,
                         "GP with given alpha and optimizing for a chaning bias")

# Building a GP that predicts both bias and alpha using lagging variables
calc_both_changing_bias_gp = Calc_Alpha_Calc_Changing_Bias.CalcBothChangingBias(sensors, sensor_time, data, true_sensors, sensor_time, true_data,
                                     space_kernel, time_kernel, kernel, noise, theta_not, bias_kernel, alpha_mean, alpha_variance)
calc_both_changing_bias_estimate = calc_both_changing_bias_gp.build(gaussian.space, gaussian.time)
calc_both_changing_bias_gt_estimate = calc_both_changing_bias_gp.build(true_sensors, true_sensor_time)
calc_both_changing_bias_gp.print_error(alpha, sensor_bias, gaussian.matrix2d, calc_both_changing_bias_estimate, true_data, calc_both_changing_bias_gt_estimate)
# calc_both_changing_bias_gp.display(gaussian.space, gaussian.time, calc_both_changing_bias_estimate,
#                                     "GP calculating both a changing bias and alpha")

# Building a GP that predicts the bias but is given alpha
opt_changing_bias_alpha_gp = Opt_Changing_Bias_and_Alpha.OptChangingBiasAndAlpha(sensors, sensor_time, data, true_sensors, sensor_time,
                                                         true_data, space_kernel, time_kernel, kernel, noise,
                                                         theta_not, bias_variance, bias_mean, bias_kernel, alpha_mean, alpha_variance, alpha, sensor_bias)
opt_changing_bias_alpha_estimate = opt_changing_bias_alpha_gp.build(gaussian.space, gaussian.time)
opt_changing_bias_alpha_gt_estimate = opt_changing_bias_alpha_gp.build(true_sensors, true_sensor_time)
opt_changing_bias_alpha_gp.print_error(alpha, sensor_bias, gaussian.matrix2d, opt_changing_bias_alpha_estimate, true_data, opt_changing_bias_alpha_gt_estimate)
opt_changing_bias_alpha_gp.display(gaussian.space, gaussian.time, opt_changing_bias_alpha_estimate,
                                   "GP optimizing for a chaning bias and alpha")

plt.show()
