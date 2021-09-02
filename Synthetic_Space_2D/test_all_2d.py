import numpy as np
import matplotlib.pyplot as plt
from Synthetic_Space_2D import Gaussian_Process
from Synthetic_Space_2D import Random_Gaussian
from Synthetic_Space_2D import Constant_Bias
from Synthetic_Space_2D import Opt_Alpha_Constant_Bias
from Synthetic_Space_2D import Calc_Alpha
from Synthetic_Space_2D import Calc_Alpha_Calc_Constant_Bias
from Synthetic_Space_2D import Wei_Model
from Synthetic_Space_2D import Calc_Bias_Changing_In_Time
from Synthetic_Space_2D import Calc_Bias_Changing_In_Time_Plus_GP
from Synthetic_Space_2D import Calc_Bias_Changing_In_Time_Integrated_GP
from Synthetic_Space_2D import Opt_Changing_Bais
from Synthetic_Space_2D import Calc_Alpha_Calc_Changing_Bias

# Variables that control the space
N_sensors = 10  # Number of sensors
N_true_sensors = 2  # Number of ground truth sensor points
N_time = 10  # Number of time samples
noise = 0.1  # random noise in the system

space_range = 10
time_range = 10
space_points = 15
time_points = 25

# Hyper-parameters
theta_space = 2
theta_time = 1
theta_not = 1

# Model Parameters
bias_variance = 1
bias_mean = 0
theta_sensor_time_bias = 8
alpha_mean = 1
alpha_variance = 0.25

# setting the seed for the program
seed = np.random.randint(100000000)
# seed = 3141592
np.random.seed(seed)
print("Seed: " + str(seed))


def space_kernel(X, Y):
    kernel = np.ndarray(shape=(len(X), len(Y)))
    for x in range(0, len(X)):
        for y in range(0, len(Y)):
            kernel[x][y] = theta_not * np.exp(- ((X[x][0] - Y[y][0]) ** 2) / (2 * theta_space ** 2)
                                              - ((X[x][1] - Y[y][1]) ** 2) / (2 * theta_space ** 2))
    return kernel


def time_kernel(X, Y):
    kernel = np.ndarray(shape=(len(X), len(Y)))
    for x in range(0, len(X)):
        for y in range(0, len(Y)):
            kernel[x][y] = np.exp(-((X[x] - Y[y]) ** 2) / (2 * theta_time ** 2))
    return kernel


def kernel(X, Y):
    kern = np.ndarray(shape=(len(X), len(Y)))
    for x in range(0, len(X)):
        for y in range(0, len(Y)):
            kern[x][y] = theta_not * np.exp(- ((X[x][0] - Y[y][0]) ** 2) / (2 * theta_space ** 2)
                                            - ((X[x][1] - Y[y][1]) ** 2) / (2 * theta_space ** 2)
                                            - ((X[x][2] - Y[y][2]) ** 2) / (2 * theta_time ** 2))
    return kern


def bias_kernel(X, Y):
    kernel = np.ndarray(shape=(len(X), len(Y)))
    for x in range(0, len(X)):
        for y in range(0, len(Y)):
            kernel[x][y] = (bias_variance ** 2) * np.exp(-((X[x] - Y[y]) ** 2) / (2 * theta_sensor_time_bias ** 2))
    return kernel


N_trials = 1
gp_error = np.zeros(4)
calc_alpha_errors = np.zeros(4)
calc_constant_bias_errors = np.zeros(4)
calc_changing_bias_error = np.zeros(4)
calc_changing_int_bias_error = np.zeros(4)
calc_both_error = np.zeros(4)


for i in range(0, N_trials):
    # Building the function
    gaussian = Random_Gaussian.RandomGaussian(space_range,
                                              time_range,
                                              space_points,
                                              time_points,
                                              space_kernel,
                                              time_kernel,
                                              np.zeros(time_points * space_points * space_points),
                                              noise)
    gaussian.display('Displaying function')

    # Select the location of the sensors, and extend them through time as they are constant
    sensors = np.mgrid[0:(space_range + 0.1):(space_range / (N_sensors - 1)),
              0:(space_range + 0.1):(space_range / (N_sensors - 1))].reshape(2, -1).T
    # Set the time interval and extend through space
    sensor_time = np.linspace(0, space_range, N_time)

    # SELECTING THE BIAS AND GAIN FOR THE SYSTEM
    # Constant alpha for the whole system
    alpha = np.random.normal(alpha_mean, alpha_variance, size=1)
    # Constant sensor bias in time
    # sensor_bias = np.outer(np.random.normal(bias_mean, bias_variance, size=N_sensors**2), np.ones(N_time)).reshape((N_sensors, N_sensors, N_time))
    # Smooth sensor bias in time
    sensor_bias = np.random.multivariate_normal(bias_mean * np.ones(N_time), bias_kernel(sensor_time, sensor_time),
                                               N_sensors ** 2).reshape((N_sensors, N_sensors, N_time))

    # Data received with sensor bias
    # data = gaussian.function(sensors, sensor_time, N_sensors) + noise*np.random.randn(N_sensors, N_sensors, N_time)
    data = alpha * (gaussian.function(sensors, sensor_time, N_sensors) + noise * np.random.randn(N_sensors, N_sensors,
                                                                                                 N_time)) + sensor_bias

    # Selecting the location of the ground truth points
    true_sensors = np.mgrid[space_range / 4:(3 * space_range / 4 + 0.1):(space_range / (2 * (N_true_sensors - 1))),
                   space_range / 4:(3 * space_range / 4 + 0.1):(space_range / (2 * (N_true_sensors - 1)))].reshape(2, -1).T
    # Setting the time matrix for the true sensors
    true_sensor_time = np.linspace(0, space_range, N_time)

    true_data = gaussian.function(true_sensors, true_sensor_time, N_true_sensors)

    # print(sensors)
    # print(sensor_time)
    # print(sensor_bias)
    # print(true_sensors)
    # print(true_data)

    # Building a basic GP
    gp = Gaussian_Process.GaussianProcess(sensors, sensor_time, data, true_sensors, true_sensor_time, true_data,
                                          space_kernel, time_kernel, noise, N_sensors)
    estimate = gp.build(gaussian.space, gaussian.time, space_points)
    gt_estimate = gp.build(true_sensors, true_sensor_time, N_true_sensors)
    gp_error = gp.print_error(alpha, sensor_bias, gaussian.matrix2d, estimate, true_data, gt_estimate)
    # gp.display(gaussian.space, space_points, gaussian.time, estimate, "Basic GP on the received data")

    # Building a GP that predicts alpha given bias
    calc_alpha = Calc_Alpha.CalcAlpha(sensors, sensor_time, data, true_sensors, sensor_time, true_data,
                                      space_kernel, time_kernel, kernel, noise, theta_not, alpha_mean, alpha_variance,
                                      sensor_bias)
    calc_alpha_estimate = calc_alpha.build(gaussian.space, gaussian.time, space_points)
    calc_alpha_gt_estimate = calc_alpha.build(true_sensors, true_sensor_time, N_true_sensors)
    calc_alpha_errors += calc_alpha.print_error(alpha, sensor_bias, gaussian.matrix2d, calc_alpha_estimate, true_data, calc_alpha_gt_estimate)

    # Building a GP that predicts the bias but is given alpha
    bias_gp = Constant_Bias.ConstantBias(sensors, sensor_time, data, true_sensors, sensor_time, true_data,
                                         space_kernel, time_kernel, kernel, noise, theta_not, bias_variance, bias_mean, alpha)
    constant_bias_estimate = bias_gp.build(gaussian.space, gaussian.time, space_points)
    constant_bias_gt_estimate = bias_gp.build(true_sensors, true_sensor_time, N_true_sensors)
    calc_constant_bias_errors += bias_gp.print_error(alpha, sensor_bias, gaussian.matrix2d, constant_bias_estimate, true_data, constant_bias_gt_estimate)

    # # Building a GP that predicts both bias and alpha using lagging variables
    # calc_both = Calc_Alpha_Calc_Constant_Bias.CalcBoth(sensors, sensor_time, data, true_sensors, sensor_time, true_data,
    #                                      space_kernel, time_kernel, kernel, noise, theta_not, bias_variance, bias_mean, alpha_mean, alpha_variance)
    # calc_both_estimate = calc_both.build(gaussian.space, gaussian.time)
    # calc_both_gt_estimate = calc_both.build(true_sensors, true_sensor_time)
    # calc_both.print_error(alpha, sensor_bias, gaussian.matrix2d, calc_both_estimate, true_data, calc_both_gt_estimate )
    #
    # # # Building a Gp that predicts bias and optimizes alpha
    # # opt_alpha_calc_bias = Opt_Alpha_Constant_Bias.OptAlphaCalcBias(sensors, sensor_time, data, true_sensors, sensor_time, true_data, space_kernel,
    # #                                                                time_kernel, kernel, noise, theta_not, bias_variance, bias_mean, alpha_variance, alpha_mean)
    # # opt_alpha_calc_bias_estimate = opt_alpha_calc_bias.build(gaussian.space, gaussian.time)
    # # opt_alpha_calc_bias_gt_estimate = opt_alpha_calc_bias.build(true_sensors, true_sensor_time)
    # # opt_alpha_calc_bias.print_error(alpha, sensor_bias, gaussian.matrix2d, opt_alpha_calc_bias_estimate, true_data, opt_alpha_calc_bias_gt_estimate)
    # #
    # # # Building a Gp that optimizes both alpha and bias
    # # opt_both = Wei_Model.OptBoth(sensors, sensor_time, data, true_sensors, sensor_time, true_data,
    # #                              space_kernel, time_kernel, kernel, noise, theta_not, bias_variance,
    # #                              bias_mean, alpha_variance, alpha_mean)
    # # opt_both_estimate = opt_both.build(gaussian.space, gaussian.time)
    # # opt_both_gt_estimate = opt_both.build(true_sensors, true_sensor_time)
    # # opt_both.print_error(alpha, sensor_bias.T[0], gaussian.matrix2d, opt_both_estimate, true_data, opt_both_gt_estimate)

    # Building a GP that predicts the bias but is given alpha
    changing_bias_gp = Calc_Bias_Changing_In_Time.ChangingBias(sensors, sensor_time, data, true_sensors, sensor_time,
                                                               true_data, space_kernel, time_kernel, kernel, noise,
                                                               theta_not, bias_variance, bias_mean, bias_kernel, alpha)
    changing_bias_estimate = changing_bias_gp.build(gaussian.space, gaussian.time, space_points)
    changing_bias_gt_estimate = changing_bias_gp.build(true_sensors, true_sensor_time, N_true_sensors)
    calc_changing_bias_error += changing_bias_gp.print_error(alpha, sensor_bias, gaussian.matrix2d, changing_bias_estimate, true_data, changing_bias_gt_estimate)
    # changing_bias_gp.display(gaussian.space, space_points, gaussian.time, changing_bias_estimate,
    #                          "GP with given alpha assuming the bias is changing in time")
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
    changing_bias_int_estimate = changing_bias_int_gp.build(gaussian.space, gaussian.time, space_points)
    changing_bias_int_gt_estimate = changing_bias_int_gp.build(true_sensors, true_sensor_time, N_true_sensors)
    calc_changing_int_bias_error += changing_bias_int_gp.print_error(alpha, sensor_bias, gaussian.matrix2d, changing_bias_int_estimate, true_data, changing_bias_int_gt_estimate)
    # changing_bias_int_gp.display(gaussian.space, gaussian.time, changing_bias_int_estimate,
    #                              "GP with given alpha assuming the bias is changing in time with integrated GP")
    #
    # # # Building a GP that predicts the bias but is given alpha
    # # opt_changing_bias_gp = Opt_Changing_Bais.OptChangingBias(sensors, sensor_time, data, true_sensors, sensor_time,
    # #                                                            true_data, space_kernel, time_kernel, kernel, noise,
    # #                                                            theta_not, bias_variance, bias_mean, changing_bias_gp.bias, alpha)
    # # opt_changing_bias_estimate = opt_changing_bias_gp.build(gaussian.space, gaussian.time)
    # # opt_changing_bias_gt_estimate = opt_changing_bias_gp.build(true_sensors, true_sensor_time)
    # # opt_changing_bias_gp.print_error(alpha, sensor_bias, gaussian.matrix2d, opt_changing_bias_estimate, true_data, opt_changing_bias_gt_estimate)
    # # opt_changing_bias_gp.display(gaussian.space, gaussian.time, changing_bias_estimate,
    # #                          "GP with given alpha and optimizing for a chaning bias")

    # Building a GP that predicts both bias and alpha using lagging variables
    calc_both_changing_bias_gp = Calc_Alpha_Calc_Changing_Bias.CalcBothChangingBias(sensors, sensor_time, data, true_sensors, sensor_time, true_data,
                                         space_kernel, time_kernel, kernel, noise, theta_not, bias_kernel, alpha_mean, alpha_variance)
    calc_both_changing_bias_estimate = calc_both_changing_bias_gp.build(gaussian.space, gaussian.time, space_points)
    calc_both_changing_bias_gt_estimate = calc_both_changing_bias_gp.build(true_sensors, true_sensor_time, N_true_sensors)
    calc_both_error += calc_both_changing_bias_gp.print_error(alpha, sensor_bias, gaussian.matrix2d, calc_both_changing_bias_estimate, true_data, calc_both_changing_bias_gt_estimate)
    # changing_bias_int_gp.display(gaussian.space, space_points, gaussian.time, calc_both_changing_bias_estimate,
    #                              "GP calculating both a changing bias and alpha with int gp")
    plt.show()
    print(i)


calc_alpha_errors = calc_alpha_errors/N_trials
calc_constant_bias_errors = calc_constant_bias_errors/N_trials
calc_changing_bias_error = calc_changing_bias_error/N_trials
calc_changing_int_bias_error = calc_changing_int_bias_error/N_trials
calc_both_error = calc_both_error/N_trials

print("Number of trails: " + str(N_trials))
print("Number of sensors: " + str(N_sensors**2))
print("Number of GT sensors: " + str(N_true_sensors**2))
print(gp_error)
print(calc_alpha_errors)
print(calc_constant_bias_errors)
print(calc_changing_bias_error)
print(calc_changing_int_bias_error)
print(calc_both_error)
