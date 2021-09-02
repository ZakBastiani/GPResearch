import numpy as np
import matplotlib.pyplot as plt
import torch

# Create of function space based on Wei's Model
# this needs to be a function in space and time with respect to PPM
# F(x,t) + b + noise

# Variables that control the space
N_sensors = 10  # Number of sensors
N_true_sensors = 2  # Number of ground truth sensor points
N_time = 10  # Number of time samples
noise = 0.01  # random noise in the system

space_range = 10
time_range = 10

# Hyper-parameters
theta_space = 8
theta_time = 10
theta_not = 1

# Model Parameters
bias_variance = 1
bias_mean = 0
alpha_mean = 1
alpha_variance = 0.25

N_space_test = 50
N_time_test = 50

# setting the seed for the program
np.random.seed(0)


# Kernel that Wei defined in his paper simplified
def space_kernel(X, Y):
    kernel = np.ndarray(shape=(len(X), len(Y)))
    for x in range(0, len(X)):
        for y in range(0, len(Y)):
            kernel[x][y] = theta_not * np.exp(- ((X[x] - Y[y]) ** 2) / theta_space)
    return kernel


def time_kernel(X, Y):
    kernel = np.ndarray(shape=(len(X), len(Y)))
    for x in range(0, len(X)):
        for y in range(0, len(Y)):
            kernel[x][y] = np.exp(-((X[x] - Y[y]) ** 2) / theta_time)
    return kernel


def kernel(X, Y):
    kern = np.ndarray(shape=(len(X), len(Y)))
    for x in range(0, len(X)):
        for y in range(0, len(Y)):
            kern[x][y] = theta_not * np.exp(
                - ((X[x][0] - Y[y][0]) ** 2) / theta_space - ((X[x][1] - Y[y][1]) ** 2) / theta_time)
    return kern


# Maximizing the predicted bias based on direct function
def calc_bias_matrix(sigma, X, Y, true_X, true_Y, alpha):
    sigma_hat_inv = np.linalg.inv(sigma * (alpha ** 2) + noise * np.eye(len(sigma)))

    def k_star_calc(x):
        k = kernel(x, X.T)
        return k

    # Build and calc A and C
    A = np.zeros(shape=(N_sensors, N_sensors))
    C = np.zeros(shape=(1, N_sensors))
    for k in range(0, N_sensors):
        extend_bias = np.zeros(shape=(N_sensors * N_time, 1))
        for j in range(0, N_time):
            extend_bias[k * N_time + j][0] = 1
        current_A = np.zeros(shape=(1, N_sensors * 10))
        current_C = 0
        for n in range(len(true_X.T)):
            k_star = k_star_calc([true_X.T[n]]).T
            holder = (k_star.T @ sigma_hat_inv @ k_star)[0][0]
            holder2 = (k_star.T @ sigma_hat_inv @ extend_bias) * (k_star.T @ sigma_hat_inv)
            current_A += holder2 / (theta_not/alpha**2 - holder)
            current_C += ((k_star.T @ sigma_hat_inv @ Y) * (k_star.T @ sigma_hat_inv @ extend_bias)
                          - true_Y[n] * (k_star.T @ sigma_hat_inv @ extend_bias)) / (theta_not/alpha**2 - holder)
        current_A += (sigma_hat_inv @ extend_bias).T
        # Need to condense current_A into b_i variables
        for i in range(0, N_sensors):
            sum = 0
            for j in range(0, N_time):
                sum += current_A[0][i * N_time + j]
            A[k][i] = sum
        A[k][k] += 1 / bias_variance

        C[0][k] = Y.T @ sigma_hat_inv @ extend_bias + current_C + bias_mean / bias_variance

    # Inverse A and multiply it by C
    A_inverse = np.linalg.inv(A)
    b = C @ A_inverse
    return b


def calc_alpha(bias_matrix, data, sigma):
    return 1


# Displaying the function that we are trying to model.
space = np.linspace(0, space_range, N_space_test)
time = np.linspace(0, time_range, N_time_test)
points = np.ndarray(shape=(N_space_test*N_time_test, 2))
for i in range(N_space_test):
    for j in range(N_time_test):
        points[i*N_time_test + j][0] = space[i]
        points[i*N_time_test + j][1] = time[j]

kern = np.kron(space_kernel(space, space), time_kernel(time, time))

mean = np.zeros(len(points))

ret_matrix = np.random.multivariate_normal(mean, kern, 1)

holder = np.ndarray(shape=(N_space_test, N_time_test))
for i in range(0, N_space_test):
    for j in range(0, N_time_test):
        holder[i][j] = ret_matrix[0][i * N_time_test + j]

# fig = plt.figure(1)
# ax = plt.axes(projection='3d')
#
# ax.scatter(points.T[0], points.T[1], ret_matrix[0])
# ax.set_xlabel('Space')
# ax.set_ylabel('Time')
# ax.set_zlabel('Data')
# ax.set_title('Basic random Gaussian function scatter')

plt.figure(1)
ax = plt.axes(projection='3d')
ax.plot_surface(np.repeat([space], N_time_test, axis=0).T, np.repeat([time], N_space_test, axis=0), holder,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('Space')
ax.set_ylabel('Time')
ax.set_zlabel('Data')
ax.set_title('Basic random Gaussian function surface')

L_function = np.linalg.cholesky(kern + noise*np.eye(len(kern)))


def f(x_points, y_points):
    Lk_function = np.linalg.solve(L_function, np.kron(space_kernel(space, x_points),
                                     time_kernel(time, y_points)))
    mu_function = np.dot(Lk_function.T, np.linalg.solve(L_function, ret_matrix[0]))

    # Should just be able to use reshape fix later
    # mu = np.reshape([mu], (test_space, test_time))
    output = np.ndarray(shape=(len(x_points), len(y_points)))
    for i in range(0, len(x_points)):
        for j in range(0, len(y_points)):
            output[i][j] = mu_function[i * len(y_points) + j]
    return output


# Select the location of the sensors, and extend them through time as they are constant
sensors = np.linspace(0, space_range, N_sensors)
# Set the time interval and extend through space
sensor_time = np.linspace(0, space_range, N_time)

# Set the bias of the Sensors and extend through time as well
sensor_bias = np.outer(np.random.normal(bias_mean, bias_variance, size=N_sensors), np.ones(N_time))
alpha = np.random.normal(alpha_mean, alpha_variance, size=1)

# Data received with sensor bias
# data = f(sensors, sensor_time) + noise*np.random.randn(N_sensors, N_time)
data = alpha*f(sensors, sensor_time) + sensor_bias + noise * np.random.randn(N_sensors, N_time)

# Selecting the location of the ground truth points
true_sensors = np.linspace(0, space_range, N_true_sensors)
# Setting the time matrix for the true sensors
true_sensor_time = np.linspace(0, space_range, N_time)

true_data = f(true_sensors, true_sensor_time)

print(sensors)
print(sensor_time)
print(sensor_bias)
print(true_sensors)
print(true_data)
#
# Plot the location of the sensors with each sensors bias
fig = plt.figure(2)
plt.clf()
plt.plot(sensors, sensor_bias.T[0], 'r+', ms=20)
plt.xlabel('Space')
plt.ylabel('Bias')
plt.title('Sensors bias')
plt.savefig('Sensors bias')
# Showing the new data received with the bias
fig = plt.figure(3)
ax = plt.axes(projection='3d')

ax.scatter(np.outer(sensors, np.ones(N_time)), np.outer(sensor_time, np.ones(N_sensors)).T, data)
ax.set_xlabel('Space')
ax.set_ylabel('Time')
ax.set_zlabel('Data')
ax.set_title('data with sensor Sensors bias')
plt.savefig('data with sensor Sensors bias')

# Build a basic GP of the overall system.
K = np.kron(space_kernel(sensors, sensors), time_kernel(sensor_time, sensor_time))
L = np.linalg.cholesky(K + noise * np.eye(N_sensors * N_time))

# Test space
test_space = np.linspace(0, space_range, N_space_test)
test_time = np.linspace(0, space_range, N_time_test)

holder = np.kron(space_kernel(sensors, test_space), time_kernel(sensor_time, test_time))
Lk = np.linalg.solve(L, np.kron(space_kernel(sensors, test_space), time_kernel(sensor_time, test_time)))
mu = np.dot(Lk.T, np.linalg.solve(L, data.flatten()))

# Should just be able to use reshape fix later
# mu = np.reshape([mu], (test_space, test_time))
holder = np.ndarray(shape=(N_space_test, N_time_test))
for i in range(0, N_space_test):
    for j in range(0, N_time_test):
        holder[i][j] = mu[i * N_time_test + j]

real_answer_test = f(test_space, test_time)
gp_error = sum(abs((real_answer_test - holder)).flatten())/(N_space_test*N_time_test)
print("Average Error in GP points: " + str(gp_error))

# Plot the GP and compare it to the original function.
# PLOTS:
plt.figure(4)
ax = plt.axes(projection='3d')
ax.plot_surface(np.repeat([test_space], N_time_test, axis=0).T, np.repeat([test_time], N_space_test, axis=0), holder,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('Space')
ax.set_ylabel('Time')
ax.set_zlabel('Data')
ax.set_title('Gaussian Model without solving for bias')
plt.savefig('Gaussian Model without solving for bias')

total_sensor_points = np.array((np.outer(sensors, np.ones(N_time)).flatten(),
                                np.outer(sensor_time, np.ones(N_sensors)).T.flatten()))
total_true_points = np.array((np.outer(true_sensors, np.ones(N_time)).flatten(),
                              np.outer(true_sensor_time, np.ones(N_true_sensors)).T.flatten()))
# Lets try calculating the bias
bias_guess = calc_bias_matrix(K, total_sensor_points, data.flatten(), total_true_points, true_data.flatten(), alpha)
data = (data - np.outer(bias_guess.T, np.ones(N_time)))/alpha

print("guess: " + str(bias_guess/alpha))
print("actual: " + str(sensor_bias.T[0]))
print("True Alpha: " + str(alpha))
print("Avg Bias Error: " + str(sum(abs((bias_guess/alpha - sensor_bias.T[0])).flatten())/N_sensors))
real_answer = f(sensors, sensor_time)
error = sum(abs((real_answer - data)).flatten())/(N_sensors*N_time)
print('Avg in sensor points Error: ' + str(error))

sensors = np.concatenate((sensors, true_sensors))
data = np.concatenate((data, true_data))

# Build a basic GP of the data after removing predicted bias
K = np.kron(space_kernel(sensors, sensors), time_kernel(sensor_time, sensor_time))
L = np.linalg.cholesky(K*(alpha**2) + noise * np.eye((N_sensors + N_true_sensors) * N_time))

# Test space
test_space = np.linspace(0, space_range, N_space_test)
test_time = np.linspace(0, space_range, N_time_test)

holder = np.kron(space_kernel(sensors, test_space), time_kernel(sensor_time, test_time))
Lk = np.linalg.solve(L, np.kron(space_kernel(sensors, test_space), time_kernel(sensor_time, test_time)))
mu = np.dot(Lk.T, np.linalg.solve(L, data.flatten()))

# Should just be able to use reshape fix later
# mu = np.reshape([mu], (test_space, test_time))
holder = np.ndarray(shape=(N_space_test, N_time_test))
for i in range(0, N_space_test):
    for j in range(0, N_time_test):
        holder[i][j] = mu[i * N_time_test + j]

gp_error = sum(abs((real_answer_test - holder)).flatten())/(N_space_test*N_time_test)
print("Average Error in GP points: " + str(gp_error))

# Plot the GP and compare it to the original function.
# PLOTS:
plt.figure(5)
ax = plt.axes(projection='3d')
ax.plot_surface(np.repeat([test_space], N_time_test, axis=0).T, np.repeat([test_time], N_space_test, axis=0), holder,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('Space')
ax.set_ylabel('Time')
ax.set_zlabel('Data')
ax.set_title('Gaussian Model with solving for bias')
plt.savefig('Gaussian Model with solving for bias')
plt.show()
