import numpy as np
import matplotlib.pyplot as plt

# Variables that control the space
N_sensors = 10  # Number of sensors
N_true_sensors = 4  # Number of ground truth sensor points
N_time = 10  # Number of time samples
noise = 0.1  # random noise in the system

space_range = 10
time_range = 10
space_points = 50
time_points = 50

# Hyper-parameters
theta_space = 8
theta_time = 10
theta_not = 1

# Model Parameters
bias_variance = 0.25
bias_mean = 0
alpha_mean = 1
alpha_variance = 0.25

N_space_test = 50
N_time_test = 50

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
            kern[x][y] = theta_not * np.exp(- ((X[x][0] - Y[y][0]) ** 2)/ theta_space - ((X[x][1] - Y[y][1]) ** 2)/theta_time )
    return kern


# Displaying the function that we are trying to model.
space = np.linspace(0, space_range, space_points)
time = np.linspace(0, time_range, time_points)
points = np.ndarray(shape=(space_points*time_points, 2))
for i in range(space_points):
    for j in range(time_points):
        points[i*time_points + j][0] = space[i]
        points[i*time_points + j][1] = time[j]

kern = np.kron(space_kernel(space, space), time_kernel(time, time))


mean = np.zeros(len(points))

ret_matrix = np.random.multivariate_normal(mean, kern, 1)

holder = np.ndarray(shape=(space_points, time_points))
for i in range(0, space_points):
    for j in range(0, time_points):
        holder[i][j] = ret_matrix[0][i * time_points + j]

fig = plt.figure(1)
ax = plt.axes(projection='3d')

ax.scatter(points.T[0], points.T[1], ret_matrix[0])
ax.set_xlabel('Space')
ax.set_ylabel('Time')
ax.set_zlabel('Data')
ax.set_title('Basic random Gaussian function scatter')

plt.figure(2)
ax = plt.axes(projection='3d')
ax.plot_surface(np.repeat([space], time_points, axis=0).T, np.repeat([time], space_points, axis=0), holder,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('Space')
ax.set_ylabel('Time')
ax.set_zlabel('Data')
ax.set_title('Basic random Gaussian function surface')

# Build a basic GP of the overall system.
K = kernel(points, points)
L = np.linalg.cholesky(K + noise * np.eye(len(points)))
# Test space
test_space = np.linspace(0, space_range, N_space_test)
test_time = np.linspace(0, time_range, N_time_test)
test_points = np.ndarray(shape=(N_space_test * N_time_test, 2))
for i in range(N_space_test):
    for j in range(N_time_test):
        test_points[i * N_time_test + j][0] = test_space[i]
        test_points[i * N_time_test + j][1] = test_time[j]

Lk = np.linalg.solve(L, kernel(points, test_points))
mu = np.dot(Lk.T, np.linalg.solve(L, ret_matrix[0]))

# Should just be able to use reshape fix later
# mu = np.reshape([mu], (test_space, test_time))
holder = np.ndarray(shape=(N_space_test, N_time_test))
for i in range(0, N_space_test):
    for j in range(0, N_time_test):
        holder[i][j] = mu[i * N_time_test + j]

plt.figure(3)
ax = plt.axes(projection='3d')
ax.scatter(test_points.T[0], test_points.T[1], mu)
ax.set_xlabel('Space')
ax.set_ylabel('Time')
ax.set_zlabel('Data')
ax.set_title('Gaussian Model scatter')

plt.figure(4)
ax = plt.axes(projection='3d')
ax.plot_surface(np.repeat([test_space], N_time_test, axis=0).T, np.repeat([test_time], N_space_test, axis=0), holder,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('Space')
ax.set_ylabel('Time')
ax.set_zlabel('Data')
ax.set_title('Gaussian Model surface')
plt.show()