import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Create of function space based on Wei's Model
# this needs to be a function in space and time with respect to PPM
# F(x,t) + b + noise

# Variables that control the space
N_sensors = 10  # Number of sensors
N_true_sensors = 4  # Number of ground truth sensor points
N_time = 10  # Number of time samples
noise = 0.01  # random noise in the system

space_range = 10
time_range = 10
space_points = 50
time_points = 50

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

plt.figure(1)
ax = plt.axes(projection='3d')
ax.plot_surface(np.repeat([space], time_points, axis=0).T, np.repeat([time], space_points, axis=0), holder,
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


# Need to alter the sensor matrix and the data matrix
X = torch.tensor([np.outer(sensors, np.ones(N_time)).flatten(),
                  np.outer(sensor_time, np.ones(N_sensors)).T.flatten()]).T
Y = torch.reshape(torch.tensor(data), (-1, 1))

Xt = torch.tensor([np.outer(true_sensors, np.ones(N_time)).flatten(),
                   np.outer(true_sensor_time, np.ones(N_true_sensors)).T.flatten()]).T
Yt = torch.reshape(torch.tensor(true_data), (-1, 1))

jitter = noise


class gpr(nn.Module):
    def __init__(self, X, Y):  # Basic constructor
        super(gpr, self).__init__()
        self.X = X
        self.Y = Y
        self.log_beta = nn.Parameter(torch.zeros(1))
        self.log_length_scale = nn.Parameter(torch.zeros(X.size(1)))
        self.log_scale = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(N_sensors))
        self.gain = nn.Parameter(torch.ones(1))

    def K_cross(self, X, X2):  # Building K, which is used to calculate sigma
        length_scale = torch.exp(self.log_length_scale).view(1, -1)

        X = X / length_scale.expand(X.size(0), -1)
        X2 = X2 / length_scale.expand(X2.size(0), -1)

        X_norm2 = torch.sum(X * X, dim=1).view(-1, 1)
        X2_norm2 = torch.sum(X2 * X2, dim=1).view(-1, 1)

        K = -2.0 * X @ X2.t() + X_norm2.expand(
            X.size(0), X2.size(0)) + X2_norm2.t().expand(
                X.size(0), X2.size(0))
        K = self.log_scale.exp() * torch.exp(-K)
        return K

    def forward(self, Xte): #Moving forward one step
        # with torch.no_grad():
        n_test = Xte.size(0)
        Sigma = self.K_cross(
            self.X, self.X) + torch.exp(self.log_beta).pow(-1) * torch.eye(
                self.X.size(0)) + jitter * torch.eye(self.X.size(0))
        kx = self.K_cross(Xte, self.X)

        y_bias = self.bias.view(-1, 1).repeat(N_time, 1)
        Y = self.Y * self.gain - y_bias
        # via cholesky decompositon
        L = torch.cholesky(Sigma)
        mean = kx @ torch.cholesky_solve(Y, L)
        alpha = L.inverse() @ kx.t()
        var_diag = self.log_scale.exp().expand(
            n_test, 1) - (alpha.t() @ alpha).diag().view(-1, 1)


        return mean, var_diag

    def neg_log_likelihood(self):
        Sigma = self.K_cross(
            self.X, self.X) + torch.exp(self.log_beta).pow(-1) * torch.eye(
                self.X.size(0)) + jitter * torch.eye(self.X.size(0))
        y_bias = self.bias.view(-1, 1).repeat(N_time, 1)
        Y = self.Y * self.gain - y_bias
        prob = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(self.X.size(0)), Sigma)
        return -prob.log_prob(Y.t())


# setting the model and then using torch to optimize
model = gpr(X, Y)
# optimizer = torch.optim.LBFGS(model.parameters(), lr=0.001)  #lr is very important, lr>0.1 lead to failure
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

lossFunc = nn.MSELoss()
for i in range(100):
    # optimizer.zero_grad()
    # LBFGS
    def closure():
        optimizer.zero_grad()
        loss = model.neg_log_likelihood()
        print('nll:', loss.item())
        loss.backward()
        return loss

    # optimizer.step(closure)

    # adam
    # check loss functions as they are returning arrays
    optimizer.zero_grad()
    # loss1 = model.neg_log_likelihood()/3      %average nll
    loss1 = model.neg_log_likelihood()

    ypred, yvar = model(Xt)
    prob = torch.distributions.multivariate_normal.MultivariateNormal(
        ypred.t().squeeze(),
        yvar.squeeze().diag_embed())
    loss2 = -prob.log_prob(Yt.t().squeeze())

    if loss2 < 0:
        loss = loss1 + loss2
    else:
        loss = loss2

    # loss =  loss1 + loss2
    loss.backward()
    optimizer.step()
    print(
        'loss1:',
        loss.item(),
        'loss2:',
        loss2.item(),
    )

print(model.bias)
print(sensor_bias.T[0])
print(model.gain)
print(alpha)

print('model alpha: ' + str(model.gain))
print('True alpha: ' + str(alpha))
print('model bias : ' + str(model.bias))
print('Predicted Model bias: ' + str(model.bias / model.gain))
print('Actual bias: ' + str(sensor_bias.T[0]))

all_sensors = np.concatenate((sensors, true_sensors))
data_with_predictions = (data - np.outer(model.bias.detach().numpy(), np.ones(N_time))) / model.gain.detach().numpy()
all_data = np.concatenate((data_with_predictions, true_data))
real_answer = f(sensors, sensor_time)
error = sum(abs(real_answer - data_with_predictions).flatten())/(N_sensors*N_time)
print('Avg Error in Sensor Points: ' + str(error))

# Build a GP of the whole system with wei's bias prediction
K_zak = np.kron(space_kernel(all_sensors, all_sensors), time_kernel(sensor_time, sensor_time))
L_zak = np.linalg.cholesky(K_zak*(model.gain.detach().numpy()**2) + noise * np.eye((N_sensors + N_true_sensors) * N_time))

# Test space
test_space = np.linspace(0, space_range, N_space_test)
test_time = np.linspace(0, space_range, N_time_test)

Lk_zak = np.linalg.solve(L_zak,
                         np.kron(space_kernel(all_sensors, test_space), time_kernel(sensor_time, test_time)))
mu_zak = np.dot(Lk_zak.T, np.linalg.solve(L_zak, all_data.flatten()))

# Should just be able to use reshape fix later
# mu = np.reshape([mu], (test_space, test_time))
holder_zak = np.ndarray(shape=(N_space_test, N_time_test))
for i in range(0, N_space_test):
    for j in range(0, N_time_test):
        holder_zak[i][j] = mu_zak[i * N_time_test + j]

real_answer_test = f(test_space, test_time)
gp_error = sum(abs((real_answer_test - holder_zak)).flatten())/(N_space_test*N_time_test)
print("Average Error in GP points afterting solving: " + str(gp_error))


# Plot the GP with wei's bais and compare it to the original function.
# PLOTS:
plt.figure(7)
ax = plt.axes(projection='3d')
ax.plot_surface(np.repeat([test_space], N_time_test, axis=0).T, np.repeat([test_time], N_space_test, axis=0),
                holder_zak,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('Space')
ax.set_ylabel('Time')
ax.set_zlabel('Data')
ax.set_title('Gaussian Model predicted bias')
# plt.figure(8)
# ax = plt.axes(projection='3d')
# ax.scatter(np.repeat(test_space, N_time_test), np.repeat([test_time], N_space_test, axis=0).flatten(), mu)
# ax.set_xlabel('Space')
# ax.set_ylabel('Time')
# ax.set_zlabel('Data')
# ax.set_title('Gaussian Model without solving for bias')
plt.show()