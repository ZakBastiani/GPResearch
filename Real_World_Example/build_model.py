import math
import torch
import pandas as pd
from Real_World_Example import Opt_all_but_bias_chi2

space = torch.tensor(pd.read_csv('real_data/space_coords.csv').to_numpy())
time = torch.tensor(pd.read_csv('real_data/time_coords.csv').to_numpy())
data = torch.tensor(pd.read_csv('real_data/PM_data.csv').to_numpy()).flatten()

gt_readings = torch.tensor(pd.read_csv('real_data/gt_space_time.csv').to_numpy())
gt_data = torch.tensor(pd.read_csv('real_data/gt_PM_data.csv').to_numpy()).flatten()

# Y values need to be zeroed
Y_mean = (sum(data) + sum(gt_data))/(len(data) + len(gt_data))

data = data - Y_mean
gt_data = gt_data - Y_mean

Y_sd = math.sqrt(sum(gt_data**2)/len(gt_data))  # Theta_not? needs ot be squared

# 'Known' Variables that need to be input
noise_sd = 6  # Set by lab experiments
bias_sd = 10  # Needs to be set by lab experiments
theta_sensor_time_bias = 4  # in std units
v = 15  # IDK how we want to set this, For tests this has been set to the number of sensors
t2 = 1.0  # IDK how we want to set this
theta_not = 400  # variance units

space_theta = 4000  # meters
time_theta = 0.25  # hours
alt_theta = 100  # meters

def bias_kernel(X, Y):
    kern = (bias_sd**2) * torch.exp(-((X.repeat(len(Y), 1) - Y.repeat(len(X), 1).T) ** 2) / (2 * theta_sensor_time_bias ** 2))
    return kern


opt_all_chi2 = Opt_all_but_bias_chi2.OptAll(space, time, data, gt_readings, gt_data, noise_sd,
                                            theta_not, bias_kernel, v, t2)
# Need to figure out how I want to save and validate the model
print(opt_all_chi2.alpha)
