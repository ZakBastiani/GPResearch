import math
import torch
import numpy as np
import pandas as pd
from Real_World_Example import Opt_all_but_bias_chi2
from Real_World_Example import Opt_alpha_calc_bias_chi2

torch.set_default_dtype(torch.float64)

space = torch.tensor(pd.read_csv('real_data/space_coords.csv', header=None).to_numpy())
time = torch.tensor(pd.read_csv('real_data/time_coords.csv', header=None).to_numpy())[:4]
data = torch.tensor(pd.read_csv('real_data/PM_data.csv', header=None).to_numpy())[:, :4].flatten()

gt_readings = torch.tensor(pd.read_csv('real_data/gt_space_time.csv', header=None).to_numpy())[:4]
gt_data = torch.tensor(pd.read_csv('real_data/gt_PM_data.csv', header=None).to_numpy()).flatten()[:4]

# Y values need to be zeroed
Y_mean = (sum(data) + sum(gt_data))/(len(data) + len(gt_data))

data = data - Y_mean
gt_data = gt_data - Y_mean

Y_sd = math.sqrt(sum(gt_data**2)/len(gt_data))  # Theta_not? needs ot be squared

# 'Known' Variables that need to be input
noise_sd = 6  # Set by lab experiments
bias_sd = 10  # Needs to be set by lab experiments
theta_sensor_time_bias = 100  # in std units
v = 15  # IDK how we want to set this, For tests this has been set to the number of sensors
t2 = 1.0  # IDK how we want to set this
theta_not = 400  # variance units

space_theta = 4000  # meters
time_theta = 0.25  # hours
alt_theta = 100  # meters


def bias_kernel(X, Y):
    kern = (bias_sd**2) * torch.exp(-((X.T[0].repeat(len(Y), 1) - Y.T[0].repeat(len(X), 1).T) ** 2) / (2 * theta_sensor_time_bias ** 2))
    return kern


print('Starting to optimize alpha calculating bias')
opt_alpha = Opt_alpha_calc_bias_chi2.OptAlphaCalcBias(space, time, data,  gt_readings, gt_data,
                                                      noise_sd, theta_not, bias_kernel, v, t2)

print("Opt_Alpha Alpha: " + str(opt_alpha.alpha))
pd.DataFrame(opt_alpha.bias.numpy()).to_csv('model\optalpha_bias.csv', header=False, index=False)

# opt_all_chi2 = Opt_all_but_bias_chi2.OptAll(space, time, data, gt_readings, gt_data, noise_sd,
#                                             theta_not, bias_kernel, v, t2)
# print("Opt_All Alpha: " + str(opt_all_chi2.alpha))
# print("Opt_All Space Theta: " + str(opt_all_chi2.space_theta))
# print("Opt_All Time Theta: " + str(opt_all_chi2.time_theta))
# print("Opt_All Alt Theta: " + str(opt_all_chi2.alt_theta))
# pd.DataFrame(opt_all_chi2.bias.numpy()).to_csv('model\optall_bias.csv', header=False, index=False)


