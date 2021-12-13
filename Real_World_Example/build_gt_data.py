import datetime
import torch
import pandas as pd
import utm
import json


# convenience/wrappers for the utm toolbox
def latlonToUTM(lat, lon):
    return utm.from_latlon(lat, lon)

f = open('real_data/DAQ-March-Dust.json',)
gt_data = json.load(f)

start = datetime.datetime(2000, 1, 1, 0, 0, 0, 0)
time_offset = 166643.25

gt_x = torch.zeros((len(gt_data), 4))
gt_y = torch.zeros((len(gt_data)))
for i in range(len(gt_data)):
    loc = latlonToUTM(gt_data[i]['Latitude'], gt_data[i]['Longitude'])
    gt_x[i][0] = loc[0]
    gt_x[i][1] = loc[1]
    if gt_data[i]['Sensor ID'] == 'Rose Park':
        gt_x[i][2] = 1289

    if gt_data[i]['Sensor ID'] == 'Hawthorne':
        gt_x[i][2] = 1310

    if gt_data[i]['Sensor ID'] == 'Copperview':
        gt_x[i][2] = 1356

    if gt_data[i]['Sensor ID'] == 'Herriman':
        gt_x[i][2] = 1524

    time_read = datetime.datetime.fromisoformat(gt_data[i]['Time'][:-1])
    gt_x[i][3] = (abs(time_read-start).total_seconds() / 3600.0) - time_offset
    gt_y[i] = gt_data[i]['PM2_5']

pd.DataFrame(gt_x.numpy()).to_csv('real_data\gt_space_time.csv', header=False, index=False)
pd.DataFrame(gt_y.numpy()).to_csv('real_data\gt_PM_data.csv', header=False, index=False)

