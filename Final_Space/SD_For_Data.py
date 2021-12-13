import math

import torch
import os

# All files ending with .txt
files = os.listdir("/Users/Zak Bastiani/PycharmProjects/GPResearch/Final_Space/Data")
models = ['Naive Gaussian Process', 'Ping Pong Fixed-Point', 'Opt Alpha', 'Opt Both']
for file in files:
    with open("Data/" + file) as f:
        line = f.readlines()
        alpha_error = [[], [], [], []]
        bias_error = [[], [], [], []]
        model_error = [[], [], [], []]
        model_num = 0

        for i in range(len(line)):
            if line[i][0] == '-':
                i += 1
                if model_num < 2:
                    alpha_error[model_num].append(abs(float(line[i][19:-2]) - float(line[i+1][20:-2])))
                else:
                    alpha_error[model_num].append(abs(float(line[i][19:-2]) - float(line[i+1][13:-1])))
                i += 2
                bias_error[model_num].append(float(line[i][26:-2]))
                if file != '0 GT Sensors.txt':
                    i += 2
                else:
                    i += 1
                model_error[model_num].append(float(line[i][29:-1]))
                model_num += 1
                if model_num >= 4:
                    model_num = 0

        print(file)
        for i in range(4):
            print(models[i])
            mean = sum(alpha_error[i])/len(alpha_error[i])
            holder = 0
            for j in range(len(alpha_error[i])):
                holder += (alpha_error[i][j] - mean)**2
            sd = math.sqrt(holder/(len(alpha_error[i])-1))
            print('SD of Alpha: ' + str(sd))

            mean = sum(bias_error[i])/len(bias_error[i])
            holder = 0
            for j in range(len(bias_error[i])):
                holder += (bias_error[i][j] - mean)**2
            sd = math.sqrt(holder/(len(bias_error[i])-1))
            print('SD of Bias: ' + str(sd))

            mean = sum(model_error[i])/len(model_error[i])
            holder = 0
            for j in range(len(model_error[i])):
                holder += (model_error[i][j] - mean)**2
            sd = math.sqrt(holder/(len(model_error[i])-1))
            print('SD of L2 Model Error: ' + str(sd))
        print()


