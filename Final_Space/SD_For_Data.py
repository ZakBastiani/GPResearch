import torch
import os

# All files ending with .txt
files = os.listdir("/Users/Zak Bastiani/PycharmProjects/GPResearch/Final_Space/Data")

setMethod = 0
for file in files:
    with open("Data/" + file) as f:
        line = f.readlines()
        for i in range(len(line)):
            if line[i][0] == '-':
                print('Ahh')

            print(line[i])
