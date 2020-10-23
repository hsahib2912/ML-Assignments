from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

mat = loadmat('Assignment_2_datasets/dataset_1.mat')
samples = mat['samples']
labels = mat['labels']

def plot(n):
    x = []
    y = []
    for i in range(len(labels[0])):
        if labels[0][i]==n:
            x.append(samples[i][0])
            y.append(samples[i][1])
    return x,y
    
sam0_x,sam0_y = plot(0)
sam1_x,sam1_y = plot(1)
plt.scatter(sam0_x,sam0_y)
plt.scatter(sam1_x,sam1_y)
plt.legend(['label : 0','label : 1'])
plt.show()



