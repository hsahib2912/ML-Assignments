from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

mat = loadmat('Assignment_2_datasets/dataset_2.mat')
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

for i in range(4):
    x,y = plot(i)
    plt.scatter(x,y)

plt.legend(['0','1','2','3'])
plt.show()