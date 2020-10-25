from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

mat = loadmat('dataset_1.mat')
labels = mat['labels']


def save_image(num):
    ind = np.where(labels == num)[1]
    for i in range(10):
        plt.imshow(mat['samples'][ind[i]],cmap='Greys')
        plt.axis('off')
        plt.savefig(str(num)+"_"+str(i)+".png",bbox_inches = 'tight',pad_inches = 0)


for i in range(10):
    save_image(i)

        
