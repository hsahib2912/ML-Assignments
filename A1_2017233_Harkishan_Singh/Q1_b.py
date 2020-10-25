from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

mat = loadmat('dataset_2.mat')
samples = mat['samples'] 
labels = mat['labels']

def plot(lab,col):
    u_samples = []
    for i in range(len(labels[0])):
        if (lab == labels[0][i]):
            u_samples.append(samples[i])
    
    u_samples = np.array(u_samples)
    s = plt.scatter(u_samples[:,0],u_samples[:,1],color = col)
    return s

plt0 = plot(0,'r')
plt1 = plot(1,'m')
plt2 = plot(2,'y')
plt3 = plot(3,'c')
plt.legend ((plt0,plt1,plt2, plt3 ),
            ('0','1','2','3'),title  = 'Categories')
plt.show()





