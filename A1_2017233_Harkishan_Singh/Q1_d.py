import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D


mat = loadmat('dataset_1.mat')
img = mat['samples']
lab = mat['labels']

img = img.reshape((50000,784))
img = img[:2000]
n_lab = lab[0][:2000]

print ("hi")
img_2d = TSNE(n_components=3,verbose = 2).fit_transform(img)


def plot(n):
    x = []
    y = []
    z = []
    for i in range(len(n_lab)):
        if(n_lab[i] == n):
            
            x.append(img_2d[i][0])
            y.append(img_2d[i][1])
            z.append(img_2d[i][2])

    return x,y,z

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(10):
    x,y,z = plot(i)
    ax.scatter3D(x,y,z)

ax.legend(['0','1','2','3','4','5','6','7','8','9'])
plt.show()

