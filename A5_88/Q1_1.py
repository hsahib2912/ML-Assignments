import numpy as np
from Q1_get_data import get_data
from sklearn.manifold import TSNE
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt

train_x,train_y,test_x,test_y = get_data()

def plot(n,X):
    x = []
    y = []
    for i in range(len(test_y)):
        if(test_y[i] == n):
            
            x.append(X[i][0])
            y.append(X[i][1])

    return x,y



print("hi")
X = TSNE(n_components=2).fit_transform(test_x)
print("hi1")
'''plt.scatter(X[:,0],X[:,1],c = test_y)
plt.legend(['1','2','3','4','5','6','7'])
plt.show()
'''

for i in range(1,8):
    x,y = plot(i,X)
    plt.scatter(x,y)

plt.title('Q1_1 : Dataset Satellite visualization')
plt.legend(['1','2','3','4','5','6','7'],title = 'Category')
plt.show()
