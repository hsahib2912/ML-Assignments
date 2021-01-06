import numpy as np
from Q2_1 import get_data
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPClassifier
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt


def plot_decision_boundary(test_x,test_y,clf,a):
    x_min, x_max = test_x[:, 0].min() - 1, test_x[:, 0].max() + 1
    y_min, y_max = test_x[:, 1].min() - 1, test_x[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(test_x[:, 0], test_x[:, 1], c=test_y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_ylabel('X[1] : y')
    ax.set_xlabel('X[0] : x')
    ax.set_xticks(())
    ax.set_yticks(())
    st = 'Alpha = '+str(a)
    ax.set_title(st)
    ax.legend()
    plt.show()
    plt.close()

    print(Z)
    print(Z.shape)

    print(x_min)
    print(x_max)
    print(y_min)
    print(y_max)




train_x,train_y,test_x,test_y = get_data()
print(train_y)
print(np.unique(train_y))
test_x = TSNE(n_components=2,verbose=2).fit_transform(test_x)
print(test_x.shape)
alp = [0,0.1,1]

for a in alp:
    clf = MLPClassifier(hidden_layer_sizes=(100,50,50),activation='logistic',verbose=2,alpha=a)
    clf.fit(test_x,test_y)
    plot_decision_boundary(test_x,test_y,clf,a)




