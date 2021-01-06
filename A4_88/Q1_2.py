from Q1_1 import get_data
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

train_x,train_y,test_x,test_y = get_data()

x = range(1,10)
rss = []
for i in x:
    km = KMeans(n_clusters=i).fit(train_x)
    rss.append(km.inertia_)
    print(km.cluster_centers_)
    print(i)

plt.plot(x,rss,marker = '^',markerfacecolor = 'red')
plt.title('Q1_2 : Elbow cluster ')
plt.xlabel('Number of clusters')
plt.ylabel('Error')
plt.show()

