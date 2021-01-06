from Q1_1 import get_data
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


train_x,train_y,test_x,test_y = get_data()
train_x = train_x.append(test_x)

km = KMeans(n_clusters=3).fit(train_x)
y = km.predict(train_x)

pca = PCA(n_components=2)
train_x = pca.fit_transform(train_x)

plt.scatter(train_x[:, 0], train_x[:, 1], c=y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Q1_3 : Scatter plot of clusters')
#plt.show()



