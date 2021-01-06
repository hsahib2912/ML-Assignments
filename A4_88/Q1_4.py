from Q1_1 import get_data
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import copy

train_x,train_y,test_x,test_y = get_data()

def ga(tmp,pred):
    return len(np.where(pred==tmp)[0])/float(len(pred))

def get_accuracy(pred,y):
    orig = np.array(y)
    l = []
    for i in range(3):
        for j in range (3):
            if i==j:
                continue
            for k in range(3):
                if i==k or k==j:
                    continue
                tmp = copy.deepcopy(y)
                tmp[y==0] = i
                tmp[y==1] = j
                tmp[y==2] = k
                l.append(ga(tmp,pred))

    return max(l)
                

km = KMeans(n_clusters=3).fit(train_x)


y_pred_train = km.predict(train_x)
y_pred_test = km.predict(test_x)

print("Train Accuracy = ",get_accuracy(y_pred_train,train_y))
print("Test/Validation Accuracy = ",get_accuracy(y_pred_test,test_y))
