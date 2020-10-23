from scipy.io import loadmat
import pandas as pd
import numpy as np
from nfold import nFold
from sklearn.metrics import mean_squared_error
import csv
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

class LogRegression(object):
    def __init__(self):
        super(LogRegression, self).__init__()
        self.arg = 0
        self.coeff = None
        self.lr = 0.001
        self.max_iter = 57

    def fit(self,train_x,train_y):
        

        self.coeff = np.zeros(train_x.shape[1])
        for i in range(self.max_iter):
            y_hat = 1/(1+np.exp(-np.dot(train_x,self.coeff)))
            dcoeff = (1/train_x.shape[1]) * np.dot(train_x.T,y_hat-train_y)
            self.coeff-= self.lr*dcoeff

            
    def predict(self,X_test):

        return np.exp(np.dot(X_test,self.coeff))
        '''y_predicted = []
        for y in y_hat:
            if(y>.5):
                y_predicted.append(1)
            else:
                y_predicted.append(0)

        return np.array(y_predicted)'''


def get_accuracy(pred,y):
    return np.sum(y == pred) / len(pred)

def get_loss(pred,y):
    return np.dot(pred-y,pred-y)/len(pred)

def plot_accuracy(iter,acc):
    plt.plot(iter,acc)

def plot_loss(iter,loss):
    plt.plot(iter,loss)

def get_binary_data(train_x,train_y,a,b):
    ind = np.where(train_y==a)[0]
    ind = np.append(ind,np.where(train_y==b)[0])
    return train_x[ind],train_y[ind]


mat = loadmat('Assignment_2_datasets/dataset_2.mat')
samples = mat['samples']
labels = mat['labels'][0]
samples = np.append(samples,np.ones((len(labels),1)),axis = 1)          #Adding a column of 1 for constant


classifiers = []
for i in range(3):      #c classes require c-1 models
    print(i)
    X,y = get_binary_data(samples,labels,i,3)
    clf = LogRegression()
    clf.fit(X,y)
    pred = clf.predict(samples)
    classifiers.append(pred)


print (classifiers[0])
print (classifiers[1])
print (classifiers[2])
lcls = []    
for i in range(len(classifiers[0])):
    print(i)
    lcls.append(1 - classifiers[0][i] - classifiers[1][i] - classifiers[2][i] )
    classifiers.append(np.array(lcls))

print(lcls)





    

    

