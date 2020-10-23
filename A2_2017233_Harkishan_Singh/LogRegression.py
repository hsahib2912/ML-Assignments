from scipy.io import loadmat
import pandas as pd
import numpy as np
from nfold import nFold
from sklearn.metrics import mean_squared_error
import csv
import matplotlib.pyplot as plt

class LogRegression(object):
    """docstring for LogRegression."""
    def __init__(self):
        super(LogRegression, self).__init__()
        self.arg = 0
        self.coeff = None
        self.lr = 0.001
        self.max_iter = 500


    """You can give any required inputs to the fit()"""
    def fit(self,train_x,train_y,test_x,test_y):
        iter = []
        train_acc = []
        train_loss = []
        test_acc = []
        test_loss = []

        self.coeff = np.zeros(train_x.shape[1])
        for i in range(self.max_iter):

            print(-np.dot(train_x,self.coeff))
            y_hat = 1/(1+np.exp(-np.dot(train_x,self.coeff)))
            

            dcoeff = (1/train_x.shape[1]) * np.dot(train_x.T,y_hat-train_y)

            self.coeff-= self.lr*dcoeff

            train_pred = self.predict(train_x)
            train_accuracy = get_accuracy(train_pred,train_y)
            train_loss_l = get_loss(train_pred,train_y)

            test_pred = self.predict(test_x)
            test_accuracy = get_accuracy(test_pred,test_y)
            test_loss_l = get_loss(test_pred,test_y)

            iter.append(i)
            train_acc.append(train_accuracy)
            train_loss.append(train_loss_l)
            test_acc.append(test_accuracy)
            test_loss.append(test_loss_l)
            

        return iter,train_acc,train_loss,test_acc,test_loss


        """Write it from scratch. Usage of sklearn is not allowed"""



    """ You can add as many methods according to your requirements, but training must be using fit(), and testing must be with predict()"""


    def predict(self,X_test):

        y_hat = 1/(1+np.exp(-np.dot(X_test,self.coeff)))
        y_predicted = []
        for y in y_hat:
            if(y>.5):
                y_predicted.append(1)
            else:
                y_predicted.append(0)


        """Write it from scratch. Usage of sklearn is not allowed"""

        """Fill your code here. predict() should only take X_test and return predictions."""
        return y_predicted


def get_accuracy(pred,y):
    return np.sum(y == pred) / len(pred)

def get_loss(pred,y):
    return np.dot(pred-y,pred-y)/len(pred)

def plot_accuracy(iter,acc):
    plt.plot(iter,acc)

def plot_loss(iter,loss):
    plt.plot(iter,loss)

mat = loadmat('Assignment_2_datasets/dataset_1.mat')
samples = mat['samples']
labels = mat['labels'][0]
samples = np.append(samples,np.ones((len(labels),1)),axis = 1)          #Adding a column of 1 for constant

tr_acc = []
te_acc = []
tr_loss = []
te_loss = []


for i in range(5):

    train_s,train_l,test_s,test_l = nFold(samples,labels,5,i)
    model = LogRegression()
    iter,train_acc,train_loss,test_acc,test_loss = model.fit(train_s,train_l,test_s,test_l)


    plot_accuracy(iter,train_acc)
    plot_accuracy(iter,test_acc)
    plt.title("Fold "+str(i)+" Accuracy : ")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Accuray")
    plt.legend(['Training Accuracy','Validation Accuracy'])
    #plt.savefig('Graphs/Acc_fold_'+str(i)+'.png')
    #plt.show()
    plt.close()

    plot_loss(iter,train_loss)
    plot_loss(iter,test_loss)
    plt.title("Fold "+str(i)+" Loss : ")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Loss")
    plt.legend(['Training Loss','Validation Loss'])
    #plt.savefig('Graphs/Loss_fold_'+str(i)+'.png')
    #plt.show()
    plt.close()

    tr_acc.append(train_acc[len(train_acc)-1])
    te_acc.append(test_acc[len(train_acc)-1])
    tr_loss.append(train_loss[len(train_loss)-1])
    te_loss.append(test_loss[len(test_loss)-1])
    print(i)
    break


    
'''with open ('Q2_c.csv','w') as file:
    writer = csv.writer(file)
    writer.writerow(['Fold Number','Train Accuracy','Validation Accuracy','Train Loss','Validation Loss'])
    for i in range(5):
        writer.writerow([i,tr_acc[i],te_acc[i],tr_loss[i],te_loss[i]])'''

print('Mean of train accuracy = ',np.mean(tr_acc))
print('Mean of train loss = ',np.mean(tr_loss))
print('Mean of test/validation accuracy = ',np.mean(te_acc))
print('Mean of test/validation loss = ',np.mean(te_loss))
