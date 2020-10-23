from scipy.io import loadmat
import pandas as pd
import numpy as np
from nfold import nFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
import csv
import matplotlib.pyplot as plt

def get_accuracy(pred,y):
    return np.sum(y == pred) / len(pred)

mat = loadmat('Assignment_2_datasets/dataset_1.mat')
samples = mat['samples']
labels = mat['labels'][0]

tr_acc = []
te_acc = []
tr_loss = []
te_loss = []


for i in range(5):

    train_s,train_l,test_s,test_l = nFold(samples,labels,5,i)
    model = LogisticRegression()
    model.fit(train_s,train_l)

    train_pred = model.predict(train_s)
    test_pred = model.predict(test_s)

    train_acc = get_accuracy(train_pred,train_l)
    test_acc = get_accuracy(test_pred,test_l)

    tr_acc.append(train_acc)
    te_acc.append(test_acc)
    print(i)


    
with open ('Q2_e.csv','w') as file:
    writer = csv.writer(file)
    writer.writerow(['Fold Number','Train Accuracy','Validation Accuracy'])
    for i in range(5):
        writer.writerow([i,tr_acc[i],te_acc[i]])

print('Mean of train accuracy = ',np.mean(tr_acc))
print('Mean of test/validation accuracy = ',np.mean(te_acc))