from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import csv

mat = loadmat('dataset_2.mat')
train_samples = mat['samples'][:14000]
test_samples = mat['samples'][14000:]
train_labels = mat['labels'][0][:14000]
test_labels = mat['labels'][0][14000:]


def accuracy(pred,gt):
    same = 0
    for i in range(len(gt)):
        if (gt[i]==pred[i]):
            same+=1
    acc = float(same)/len(gt)
    return acc

def compute(depth):
    cls = DecisionTreeClassifier(max_depth=depth)
    cls.fit(train_samples,train_labels)

    train_y_pred = cls.predict(train_samples) 
    test_y_pred = cls.predict(test_samples)

    train_acc = accuracy(train_y_pred,train_labels)
    test_acc = accuracy(test_y_pred,test_labels)

    return train_acc,test_acc

tabel = []
tabel.append(['Depth','Training Accuracy','Validation Accuracy'])
for i in range(1,20):
    train_acc, test_acc = compute(i)
    tabel.append([i,train_acc,test_acc])

with open ('Q2_b.csv','w') as file:
    writer = csv.writer(file)
    for i in tabel:
        writer.writerow(i)


