import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('PRSA_data_2010.1.1-2014.12.31.csv')
df.drop('No',axis = 'columns', inplace = True)
df.drop('cbwd',axis = 'columns', inplace = True)

df.dropna(subset = ['pm2.5'],inplace = True)
df = df.reset_index(drop = True)

samples = df.loc[:, df.columns != 'month']
labels = df['month']

train_samples = samples[:int(0.8*len(samples))]
train_labels = labels[:int(0.8*len(labels))]
test_samples = samples[int(0.8*len(samples)):]
test_labels = labels[int(0.8*len(labels)):]

def accuracy(pred,gt):
    return accuracy_score(pred,gt)

def compute(depth):
    cls = DecisionTreeClassifier(max_depth=depth)
    cls.fit(train_samples,train_labels)

    train_y_pred = cls.predict(train_samples) 
    test_y_pred = cls.predict(test_samples)

    train_acc = accuracy(train_y_pred,train_labels)
    test_acc = accuracy(test_y_pred,test_labels)

    return train_acc,test_acc


tabel = []
train_y = []
test_y = []
tabel.append(['Depth','Training Accuracy','Validation Accuracy'])
depth = [2, 4, 8, 10, 15, 30]
for i in depth:
    train_acc, test_acc = compute(i)
    train_y.append(train_acc)
    test_y.append(test_acc)
    tabel.append([i,train_acc,test_acc])

with open ('Q3_b.csv','w') as file:
    writer = csv.writer(file)
    for i in tabel:
        writer.writerow(i)

plt.plot(depth,train_y)
plt.plot(depth,test_y)
plt.xlabel("Depth of the tree")
plt.ylabel("Accuracy")
plt.title('Q3_b')
plt.legend(['training','validation'])
plt.show()