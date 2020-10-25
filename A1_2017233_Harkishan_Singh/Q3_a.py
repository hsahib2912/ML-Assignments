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

def train(ctr):
    cls = DecisionTreeClassifier(criterion = ctr)
    cls.fit(train_samples,train_labels)

    pred = cls.predict(test_samples)
    return accuracy(pred,test_labels)


gini_accuracy = train('gini')
entropy_accuracy = train('entropy')
print(gini_accuracy)
print (entropy_accuracy)


