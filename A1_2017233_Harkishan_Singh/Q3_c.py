import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import random

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


train_samples[0]

def accuracy(pred,gt):
    return accuracy_score(pred,gt)


dt = []

for i in range(100):
    print(i)
    lst = random.sample(range(0,len(train_samples)),int(len(train_samples)*.5))

    tr_smp = []
    tr_lab = []
    for j in lst:
        tr_smp.append(list(train_samples.iloc(0)[j]))
        tr_lab.append(train_labels.iloc(0)[j])

    cls = DecisionTreeClassifier(max_depth=3)
    cls.fit(tr_smp,tr_lab)
    dt.append(cls)


maj_vote = []
for i in range(100):
    lst = dt[i].predict(test_samples)
    for j in range(len(lst)):
        maj_vote[]
    maj_vote.append(lst)

final = []

for i in range(len(maj_vote)):

