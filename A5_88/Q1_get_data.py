import numpy as np
import csv
import pandas as pd


def load(file):
    x=[]
    y=[]
    with open (file,'r') as f:
        reader = csv.reader(f,delimiter = ' ')
        for i in reader:
            tmp = [int(j) for j in i]
            x.append(tmp[:36])
            y.append(tmp[36])

    return np.array(x),np.array(y)

def get_data():
    train_x,train_y = load('sat.trn')
    test_x,test_y = load('sat.tst')
    return train_x,train_y,test_x,test_y

