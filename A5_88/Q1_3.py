import numpy as np
from Q1_get_data import get_data
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

train_x,train_y,test_x,test_y = get_data()

def get_acc(pred,y):
    return len(np.where(pred == y)[0])/float(len(y))

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(train_x,train_y)

train_pred = clf.predict(train_x)
test_pred = clf.predict(test_x)

train_acc = get_acc(train_pred,train_y)
test_acc = get_acc(test_pred,test_y)

print("Training accuracy : ",train_acc)
print("Testing/Validation accuracy : ",test_acc)