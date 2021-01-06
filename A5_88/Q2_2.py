from sklearn.neural_network import MLPClassifier
import numpy as np
from Q2_1 import get_data

train_x,train_y,test_x,test_y = get_data()

def get_acc(pred,y):
    return len(np.where(pred == y)[0])/float(len(y))

clf = MLPClassifier(hidden_layer_sizes=(100,50,50),activation='logistic',verbose=2)
clf.fit(train_x,train_y)
pred = clf.predict(test_x)

print("Validation accuracy = ",get_acc(pred,test_y))

