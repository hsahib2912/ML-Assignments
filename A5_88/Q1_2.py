import numpy as np
from Q1_get_data import get_data
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import copy
import timeit

train_x,train_y,test_x,test_y = get_data()

class kNN:
    def __init__ (self,k,train_x,train_y):
        self.k = k
        self.train_x = train_x
        self.train_y = train_y

    def predict(self,test_x):
        pred = []
        for test_inst in test_x:
            dist = []
            for train_inst in train_x:
                dist.append(np.sum(abs(train_inst-test_inst)))
            
            pred.append(self.get_label(dist))

        return pred
            
    
    def get_label(self,dist):
        labels = []
        tmp = copy.deepcopy(dist)
        for i in range(self.k):
            m = min(tmp)
            labels.append(train_y[dist.index(m)])
            tmp.remove(m)
        return max(set(labels), key = labels.count)

def get_acc(pred,y):
    return len(np.where(pred == y)[0])/float(len(y))

def get_err(pred,y):
    return (len(y) - len(np.where(pred == y)[0]))/float(len(y))

k = []
acc = []
err = []
for i in range(2,15):
    start = timeit.default_timer()
    k.append(i)
    print(i)
    clf = kNN(i,train_x,train_y)
    pred = clf.predict(test_x)
    acc.append(get_acc(pred,test_y))
    err.append(get_err(pred,test_y))
    end = timeit.default_timer()
    print(end-start)

for i in range(len(k)):
    print('k = ',k[i],' Error = ',err[i])

'''plt.plot(k,acc,marker = 'o',markerfacecolor = 'red')
plt.xlabel("Number of neighbours")
plt.ylabel("Accuracy")
plt.title('Question 1 part 2 : Accuracy plot')
plt.show()
'''
plt.plot(k,err,marker = 'o',markerfacecolor = 'red')
plt.xlabel("Number of neighbours")
plt.ylabel("Error")
plt.title('Question 1 part 2 : Error plot')
plt.show()