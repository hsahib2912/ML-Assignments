from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


mat = loadmat('dataset_2.mat')
train_samples = mat['samples'][:14000]
test_samples = mat['samples'][14000:]
train_labels = mat['labels'][0][:14000]
test_labels = mat['labels'][0][14000:]

def accuracy(y_pred):
    same = 0
    for i in range(len(y_pred)):
        if (y_pred[i]==test_labels[i]):
            same+=1
    acc = float(same)/len(y_pred)
    return acc


def compute(depth):
    cls = DecisionTreeClassifier(max_depth=depth)
    cls.fit(train_samples,train_labels)
    y_pred = cls.predict(test_samples)
    return accuracy(y_pred)

x = []
y = []
for i in range(1,20):
    print(i)
    x.append(i)
    y.append(compute(i))

plt.plot(x, y, marker='o',markerfacecolor='red')
plt.xlabel("Depth of the tree")
plt.ylabel("Accuracy")
plt.title("Question 2 a) Accuracy of Classification")
plt.show()
