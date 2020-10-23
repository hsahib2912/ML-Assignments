from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nfold import nFold


mat = loadmat('Assignment_2_datasets/dataset_2.mat')
samples = mat['samples']
labels = mat['labels'][0]
train_samples,train_labels,test_samples,test_labels = nFold(samples,labels,5,4)

clf = LogisticRegression(random_state=0).fit(samples, labels)
pred = clf.predict(samples)
print(accuracy_score(pred,labels))
print(clf.coef_)
print(clf.intercept_)