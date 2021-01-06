from Q2_3 import *
from Q2_1 import get_data
import numpy as np
from sklearn.naive_bayes import MultinomialNB


train_x,train_y,test_x,test_y = get_data()

vocab = get_vocab(train_x)

train_samples = get_doc_word_mat(train_x,vocab)
test_samples = get_doc_word_mat(test_x,vocab)

train_labels = np.array(train_y)
test_labels = np.array(test_y)

def get_accuracy(pred,y):
    return len(np.where(pred==y)[0])/float(len(pred))

def get_miss(pred,y,samples):
    miss = np.where(pred!=y)[0]
    for i in miss:
        print(i+700)
        print(samples[i])





clf = MultinomialNB(alpha=1)
clf.fit(train_samples,train_labels)


train_pred = clf.predict(train_samples)
test_pred = clf.predict(test_samples)


print("Train Accuracy = ",get_accuracy(train_pred,train_labels))
print("Test/Validation Accuracy = ",get_accuracy(test_pred,test_labels))

#get_miss(test_pred,test_labels,test_x)