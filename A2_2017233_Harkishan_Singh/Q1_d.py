import numpy as np
import pandas as pd
import csv
from nfold import nFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

def load_data():
    df = pd.read_csv('Assignment_2_datasets/regression_data/nDataset.csv',header = None)
    samples = df[df.columns[:8]].to_numpy()
    labels = df[df.columns[8]].to_numpy()
    return samples,labels

def mse(pred,test_l):
    k = pred-test_l
    k = np.multiply(k,k)
    return np.mean(k)


samples,labels = load_data()

fold = []
tr_err = []
te_err = []
sk_tr_err = []
sk_te_err = []

for i in range (5):     #FOR 5 FOLDS
    train_s,train_l,test_s,test_l = nFold(samples,labels,5,i)
    model = LinearRegression()
    model.fit(train_s,train_l)


    train_pred = model.predict(train_s)
    test_pred = model.predict(test_s)
    
    fold.append(i)
    tr_err.append(mse(train_pred,train_l))
    te_err.append(mse(test_pred,test_l))
    sk_tr_err.append(mean_squared_error(train_pred,train_l))
    sk_te_err.append(mean_squared_error(test_pred,test_l))

with open ('Q1_d.csv','w') as file:
    writer = csv.writer(file)
    writer.writerow(['Fold Number','Train Error','Validation Error','Sklearn Train Error','Sklearn Validation Error'])
    for i in fold:
        writer.writerow([i,tr_err[i],te_err[i],sk_tr_err[i],sk_te_err[i]])


print('Mean of train MSE = ',np.mean(tr_err))
print('Mean of train MSE of Sklearn = ',np.mean(sk_tr_err))
print('Mean of test/validation MSE = ',np.mean(te_err))
print('Mean of test/validation MSE of Sklearn = ',np.mean(sk_te_err))
