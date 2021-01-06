import pandas as pd

df = pd.read_csv('iris.data',header=None)
df[4].replace({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2},inplace = True)

df = df.sample(frac =1 )


def get_data():
    return df[:105].iloc[:,:4],df[:105].iloc[:,4],df[105:].iloc[:,:4],df[105:].iloc[:,4]

