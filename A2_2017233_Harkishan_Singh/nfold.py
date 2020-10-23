import numpy as np
def nFold(samples,labels,n,i):   #i is used for testing ith fold
    l = len(samples)
    start = int(l/n)*i
    end = int(l/n)*(i+1)
    return np.concatenate((samples[0:start],samples[end:l]),axis = 0),np.concatenate((labels[0:start],labels[end:l]),axis = None),samples[start:end],labels[start:end]

