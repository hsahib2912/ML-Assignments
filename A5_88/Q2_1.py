from h5py import File
import numpy as np
import random

def get_data():
    data = File('MNIST_Subset.h5','r')
    split = int(.8*len(data['X']))

    random.seed(42)
    
    train_x = data['X'][:split]
    train_x = train_x.reshape(11400,28*28)
    train_y = data['Y'][:split]

    test_x = data['X'][split:]
    test_x = test_x.reshape(2851,28*28)
    test_y = data['Y'][split:]

    return train_x,train_y,test_x,test_y