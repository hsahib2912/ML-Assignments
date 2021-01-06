import numpy as np
from Q2_1 import get_data

train_x,train_y,test_x,test_y = get_data()

def get_vocab(x):
    vocab = []

    for l in x:
        for word in l:
            if word not in vocab:
                vocab.append(word)
    return vocab

def get_doc_word_mat(x,vocab):
    mat = []
    for doc in x:
        tmp = []
        for word in vocab:
            tmp.append(doc.count(word))
        mat.append(tmp)
    
    return np.array(mat)
        
            
vocab = get_vocab(train_x)
mat = get_doc_word_mat(train_x,vocab)
