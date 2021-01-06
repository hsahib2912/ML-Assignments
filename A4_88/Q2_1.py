import numpy as np
from nltk.corpus import stopwords
x = []
y = []
with open('yelp_labelled.txt','r') as f:
    reader = f.readlines()
    for i in reader:
        i = i.split('\t')
        x.append(i[0].lower())
        y.append(int(i[1][0]))
        

def remove_punctuations(x):
    pun = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

    for i in range(len(x)):
        for exp in pun:
            if exp in x[i]:
                x[i] = x[i].replace(exp,'')
    return x

def tokenize_doc(x):
    return [i.split() for i in x]

def remove_stopwords(x):
    stop_words = set(stopwords.words('english'))
    for i in range(len(x)):
        for word in stop_words:
            while(word in x[i]):
                x[i].remove(word)


    return x

x = remove_punctuations(x)
x = tokenize_doc(x)
x = remove_stopwords(x)

def get_data():
    return x[:700],y[:700],x[700:],y[700:]