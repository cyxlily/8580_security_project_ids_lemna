import numpy as np
import pandas
import re
import h5py
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import log_loss, auc, roc_curve
from keras.layers.core import Masking
from keras.layers import Dense, LSTM, Dropout, Embedding
from keras.layers.wrappers import Bidirectional
from keras.models import Model, Sequential
from keras.utils import np_utils
from keras.models import Model, Sequential
from keras.layers import Embedding, TimeDistributed
from tensorflow.python.client import device_lib
from lxml import etree
from itertools import groupby
from gensim.models import Word2Vec
import glob
import math
import itertools
from sklearn.metrics import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

##
## Load data
##
print("Loading data...")
xml_list = glob.glob('data/labeled_flows_xml/*xml')
print(xml_list)

parser = etree.XMLParser(recover=True)

def xml2df(xml_data):
    root = etree.fromstring(xml_data, parser=parser) # element tree
    all_records = []
    for i, child in enumerate(root):
        record = {}
        for subchild in child:
            record[subchild.tag] = subchild.text
            all_records.append(record)
    return pandas.DataFrame(all_records)

dfs = []
for ii in xml_list:
    print(ii)
    xml_data = open(ii).read()
    dfs.append(xml2df(xml_data))
    
data = pandas.concat(dfs,sort=False)
data = data.drop_duplicates()
#data = data.sort_values("startDateTime")
del dfs

##
## Create IP-dyad hours
##

#print(data.shape)#(1889172, 22)
print("De-dup Flows: "+str(len(data)))#1889172
data = data.sort_values('startDateTime')
data['totalBytes'] = data.totalSourceBytes.astype(float) + data.totalDestinationBytes.astype(float)
data['lowIP'] = data[['source','destination']].apply(lambda x: x[0] if x[0] <= x[1] else x[1], axis=1)
data['highIP'] = data[['source','destination']].apply(lambda x: x[0] if x[0] > x[1] else x[1], axis=1)
data['seqId'] = data['lowIP'] + '_' + data['highIP']  + '_' + data['startDateTime'].str[:13]
data['protoBytes'] = data[['protocolName','totalBytes']].apply(lambda x: str(x[0])[0] + str(math.floor(np.log2(x[1] + 1.0))), axis=1)
print(list(data))
print(data.iloc[0,:])
##
## Group by key and produce sequences
## 
key = data.groupby('seqId')[['Tag','protoBytes']].agg({"Tag":lambda x: "%s" % ','.join([a for a in x]),"protoBytes":lambda x: "%s" % ','.join([str(a) for a in x])})
#print "key",key
attacks = [a.split(",") for a in key.Tag.tolist()]
#print "attacks",attacks
sequences = [a.split(",") for a in key.protoBytes.tolist()]
#print "sequences",sequences

unique_tokens = list(set([a for b in sequences for a in b]))
print "unique tokens", unique_tokens
np.savetxt('unique_tokens.csv',unique_tokens,delimiter = ',')
print('len unique tokens: ',len(unique_tokens))#53
le = LabelEncoder()
le.fit(unique_tokens)
sequences = [le.transform(s).tolist() for s in sequences]
sequences = [[b+1 for b in a] for a in sequences]

sequence_attack = zip(attacks, sequences)


##
## Produce sequences for modeling
##
na_value = 0.
seq_len = 10
seq_index = []
seq_x = []
seq_y = []
seq_attack = []
for si, (sa, ss) in enumerate(sequence_attack):
    prepend = [0.] * (seq_len)
    seq =  prepend + ss
    seqa = prepend + sa
    for ii in range(seq_len, len(seq)):
        subseq = seq[(ii-seq_len):(ii)]
        vex = []
        for ee in subseq:
            try:
                vex.append(ee)
            except:
                vex.append(na_value)
        seq_x.append(vex)
        seq_y.append(seq[ii])
        seq_index.append(si)
        seq_attack.append(seqa[ii])
print(seq_index[0])#0
print(seq_attack[0])#Normal
print(set(seq_attack))#['Attack', 'Normal']
print(seq_x[0])#[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
print(seq_y[0])#35
#print(type(seq_x))#list
#print(type(seq_y))#list

X = np.array(seq_x)
Y = np.asarray(seq_y).reshape(-1,1)
print(X.shape)
print(Y.shape)
print(X[0])#[0 0 0 0 0 0 0 0 0 0]
print(Y[0])#35
print(X.dtype)
print(Y.dtype)

np.savetxt('X.csv',X,fmt='%d',delimiter = ',')
np.savetxt('Y.csv',Y,fmt='%d',delimiter = ',')
np.savetxt('seq_index.csv',seq_index,fmt='%d',delimiter = ',')
np.savetxt('seq_attack.csv',seq_attack,fmt='%s',delimiter = ',')
'''
#np.random.seed(123)
X = np.loadtxt('X.csv',dtype = int, delimiter=',')
Y = np.loadtxt('Y.csv',dtype = int, delimiter=',')
print("read data")

print(X.shape)#(1889171,10)
print(Y.shape)#(1889171,)
#x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3)
train_len = int(X.shape[0]*0.7)
print(train_len)
x_train = X[:train_len,:]
y_train = Y[:train_len]
x_test = X[train_len:,:]
y_test = Y[train_len:]
print(x_train.shape)#(1322419, 10)
print(y_train.shape)#(1322419,)
print(x_test.shape)#(566752, 10)
print(y_test.shape)#(566752,)
np.savetxt('x_train.csv',x_train,fmt='%d',delimiter = ',')
np.savetxt('x_test.csv',x_test,fmt='%d',delimiter = ',')
np.savetxt('y_train.csv',y_train,fmt='%d',delimiter = ',')
np.savetxt('y_test.csv',y_test,fmt='%d',delimiter = ',')
'''