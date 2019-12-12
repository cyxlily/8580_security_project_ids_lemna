import numpy as np
import pandas as pd
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
import random

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
    return pd.DataFrame(all_records)

dfs = []
for ii in xml_list:
    print(ii)
    xml_data = open(ii).read()
    dfs.append(xml2df(xml_data))
    
data = pd.concat(dfs,sort=False)
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
attacks = [a.split(",") for a in key.Tag.tolist()]
sequences = [a.split(",") for a in key.protoBytes.tolist()]
#print "sequences",sequences
unique_tokens = list(set([a for b in sequences for a in b]))
print "unique tokens", unique_tokens
print'len unique tokens: ',len(unique_tokens)#53
pd_unique_tokens = pd.DataFrame(data=unique_tokens)

pd_unique_tokens.to_csv('unique_tokens54.csv',header=None,index=False)

le = LabelEncoder()
le.fit(unique_tokens)
unique_tokens_en = le.transform(unique_tokens)#0~53
unique_tokens_en54 = [i+1 for i in unique_tokens_en]
print 'unique_tokens_en54: ',unique_tokens_en54
pd_unique_tokens_en = pd.DataFrame(data=unique_tokens_en54)
pd_unique_tokens_en.to_csv('unique_tokens_en54.csv',header=None,index=False)

sequences = [le.transform(s).tolist() for s in sequences]
sequences = [[b+1 for b in a] for a in sequences]

sequence_attack = zip(attacks, sequences)
print "len sequence attack",len(sequence_attack)

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
    #print "attack",sa
    #print "sequence",ss
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

y_binary = []
for label in seq_attack:
    if label == 'Attack':
        y_binary.append([1])
    elif label == 'Normal':
        y_binary.append([0])
    else:
        print "wrong label",label
y_binary = np.array(y_binary)
X = np.array(seq_x)
print y_binary
print y_binary.shape
print y_binary.dtype
print X.shape
print X[0]#[0 0 0 0 0 0 0 0 0 0]
print X.dtype
np.savetxt('X_binary54.csv',X,fmt='%d',delimiter = ',')
np.savetxt('Y_binary54.csv',y_binary,fmt='%d',delimiter = ',')


##
## Generate train and test dataset
##
test_ratio= 0.3
len_attacks = len(attacks)
idx = random.sample(xrange(len_attacks),int(len_attacks*test_ratio))
train_attacks = []
test_attacks = []
train_sequences = []
test_sequences = []
for i in xrange(len_attacks):
    if i in idx:
        test_attacks.append(attacks[i])
        test_sequences.append(sequences[i])
    else:
        train_attacks.append(attacks[i])
        train_sequences.append(sequences[i])
print "total len",len_attacks
print "test len",len(test_attacks)
print "train len",len(train_attacks)

train_sequence_attack = zip(train_attacks, train_sequences)
test_sequence_attack = zip(test_attacks,test_sequences)

##
## Train
##

na_value = 0.
seq_len = 10
seq_index = []
seq_x = []
seq_y = []
seq_attack = []
for si, (sa, ss) in enumerate(train_sequence_attack):
    prepend = [0.] * (seq_len)
    seq =  prepend + ss
    seqa = prepend + sa
    print "seq",seq
    print "seqa",seqa
    for ii in range(seq_len, len(seq)):
        subseq = seq[(ii-seq_len):(ii)]
        print "subseq",subseq
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

y_binary = []
for label in seq_attack:
    if label == 'Attack':
        y_binary.append([1])
    elif label == 'Normal':
        y_binary.append([0])
    else:
        print "wrong label",label
y_binary = np.array(y_binary)
X = np.array(seq_x)
print y_binary
print y_binary.shape
print y_binary.dtype
print X.shape
print X[0]#[0 0 0 0 0 0 0 0 0 0]
print X.dtype
np.savetxt('X_train_binary54.csv',X,fmt='%d',delimiter = ',')
np.savetxt('Y_train_binary54.csv',y_binary,fmt='%d',delimiter = ',')


##
## Test
##
na_value = 0.
seq_len = 10
seq_index = []
seq_x = []
seq_y = []
seq_attack = []
for si, (sa, ss) in enumerate(test_sequence_attack):
    #print "attack",sa
    #print "sequence",ss
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

y_binary = []
for label in seq_attack:
    if label == 'Attack':
        y_binary.append([1])
    elif label == 'Normal':
        y_binary.append([0])
    else:
        print "wrong label",label
y_binary = np.array(y_binary)
X = np.array(seq_x)
print y_binary
print y_binary.shape
print y_binary.dtype
print X.shape
print X[0]#[0 0 0 0 0 0 0 0 0 0]
print X.dtype
np.savetxt('X_test_binary54.csv',X,fmt='%d',delimiter = ',')
np.savetxt('Y_test_binary54.csv',y_binary,fmt='%d',delimiter = ',')
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