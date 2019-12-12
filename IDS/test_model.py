import numpy as np
import re
import h5py
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import log_loss, auc, roc_curve
from keras.layers.core import Masking
from keras.layers import Dense, LSTM, Dropout, Embedding
from keras.layers.wrappers import Bidirectional
from keras.models import Model, Sequential
from keras.utils import np_utils
from keras.layers import Embedding, TimeDistributed
from tensorflow.python.client import device_lib
from lxml import etree
from itertools import groupby
from gensim.models import Word2Vec
import glob
import math
import itertools
from sklearn.metrics import *
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import load_model

print("Load data ...")
X = np.loadtxt('X.csv',dtype = np.int32, delimiter=',')
Y = np.loadtxt('Y.csv',dtype = np.int32, delimiter=',')
seq_index = np.loadtxt('seq_index.csv',dtype = np.int32, delimiter=',').tolist()
seq_attack = pd.read_csv('seq_attack.csv', dtype=str,header=None).iloc[:,0]

print seq_attack.shape#(1889171,)
seq_attack = seq_attack.values.tolist()

print(X.dtype)
print(Y.dtype)
print(X.shape)#(1889171, 10)
print(Y.shape)#(1889171,)

print(X[0])
print(Y[0])
print len(seq_index)#1889171
print len(seq_attack)#1889171

##
## Initialize One-hot-encoder
##
ohe = OneHotEncoder(sparse=False)#common matrix
ohe_y = ohe.fit_transform(Y.reshape(-1, 1)).astype(np.int32)
print(ohe_y[0])
print(ohe_y.shape)#(1322419, 53)
print(ohe_y.dtype)#int32

print "Load model ..."
model = load_model("models/protobytes_dirty.h5")
preds = model.predict_proba(X, batch_size=512)#(1889171, 53)
print "preds_shape",preds.shape
test_index = range(0,len(preds))
indexed_preds = zip(np.asarray(seq_index)[np.asarray(test_index)], preds, ohe_y[np.asarray(test_index)], np.asarray(seq_attack)[np.asarray(test_index)])
#from keras.utils import plot_model
#plot_model(model, to_file='graphics/protobytes_model.jpg')
logloss_list = []
for (ii, pp, yy, aa) in indexed_preds:
    ll = -math.log(pp[np.argmax(yy)])
    logloss_list.append(ll)

fpr, tpr, thresholds = roc_curve(np.asarray(seq_attack)[np.asarray(test_index)],logloss_list, pos_label="Attack")

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc(fpr,tpr))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
key_ll = zip(seq_index, logloss_list, seq_attack)
dictionary = dict()
for (key, ll, aa) in key_ll: #
    current_value = dictionary.get(str(key), ([],[]))
    dictionary[str(key)] = (current_value[0] + [ll], current_value[1] + [aa])

agg_ll = []
agg_bad = []
for key, val in dictionary.iteritems():
    bad = str(np.mean([v=="Attack" for v in val[1]]) > 0.)
    score = np.max(val[0])
    agg_bad.append(bad)
    agg_ll.append(score)
    
fpr, tpr, thresholds = roc_curve(agg_bad, agg_ll, pos_label="True")

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc(fpr,tpr))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Proto-bytes Dirty Baseline')
plt.legend(loc="lower right")
plt.savefig("graphics/protobytes_dirty.png",format="png")
plt.show()