import numpy as np
import re
import h5py
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import log_loss, auc, roc_curve
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
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.layers.core import Masking
from keras.layers import Dense, LSTM, Dropout, Embedding
from keras.layers.wrappers import Bidirectional
from keras.models import Model, Sequential
from keras.utils import np_utils
from keras.utils import to_categorical
from keras.layers import Embedding, TimeDistributed
from tensorflow.python.client import device_lib

from keras.models import load_model

print("Load data ...")
X = np.loadtxt('X_test_binary.csv',dtype = np.int32, delimiter=',')
y = np.loadtxt('Y_test_binary.csv',dtype = np.int32, delimiter=',')
y = y.reshape(-1,1)
print y

class_names=['Normal','Attack']

print(X.dtype)
print(y.dtype)
print(X.shape)#(1889171, 10)
print(y.shape)#(1889171,)
print(X[0])
print(y[0])

print "Load model ..."
model = load_model("models/binary_protobytes_dirty.h5")

#from keras.utils import plot_model
#plot_model(model, to_file='graphics/binary_protobytes_model.jpg')


preds = model.predict_proba(X, batch_size=512)#(1889171, 53)
y_preds = np.argmax(preds, axis=1)
y_preds = y_preds.reshape(-1,1)
print y_preds

print "preds_sample",y_preds[0]#0
print "preds_shape",y_preds.shape#(571151,)
print "preds_dtype",y_preds.dtype

# Calculate the confusion matrix.
cm = confusion_matrix(y, y_preds)
# Plot the confusion matrix.
plt.matshow(cm,cmap=plt.cm.Greens)
plt.colorbar()
for x in range(len(cm)):
    for y in range(len(cm)):
        plt.annotate(cm[x,y],xy=(x,y),horizontalalignment='center',verticalalignment='center')
plt.ylabel('True label')
plt.xlabel('Predicted label')
#figure = plot_confusion_matrix(cm, class_names=class_names)
#cm_image = plot_to_image(figure)
plt.savefig('val.png')


f = open('report.txt','w')

#y = [1,0,1]
#y_preds = [1,0,1]
accuracy_score = accuracy_score(y, y_preds)
precision_score = precision_score(y, y_preds)
recall_score = recall_score(y, y_preds)
f1_score = f1_score(y, y_preds)
report  = classification_report(y, y_preds, target_names=class_names)
print 'accuracy: %f'%accuracy_score
print 'precision: %f'%precision_score
print 'recall: %f'%recall_score
print 'f1-score: %f'%f1_score
print(report)
f.write('accuracy: %f\n'%accuracy_score)
f.write('precision: %f\n'%precision_score)
f.write('recall: %f\n'%recall_score)
f.write('f1-score: %f\n'%f1_score)
f.write(report)
f.close()

f = open('report.csv','w')
f.write(report)
f.close()