import numpy as np
import pandas
import re
import h5py
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import log_loss, auc, roc_curve
from tensorflow.python.client import device_lib
from lxml import etree
from itertools import groupby
from gensim.models import Word2Vec
import glob
import math
import itertools
from sklearn.metrics import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.utils import to_categorical
from keras.layers.core import Masking
from keras.layers import Dense, LSTM, Dropout, Embedding
from keras.layers.wrappers import Bidirectional
from keras.models import Model, Sequential
from keras.utils import np_utils
from keras.models import Model, Sequential
from keras.layers import Embedding, TimeDistributed
from keras.callbacks import EarlyStopping 
'''
print("Load data ...")
X = np.loadtxt('X.csv',dtype = int, delimiter=',')
Y = np.loadtxt('Y.csv',dtype = int, delimiter=',')
print(X.dtype)#float64
print(Y.dtype)#int64
print(X.shape)
print(Y.shape)
print(X[0])
print(Y[0])

##
## Initialize One-hot-encoder
##
ohe = OneHotEncoder(sparse=False)#common matrix
ohe_y = ohe.fit_transform(Y.reshape(-1, 1)).astype(np.int64)

print(ohe_y[0])
#print(X.shape)#((1889171, 10)
#print(ohe_y.shape)#(1889171, 53)
print(ohe_y.dtype)#float64

'''

def binary_focal_loss(gamma=2, alpha=0.25):
    """
    Binary form of focal loss.
    
    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true*alpha + (K.ones_like(y_true)-y_true)*(1-alpha)
    
        p_t = y_true*y_pred + (K.ones_like(y_true)-y_true)*(K.ones_like(y_true)-y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true)-p_t),gamma) * K.log(p_t)
        return K.mean(focal_loss)
    return binary_focal_loss_fixed


print("Load data ...")
X = np.loadtxt('X_train_binary.csv',dtype = np.int32, delimiter=',')
Y = np.loadtxt('Y_train_binary.csv',dtype = np.int32, delimiter=',')
Y=Y[:,np.newaxis]
y_binary = to_categorical(Y)

print y_binary
print(X.dtype)
print(Y.dtype)
print(X.shape)
print(Y.shape)
print(X[0])
print(Y[0])

class BatchGenerator(object):
    def __init__(self, batch_size, x, y, ohe):
        self.batch_size = batch_size
        self.n_batches = int(math.floor(np.shape(x)[0] / batch_size))
        self.batch_index = [a * batch_size for a in range(0, self.n_batches)]
        self.x = x
        self.y = y
        self.ohe = ohe
        
    def __iter__(self):
        for bb in itertools.cycle(self.batch_index):
            y = self.y[bb:(bb+self.batch_size)]
            ohe_y = self.ohe.transform(y.reshape(len(y), 1))
            yield (self.x[bb:(bb+self.batch_size),], ohe_y)
            
print("Ready to Go!")
#np.random.seed(123)

model = Sequential()

# model.add(Masking(0., input_shape=(seq_len,w2v_size)))
model.add(Embedding(output_dim=100, input_dim=53, mask_zero=True))
model.add(Bidirectional(LSTM(50, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(50, activation="relu", return_sequences=False)))
model.add(Dropout(0.2))
model.add(Dense(50, activation="linear"))
model.add(Dropout(0.2))
model.add(Dense(y_binary.shape[1], activation="softmax"))

model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], optimizer="nadam", metrics=["accuracy"])

#training_data = BatchGenerator(1000, X[np.asarray(train_index)], le_y[np.asarray(train_index)], ohe)
early_stopping=EarlyStopping(monitor='val_loss', min_delta=0,
                              patience=0, verbose=0, mode='auto',
                              baseline=None, restore_best_weights=False)
model.fit(X, y_binary, epochs=1, batch_size=512, verbose=1,validation_split=0.2,callbacks=[early_stopping])
#model.fit_generator(training_data.__iter__(),steps_per_epoch=training_data.n_batches,epochs=30, verbose=1)

model.save("models/binary_focal_protobytes_dirty.h5")

model.summary()
from keras.utils import plot_model
plot_model(model, to_file='graphics/binary_focal_protobytes_model.jpg')

#***************************TEST**********************************
print("Load data ...")
X = np.loadtxt('X_test_binary.csv',dtype = np.int32, delimiter=',')
y = np.loadtxt('Y_test_binary.csv',dtype = np.int32, delimiter=',')

class_names=['Normal','Attack']

print(X.dtype)
print(y.dtype)
print(X.shape)#(1889171, 10)
print(y.shape)#(1889171,)

print(X[0])
print(y[0])

#print "Load model ..."
#model = load_model("models/binary_protobytes_dirty.h5")
#model = load_model("models/binary_focal_protobytes_dirty.h5",custom_objexts={'binary_focal_loss':binary_focal_loss})
#from keras.utils import plot_model
#plot_model(model, to_file='graphics/binary_protobytes_model.jpg')


preds = model.predict_proba(X, batch_size=512)#(1889171, 53)
y_preds = np.argmax(preds, axis=1)

print "preds_sample",y_preds[0]#0
print "preds_shape",y_preds.shape#(571151,)

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
plt.savefig('val_focal.png')


f = open('report_focal.txt','w')

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

f = open('report_focal.csv','w')
f.write(report)
f.close()