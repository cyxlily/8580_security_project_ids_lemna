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
from keras.utils import to_categorical
from keras.models import Model, Sequential
from keras.layers import Embedding, TimeDistributed
from keras.callbacks import EarlyStopping
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

model.compile(loss="categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])

#training_data = BatchGenerator(1000, X[np.asarray(train_index)], le_y[np.asarray(train_index)], ohe)
early_stopping=EarlyStopping(monitor='val_loss', min_delta=0,
                              patience=0, verbose=0, mode='auto',
                              baseline=None, restore_best_weights=False)
model.fit(X, y_binary, epochs=1, batch_size=512, verbose=1,validation_split=0.2,callbacks=[early_stopping])
#model.fit_generator(training_data.__iter__(),steps_per_epoch=training_data.n_batches,epochs=30, verbose=1)

model.save("models/binary_protobytes_dirty.h5")

model.summary()
from keras.utils import plot_model
plot_model(model, to_file='graphics/binary_protobytes_model.jpg')