import argparse
import os
#os.environ["THEANO_FLAGS"] = "device=gpu,floatX=float32"
import numpy as np
import cPickle as pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from scipy import io
from rpy2 import robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
from keras.models import load_model

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
import csv

np.random.seed(1234)

r = robjects.r
rpy2.robjects.numpy2ri.activate()

#np.set_printoptions(threshold = 1e6)
importr('genlasso')
importr('gsubfn')

def perf_measure(y_true, y_pred):
    TP_FN = np.count_nonzero(y_true)
    FP_TN = y_true.shape[0] * y_true.shape[1] - TP_FN
    FN = np.where((y_true - y_pred) == 1)[0].shape[0]
    TP = TP_FN - FN
    FP = np.count_nonzero(y_pred) - TP
    TN = FP_TN - FP
    Precision = float(float(TP) / float(TP + FP + 1e-9))
    Recall = float(float(TP) / float((TP + FN + 1e-9)))
    accuracy = float(float(TP + TN) / float((TP_FN + FP_TN + 1e-9)))
    F1 =  2*((Precision * Recall) / (Precision + Recall))
    return Precision, Recall, accuracy

class xai_rnn(object):
    """class for explaining the rnn prediction"""
    def __init__(self, model, data):
        """
        Args:
            model: target rnn model.
            data: data sample needed to be explained.
            start: value of function start.
        """
        self.model = model
        self.data = data
        self.seq_len = data.shape[1]#10
        self.pred = self.model.predict(self.data, verbose = 0)

    def xai_feature(self, samp_num):
        """extract the important features from the input data
        Arg:
            fea_num: number of features that needed by the user
            samp_num: number of data used for explanation
        return:
            fea: extracted features
        """
        sample = np.random.randint(1, self.seq_len, samp_num)#remove count
        features_range = range(self.seq_len)
        data_sampled = np.copy(self.data)
        for i, size in enumerate(sample, start=1):
            inactive = np.random.choice(features_range, size, replace=False)
            tmp_sampled = np.copy(self.data)
            tmp_sampled[:,inactive] = 0
            data_sampled = np.concatenate((data_sampled, tmp_sampled), axis=0)
        print "data_sampled",data_sampled
        label_sampled = self.model.predict(data_sampled, verbose = 0)[:, 1]
        print "label_sampled",label_sampled
        print "label_sampled_shape",label_sampled.shape#(501,)
        label_sampled = label_sampled.reshape(label_sampled.shape[0], 1)
        print "label_sampled_shape",label_sampled.shape#(501,1)
        X = r.matrix(data_sampled, nrow = data_sampled.shape[0], ncol = data_sampled.shape[1])
        Y = r.matrix(label_sampled, nrow = label_sampled.shape[0], ncol = label_sampled.shape[1])

        n = r.nrow(X)
        p = r.ncol(X)
        results = r.fusedlasso1d(y=Y,X=X)
        result = np.array(r.coef(results, np.sqrt(n*np.log(p)))[0])[:,-1]
        print "result",result

        importance_score = np.argsort(result)[::-1]
        self.fea = importance_score
        self.fea = self.fea[np.where(self.fea<200)]
        self.fea = self.fea[np.where(self.fea>=0)]
        return self.fea
    
    def record_instance(self,writer,X,le,data_idx):
        writer.writerow('*************************************************')
        writer.writerow([data_idx])#index
        data = X[data_idx]
        writer.writerow(data)#data number
        sequence = le.inverse_transform(data)
        writer.writerow(sequence)#data str
        idx = [0] * 10
        order = 1
        for i in self.fea:
            idx[i]=order
            order = order+1
        writer.writerow(idx)#importance order
        writer.writerow(self.fea)#importance idx
        
        writer.writerow([data_idx+1])#next index
        next_data = X[data_idx+1]
        writer.writerow(next_data)
        next_sequence = le.inverse_transform(next_data)
        writer.writerow(next_sequence)
        

class fid_test(object):
    def __init__(self, xai_rnn):
        self.xai_rnn = xai_rnn

    def pos_boostrap_exp(self, num_fea):
        """
        feature deduction test.
        :param num_fea: number of selected features.
        :return: generated testing sample, probability of our method and random selection.
        """
        test_data = np.copy(self.xai_rnn.data)
        selected_fea = self.xai_rnn.fea[0:num_fea]
        test_data[0, selected_fea] = 0
        P1 = self.xai_rnn.model.predict(test_data, verbose=0)[0, 1]
        #test_data[0, self.xai_rnn.sp] = self.xai_rnn.start
        #P2 = self.xai_rnn.model.predict(test_data, verbose=0)[0, 1]

        random_fea = np.random.randint(0, self.xai_rnn.seq_len, num_fea)
        test_data_1 = np.copy(self.xai_rnn.data)
        test_data_1[0, random_fea] = 0
        P2 = self.xai_rnn.model.predict(test_data_1, verbose=0)[0, 1]
        return test_data, P1, P2


    def neg_boostrap_exp(self, test_seed, num_fea):
        """
        feature augmentation test.
        :param num_fea: number of selected features.
        :return: generated testing sample, probability of our method and random selection.
        """
        test_seed = test_seed.reshape(1, self.xai_rnn.seq_len)
        test_data = np.copy(test_seed)
        selected_fea = self.xai_rnn.fea[0:num_fea]
        test_data[0, selected_fea] = self.xai_rnn.data[0,selected_fea]
        P_neg_1 = self.xai_rnn.model.predict(test_data, verbose=0)[0, 1]

        random_fea = np.random.randint(0, self.xai_rnn.seq_len, num_fea)
        test_data_1 = np.copy(test_seed)
        test_data_1[0, random_fea] = self.xai_rnn.data[0,random_fea]
        P_neg_2 = self.xai_rnn.model.predict(test_data_1, verbose=0)[0, 1]

        return test_data, P_neg_1, P_neg_2


    def new_test_exp(self, num_fea):
        """
        Synthetic test.
        :param num_fea: number of selected features.
        :return: generated testing sample, probability of our method and random selection.
        """
        test_data = np.zeros_like(self.xai_rnn.data)
        # test_data = np.ones_like(self.xai_rnn.data) # either use one or zero to pad the missing part.
        selected_fea = self.xai_rnn.fea[0:num_fea]
        test_data[0, selected_fea] = self.xai_rnn.data[0, selected_fea]
        P_test_1 = self.xai_rnn.model.predict(test_data, verbose=0)[0, 1]

        random_fea = np.random.randint(0, self.xai_rnn.seq_len, num_fea)
        test_data_1 = np.zeros_like(self.xai_rnn.data)
        test_data_1[0, random_fea] = self.xai_rnn.data[0, random_fea]
        P_test_2 = self.xai_rnn.model.predict(test_data_1, verbose=0)[0, 1]

        return test_data, P_test_1, P_test_2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage="it's usage tip.", description="help info.")
    parser.add_argument("--f", type=int, default=5, required=True, help="the select feature number.")  
    args = parser.parse_args() 

    print '[Load model...]'
    model = load_model('../model/binary_protobytes_dirty.h5')
    n_fea_select = args.f

    print '[Load data...]'
    X = np.loadtxt('../data/X_test_binary.csv',dtype = np.int32, delimiter=',')
    y = np.loadtxt('../data/Y_test_binary.csv',dtype = np.int32, delimiter=',')

    print "X_dtype",X.dtype#int32
    print "y_dtype",y.dtype#int32
    print "X_shape",X.shape#(571151, 10)
    print "y_shape",y.shape#(571151,)

    print "X_sample",X[1]#[0 0 0 0 0 0 0 0 0 35]
    print "y_sample",y[1]#0
    
    data_num = X.shape[0]
    print 'Data_num:',data_num#571151
    seq_len = X.shape[1]
    print 'Sequence length:', seq_len#10
##
## Initialize One-hot-encoder
##
    y_test = to_categorical(y)
    y_test=y_test[:,np.newaxis,:]
    print "y_test_sample",y_test[1]#[1,0]
    print "y_test_dtype",y_test.dtype#float32

    # Padding sequence and prepare labelss
    x_test = X

    print "x_test_shape",x_test.shape#(571151, 10)
    print "y_test_shape",y_test.shape#(571151, 1, 2)
    # extract all the function starts for explanation
    print "nozero",np.nonzero(y)#nozero (array([ 61014,  61015,  61016, ..., 570614, 570615, 570676]),)
    idx = np.nonzero(y)[0].tolist()[:1000]

    n1 = len(idx)
    print n1#10216

    n_pos = 0
    n_new = 0
    n_neg = 0

    n_pos_rand = 0
    n_new_rand = 0
    n_neg_rand = 0
    n = 0
    '''
    example_number = 10#record first 10 explain data
    csvFile = open("instance.csv", "w+")
    writer = csv.writer(csvFile)
    unique_tokens = pd.read_csv('../data/unique_tokens.csv',header=None).iloc[:,0].tolist()
    le = LabelEncoder()
    le.fit(unique_tokens)
    '''
    for i in idx:
        print('%d of %d' % (i, len(x_test)))#61014 of 571151
        x_test_d = x_test[i:i + 1,:]
        print "x_test_d",x_test_d#[[ 0  0 32 32 33 32 32 32 32 32]]
        print "x_test_d_shape",x_test_d.shape#(1,10)
        print "lebal y",y[i]#1
        print "one hot y",y_test[i]#[[0. 1.]]
        xai_test = xai_rnn(model, x_test_d)
        print "pred",xai_test.pred#[[0.29173648 0.7082635 ]]
        print "xai_test_pred",xai_test.pred[0,1]#0.7082635
        if xai_test.pred[0, 1] > 0.5:
            print "successfully predict attack!"
            n = n + 1
            xai_fea = xai_test.xai_feature(500)
            print "xai_fea",xai_fea
            fea = np.zeros_like(xai_test.data)
            fea[0, xai_fea[0:n_fea_select]] = xai_test.data[0, xai_fea[0:n_fea_select]]
            print "fea",fea
            '''
            if example_number>0:
                example_number = example_number-1
                xai_test.record_instance(writer,X,le,i)
            '''
            fid_tt = fid_test(xai_test)

            test_data, P1, P2 = fid_tt.pos_boostrap_exp(n_fea_select)
            if P1 > 0.5:
                n_pos = n_pos + 1
            if P2 > 0.5:
                n_pos_rand = n_pos_rand + 1

            test_data, P_test_1, P_test_2 = fid_tt.new_test_exp(n_fea_select)
            if P_test_1> 0.5:
                n_new = n_new + 1
            if P_test_2 > 0.5:
                n_new_rand = n_new_rand + 1

            test_seed = x_test[0, ]
            neg_test_data, P_neg_1, P_neg_2 = fid_tt.neg_boostrap_exp(test_seed, n_fea_select)
            if P_neg_1 > 0.5:
                n_neg = n_neg + 1
            if P_neg_2 > 0.5:
                n_neg_rand = n_neg_rand + 1

    print n
    print 'n_fea_select:',n_fea_select
    print 'Our method'
    print 'Acc pos:', float(n_pos)/n
    print 'Acc new:', float(n_new)/n
    print 'Acc neg:', float(n_neg)/n

    print 'Random'
    print 'Acc pos:', float(n_pos_rand) / n
    print 'Acc new:', float(n_new_rand) / n
    print 'Acc neg:', float(n_neg_rand) / n