import os
from keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np
#from lime.lime_text import LimeTextExplainer


#from rpy2 import robjects

#from rpy2.robjects.packages import importr
#import rpy2.robjects.numpy2ri

np.random.seed(1234)

#r = robjects.r
#rpy2.robjects.numpy2ri.activate()
#np.set_printoptions(threshold = 1e6)
#importr('genlasso')
#importr('gsubfn')

MODEL_DIR = '../model/O1_Bi_Rnn.h5'
#DATA_DIR = '../data/elf_x86_32_gcc_O1_train.pkl'#14006
DATA_DIR = '../data/elf_x86_32_gcc_O1_test.pkl'

class xai_rnn(object):
    """class for explaining the rnn prediction"""
    def __init__(self, model, data, start_binary, real_start_sp):
        """
        Args:
            model: target rnn model.
            data: data sample needed to be explained.
            start: value of function start.
        """
        self.model = model
        self.data = data
        self.seq_len = data.shape[1]
        self.start = start_binary
        self.sp = np.where((self.data == self.start))
        self.real_sp = real_start_sp
        self.pred = self.model.predict(self.data, verbose = 0)[self.sp]

    def truncate_seq(self, trunc_len):
        """ Generate truncated data sample
        Args:
            trun_len: the lenght of the truncated data sample.

        return:
            trunc_data: the truncated data samples.
        """
        self.trunc_data_test = np.zeros((1, self.seq_len), dtype=int)
        #self.trunc_data_test = np.ones((1, self.seq_len),dtype = int)
        self.tl = trunc_len#40
        cen = self.seq_len/2
        half_tl = trunc_len/2

        if self.real_sp < half_tl:
            self.trunc_data_test[0, (cen - self.real_sp):cen] = self.data[0, 0:self.real_sp]
            self.trunc_data_test[0, cen:(cen+half_tl+1)] = self.data[0, self.real_sp:(self.real_sp+half_tl+1)]

        elif self.real_sp >= self.seq_len - half_tl:
            self.trunc_data_test[0, (cen - half_tl):cen] = self.data[0, (self.real_sp-half_tl):self.real_sp]
            self.trunc_data_test[0, cen:(cen + (self.seq_len-self.real_sp))] = self.data[0, self.real_sp:self.seq_len]

        else:
            self.trunc_data_test[0, (cen - half_tl):(cen + half_tl + 1)] = self.data[0, (self.real_sp - half_tl):(self.real_sp + half_tl + 1)]

        self.trunc_data = self.trunc_data_test[0, (cen - half_tl):(cen + half_tl + 1)]
        return self.trunc_data

    def xai_feature(self, samp_num, option= 'None'):
        """extract the important features from the input data
        Arg:
            fea_num: number of features that needed by the user
            samp_num: number of data used for explanation
            option: 
        return:
            fea: extracted features
        """
        cen = self.seq_len/2
        print('cen:',cen)#200/2=100
        half_tl = self.tl/2
        print('half_tl: ', half_tl)#40/2=20
        sample = np.random.randint(1, self.tl+1, samp_num)#[ 6 17 22 37 37 20 21 32  6 20 14...,500 number in [1,40]
        print('sample: ', sample)#
        features_range = range(self.tl+1)
        data_explain = np.copy(self.trunc_data).reshape(1, self.trunc_data.shape[0])
        print('data_explain: ',data_explain)#[[  0...0 86 138 230  84 132 237 5 233  1  1  1  1  92 130 196 193 189  13  1 140 148]]
        data_sampled = np.copy(self.data)
        for i, size in enumerate(sample, start=1):
            #print 'i: ',i#1
            #print 'size: ',size#6
            inactive = np.random.choice(features_range, size, replace=False)
            #print 'inactive: ',inactive#[19 24 26 38 10 31]
            tmp_sampled = np.copy(self.trunc_data)
            tmp_sampled[inactive] = 0
            tmp_sampled = tmp_sampled.reshape(1, self.trunc_data.shape[0])
            #print 'tmp_sampled: ',tmp_sampled# [[  0...0  86 138 230  84  0 237  0 233  1  1  1  0  92 130 196 193 189  13  0 140 148]]
            data_explain = np.concatenate((data_explain, tmp_sampled), axis=0)
            #print 'data_explain: ',data_explain#save tmp_sampled in data_explain,40
            data_sampled_mutate = np.copy(self.data)
            if self.real_sp < half_tl:
                data_sampled_mutate[0, 0:tmp_sampled.shape[1]] = tmp_sampled
            elif self.real_sp >= self.seq_len - half_tl:
                data_sampled_mutate[0, (self.seq_len - tmp_sampled.shape[1]): self.seq_len] = tmp_sampled
            else:
                data_sampled_mutate[0, (self.real_sp - half_tl):(self.real_sp + half_tl + 1)] = tmp_sampled
            #print 'data_sampled_mutate: ',data_sampled_mutate
            data_sampled = np.concatenate((data_sampled, data_sampled_mutate),axis=0)#save data_sampled_mutate in data_sampled,200
        print('data_sampled: ',data_sampled)

        if option == "Fixed":
            print("Fix start points")
            data_sampled[:, self.real_sp] = self.start

        label_sampled = self.model.predict(data_sampled, verbose = 0)[:, self.real_sp, 1]
        print('label_sampled: ',label_sampled)
        label_sampled = label_sampled.reshape(label_sampled.shape[0], 1)
        X = r.matrix(data_explain, nrow = data_explain.shape[0], ncol = data_explain.shape[1])
        print('X: ',X)
        Y = r.matrix(label_sampled, nrow = label_sampled.shape[0], ncol = label_sampled.shape[1])
        print('Y: ',Y)

        n = r.nrow(X)
        p = r.ncol(X)
        results = r.fusedlasso1d(y=Y,X=X)
        print('results: ',results)
        result = np.array(r.coef(results, np.sqrt(n*np.log(p)))[0])[:,-1]
        print('result: ',result)

        importance_score = np.argsort(result)[::-1]
        self.fea = (importance_score-self.tl/2)+self.real_sp
        self.fea = self.fea[np.where(self.fea<200)]
        self.fea = self.fea[np.where(self.fea>=0)]
        print('fea: ',self.fea)
        return self.fea


#load model
model = load_model(MODEL_DIR)

#load data
with open(DATA_DIR, 'rb') as f:  
    data = pickle.load(f)

print('Data size :',len(data))#2,data and label
data_num = len(data[0])
print('Data num: ',data_num)#6003
seq_len = len(data[0][0])
print('Sequence length: ', seq_len)#feature length:200
print('data0_sample: ',data[0][0])
print('data1_sample: ',data[1][0])

# Padding sequence and prepare labelss
x_test = pad_sequences(data[0], maxlen=seq_len, dtype='int32', padding='post', truncating='post', value=0)
x_test = x_test + 1
y = pad_sequences(data[1], maxlen=seq_len, dtype='int32', padding='post', truncating='post', value=0)
y_test = np.zeros((data_num, seq_len, 2), dtype=y.dtype)
for test_id in range(data_num):
    y_test[test_id, np.arange(seq_len), y[test_id]] = 1

#x_test_l=x_test[:, : ,np.newaxis]#(14006,200,1)
print('x_test shape: ',x_test.shape)#(14006,200)
print('y_test shape: ',y_test.shape)#(14006,200,2)
print('y_test_sample: ',y_test[0,0])#[0 1]

idx = np.nonzero(y)[0]
print('has1 idx: ',idx)
start_points = np.nonzero(y)[1]
print('start point: ',start_points)

#class_names = ['not_head', 'head']
#explainer = LimeTextExplainer(class_names=class_names)
#print("successfully set lime!")

for i in range(len(x_test)):
    if i % 200 == 0:
        print('%d of %d' % (i, len(x_test)))
    if i in idx:
        idx_Col = np.where(idx == i)
        idx_Row = start_points[idx_Col]
        binary_func_start = x_test[i][idx_Row]
        x_test_d = x_test[i:i + 1]#(200,1)
        #x_test_d = x_test[i]#(1,200)
        print('x_test_d shape: ',x_test_d.shape)#(200,1)
        print('x_test_d: ',x_test_d)
        #exp = explainer.explain_instance(x_test_d, model.predict, num_features=25)
        #print(exp.as_list())
        
        for j in range(len(idx_Row)):
            print('binary_func_start j',binary_func_start[j])
            print('idx_Row j',idx_Row[j])
            
            xai_test = xai_rnn(model, x_test_d, binary_func_start[j], idx_Row[j])
            print('xai_test data: ',xai_test.data)
            print('xai_test seq_len: ',xai_test.seq_len)#200
            print('xai_test start: ',xai_test.start)#86
            print('xai_test sp:',xai_test.sp)
            print('xai_test real_sp',xai_test.real_sp)#0
            print('xai_test pred',xai_test.pred)#[[0.05681201 0.94318795]]
            if xai_test.pred[0, 1] > 0.5:
                truncate_seq_data = xai_test.truncate_seq(40)#gengerate self.tl=40 and self.trunc_data
                print('truncate_seq_data: ',truncate_seq_data)
                #exp = explainer.explain_instance(x_test_d, model.predict, num_features=25, labels=(1,))
                xai_fea = xai_test.xai_feature(500)
                print('xai_fea: ',xai_fea)
                print('xai_fea shape: ',xai_fea.shape)
                fea = np.zeros_like(xai_test.data)
                fea[0, xai_fea[0:25]] = xai_test.data[0, xai_fea[0:25]]
                os.system("pause")
        