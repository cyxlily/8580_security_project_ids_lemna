# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

from lime import lime_tabular

df = pd.read_csv('data/co2_data.csv', index_col=0, parse_dates=True)
print(df)
print('X sample: ',df.iloc[0:13,0:2])
print('y sample: ',df.iloc[0:13,2])
def reshape_data(seq, n_timesteps):
    N = len(seq) - n_timesteps - 1
    nf = seq.shape[1]
    if N <= 0:
        raise ValueError('I need more data!')
    new_seq = np.zeros((N, n_timesteps, nf))
    for i in range(N):
        new_seq[i, :, :] = seq[i:i+n_timesteps]
    return new_seq

N_TIMESTEPS = 12  # Use 1 year of lookback
data_columns = ['co2', 'co2_detrended']
target_columns = ['rising']

scaler = MinMaxScaler(feature_range=(-1, 1))
X_original = scaler.fit_transform(df[data_columns].values)
X = reshape_data(X_original, n_timesteps=N_TIMESTEPS)
y = to_categorical((df[target_columns].values[N_TIMESTEPS:-1]).astype(int))
print(X.shape, y.shape)#(2270, 12, 2) (2270, 2)
print('X',X)
print('y',y)
# Train on the first 2000, and test on the last 276 samples
X_train = X[:2000]
y_train = y[:2000]
X_test = X[2000:]
y_test = y[2000:]



'''
model = Sequential()
model.add(LSTM(32, input_shape=(N_TIMESTEPS, len(data_columns))))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

optimizer = Adam(lr=1e-4)
model.compile(loss='binary_crossentropy', optimizer=optimizer)
model.fit(X_train, y_train, batch_size=100, epochs=50,
          validation_data=(X_test, y_test),
          verbose=2)
model.save('model/co2_model.h5')
print("model save!")
'''
model = load_model('model/co2_model.h5')
print("load model!")

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print(classification_report(y_true, y_pred))

explainer = lime_tabular.RecurrentTabularExplainer(X_train, training_labels=y_train, feature_names=data_columns,
                                                   discretize_continuous=True,
                                                   class_names=['Falling', 'Rising'],
                                                   discretizer='decile')
print('lime successful!')
print('instance shape: ',X_test[50].shape)#(12,2)
print('X test 50: ',X_test[50])
data_row = X_test[50].T.reshape(12 * 2)
print(data_row)
print(data_row.shape)#(24,)

exp = explainer.explain_instance(X_test[50], model.predict, num_features=10, labels=(1,))
print(exp.as_list())