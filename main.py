# First we will import the necessary Library 

import datetime as dt
import math
import os
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
from plotly.subplots import make_subplots
from sklearn.metrics import (accuracy_score, explained_variance_score,
                             mean_absolute_error, mean_gamma_deviance,
                             mean_poisson_deviance, mean_squared_error,
                             r2_score)
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

# For Evalution we will use these library


# For model building we will use these library

'''
“data = pd.read_csv(‘bitcoin.csv’)”: ‘bitcoin.csv’ dosyasından verileri okur ve “data” adlı bir DataFrame nesnesine yükler.
“data = data.dropna()”: DataFrame’deki tüm eksik değerleri içeren satırları kaldırır.
“data = data[[‘Close’]]”: DataFrame’deki sadece “Close” sütununu seçer.
“data = data.values”: DataFrame’i bir NumPy dizisine dönüştürür.
“train_data = data[:int(len(data)*0.8)]”: Verilerin %80’ini eğitim verileri olarak kullanmak üzere ayırırken “test_data = data[int(len(data)*0.8):]” geri kalan %20’yi test verileri olarak ayırır.
“scaler = MinMaxScaler(feature_range=(0,1))”: MinMaxScaler sınıfından bir örnek oluşturur ve özellik aralığını (0,1) olarak ayarlar.
“train_data = scaler.fit_transform(train_data)”: Eğitim verilerini ölçeklendirir.
“test_data = scaler.transform(test_data)”: Test verilerini ölçeklendirir.
“create_dataset()” fonksiyonu, verileri belirli bir zaman aralığına dayalı olarak X ve Y veri setlerine ayırır. “look_back” parametresi, her bir X örneğinin kaç zaman adımına dayandığını belirler.
“X_train” ve “X_test” dizileri, LSTM modelinin girdisi olarak kullanılacak şekilde yeniden şekillendirilir.
LSTM modeli oluşturulur ve eğitilir.
“train_predict” ve “test_predict”, modelin tahminleri olarak hesaplanır ve ters ölçeklendirilir.
RMSE skorları hesaplanır.
'''

# For PLotting we will use these library


data = pd.read_csv('bitcoin.csv')
data = data.dropna()
data = data[['Close']]
data = data.values

train_data = data[:int(len(data)*0.8)]
test_data = data[int(len(data)*0.8):]

scaler = MinMaxScaler(feature_range=(0,1))
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data)-look_back-1):
        X.append(data[i:(i+look_back), 0])
        Y.append(data[i+look_back, 0])
    return np.array(X), np.array(Y)

look_back = 5
X_train, Y_train = create_dataset(train_data, look_back)
X_test, Y_test = create_dataset(test_data, look_back)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, Y_train, epochs=10, batch_size=32)

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform([Y_train])
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform([Y_test])

train_score = math.sqrt(mean_squared_error(Y_train[0], train_predict[:,0]))
print("Train Score: %.2f RMSE" % (train_score))
test_score = math.sqrt(mean_squared_error(Y_test[0], test_predict[:,0]))
print('Test Score: %.2f RMSE' % (test_score))