import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from Backtest import backtest_func
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import TimeDistributed
import tensorflow as tf
import keras
from tensorflow.keras.optimizers import Adam, Adamax
from keras.callbacks import History
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# tf.random.set_seed(20)
np.random.seed(10)


# (Bidirectional(LSTM(20,

def build_model(n_inputs, n_features):
    model = Sequential()
    model.add(LSTM(units=400, return_sequences=True, input_shape=(n_inputs, n_features)))
    model.add(Dropout(0.1))
    model.add(LSTM(units=400))
    model.add(Dropout(0.1))
    # model.add(Dense(32, kernel_initializer="uniform", activation='relu'))
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    # history = model.fit
    return model


def Bi_LSTM(n_inputs, n_features):
    optim = Adam(learning_rate=0.01)
    model = Sequential()
    model.add(Bidirectional(LSTM(units=800, return_sequences=True, input_shape=(n_inputs, n_features))))
    model.add(Dropout(0.1))
    model.add(Bidirectional(LSTM(units=800)))
    model.add(Dropout(0.1))
    # model.add(Dense(32, kernel_initializer="uniform", activation='relu'))
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer=optim, loss='mean_squared_error')
    # history = model.fit
    return model

def class_LSTM(n_inputs, n_features):
    model = Sequential()
    model.add(LSTM(units=200, return_sequences=True, input_shape=(n_inputs, n_features)))
    model.add(Dropout(0.01))
    model.add(LSTM(100))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def LSTM_HPO(n_inputs, n_features):
    opt = Adamax(learning_rate=0.01)
    nn = Sequential()
    nn.add(Bidirectional(LSTM(units=175, return_sequences=True, input_shape=(n_inputs, n_features))))
    nn.add(Dropout(0.01))
    nn.add(Bidirectional(LSTM(units=175)))
    nn.add(Dense(1, activation='linear'))
    nn.compile(loss='mse', optimizer=opt, metrics=['mse', 'mae'])
    return nn


def prepare_data(data_set_scaled, backcandles, liste, splitlimit):
    X = []
    for j in range(len(liste)):  # data_set_scaled[0].size):#2 columns are target not X
        X.append([])
        for i in range(backcandles, data_set_scaled.shape[0]):  # backcandles+2
            X[j].append(data_set_scaled[i - backcandles:i, liste[j]])

    # move axis from 0 to position 2
    X = np.moveaxis(X, [0], [2])

    X, yi = np.array(X), np.array(data_set_scaled[backcandles:, -1])
    y = np.reshape(yi, (len(yi), 1))

    X_train, X_test = X[:splitlimit], X[splitlimit:]
    y_train, y_test = y[:splitlimit], y[splitlimit:]

    return X_train, X_test, y_train, y_test


def main():
    '''data = yf.download(tickers='^GSPC', start='2021-01-01', end='2022-12-27')
    data.head(10)

    # Adding indicators
    data['RSI'] = ta.rsi(data.Open, length=15)
    data['EMAF'] = ta.ema(data.Open, length=20)
    data['EMAM'] = ta.ema(data.Open, length=100)
    data['EMAS'] = ta.ema(data.Open, length=150)

    data["log_Open"] = np.log(data["Open"])
    data['log_Close'] = np.log(data['Close'])
    data["Return_open"] = data["log_Open"] - data["log_Open"].shift(+1)
    data["Return_close"] = data["log_Close"] - data["log_Close"].shift(+1)
    data["Return_target"] = data["log_Open"].shift(-1) - data["log_Open"]
    # Class: 1 = positive return, 0 = negative return
    # data["Class"] = [1 if data.Return[i] > 0 else 0 for i in range(len(data))]
    # data['TargetClass'] = [1 if data.Target[i]>0 else 0 for i in range(len(data))]
    data.dropna(inplace=True)
    data.reset_index(inplace=True)
    data.drop(['Close', 'Date'], axis=1, inplace=True)
    data.head(20)'''

    # import data
    data = pd.read_csv('featureSelectionDataset_Paul.csv', sep=',', header=0, index_col=0, parse_dates=True,
                       decimal=".")
    print(data)
    data = data.iloc[-600:,:]
    data.dropna(inplace=True)
    y_open = np.asarray([1 if data.Return_open[i]>0 else 0 for i in range(len(data))]).reshape(-1, 1)
    y_close = np.asarray([1 if data.Return_close[i]>0 else 0 for i in range(len(data))]).reshape(-1, 1)
    y_target = np.asarray([1 if data.Target[i]>0 else 0 for i in range(len(data))]).reshape(-1, 1)
    print(y_target, y_target.shape)
    print(y_open)
    data = data.drop(['Target'], axis=1)
    print(data)

    # scale data
    scaler = StandardScaler()
    data_set_scaled = scaler.fit_transform(data)
    data_set_scaled = np.concatenate((data_set_scaled, y_open), axis=1)
    data_set_scaled = np.concatenate((data_set_scaled, y_close), axis=1)
    data_set_scaled = np.concatenate((data_set_scaled, y_target), axis=1)
    print('data_scaled', data_set_scaled, data_set_scaled.shape)
    

    # choose how many look back days
    backcandles = 50

    # choose columns: all but open price and target
    liste = list(range(0, data.shape[1] - 1))
    print(data.iloc[:, liste])

    # split data into train test sets
    splitlimit = (len(data) - 60)

    # prepare data for lstm
    X_train, X_test, y_train, y_test = prepare_data(data_set_scaled, backcandles, liste, splitlimit)
    print('test', y_test)

    # build model

    nb_epoch = 50
    batch_size = 10
    n_inputs = X_train.shape[1]
    n_features = X_train.shape[2]

    model = class_LSTM(n_inputs, n_features)
    # model = Bi_LSTM(n_inputs, n_features)
    early_stopping = EarlyStopping(monitor="loss", patience=10, mode='auto', min_delta=0)
    history = model.fit(X_train, y_train, batch_size=batch_size,
                        epochs=nb_epoch, callbacks=[early_stopping])

    '''lstm_input = Input(shape=(n_inputs, n_features), name='lstm_input')
    inputs = LSTM(150, name='first_layer')(lstm_input)
    inputs = Dense(1, name='dense_layer')(inputs)
    output = Activation('linear', name='output')(inputs)
    model = Model(inputs=lstm_input, outputs=output)
    adam = Adam()
    model.compile(optimizer=adam, loss='mse')
    model.fit(X_train, y_train, batch_size=10, epochs=93)'''

    # predict
    y_pred = model.predict(X_test)
    print('y_pred', y_pred)
    
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = -1
    
    y_test[np.where(y_test == 0)] = -1
    print(y_test)
    

    # compare decisions
    print('Accuracy:', accuracy_score(y_pred, y_test))

    backtest_func(data[-len(y_test):], y_pred)
    print('this was for our prediction')

    backtest_func(data[-len(y_test):], y_test)
    print('this was for the real value')


if __name__ == '__main__':
    main()
