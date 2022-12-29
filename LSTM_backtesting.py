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
from keras.optimizers import Adam, Adamax
from keras.callbacks import History
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
import numpy as np

from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

#tf.random.set_seed(20)
np.random.seed(10)
# (Bidirectional(LSTM(20,

def build_model(n_inputs, n_features):
    model = Sequential()
    model.add(LSTM(units=150, return_sequences=True, input_shape=(n_inputs, n_features)))
    model.add(Dropout(0.1))
    model.add(LSTM(units=150))
    model.add(Dropout(0.2))
    #model.add(Dense(32, kernel_initializer="uniform", activation='relu'))
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    #history = model.fit
    return model

def Bi_LSTM(n_inputs, n_features):
    optim = Adam(learning_rate=0.01)
    model = Sequential()
    model.add(Bidirectional(LSTM(units=200, return_sequences=True, input_shape=(n_inputs, n_features))))
    model.add(Dropout(0.1))
    #model.add(Bidirectional(LSTM(units=200)))
    #model.add(Dropout(0.1))
    #model.add(Dense(32, kernel_initializer="uniform", activation='relu'))
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer=optim, loss='mean_squared_error')
    #history = model.fit
    return model

    '''OrderedDict([('activation', 'elu'), ('batch_size', 10), 
                 ('dropout', 0.02213745001720729), 
                 ('dropout_rate', 0.22246482863657333), 
                 ('layers2', 0), ('learning_rate', 0.009968434553604313), 
                 ('nb_epoch', 93), ('optimizerL', 'Adamax'), 
                 ('unit1', 175), ('unit2', 175)])
            optimizerD = {'Adam': Adam(learning_rate=learning_rate), 'SGD': SGD(learning_rate=learning_rate),
                      'Adadelta': Adadelta(learning_rate=learning_rate),
                      'Adagrad': Adagrad(learning_rate=learning_rate), 'Adamax': Adamax(learning_rate=learning_rate),
                      'Nadam': Nadam(learning_rate=learning_rate), 'Ftrl': Ftrl(learning_rate=learning_rate)}
        #      'RMSprop': RMSprop(learning_rate=learning_rate),

        opt = optimizerD[optimizerL]
        nn = Sequential()
        nn.add(Bidirectional(LSTM(units=unit1, return_sequences=True, input_shape=(n_inputs, n_features))))
        if dropout > 0.5:
            nn.add(Dropout(dropout_rate))
        for i in range(layers1):
            nn.add(Bidirectional(LSTM(units=unit2, return_sequences=True)))
            if dropout > 0.5:
                nn.add(Dropout(dropout_rate))
        for i in range(layers1):
            nn.add(Bidirectional(LSTM(units=unit2)))
            if dropout > 0.5:
                nn.add(Dropout(dropout_rate))
        for i in range(layers2):
            nn.add(Dense(32, activation=activation))
        nn.add(Dense(1, activation='linear'))
        nn.compile(loss='mse', optimizer=opt, metrics=['mse', 'mae'])
        early_stopping = EarlyStopping(monitor="loss", patience=6, mode='auto', min_delta=0)
        history = nn.fit(X_train, y_train, batch_size=batch_size,
                         epochs=nb_epoch, callbacks=[early_stopping])'''

def LSTM_HPO(n_inputs, n_features):
    opt = Adamax(learning_rate=0.01)
    nn = Sequential()
    nn.add(Bidirectional(LSTM(units=175, return_sequences=True, input_shape=(n_inputs, n_features))))
    nn.add(Dropout(0.01))
    nn.add(Bidirectional(LSTM(units=175)))
    nn.add(Dense(1, activation='linear'))
    nn.compile(loss='mse', optimizer=opt, metrics=['mse', 'mae'])
    return nn

def prepare_data(data_set_scaled, backcandles, list, splitlimit):
    X = []
    for j in range(len(list)):  # data_set_scaled[0].size):#2 columns are target not X
        X.append([])
        for i in range(backcandles, data_set_scaled.shape[0]):  # backcandles+2
            X[j].append(data_set_scaled[i - backcandles:i, list[j]])

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
    data = pd.read_csv('featureSelectionDataset_Paul.csv', sep=',', header=0, index_col=0, parse_dates=True, decimal=".")
    data.dropna(inplace=True)
    print(data)

    # scale data
    scaler = StandardScaler()
    data_set_scaled = scaler.fit_transform(data)
    data_set_scaled.shape[0]

    # choose how many look back days
    backcandles = 30

    # choose columns: all
    liste = list(range(0, data.shape[1] - 1))
    print(liste)
    print(data.iloc[:, liste])

    # split data into train test sets
    splitlimit = (len(data) - 60)

    # prepare data for lstm
    X_train, X_test, y_train, y_test = prepare_data(data_set_scaled, backcandles, liste, splitlimit)
    print('test', y_test.shape)

    # build model
    '''n_inputs = X_train.shape[1]
    n_features = X_train.shape[2]
    nb_epoch = 93
    batch_size = 10
    #model = build_model(n_inputs, n_features)
    model = build_model(n_inputs, n_features)
    early_stopping = EarlyStopping(monitor="loss", patience=10, mode='auto', min_delta=0)
    history = model.fit(X_train, y_train, batch_size=batch_size,
                        epochs=nb_epoch, callbacks=[early_stopping])'''

    lstm_input = Input(shape=(backcandles, len(list)), name='lstm_input')
    inputs = LSTM(150, name='first_layer')(lstm_input)
    inputs = Dense(1, name='dense_layer')(inputs)
    output = Activation('linear', name='output')(inputs)
    model = Model(inputs=lstm_input, outputs=output)
    adam = Adam()
    model.compile(optimizer=adam, loss='mse')
    model.fit(X_train, y_train, batch_size=10, epochs=93)

    # predict
    y_pred = model.predict(X_test)

    plt.figure(figsize=(16, 8))
    plt.plot(y_test, color='black', label='Test')
    plt.plot(y_pred, color='green', label='pred')
    plt.axhline(y=0, color='r', linestyle='-', label="Zero")
    plt.legend()
    plt.title('Scaled Returns')
    plt.show()

    print('MSE Scaled:', mean_squared_error(y_test, y_pred))
    print('MAPE Scaled:', mean_absolute_error(y_test, y_pred))
    print('MAE scaled:', mean_absolute_percentage_error(y_test, y_pred))

    # inverse scaling
    y_test = np.tile(y_test.reshape(-1, 1), (1, data.shape[1]))
    y_test = scaler.inverse_transform(y_test)
    y_test = y_test[:, -1]

    y_pred= np.tile(y_pred.reshape(-1, 1), (1, data.shape[1]))
    y_pred = scaler.inverse_transform(y_pred)
    y_pred = y_pred[:, -1]

    print('MSE:', mean_squared_error(y_test, y_pred))
    print('MAPE:', mean_absolute_error(y_test, y_pred))
    print('MAE:', mean_absolute_percentage_error(y_test, y_pred))

    # create decision signal
    decision = [0] * len(y_pred)
    for i, yp in enumerate(y_pred):
        # if i == 0:
        if yp < 0:
            decision[i] = -1
        elif yp > 0:
            decision[i] = 1
        else:
            decision[i] = 0

    # create decision signal, if we would have the right prediction
    real_signal = [0] * len(y_test)
    for i, yp in enumerate(y_test):
        # if i == 0:
        if yp < 0:
            real_signal[i] = -1
        elif yp > 0:
            real_signal[i] = 1
        else:
            real_signal[i] = 0

    # compare decisions
    print('decision', decision, real_signal)
    print('Accuracy:', accuracy_score(real_signal, decision))

    backtest_func(data[-len(y_test):], decision)
    print('this was for our prediction')

    backtest_func(data[-len(y_test):], real_signal)
    print('this was for the real value')

    plt.figure(figsize=(16, 8))
    plt.plot(y_test, color='black', label='Test')
    plt.plot(y_pred, color='green', label='pred')
    plt.axhline(y=0, color='r', linestyle='-', label="Zero")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()