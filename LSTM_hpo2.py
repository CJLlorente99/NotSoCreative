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
from skopt.space import Real, Categorical, Integer
from tensorflow.keras.optimizers import Adam, SGD, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from keras.callbacks import History
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
import numpy as np
import shap
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from keras.layers import LeakyReLU
from keras.wrappers.scikit_learn import KerasRegressor
from skopt import BayesSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
from sklearn.utils.fixes import loguniform
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# tf.random.set_seed(20)
np.random.seed(10)


# (Bidirectional(LSTM(20,
def build_model(n_inputs, n_features):
    model = Sequential()
    model.add(LSTM(units=150, return_sequences=True, input_shape=(n_inputs, n_features)))
    model.add(Dropout(0.1))
    model.add(LSTM(units=150))
    model.add(Dropout(0.2))
    # model.add(Dense(32, kernel_initializer="uniform", activation='relu'))
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    # history = model.fit
    return model


def Bi_LSTM(n_inputs, n_features):
    optim = Adam(learning_rate=0.01)
    model = Sequential()
    model.add(Bidirectional(LSTM(units=200, return_sequences=True, input_shape=(n_inputs, n_features))))
    model.add(Dropout(0.1))
    # model.add(Bidirectional(LSTM(units=200)))
    # model.add(Dropout(0.1))
    # model.add(Dense(32, kernel_initializer="uniform", activation='relu'))
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer=optim, loss='mean_squared_error')
    # history = model.fit
    return model


def main():
    data = yf.download(tickers='^GSPC', start='2021-01-01', end='2022-12-22')
    data.head(10)

    # Adding indicators
    data['RSI'] = ta.rsi(data.Open, length=15)
    data['EMAF'] = ta.ema(data.Open, length=20)
    data['EMAM'] = ta.ema(data.Open, length=100)
    data['EMAS'] = ta.ema(data.Open, length=150)

    data["log(Open)"] = np.log(data["Open"])
    # Target variable = log(price(t+1)-log(t))
    data["Return_before"] = data["log(Open)"] - data["log(Open)"].shift(+1)
    data["Return"] = data["log(Open)"].shift(-1) - data["log(Open)"]
    # Class: 1 = positive return, 0 = negative return
    # data["Class"] = [1 if data.Return[i] > 0 else 0 for i in range(len(data))]
    # data['TargetClass'] = [1 if data.Target[i]>0 else 0 for i in range(len(data))]
    data.dropna(inplace=True)
    data.reset_index(inplace=True)
    data.drop(['Volume', 'Close', 'Date'], axis=1, inplace=True)
    data.head(20)
    print(data)

    sc = StandardScaler()
    data_set_scaled = sc.fit_transform(data)
    data_set_scaled.shape[0]

    # multiple feature from data provided to the model
    X = []
    # print(data_set_scaled[0].size)
    # data_set_scaled=data_set.values
    backcandles = 30
    print(data_set_scaled.shape[0])
    # Indpendent variables
    # deleted range(8)
    # Columns as an input = list
    # 4,5,6,7,

    list = [4, 5, 6, 7, 9]
    for j in range(len(list)):  # data_set_scaled[0].size):#2 columns are target not X
        X.append([])
        for i in range(backcandles, data_set_scaled.shape[0]):  # backcandles+2
            X[j].append(data_set_scaled[i - backcandles:i, list[j]])

    # move axis from 0 to position 2
    X = np.moveaxis(X, [0], [2])

    # Erase first elements of y because of backcandles to match X length
    # del(yi[0:backcandles])
    # X, yi = np.array(X), np.array(yi)
    # Choose -1 for last column, classification else -2...
    X, yi = np.array(X), np.array(data_set_scaled[backcandles:, -2])
    # yi = yi * 1.5
    y = np.reshape(yi, (len(yi), 1))

    # split data into train test sets
    splitlimit = (len(X) - 60)
    # splitlimit = int(len(X)*0.9)
    print(splitlimit)
    X_train, X_test = X[:splitlimit], X[splitlimit:]
    y_train, y_test = y[:splitlimit], y[splitlimit:]
    n_inputs = X_train.shape[1]
    n_features = X_train.shape[2]

    param_nn_bs = {
        'unit1': Integer(50, 200),
        'unit2': Integer(50, 200),
        'activation': Categorical(['relu', 'sigmoid', 'tanh', 'elu']),
        'learning_rate': Real(0.001, 0.1),
        'layers1': Integer(0, 1),
        'layers2': Integer(0, 1),
        'dropout': Real(0, 1),
        'dropout_rate': Real(0, 0.3),
        'nb_epoch': Integer(50, 100),
        'batch_size': Integer(10, 100),
        # 'normalization': Real(0, 1),
        'optimizerL': ['Adam', 'RMSprop', 'Adagrad', 'Adamax', 'Adadelta']}
    # 'layers1': Integer(0, 1),

    print('build model')

    def build_LSTM(learning_rate=0.01, unit1=100, unit2=100, activation='relu', layers1=1, layers2=1, dropout=0.3,
                   dropout_rate=0.23,
                   nb_epoch=20, batch_size=20, optimizerL='Adam'):
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
        '''for i in range(layers1):
            nn.add(Bidirectional(LSTM(units=unit2, return_sequences=True)))
            if dropout > 0.5:
                nn.add(Dropout(dropout_rate))'''
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
                         epochs=nb_epoch, callbacks=[early_stopping])

        return nn

    nn_reg = KerasRegressor(build_fn=build_LSTM, verbose=0)

    n_iter_search = 20

    # optimize
    print('opt')
    search_reg = BayesSearchCV(nn_reg, param_nn_bs, n_iter=n_iter_search, scoring='neg_mean_squared_error', cv=5)
    search_reg.fit(X_train, y_train)
    print(search_reg.best_params_)
    print(f'max error: {search_reg.best_score_}')

    # test prediction
    y_pred = search_reg.predict(X_test)

    decision = [0] * len(y_pred)
    for i, yp in enumerate(y_pred):
        # if i == 0:
        if yp < -0.05:
            decision[i] = -1
        elif yp > 0.05:
            decision[i] = 1
        else:
            decision[i] = 0

    real_signal = [0] * len(y_test)
    for i, yp in enumerate(y_test):
        # if i == 0:
        if yp < -0.05:
            real_signal[i] = -1
        elif yp > 0.05:
            real_signal[i] = 1
        else:
            real_signal[i] = 0

    print('Accuracy:', accuracy_score(real_signal, decision))
    print('y_test', y_test, 'y_pred', y_pred)

    '''else:   
            if y_pred[i] - y_pred[i - 1] < 0:
                decision[i] = -1
            elif y_pred[i] - y_pred[i - 1] > 0:
                decision[i] = 1
            else:
                decision[i] = 0'''

    backtest_func(data[-len(y_test):], decision)

    plt.figure(figsize=(16, 8))
    plt.plot(y_test, color='black', label='Test')
    plt.plot(y_pred, color='green', label='pred')
    plt.axhline(y=0, color='r', linestyle='-', label="Zero")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()