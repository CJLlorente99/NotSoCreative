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
from scipy.stats import uniform, randint
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


def class_LSTM(n_inputs, n_features):
    model = Sequential()
    model.add(LSTM(units=200, return_sequences=True, input_shape=(n_inputs, n_features)))
    model.add(Dropout(0.01))
    model.add(LSTM(100))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def Bi_LSTM(n_inputs, n_features):
    optim = Adam(learning_rate=0.01)
    model = Sequential()
    model.add(Bidirectional(LSTM(units=400, return_sequences=True, input_shape=(n_inputs, n_features))))
    # model.add(Dropout(0.1))
    # model.add(Bidirectional(LSTM(units=200)))
    # model.add(Dropout(0.1))
    # model.add(Dense(32, kernel_initializer="uniform", activation='relu'))
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer=optim, loss='mean_squared_error')
    # history = model.fit
    return model

def prepare_data(data_set_scaled, backcandles, liste, pred_days):
    X = []
    for j in range(len(liste)):  # data_set_scaled[0].size):#2 columns are target not X
        X.append([])
        for i in range(backcandles, data_set_scaled.shape[0]):  # backcandles+2
            X[j].append(data_set_scaled[i - backcandles:i, liste[j]])

    # move axis from 0 to position 2
    X = np.moveaxis(X, [0], [2])

    X, yi = np.array(X), np.array(data_set_scaled[backcandles:, -1])
    y = np.reshape(yi, (len(yi), 1))
    splitlimit = X.shape[0] - pred_days

    X_train, X_test = X[:splitlimit], X[splitlimit:]
    y_train, y_test = y[:splitlimit], y[splitlimit:]
    print('shape', X_train)
    print('shapes', y_train.shape, y_test.shape)

    return X_train, X_test, y_train, y_test

def main():
    '''data = yf.download(tickers='^GSPC', start='2021-01-01', end='2022-12-22')
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
    print(data)'''
    # import data
    '''data = pd.read_csv('featureSelectionDataset_Paul.csv', sep=',', header=0, index_col=0, parse_dates=True,
                       decimal=".")
    data = data.iloc[-600:, :]
    data.dropna(inplace=True)
    y_open = np.asarray([1 if data.Return_open[i] > 0 else 0 for i in range(len(data))]).reshape(-1, 1)
    y_close = np.asarray([1 if data.Return_close[i] > 0 else 0 for i in range(len(data))]).reshape(-1, 1)
    y_target = np.asarray([1 if data.Target[i] > 0 else 0 for i in range(len(data))]).reshape(-1, 1)
    data = data.drop(['Target'], axis=1)
    print(data)

    # scale data
    scaler = StandardScaler()
    data_set_scaled = scaler.fit_transform(data)
    data_set_scaled = np.concatenate((data_set_scaled, y_open), axis=1)
    data_set_scaled = np.concatenate((data_set_scaled, y_close), axis=1)
    data_set_scaled = np.concatenate((data_set_scaled, y_target), axis=1)
    print('data_scaled', data_set_scaled.shape)'''
    
    # no return Open inside, but Class_Close inside
    data = pd.read_csv('featureSelectionDataset_Paul_Class.csv', sep=',', header=0, index_col=0, parse_dates=True,
                       decimal=".")
    data = data.iloc[-600:, :]
    data.dropna(inplace=True)
    
    y_close = np.asarray([1 if data.Return_close[i] > 0 else 0 for i in range(len(data))]).reshape(-1, 1)
    y_target = np.asarray([1 if data.Target[i] > 0 else 0 for i in range(len(data))]).reshape(-1, 1)
    data = data.drop(['Target', 'Class_close'], axis=1)
    
    # scale data
    scaler = StandardScaler()
    data_set_scaled = scaler.fit_transform(data)
    data_set_scaled = np.concatenate((data_set_scaled, y_close), axis=1)
    data_set_scaled = np.concatenate((data_set_scaled, y_target), axis=1)
    print('data_scaled', data_set_scaled.shape)
    
    

    # choose how many look back days
    backcandles = 30

    # choose columns: all but target variable (its last column)
    liste = list(range(0, data.shape[1] - 1))

    # print(data.iloc[:, liste])

    # split data into train test sets
    pred_days = 10

    # prepare data for lstm
    X_train, X_test, y_train, y_test = prepare_data(data_set_scaled, backcandles, liste, pred_days)
    n_inputs = X_train.shape[1]
    n_features = X_train.shape[2]


    param_nn_rand = {
        'unit1': randint(125, 250),
        'unit2': randint(125, 250),
        # 'activation': ['relu', 'sigmoid', 'tanh', 'elu'],
        'learning_rate': loguniform(0.001, 0.1),
        'layers1': [0, 1],
        'layers2': [0, 1],
        # 'dropout': uniform(0, 1),
        # 'dropout_rate': uniform(0, 0.3),
        'nb_epoch': randint(50, 150),
        'batch_size': randint(2, 70),
        # 'normalization': uniform(0, 1),
        'optimizerL': ['Adam', 'Adagrad', 'Adamax', 'RMSprop', 'Adadelta']}

    param_nn_bs = {
        'unit1': Integer(125, 400),
        'unit2': Integer(125, 400),
        'learning_rate': Real(0.001, 0.1),
        # 'layers1': Integer(0, 1),
        # 'layers2': Integer(0, 1),
        'nb_epoch': Integer(10, 40),
        'batch_size': Integer(2, 70),
        'optimizerL': ['Adam', 'Adagrad', 'Adamax', 'Adadelta']}
    # 'layers1': Integer(0, 1),
    # 'activation': Categorical(['sigmoid', 'tanh', 'elu']),
    #        'dropout': Real(0, 1),
    # 'dropout_rate': Real(0, 0.3),

    print('build model')

    def build_LSTM(learning_rate=0.01, unit1=200, unit2=200,
                   nb_epoch=20, batch_size=20, optimizerL='Adam'):
        # activation='relu', layers1=1, layers2=1, dropout=0.3, dropout_rate=0.23
        optimizerD = {'Adam': Adam(learning_rate=learning_rate),
                      'Adadelta': Adadelta(learning_rate=learning_rate),
                      'Adagrad': Adagrad(learning_rate=learning_rate),
                      'Adamax': Adamax(learning_rate=learning_rate)}
        # 'Nadam': Nadam(learning_rate=learning_rate),
        # 'Ftrl': Ftrl(learning_rate=learning_rate)}
        #      'RMSprop': RMSprop(learning_rate=learning_rate),

        opt = optimizerD[optimizerL]
        nn = Sequential()
        nn.add((LSTM(units=unit1, return_sequences=True, input_shape=(n_inputs, n_features))))
        nn.add(Dropout(0.01))
        # for i in range(layers1):
        #   nn.add((LSTM(units=unit2, return_sequences=True)))
        #   nn.add(Dropout(0.01))
        nn.add((LSTM(units=unit2)))
        nn.add(Dropout(0.1))
        # for i in range(layers2):
        #    nn.add(Dense(32, activation='relu'))
        nn.add(Dense(1, activation='sigmoid'))
        nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor="accuracy", patience=6, mode='auto', min_delta=0)
        history = nn.fit(X_train, y_train, batch_size=batch_size,
                         epochs=nb_epoch, callbacks=[early_stopping])
        return nn

    nn = KerasRegressor(build_fn=build_LSTM, verbose=0)

    n_iter_search = 40

    # optimize
    print('opt')
    # search_reg = RandomizedSearchCV(nn_reg, param_nn_rand, n_iter=n_iter_search, scoring='neg_mean_squared_error')
    search_reg = BayesSearchCV(nn, param_nn_bs, n_iter=n_iter_search, scoring='accuracy', cv=5)
    search_reg.fit(X_train, y_train)
    print(search_reg.best_params_)
    print(f'best score: {search_reg.best_score_}')

    # test prediction
    y_pred = search_reg.predict(X_test)
    
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = -1

    y_test[np.where(y_test == 0)] = -1

    # compare decisions
    print('Accuracy:', accuracy_score(y_pred, y_test))

    backtest_func(data[-len(y_test):], y_pred)
    print('this was for our prediction')

    backtest_func(data[-len(y_test):], y_test)
    print('this was for the real value')

if __name__ == '__main__':
    main()
    

