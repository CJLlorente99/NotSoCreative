import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from Backtest import backtest_func
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional
import tensorflow as tf
import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.optimizers import Adam, Adamax
from keras.callbacks import History
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
import numpy as np
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
from tensorflow.keras.utils import to_categorical
#from keras.layers.normalization import BatchNormalization

# tf.random.set_seed(20)
np.random.seed(10)


def build_model(n_inputs, n_features):
    model = Sequential()
    model.add(LSTM(units=200, return_sequences=True, input_shape=(n_inputs, n_features)))
    model.add(Dropout(0.01))
    model.add(LSTM(units=200))
    model.add(Dropout(0.01))
    # model.add(Dense(32, kernel_initializer="uniform", activation='relu'))
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    # history = model.fit
    return model

# usual data
def class_LSTM(n_inputs, n_features):
    model = Sequential()
    model.add(LSTM(units=250, return_sequences=True, input_shape=(n_inputs, n_features)))
    model.add(Dropout(0.1))
    model.add(LSTM(175))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model
# only things with 6 values
def classsix_LSTM(n_inputs, n_features):
    opt = Adamax(learning_rate=0.06)
    model = Sequential()
    model.add(LSTM(units=200, return_sequences=True, input_shape=(n_inputs, n_features)))
    model.add(Dense(32, activation='relu'))
    #model.add(BatchNormalization())
    model.add(LSTM(150))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

'''OrderedDict([('batch_size', 8), ('learning_rate', 0.0087051376109549), ('nb_epoch', 35), 
('optimizerL', 'Adamax'), ('unit1', 176), ('unit2', 125)])'''

'''{'batch_size': 14, 'learning_rate': 0.0015336004104650077, 'nb_epoch': 34, 'unit1': 165, 'unit2': 151}
best score: 0.9319317160826595'''

def classHPO_LSTM(n_inputs, n_features):
    opt = Adamax(learning_rate=0.009)
    model = Sequential()
    model.add(LSTM(units=175, return_sequences=True, input_shape=(n_inputs, n_features)))
    model.add(Dropout(0.1))
    model.add(LSTM(125))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
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


    return X_train, X_test, y_train, y_test


def fit_model(X_train, y_train, epochs, batch_size):
    # define neural network model
    n_inputs = X_train.shape[1]
    n_features = X_train.shape[2]
    model = class_LSTM(n_inputs, n_features)
    # model.fit(X_train, y_train, batch_size=10, epochs=93)
    # fit the model on the training dataset
    early_stopping = EarlyStopping(monitor="accuracy", patience=10, mode='auto', min_delta=0)
    model.fit(X_train, y_train, verbose=2, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])
    return model


def fit_ensemble(n_members, X_train, X_test, y_train, y_test, epochs, batch_size):
    ensemble = list()
    y_pred = np.empty((n_members, y_test.shape[0]))
    accuracy  = []
    prob = np.empty((n_members, y_test.shape[0]))
    for i in range(n_members):
        # define and fit the model on the training set
        model = fit_model(X_train, y_train, epochs, batch_size)
        # evaluate model on the test set
        yhat = model.predict(X_test, verbose=2)
        print('yhat', yhat.shape)
        prob[i, :] = yhat.flatten()
        
        yhat[yhat > 0.5] = 1
        yhat[yhat <= 0.5] = 0
        
        
        acc = accuracy_score(y_test, yhat)
        print('>%d, acc: %.3f' % (i + 1, acc))
        accuracy.append(acc)
        # store the model and prediction
        ensemble.append(model)
        y_pred[i, :] = yhat.flatten()
    return ensemble, y_pred, accuracy, prob



def majority_vote(yhat):
     # if majority is 1 -> signal = 1 -> buy
    y_mean = []
    probs = []
    for i in range(yhat.shape[1]):
        y_10 = yhat[:, i]
        probs.append(np.sum(y_10))
        n_one = np.count_nonzero(y_10 == 1)
        length =  round(y_10.shape[0] * 0.5)
        if n_one > length:
            y_mean.append(1)
        else:
            y_mean.append(-1)
    #error here
    #probs_mean = probs / len(probs)
    return y_mean #, probs_mean


# make predictions with the ensemble and calculate a prediction interval
'''def predict_with_pi(ensemble, X):
    # make predictions
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    # calculate 95% gaussian prediction interval
    interval = 1.96 * yhat.std()
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper'''


def main():
    '''data = yf.download(tickers='^GSPC', start='2020-01-01', end='2023-01-03')
    data.head(10)
    data['Volume_t_1'] = data.Volume.shift(1)
    data['High_t_1'] = data.High.shift(1)
    data['Adj_Close_t_1'] = data['Adj Close'].shift(1)
    data['Close'] = data['Close'].shift(+1)
    data['Low_t_1'] = data.Low.shift(1)
    
    data['log_Close'] = np.log(data['Close'])
    data["log_Open"] = np.log(data["Open"])
    data["Return_close"] = data["log_Close"] - data["log_Close"].shift(+1)
    data["Return_open"] = data["log_Open"] - data["log_Open"].shift(+1)
    data["Return_intraday"] = data["log_Close"] - data["log_Open"].shift(+1)
    data['Return_interday'] = data["log_Open"] - data["log_Close"]
    data['Diff_open'] = data["Open"] - data["Open"].shift()
    data['Diff_close'] = data['Close'] - data['Close'].shift()
    data['Diff_intraday'] = data['Close'] - data['Open'].shift(1)
    data['Diff_interday'] = data['Open'] - data['Close']
    data['Target'] = data["log_Open"].shift(-1) - data["log_Open"]
    data.drop(['Close', 'Volume', 'High', 'Low', 'Adj Close'], axis=1, inplace=True)
    # data['TargetClass'] = [1 if data.Target[i]>0 else 0 for i in range(len(data))]
    data.dropna(inplace=True)
    print(data)'''
    
    # target: open to open but with values shifted -> no close, volume for day of decision
    '''data = pd.read_csv('featureSelectionDataset_Paul_Class_shift_open.csv', sep=',', header=0, index_col=0, parse_dates=True,
                       decimal=".")
    #print(data)
    data['Return_open'] = np.log(data.Open) - np.log(data.Open.shift())
    data = data.iloc[-600:,:]
    data.dropna(inplace=True)

    y_target = np.asarray([1 if data.Target[i] > 0 else 0 for i in range(len(data))]).reshape(-1, 1)
    data = data.drop(['Target'], axis=1)

    scaler = StandardScaler()
    data_set_scaled = scaler.fit_transform(data)        
    #data_set_scaled = np.concatenate((data_set_scaled, y_open), axis=1)
    #data_set_scaled = np.concatenate((data_set_scaled, y_close), axis=1)
    data_set_scaled = np.concatenate((data_set_scaled, y_target), axis=1)
    print(data_set_scaled)'''
    

    
    
    # target: return: close open from same day
    data = pd.read_csv('featureSelectionDataset_Paul_Class_shift_intra.csv', sep=',', header=0, index_col=0, parse_dates=True,
                       decimal=".")
    data = data.iloc[-600:,:]
    print(data.Return_intraday)
    print(data)
    y_target = np.asarray([1 if data.Target[i] > 0 else 0 for i in range(len(data))]).reshape(-1, 1)
    data = data.drop(['Target'], axis=1)

    scaler = StandardScaler()
    data_set_scaled = scaler.fit_transform(data)        
    #data_set_scaled = np.concatenate((data_set_scaled, y_open), axis=1)
    #data_set_scaled = np.concatenate((data_set_scaled, y_close), axis=1)
    data_set_scaled = np.concatenate((data_set_scaled, y_target), axis=1)
    print(data_set_scaled)
    
    
    backcandles = 40

    # choose columns: all but target variable (its last column)
    liste = list(range(0, data.shape[1] - 1))

    # print(data.iloc[:, liste])

    # split data into train test sets
    pred_days = 10

    # prepare data for lstm
    X_train, X_test, y_train, y_test = prepare_data(data_set_scaled, backcandles, liste, pred_days)
    print('test', y_test)
    print(X_test)
    

    # train and predict
    # n_members -> how many predictors we wanted to use
    n_members = 10
    epochs = 35
    batch_size = 8
    ensemble, y_pred, accuracy, prob = fit_ensemble(n_members, X_train, X_test, y_train, y_test, epochs, batch_size)
    print('probabilites', prob)
    print('y_pred', y_pred)
    
    # majority vote, probs for amount
    y_mean = majority_vote(y_pred)
    #print('probs', probs_mean)

    # just to make a decision signal for the backtest.py -> there is -1 for sell and +1 for buy, so I have to switch the 0 to a -1
    y_test[np.where(y_test == 0)] = -1
    
    # compare decisions
    print(f'Accuracy:{accuracy_score(y_test, y_mean)*100}%')
    print('Accuracies from all ensembles', accuracy)

    backtest_func(data[-len(y_test):], y_mean)
    print('this was for our prediction')

    backtest_func(data[-len(y_test):], y_test)
    print('this was for the real value')
    
    print('prob_mean', np.sum(prob, axis=0) / n_members)

if __name__ == '__main__':
    main()

