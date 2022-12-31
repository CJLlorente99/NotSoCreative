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
    model.add(Bidirectional(LSTM(units=200, return_sequences=True, input_shape=(n_inputs, n_features))))
    model.add(Dropout(0.1))
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




def fit_model(X_train, y_train, epochs, batch_size):
    # define neural network model
    n_inputs = X_train.shape[1]
    n_features = X_train.shape[2]
    #model = build_model(n_inputs, n_features)
    lstm_input = Input(shape=(n_inputs, n_features), name='lstm_input')
    inputs = LSTM(150, name='first_layer')(lstm_input)
    inputs = Dense(1, name='dense_layer')(inputs)
    output = Activation('linear', name='output')(inputs)
    model = Model(inputs=lstm_input, outputs=output)
    adam = Adam()
    model.compile(optimizer=adam, loss='mse')
    #model.fit(X_train, y_train, batch_size=10, epochs=93)
    # fit the model on the training dataset
    early_stopping = EarlyStopping(monitor="loss", patience=10, mode='auto', min_delta=0)
    model.fit(X_train, y_train, verbose=2, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])
    return model


def fit_ensemble(n_members, X_train, X_test, y_train, y_test, epochs, batch_size):
    ensemble = list()
    y_pred = np.empty((n_members, y_test.shape[0]))

    for i in range(n_members):
        # define and fit the model on the training set
        model = fit_model(X_train, y_train, epochs, batch_size)
        # evaluate model on the test set
        yhat = model.predict(X_test, verbose=2)
        print('yhat', yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        print(f'MAPE:', mean_absolute_percentage_error(y_test, yhat))
        # store the model and prediction
        ensemble.append(model)
        y_pred[i, :] = yhat.flatten()
    return ensemble, y_pred


def inverse_scaling(data, yhat, scaler):
    y_preds = np.empty((yhat.shape[0], yhat.shape[1]))

    for i in range(yhat.shape[0]):
        y_pred = np.tile(yhat[i, :].reshape(-1, 1), (1, data.shape[1]))
        y_pred = scaler.inverse_transform(y_pred)
        y_pred = y_pred[:, -1]
        y_preds[i, :] = y_pred
    return y_preds


def calculate_bounds(yhat):
    lower = []
    upper = []
    y_mean = []
    for i in range(yhat.shape[1]):
        interval = 1.96 * yhat[:, i].std()
        y_mean.append(yhat[:, i].mean())
        lower.append(yhat[:, i].mean() - interval)
        upper.append(yhat[:, i].mean() + interval)
    return lower, y_mean, upper


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
    data = pd.read_csv('featureSelectionDataset_Paul.csv', sep=',', header=0, index_col=0, parse_dates=True, decimal=".")
    data = data.iloc[-600:,:]
    data.dropna(inplace=True)
    print(data)

    # scale data
    scaling = 1
    if scaling == 1:
        scaler = StandardScaler()
        data_set_scaled = scaler.fit_transform(data)
        data_set_scaled.shape[0]
    else:
        data_set_scaled = np.asarray(data)
        

    # choose how many look back days
    backcandles = 30

    # choose columns
    liste = list(range(0, data.shape[1] - 1))
    print(liste)
    print(data.iloc[:, liste])

    # split data into train test sets
    pred_days = 10

    # prepare data for lstm
    X_train, X_test, y_train, y_test = prepare_data(data_set_scaled, backcandles, liste, pred_days)
    print('test', y_test.shape)


    # train and predict
    n_members = 10
    epochs = 30
    batch_size = 10
    ensemble, y_pred_scale = fit_ensemble(n_members, X_train, X_test, y_train, y_test, epochs, batch_size)
    
    # bounds scaled
    lower_scale, y_mean_scale, upper_scale = calculate_bounds(y_pred_scale)
    
    # errors
    print('MSE scaled:', mean_squared_error(y_test, y_mean_scale))
    print('MAPE scaled:', mean_absolute_error(y_test, y_mean_scale))
    print('MAE scaled:', mean_absolute_percentage_error(y_test, y_mean_scale))

    plt.figure(figsize=(16, 8))
    plt.plot(y_test, color='black', label='Test')
    plt.plot(lower_scale, color='green', label='lower scale')
    plt.plot(upper_scale, color='green', label='upper scale')
    plt.plot(y_mean_scale, color='green', label='y_mean_scale')
    plt.axhline(y=0, color='r', linestyle='-', label="Zero")
    plt.legend()
    plt.title('Scaled Returns')
    plt.show()
    
    # inverse scaling
    if scaling == 1:
        y_pred = inverse_scaling(data, y_pred_scale, scaler)
        y_test = np.tile(y_test.reshape(-1, 1), (1, data.shape[1]))
        y_test = scaler.inverse_transform(y_test)
        y_test = y_test[:, -1]

    # bounds unscaled
    lower, y_mean, upper = calculate_bounds(y_pred)
    
    plt.figure(figsize=(16, 8))
    plt.plot(y_test, color='black', label='Test')
    plt.plot(lower, color='green', label='lower')
    plt.plot(upper, color='green', label='upper')
    plt.plot(y_mean, color='green', label='y_mean')
    plt.axhline(y=0, color='r', linestyle='-', label="Zero")
    plt.legend()
    plt.title('Returns')
    plt.show()
    
    plt.figure(figsize=(16, 8))
    for i in range((y_pred.shape[0])):
        plt.plot(y_pred[i, :])
    plt.axhline(y=0, color='r', linestyle='-', label="Zero")
    plt.plot(y_test, color='black', linewidth=4, label='Test')
    plt.title('all Returns')
    plt.legend()
    plt.show()
    
    fig = go.Figure()
    for i in range((y_pred.shape[0])):
        fig.add_trace(go.Scatter(y_pred[i, :], name=f'y_pred_{i}'))
    fig.add_hline(y=0)
    fig.add_trace(go.Scatter(y_test, name='y_test'))
    fig.update_layout(
        title=f'Return',
        xaxis_title=f'Time',
        yaxis_title=f'Return',
        font=dict(family="Tahoma", size=18, color="Black"))
    fig.show()
        
        
    
    # errors
    print('MSE:', mean_squared_error(y_test, y_mean))
    print('MAPE:', mean_absolute_error(y_test, y_mean))
    print('MAE:', mean_absolute_percentage_error(y_test, y_mean))

    
    # create decision signal
    decision = [0] * len(y_mean)
    for i, yp in enumerate(y_mean):
        # if i == 0:
        if yp < 0:
            decision[i] = -1
        elif yp > 0:
            decision[i] = 1
        else:
            decision[i] = 0

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
    print('Accuracy:', accuracy_score(real_signal, decision))

    backtest_func(data[-len(y_test):], decision)
    print('this was for our prediction')

    backtest_func(data[-len(y_test):], real_signal)
    print('this was for the real value')
    


if __name__ == '__main__':
    main()

