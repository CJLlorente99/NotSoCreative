from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import pandas_datareader as web
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras import optimizers
from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
import pandas as pd
from keras.layers import LeakyReLU
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
LeakyReLU = LeakyReLU(alpha=0.01)


def plot_price(df):
    plt.figure(figsize=(10, 6))
    plt.grid()
    plt.plot(df)
    plt.xlabel('Date')
    plt.ylabel('Opening Prices')
    plt.title('S&P 500 opening price')
    plt.legend()
    plt.show()


def build_model(n_inputs, n_features):
    model = Sequential()
    model.add(LSTM(units=150, return_sequences=True, input_shape=(n_inputs, n_features)))
    model.add(Dropout(0.1))
    model.add(LSTM(units=150, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(units=150))
    model.add(Dropout(0.2))
    model.add(Dense(32, kernel_initializer="uniform", activation='relu'))
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    #history = model.fit
    return model

def build_model_multistep(n_inputs, n_outputs, n_features):
    model = Sequential()
    model.add(LSTM(units=150, return_sequences=True, input_shape=(n_inputs, n_features)))
    model.add(Dropout(0.1))
    model.add(LSTM(units=150, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(units=150))
    model.add(Dropout(0.2))
    model.add(Dense(32, kernel_initializer="uniform", activation='relu'))
    model.add(Dense(units=n_outputs, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    #history = model.fit
    return model


'''('activation'LeakyReLU), ('batch_size', 25), ('dropout', 0.5469492804461135), ('dropout_rate', 0.015766818556683073)
('layers1', 1), ('layers2', 1), ('learning_rate', 0.09699768160862692), ('nb_epoch', 28), ('optimizerL', 'Adadelta'), ('unit', 53)])
max error: -0.0017390910866125163'''

def build_LSTM(n_inputs, n_features):
    opt = Adadelta(learning_rate=0.01)
    nn = Sequential()
    nn.add(LSTM(units=256, return_sequences=True, input_shape=(n_inputs, n_features)))
    nn.add(Dropout(0.2))
    nn.add(LSTM(units=53, return_sequences=True))
    nn.add(Dropout(0.05))
    nn.add(LSTM(units=53))
    nn.add(Dropout(0.05))
    nn.add(Dense(32, activation=LeakyReLU))
    nn.add(Dense(1, activation='linear'))
    nn.compile(loss='mse', optimizer=opt, metrics=['mse', 'mae'])
    return nn


def create_dataset2(dataset, time_step=1, n_features=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    dataX, dataY = np.array(dataX), np.array(dataY)
    dataX = dataX.reshape(dataX.shape[0], dataX.shape[1], n_features)
    return dataX, dataY

def preprocess_lstm(sequence, n_steps_in, n_features=1):
    # predicts one day in future
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        # check if we are beyond the sequence
        if end_ix >= len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)

    X = np.array(X)
    y = np.array(y)

    X = X.reshape((X.shape[0], X.shape[1], n_features))
    return X, y


def preprocess_multistep_lstm(sequence, n_steps_in, n_steps_out, n_features=1):
    # predicts n_steps_out in future
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)

    X = np.array(X)
    y = np.array(y)

    X = X.reshape((X.shape[0], X.shape[1], n_features))

    return X, y


def add_features(df, horizons):
    horizons = [2, 5, 60, 250, 500]
    new_predictors = []
    for horizon in horizons:
        rolling_averages = df.rolling(horizon).mean()

        ratio_column = f"Close_Ratio_{horizon}"
        df[ratio_column] = df["Open"] / rolling_averages["Open"]

        trend_column = f"Trend_{horizon}"
        # anstatt sum() mean()
        df[trend_column] = df.shift(1).rolling(horizon).sum()["Target"]

        new_predictors += [ratio_column, trend_column]
    df.dropna()
    return df

def test_stationarity(timeseries):
    # Null Hypothesis: The series has a unit root (value of a =1)
    # Alternate Hypothesis: The series has no unit root.
    # p-value is greater than 0.05 so we cannot reject the Null hypothesis
    #Determing rolling statistics
    rollmean = timeseries.rolling(12).mean()
    rollstd = timeseries.rolling(12).std()
    #Plot rolling statistics:
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rollmean, color='red', label='Rolling Mean')
    plt.plot(rollstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show()
    print("Results of dickey fuller test")
    adft = adfuller(timeseries, autolag='AIC')
    output = pd.Series(adft[0:4],index=['Test Statistics', 'p-value', 'No. of lags used', 'Number of observations used'])
    for key, values in adft[4].items():
        output['critical value (%s)'%key] = values
    print(output)

def main():
    # load data
    start = dt.datetime(2018, 1, 1)
    end = dt.datetime(2022, 1, 1)
    ticker = '^GSPC'
    df_open = web.DataReader(ticker, 'yahoo', start, end)
    #print(df_open.columns.values.tolist())
    df_open = df_open['Open']
    df_or = df_open

    print('raw data')
    test_stationarity(df_open)

    transformation = 3
    if transformation == 1:
        df_log = np.log(df_open)
        # for me
        df_open = df_log

    elif transformation == 2:
        df_log = np.log(df_open)
        df_log_diff = df_log - df_log.shift()
        df_open = df_log_diff
        df_open.dropna(inplace=True)

    elif transformation == 3:
        df_log = np.log(df_open)
        df_tf = np.sqrt(df_log)
        df_log_diff = df_tf - df_tf.shift()
        df_open = df_log_diff
        df_open.dropna(inplace=True)

    # test stationarity
    test_stationarity(df_open)

    # create data set for lstm: pred_day how much days I use to predict next day
    # X_train shape: (train_size,) as pd, after scaler: (train_size, 1) as np.array
    # after preprocess: X shape: (train_size - pre_day, pre_day, n_feature) -> n_feature = 1 -> only time series
    n_days_in = 60

    n_features = 1
    #Xp, yp = create_dataset2((np.asarray(df_open)).reshape(-1, 1), n_days_in, n_features)
    Xp, yp = preprocess_lstm((np.asarray(df_open)).reshape(-1, 1), n_days_in, n_features)

    # train, test split for original index
    test_days = round(0.4 * df_open.shape[0])
    train_or = df_or.iloc[:-test_days]
    test_or = df_or.iloc[-test_days:]

    # split for fitting and testing
    Xp_train, yp_train = Xp[:-test_days], yp[:-test_days]
    Xp_test, yp_test = Xp[-test_days:], yp[-test_days:]

    # build LSTM
    n_inputs = Xp_train.shape[1]
    nb_epoch = 10
    batch_size = 32
    model = build_model(n_inputs, n_features)
    early_stopping = EarlyStopping(monitor="loss", patience=10, mode='auto', min_delta=0)
    history = model.fit(Xp_train, yp_train, batch_size=batch_size,
                        epochs=nb_epoch, callbacks=[early_stopping])


    # predict training and testing data
    ytr_pred = model.predict(Xp_train)
    y_pred = model.predict(Xp_test)
    print('shape ytr', ytr_pred.shape)
    print('y_pred', y_pred.shape)

    # error
    print('RMSE TRAIN transf:', np.sqrt(mean_squared_error(yp_train, ytr_pred)))
    print('RMSE TEST transf:', np.sqrt(mean_squared_error(yp_test, y_pred)))

    # if i want to get real error: I have to transform yp_train/test in the way below
    # for preprocess only +1 not +1+1
    if transformation == 1:
        # flat prediction
        train_idx = train_or.iloc[n_days_in+1:]
        ytr_pred = pd.DataFrame(ytr_pred[:, 0], train_idx.index, columns=['Open'])
        ytr_pred = np.exp(ytr_pred)
        y_pred = pd.DataFrame(y_pred[:, 0], test_or.index, columns=['Open'])
        y_pred = np.exp(y_pred)

    elif transformation == 2:
        train_idx = train_or.iloc[n_days_in + 1:]
        ytr_pred = pd.DataFrame(ytr_pred[:, 0], train_idx.index, columns=['Open'])
        ytr_pred['Open'] = ytr_pred['Open'] + df_log.shift().values[n_days_in+1:-test_days]
        ytr_pred = np.exp(ytr_pred)
        y_pred = pd.DataFrame(y_pred[:, 0], test_or.index, columns=['Open'])
        y_pred['Open'] = y_pred['Open'] + df_log.shift().values[-test_days:]
        y_pred = np.exp(y_pred)

    elif transformation == 3:
        train_idx = train_or.iloc[n_days_in + 1:]
        ytr_pred = pd.DataFrame(ytr_pred[:, 0], train_idx.index, columns=['Open'])
        ytr_pred['Open'] = ytr_pred['Open'] + df_tf.shift().values[n_days_in+1:-test_days]
        ytr_pred = (ytr_pred ** 2)
        ytr_pred = np.exp(ytr_pred)

        y_pred = pd.DataFrame(y_pred[:, 0], test_or.index, columns=['Open'])
        y_pred['Open'] = y_pred['Open'] + df_tf.shift().values[-test_days:]
        y_pred = (y_pred ** 2)
        y_pred = np.exp(y_pred)

        y_train = pd.DataFrame(yp_train, train_idx.index, columns=['Open'])
        y_train['Open'] = y_train['Open'] + df_tf.shift().values[n_days_in+1:-test_days]
        y_train = (y_train ** 2)
        y_train = np.exp(y_train)



    print('RMSE TRAIN true:', np.sqrt(mean_squared_error(y_train, ytr_pred)))
    print('RMSE TEST true:', np.sqrt(mean_squared_error(test_or, y_pred)))
    #print('train or rmse', np.sqrt(mean_squared_error(train_or[-n_days_in-1], ytr_pred)))

    # plot
    plt.figure(figsize=(10, 6))
    plt.grid()
    plt.plot(df_or, color='blue', label=f'actual prices of S&P 500')
    plt.plot(ytr_pred, color='green', label='Predicted Train prices')
    plt.plot(y_pred, color='red', label='Predicted Test prices')
    plt.xlabel('Date')
    plt.ylabel('Opening Prices')
    plt.title('S&P 500 opening price')
    plt.legend()
    plt.show()

    # plot
    plt.figure(figsize=(10, 6))
    plt.grid()
    plt.plot(test_or, color='blue', label=f'actual prices of S&P 500')
    plt.plot(y_pred, color='red', label='Predicted Test prices')
    plt.xlabel('Date')
    plt.ylabel('Opening Prices')
    plt.title('S&P 500 opening price')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'])
    plt.title('Training loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

    '''plt.figure(figsize=(10, 6))
    plt.grid()
    plt.plot(df, color='blue', label=f'actual prices of S&P 500')
    plt.plot(X_train, color='green', label='Train prices')
    plt.plot(X_test, color='orange', label='Test prices')
    plt.xlabel('Date')
    plt.ylabel('Opening Prices')
    plt.title('S&P 500 opening price')
    plt.legend()
    plt.show()'''


if __name__ == '__main__':
    main()
    # für buy/sell strategy, (0, 1) up or down prediction
    # features einbauen? TA-lib einlesen
    # buy/sell strategy: RF classify

