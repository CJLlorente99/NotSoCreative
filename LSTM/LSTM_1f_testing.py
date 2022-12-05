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
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import pandas as pd
from keras.layers import LeakyReLU
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
LeakyReLU = LeakyReLU(alpha=0.01)

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

def build_model_or(n_inputs, n_features):
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

'''('activation'LeakyReLU), ('batch_size', 25), ('dropout', 0.5469492804461135), ('dropout_rate', 0.015766818556683073)
('layers1', 1), ('layers2', 1), ('learning_rate', 0.09699768160862692), ('nb_epoch', 28), ('optimizerL', 'Adadelta'), ('unit', 53)])
max error: -0.0017390910866125163'''

def stack2(z, window_size):
    X, y = [], []
    for i in range(window_size, len(z)):
        X.append(z[i - window_size: i])
        y.append(z[i])
        X = np.array(X)
        y = np.array(y)
    return X, y

def multi2(sequence, n_steps_in, n_steps_out):
    X = []
    y = []
    for i in range(n_steps_in, len(sequence) - n_steps_out + 1):
        X.append(sequence[i - n_steps_in: i])
        y.append(sequence[i: i + n_steps_out])
    X = np.array(X)
    y = np.array(y)
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
    '''plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rollmean, color='red', label='Rolling Mean')
    plt.plot(rollstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show()'''

    print("Results of dickey fuller test")
    adft = adfuller(timeseries, autolag='AIC')
    output = pd.Series(adft[0:4], index=['Test Statistics', 'p-value', 'No. of lags used', 'Number of observations used'])
    for key, values in adft[4].items():
        output['critical value (%s)'%key] = values
    print(output)

    '''# Building the model
model = Sequential()
# Adding a Bidirectional LSTM layer
model.add(Bidirectional(LSTM(64,return_sequences=True, dropout=0.5, input_shape=(X_train.shape[1], X_train.shape[-1]))))
model.add(Bidirectional(LSTM(20, dropout=0.5)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='rmsprop'''

def main():
    # automatic: if date = -> n_future 10 -1

    # load data
    #start = dt.datetime(2018, 1, 1)
    #end = dt.datetime(2022, 11, 1)
    # for us
    ticker = '^GSPC'
    start = dt.datetime(2019, 1, 1)
    end = dt.datetime(2022, 3, 1)
    df = web.DataReader(ticker, 'yahoo', start, end)
    #df.to_csv(r'/Users/paulheller/PycharmProjects/DataScience/SavedData/datatillnov.csv', header=True)
    #df = pd.read_csv(r'/Users/paulheller/PycharmProjects/DataScience/SavedData/datatillnov.csv', index_col=0)
    df_or = df.Open.copy()
    df_open = df['Open']

    # future data to test
    #start_fut = dt.datetime(2022, 11, 2)
    #end_fut = dt.datetime(2022, 11, 15)
    start_fut = dt.datetime(2022, 3, 2)
    end_fut = dt.datetime(2022, 3, 30)
    df_fut = web.DataReader(ticker, 'yahoo', start_fut, end_fut)
    #df.to_csv(r'/Users/paulheller/PycharmProjects/DataScience/SavedData/datafromnov.csv', header=True)
    #df = pd.read_csv(r'/Users/paulheller/PycharmProjects/DataScience/SavedData/datafromnov.csv', index_col=0)
    df_fut = df_fut['Open']


    print('raw data')
    #test_stationarity(df_open)

    transform = 0
    if transform == 0:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(np.asarray(df_open).reshape(-1, 1))
        df_open = scaler.transform(np.asarray(df_open).reshape(-1, 1))
    elif transform == 1:
        df_log = np.log(df_open)
        df_tf = np.sqrt(df_log)
        df_log_diff = df_tf - df_tf.shift()
        df_open = df_log_diff
        print(df_open)
        df_open.dropna(inplace=True)

    # test stationarity
    #test_stationarity(df_open)

    # create data set for lstm: pred_day how much days I use to predict next day
    n_days_in = 30
    n_days_out = 1

    n_features = 1
    #Xp_stack, yp_stack = stack2((np.asarray(df_open)), window_size=n_days_in)
    Xp, yp = preprocess_lstm((np.asarray(df_open)).reshape(-1, 1), n_steps_in=n_days_in, n_features=1)
    #Xp_stack, yp_stack = multi2((np.asarray(df_open)).reshape(-1, 1), n_days_in, n_days_out)

    # build LSTM
    n_inputs = Xp.shape[1] # n_days_in
    nb_epoch = 10
    batch_size = 32
    model = build_model_or(n_inputs, n_features)
    early_stopping = EarlyStopping(monitor="loss", patience=10, mode='auto', min_delta=0)
    history = model.fit(Xp, yp, batch_size=batch_size,
                        epochs=nb_epoch, callbacks=[early_stopping])

    # generate the forecasts
    x_pred = Xp[-1:, :, :]  # last observed input sequence (1, 60, 1)
    y_pred = yp[-1]  # 1
    y_future = []
    n_future = len(df_fut)

    # future prediction
    for i in range(n_future):
        # feed the last forecast back to the model as an input
        x_pred = np.append(x_pred[:, 1:, :], y_pred.reshape(1, 1, 1), axis=1)

        # generate the next forecast
        y_pred = model.predict(x_pred)

        # save the forecast
        y_future.append(y_pred.flatten()[0])

    y_future = np.array(y_future).reshape(-1, 1)


    if transform == 0:
        y_future = scaler.inverse_transform(y_future)


    # organize the results in a data frame
    df_past = df[['Open']].reset_index()
    df_past.rename(columns={'index': 'Date'}, inplace=True)
    df_past['Date'] = pd.to_datetime(df_past['Date'])
    df_past['Forecast'] = np.nan

    df_future = pd.DataFrame(columns=['Date', 'Open', 'Forecast'])
    df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_future)
    df_future['Forecast'] = y_future.flatten()

    if transform == 1:
        df_future.loc[0, ('Forecast')] = df_future.loc[0, ('Forecast')] + df_tf[-1]
        for i in range(n_future-1):
            df_future.loc[i+1, ('Forecast')] = df_future.loc[i+1, ('Forecast')] + \
                                               df_future.loc[i, ('Forecast')]
        print(df_future['Forecast'])
        df_future['Forecast'] = df_future['Forecast'] ** 2
        df_future['Forecast'] = np.exp(df_future['Forecast'])
    df_future['Open'] = np.nan

    print(f'forecast_future {df_future.Forecast} ,actual future, {df_fut}')
    print('MSE', mean_squared_error(df_fut, df_future['Forecast'].values))
    print('MAE', mean_absolute_error(df_fut, df_future['Forecast'].values))
    print('MAPE :', mean_absolute_percentage_error(df_fut, df_future['Forecast'].values))
    print(f'mape self: {np.mean(np.abs((df_fut.values - df_future.Forecast.values) / df_fut.values)) * 100} %')
    results = df_past.append(df_future).set_index('Date')
    results.plot(title='S&P 500')
    plt.show()

    print(results[-60:])
    results[-60:].plot(title='S&P 500')
    plt.show()
    print(df_fut)
    results[-n_future:].plot(title='S&P 500')
    plt.plot(df_fut, color='blue')
    plt.show()
# to do:
# test this on real data
    # plot
    '''plt.figure(figsize=(10, 6))
    plt.grid()
    plt.plot(df_or, color='blue', label=f'actual prices of S&P 500')
    plt.plot(results_future_idx, results[-n_future:], color='green', label='Predicted Train prices')
    plt.plot(y_pred, color='red', label='Predicted Test prices')
    plt.xlabel('Date')
    plt.ylabel('Opening Prices')
    plt.title('S&P 500 opening price')
    plt.legend()
    plt.show()'''

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'])
    plt.title('Training loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()


if __name__ == '__main__':
    main()
    # für buy/sell strategy, (0, 1) up or down prediction
    # features einbauen? TA-lib einlesen
    # buy/sell strategy: RF classify

# mean portfolio value
# confidence means how much we risk: divide et impera how to look on these metrics
# price higher predicted -> buy
# trend
# probability


#start = dt.datetime(20016, 1, 1)
#end = dt.datetime(2018, 9, 11)
#start_fut = dt.datetime(2018, 9, 12)
#end_fut = dt.datetime(2018, 9, 26)
#MSE 694.4408847505935
#MAE 24.519917674805335