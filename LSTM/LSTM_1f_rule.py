from sklearn.preprocessing import MinMaxScaler
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
    start = dt.datetime(2020, 1, 1)
    end = dt.datetime.today()
        #(2022, 3, 1)
    df = web.DataReader(ticker, 'yahoo', start, end)
    #df.to_csv(r'/Users/paulheller/PycharmProjects/DataScience/SavedData/datatillnov.csv', header=True)
    #df = pd.read_csv(r'/Users/paulheller/PycharmProjects/DataScience/SavedData/datatillnov.csv', index_col=0)
    df_or = df.Open.copy()
    df_open = df['Open']

    # future data to test
    #start_fut = dt.datetime(2022, 11, 2)
    #end_fut = dt.datetime(2022, 11, 15)
    start_fut = dt.datetime(2022, 3, 1)
    end_fut = dt.datetime(2022, 3, 30)
    df_fut = web.DataReader(ticker, 'yahoo', start_fut, end_fut)
    #df.to_csv(r'/Users/paulheller/PycharmProjects/DataScience/SavedData/datafromnov.csv', header=True)
    #df = pd.read_csv(r'/Users/paulheller/PycharmProjects/DataScience/SavedData/datafromnov.csv', index_col=0)
    df_fut = df_fut['Open']

    df_fut.reset_index(drop=True, inplace=True)



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
    n_days_in = 10
    n_days_out = 1

    n_features = 1
    #Xp_stack, yp_stack = stack2((np.asarray(df_open)), window_size=n_days_in)
    Xp, yp = preprocess_lstm((np.asarray(df_open)).reshape(-1, 1), n_steps_in=n_days_in, n_features=1)
    #Xp_stack, yp_stack = multi2((np.asarray(df_open)).reshape(-1, 1), n_days_in, n_days_out)

    # build LSTM
    n_inputs = Xp.shape[1] # n_days_in
    nb_epoch = 5
    batch_size = 32
    model = build_model_or(n_inputs, n_features)
    early_stopping = EarlyStopping(monitor="loss", patience=10, mode='auto', min_delta=0)
    history = model.fit(Xp, yp, batch_size=batch_size,
                        epochs=nb_epoch, callbacks=[early_stopping])

    # generate the forecasts
    x_pred_pred = Xp[-1:, :, :] # last observed input sequence (1, 60, 1)
    x_pred_act = Xp[-1:, :, :]
    y_pred_pred = yp[-1]
    y_pred_feed = yp[-1]
    y_future_pred = []
    y_future_act = []
    n_future = len(df_fut)-1
    print('len fut', n_future)
    print('fut arr', np.asarray(df_fut))
    #df_fut_arr = scaler.fit_transform(np.asarray(df_fut).reshape(-1, 1))
    if transform == 1:
        df_fut_log = np.log(df_fut)
        df_fut_tf = np.sqrt(df_fut_log)
        df_log_diff = df_fut_tf - df_fut_tf.shift()
        df_fut_arr = df_log_diff
        df_fut_arr.dropna(inplace=True)
        df_fut_arr = np.asarray(df_fut_arr)

    print('size fut_arr', df_fut_arr)

    df_fut = df_fut.iloc[1:]


    # future prediction
    for i in range(n_future):
        # feed the last forecast back to the model as an input
        x_pred_pred = np.append(x_pred_pred[:, 1:, :], y_pred_pred.reshape(1, 1, 1), axis=1)
        x_pred_act = np.append(x_pred_act[:, 1:, :], y_pred_feed.reshape(1, 1, 1), axis=1)
        #print('pred, pred', x_pred_pred)
        #print('pred, act', x_pred_act)

        # generate the next forecast
        y_pred_pred = model.predict(x_pred_pred)
        y_pred_act = model.predict(x_pred_act)
        y_pred_feed = df_fut_arr[i]
        #print('y_pred_feed', y_pred_feed)

        # save the forecast
        y_future_pred.append(y_pred_pred.flatten()[0])
        y_future_act.append(y_pred_act.flatten()[0])

    y_future_pred = np.array(y_future_pred).reshape(-1, 1)
    y_future_act = np.array(y_future_act).reshape(-1, 1)
    #print('y_f_pred', y_future_pred)
    #print('y_f_act', y_future_act)


    if transform == 0:
        y_future_pred = scaler.inverse_transform(y_future_pred)
        y_future_act = scaler.inverse_transform(y_future_act)


    # organize the results in a data frame for predicted
    df_past = df[['Open']].reset_index()
    df_past.rename(columns={'index': 'Date'}, inplace=True)
    df_past['Date'] = pd.to_datetime(df_past['Date'])
    df_past['Forecast'] = np.nan

    df_past_act = df_past.copy()

    df_future_pred = pd.DataFrame(columns=['Date', 'Open', 'Forecast'])
    df_future_pred['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_future)
    df_future_pred['Forecast'] = y_future_pred.flatten()


    if transform == 1:
        df_future_pred.loc[0, ('Forecast')] = df_future_pred.loc[0, ('Forecast')] + df_tf[-1]
        for i in range(n_future-1):
            df_future_pred.loc[i + 1, ('Forecast')] = df_future_pred.loc[i + 1, ('Forecast')] + \
                                                      df_future_pred.loc[i, ('Forecast')]
        print(df_future_pred['Forecast'])
        df_future_pred['Forecast'] = df_future_pred['Forecast'] ** 2
        df_future_pred['Forecast'] = np.exp(df_future_pred['Forecast'])
    df_future_pred['Open'] = np.nan

    # organize the results in a data frame for actual
    df_future_act = pd.DataFrame(columns=['Date', 'Open', 'Forecast'])
    df_future_act['Date'] = pd.date_range(start=df_past_act['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_future)
    df_future_act['Forecast'] = y_future_act.flatten()

    if transform == 1:
        df_future_act.loc[0, ('Forecast')] = df_future_act.loc[0, ('Forecast')] + df_tf[-1]
        for i in range(n_future-1):
            df_future_act.loc[i + 1, ('Forecast')] = df_future_act.loc[i + 1, ('Forecast')] + \
                                                     df_future_act.loc[i, ('Forecast')]
        print(df_future_act['Forecast'])
        df_future_act['Forecast'] = df_future_act['Forecast'] ** 2
        df_future_act['Forecast'] = np.exp(df_future_act['Forecast'])
    df_future_act['Open'] = np.nan

    #print(f'forecast_future pred{df_future_pred.Forecast} ,actual future, {df_fut}')
    print('MSE pred:', mean_squared_error(df_fut, df_future_pred['Forecast'].values))
    print('MAE pred:', mean_absolute_error(df_fut, df_future_pred['Forecast'].values))
    #print('MAPE pred:', mean_absolute_percentage_error(df_fut, df_future_act['Forecast'].values))
    print(f'mape self pred: {np.mean(np.abs((df_fut.values - df_future_pred.Forecast.values) / df_fut.values)) * 100} %')

    #print(f'forecast_future act{df_future_act.Forecast} ,actual future, {df_fut}')
    print('MSE act:', mean_squared_error(df_fut, df_future_act['Forecast'].values))
    print('MAE act:', mean_absolute_error(df_fut, df_future_act['Forecast'].values))
    #print('MAPE pred:', mean_absolute_percentage_error(df_fut, df_future_act['Forecast'].values))
    print(f'mape self act: {np.mean(np.abs((df_fut.values - df_future_act.Forecast.values) / df_fut.values)) * 100} %')


    # have some 'error' because of weekends
    results_fut = df_past.append(df_future_pred).set_index('Date')
    results_act = df_past_act.append(df_future_act).set_index('Date')


    results_fut.reset_index(drop=True, inplace=True)
    results_act.reset_index(drop=True, inplace=True)
    print('fut together', results_fut[-30:])
    print('act together', results_act[-30:])
    print('df fut', df_fut)

    results_fut.plot(title='S&P 500 pred')
    results_act.plot(title='S&P 500 act pred')
    plt.show()

    y_f_pred = results_fut[-n_future:]
    y_f_pred = y_f_pred['Forecast']
    y_f_act = results_act[-n_future:]
    y_f_act = y_f_act['Forecast']
    y_f_pred.reset_index(drop=True, inplace=True)
    y_f_act.reset_index(drop=True, inplace=True)
    print('act only', y_f_act)
    print('pred only', y_f_pred)

    plt.figure()
    plt.plot(y_f_pred, color='red', label='pred on pred')
    plt.plot(y_f_act, color='green', label='act on train')
    plt.plot(df_fut, color='blue', label='actual')
    plt.ylabel('price')
    plt.xlabel('date')
    plt.title('s&p 500 price')
    plt.legend()
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