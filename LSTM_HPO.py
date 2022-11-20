from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import pandas_datareader as web
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import Sequential
from keras.layers import LSTM, Dense, BatchNormalization, Dropout
from keras import optimizers
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from keras.models import Sequential
from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from keras.callbacks import EarlyStopping, ModelCheckpoint
from skopt import BayesSearchCV
from keras.layers import LeakyReLU
from keras.wrappers.scikit_learn import KerasRegressor
LeakyReLU = LeakyReLU(alpha=0.01)

''' yticker = yf.Ticker('^GSPC')
    df_y = ticker.history(start, end, interval='1d')
    print(df_y.columns.values.tolist())
    print(df_y)'''

'''def create_model(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # first layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True), batch_input_shape=(None, sequence_length, n_features)))
            else:
                model.add(cell(units, return_sequences=True, batch_input_shape=(None, sequence_length, n_features)))
        elif i == n_layers - 1:
            # last layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        # add dropout after each layer
             model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model'''

def create_dataset(data, pred_day):
    X = []
    y = []
    for i in range(pred_day, data.shape[0]):
        X.append(data[i - pred_day:i, 0])
        y.append(data[i, 0])
    X, y = np.asarray(X), np.asarray(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y

def create_dataset2(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    dataX, dataY = np.array(dataX), np.array(dataY)
    dataX = dataX.reshape(dataX.shape[0], dataX.shape[1], 1)
    return dataX, dataY


def plot_price(df):
    plt.figure(figsize=(10, 6))
    plt.grid()
    plt.plot(df)
    plt.xlabel('Date')
    plt.ylabel('Opening Prices')
    plt.title('S&P 500 opening price')
    plt.legend()
    plt.show()


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

def main():
    # load data
    start = dt.datetime(2018, 1, 1)
    end = dt.datetime(2022, 1, 1)
    ticker = '^GSPC'
    df_open = web.DataReader(ticker, 'yahoo', start, end)
    # print(df_open.columns.values.tolist())
    df_open = df_open['Open']
    df_or = df_open

    print('raw data')
    #test_stationarity(df_open)


    # transformation
    df_log = np.log(df_open)
    df_tf = np.sqrt(df_log)
    df_log_diff = df_tf - df_tf.shift()
    df_open = df_log_diff
    df_open.dropna(inplace=True)

    # log and difference
    print('transformed data')
    #test_stationarity(df_open)

    # create data set for lstm: pred_day how much days I use to predict next day
    # X_train shape: (train_size,) as pd, after scaler: (train_size, 1) as np.array
    # after preprocess: X shape: (train_size - pre_day, pre_day, n_feature) -> n_feature = 1 -> only time series
    n_days_in = 60

    n_features = 1
    # Xp, yp = create_dataset2((np.asarray(df_open)).reshape(-1, 1), n_days_in, n_features)
    Xp, yp = preprocess_lstm((np.asarray(df_open)).reshape(-1, 1), n_days_in, n_features)
    print('full data processed', Xp.shape, yp.shape)

    # train, test split for original index
    test_days = round(0.4 * df_open.shape[0])
    train_or = df_or.iloc[:-test_days]
    test_or = df_or.iloc[-test_days:]

    # split for fitting and testing
    Xp_train, yp_train = Xp[:-test_days], yp[:-test_days]
    Xp_test, yp_test = Xp[-test_days:], yp[-test_days:]
    print('train processed', Xp_train.shape)
    print('test processed', yp_test.shape)


    # build LSTM
    n_inputs = Xp_train.shape[1]
    param_nn_bs = {
        'unit': Integer(30, 100),
        'activation': Categorical(['relu', 'sigmoid', 'tanh', 'elu', LeakyReLU]),
        'learning_rate': Real(0.001, 0.1),
        'layers1': Integer(1, 3),
        'layers2': Integer(0, 1),
        'dropout': Real(0, 1),
        'dropout_rate': Real(0, 0.3),
        'nb_epoch': Integer(10, 100),
        'batch_size': Integer(10, 100),
        #'normalization': Real(0, 1),
        'optimizerL': ['Adam', 'RMSprop', 'Adagrad', 'Adamax', 'Adadelta']}
    # 'optimizerL': Categorical(['SGD', 'Adam', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl'])}


    print('build model')
    def build_LSTM(learning_rate=0.01, unit=12, activation='relu', layers1=1, layers2=1, dropout=0.3, dropout_rate=0.23,
                   nb_epoch=20, batch_size=20, optimizerL='Adam'):

        optimizerD = {'Adam': Adam(learning_rate=learning_rate), 'SGD': SGD(learning_rate=learning_rate),
                      'RMSprop': RMSprop(learning_rate=learning_rate),
                      'Adadelta': Adadelta(learning_rate=learning_rate),
                      'Adagrad': Adagrad(learning_rate=learning_rate), 'Adamax': Adamax(learning_rate=learning_rate),
                      'Nadam': Nadam(learning_rate=learning_rate), 'Ftrl': Ftrl(learning_rate=learning_rate)}

        opt = optimizerD[optimizerL]
        nn = Sequential()
        nn.add(LSTM(units=256, return_sequences=True, input_shape=(n_inputs, 1)))
        nn.add(Dropout(0.2))
        for i in range(layers1):
            nn.add(LSTM(units=unit, return_sequences=True))
            if dropout > 0.5:
                nn.add(Dropout(dropout_rate))
        nn.add(LSTM(units=unit))
        if dropout > 0.5:
            nn.add(Dropout(dropout_rate))
        for i in range(layers2):
            nn.add(Dense(32, activation=activation))
        nn.add(Dense(1, activation='linear'))
        nn.compile(loss='mse', optimizer=opt, metrics=['mse', 'mae'])
        early_stopping = EarlyStopping(monitor="loss", patience=6, mode='auto', min_delta=0)
        history = nn.fit(Xp_train, yp_train, batch_size=batch_size,
                         epochs=nb_epoch, callbacks=[early_stopping])

        return nn

    nn_reg_ = KerasRegressor(build_fn=build_LSTM, verbose=0)

    n_iter_search = 10

    # optimize
    print('opt')
    search_reg = BayesSearchCV(nn_reg_, param_nn_bs, n_iter=n_iter_search, scoring='neg_mean_squared_error', cv=5)
    search_reg.fit(Xp_train, yp_train)
    print(search_reg.best_params_)
    print(f'max error: {search_reg.best_score_}')
    # test prediction
    y_pred = search_reg.predict(Xp_test)

    # error
    print('RMSE TEST transf:', np.sqrt(mean_squared_error(yp_test, y_pred)))
    print('MAE TEST transf:', mean_absolute_error(yp_test, y_pred))

    # inverse transformation
    y_pred = pd.DataFrame(y_pred[:, 0], test_or.index, columns=['Open'])
    y_pred['Open'] = y_pred['Open'] + df_tf.shift().values[-test_days:]
    y_pred = (y_pred ** 2)
    y_pred = np.exp(y_pred)

    # plot
    plt.figure(figsize=(10, 6))
    plt.grid()
    plt.plot(df_or, color='blue', label=f'actual prices of S&P 500')
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



if __name__ == '__main__':
    main()

'''('activation'LeakyReLU), ('batch_size', 25), ('dropout', 0.5469492804461135), ('dropout_rate', 0.015766818556683073)
('layers1', 1), ('layers2', 1), ('learning_rate', 0.09699768160862692), ('nb_epoch', 28), ('optimizerL', 'Adadelta'), ('unit', 53)])
max error: -0.0017390910866125163'''