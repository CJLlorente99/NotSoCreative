import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Backtest_days import backtest_func
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional
from keras import initializers
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, accuracy_score
from keras.optimizers import Adam, Adamax
from keras.callbacks import History
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
import numpy as np
from keras.utils.vis_utils import plot_model
from keras import initializers

import keras
import pydot as pyd
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
keras.utils.vis_utils.pydot = pyd

#Visualize Model

def visualize_model(model):
  return SVG(model_to_dot(model).create(prog='dot', format='svg'))
#create your model

# from keras.layers.normalization import BatchNormalization

# tf.random.set_seed(20)
# np.random.seed(10)
# 100 neurons ?

# Bidirectional

def build_model(n_inputs, n_features, n_outputs):
    opt = Adam(learning_rate=0.001)
    model = Sequential()
    model.add(LSTM(units=200, return_sequences=True, bias_initializer=initializers.Constant(0.01),
                   kernel_initializer='he_uniform', input_shape=(n_inputs, n_features)))
    model.add(Dropout(0.1))
    model.add(LSTM(units=200))
    model.add(Dropout(0.1))
    # model.add(Dense(32, kernel_initializer="uniform", activation='relu'))
    model.add(Dense(units=n_outputs, activation='linear'))
    model.compile(optimizer=opt, loss='mse')
    # history = model.fit
    return model


def build_Bimodel(n_inputs, n_features, n_outputs):
    opt = Adam(learning_rate=0.001)
    model = Sequential()
    model.add(Bidirectional(LSTM(units=200, return_sequences=True, bias_initializer=initializers.Constant(0.01),
                                 kernel_initializer='he_uniform', input_shape=(n_inputs, n_features))))
    model.add(Dropout(0.1))
    model.add(Bidirectional(LSTM(units=200)))
    model.add(Dropout(0.1))
    # model.add(Dense(32, kernel_initializer="uniform", activation='relu'))
    model.add(Dense(units=n_outputs, activation='linear'))
    model.compile(optimizer=opt, loss='mse')
    # history = model.fit
    return model


def build_model2(n_inputs, n_features, n_outputs):
    opt = Adam(learning_rate=0.001)
    model = Sequential()
    model.add(LSTM(units=250, return_sequences=True, bias_initializer=initializers.Constant(0.01),
                   kernel_initializer='he_uniform', input_shape=(n_inputs, n_features)))
    model.add(Dropout(0.1))
    model.add(LSTM(units=100))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(units=n_outputs, activation='linear'))
    model.compile(optimizer=opt, loss='mse')
    # history = model.fit
    return model


'''OrderedDict([('batch_size', 9), ('layers1', 0), ('layers2', 1), 
('learning_rate', 0.001), ('nb_epoch', 30), ('unit1', 167), ('unit2', 250)])'''

'''{'batch_size': 13, 'kernel_initializer': 'he_normal', 'learning_rate': 
0.0035887645732755992, 'nb_epoch': 49, 'unit1': 229, 'unit2': 140}'''


def build_modelHPO(n_inputs, n_features, n_outputs):
    opt = Adam(learning_rate=0.0035)
    model = Sequential()
    model.add(LSTM(units=250, return_sequences=True, bias_initializer=initializers.Constant(0.01),
                   kernel_initializer='he_uniform', input_shape=(n_inputs, n_features)))
    model.add(Dropout(0.1))
    model.add(LSTM(units=200))
    model.add(Dropout(0.1))
    # model.add(Dense(32, activation='relu'))
    model.add(Dense(units=n_outputs, activation='linear'))
    model.compile(optimizer=opt, loss='mse')
    # history = model.fit
    return model


'''OrderedDict([('batch_size', 2), ('kernel_initializer', 'he_uniform'), 
             ('learning_rate', 0.0001), ('nb_epoch', 30), ('unit1', 235), ('unit2', 185)])'''


# learn slower 0.666, 0.88

# learning rate should be even lower i think, so test it again with HPO
def build_modelHPO2(n_inputs, n_features, n_outputs):
    opt = Adam(learning_rate=0.0001)
    model = Sequential()
    model.add(LSTM(units=235, return_sequences=True, bias_initializer=initializers.Constant(0.01),
                   kernel_initializer='he_uniform', input_shape=(n_inputs, n_features)))
    model.add(Dropout(0.1))
    model.add(LSTM(units=185))
    model.add(Dropout(0.1))
    # model.add(Dense(32, activation='relu'))
    model.add(Dense(units=n_outputs, activation='linear'))
    model.compile(optimizer=opt, loss='mse')
    # history = model.fit
    return model


def build_model_app(n_inputs, n_features, n_outputs):
    opt = Adam(learning_rate=0.000224)
    model = Sequential()
    model.add(Bidirectional(LSTM(units=150, return_sequences=True, bias_initializer=initializers.Constant(0.01),
                                 kernel_initializer='he_uniform', input_shape=(n_inputs, n_features))))
    model.add(Dropout(0.1))
    model.add(Bidirectional(LSTM(units=100)))
    model.add(Dropout(0.1))
    # model.add(Dense(32, kernel_initializer="uniform", activation='relu'))
    model.add(Dense(units=n_outputs, activation='linear'))
    model.compile(optimizer=opt, loss='mse')
    # history = model.fit
    return model


'''def build_model(n_inputs, n_features, n_outputs):
    model = Sequential()
    model.add(LSTM(units=200, return_sequences=True, input_shape=(n_inputs, n_features)))
    model.add(Dropout(0.3))
    model.add(LSTM(units=200))
    model.add(Dropout(0.1))
    # model.add(Dense(32, kernel_initializer="uniform", activation='relu'))
    model.add(Dense(units=n_outputs, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    # history = model.fit
    return model'''


def fit_model(X_train, y_train, epochs, batch_size):
    # define neural network model
    n_inputs = X_train.shape[1]
    n_features = X_train.shape[2]
    n_outputs = y_train.shape[1]
    model = build_model_app(n_inputs, n_features, n_outputs)
    input_shape = (None, n_inputs, n_features)
    model.build(input_shape)

    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    # fit the model on the training dataset
    early_stopping = EarlyStopping(monitor="loss", patience=10, mode='auto', min_delta=0)
    model.fit(X_train, y_train, verbose=2, epochs=epochs, batch_size=batch_size, validation_split=0.15,
              callbacks=[early_stopping])
    model.summary()
    return model


def fit_ensemble(n_members, X_train, X_test, y_train, y_test, epochs, batch_size):
    # ensemble consists of several predictors
    ensemble = list()

    # to save arrays of prediction in this list
    y_pred = []

    for i in range(n_members):
        # define and fit the model on the training set
        model = fit_model(X_train, y_train, epochs, batch_size)

        # evaluate model on the test set
        yhat = model.predict(X_test, verbose=2)
        # print('yhat', yhat.shape, yhat)
        mae = mean_absolute_error(y_test, yhat)
        mse = mean_squared_error(y_test, yhat)
        print(f'MAE: {i + 1}, {mae}')
        print(f'MSE: {i + 1}, {mse}')
        print(f'MAPE:', mean_absolute_percentage_error(y_test, yhat))

        # store the model and prediction
        ensemble.append(model)
        y_pred.append(yhat)

    # print('y_pred_list', y_pred, len(y_pred))
    return ensemble, y_pred


def inverse_scaling(data, y_p, scaler):
    y_ensemble = []
    y_preds = np.zeros((y_p[0].shape[0], y_p[0].shape[1]))
    # select ensemble
    for j in range(len(y_p)):
        y_t = y_p[j]
        # select row of prediction of ensemble j
        for i in range(y_t.shape[0]):
            yhat = y_t[i, :]
            yhat = yhat.reshape(-1, 1)
            y_pred = np.tile(yhat, (1, data.shape[1]))
            y_pred = scaler.inverse_transform(y_pred)
            y_pred = y_pred[:, -1]
            y_preds[i, :] = y_pred
        y_ensemble.append(y_preds)
    return y_ensemble


def calculate_bounds(y_ensemble):
    # calculating mean of ensemble predictions for every row of prediction
    # each row consists of step_out predictions and there exists n_rows
    y_pji = np.zeros((y_ensemble[0].shape[0], y_ensemble[0].shape[1]))
    for i in range(y_ensemble[0].shape[0]):
        for j in range(len(y_ensemble)):
            # pick ensemble
            y_e = y_ensemble[j]
            # add each row of every ensemble together then ->
            y_pji[i, :] += y_e[i, :]

    # then: divide through numbers of predictors
    y_pji = y_pji / len(y_ensemble)
    return y_pji


def prepare_multidata(data_set_scaled, backcandles, pred_days, step_out):
    # preparing data
    # just to plot  I use test_d
    test_d = step_out + pred_days - 1
    X = list()
    y = list()
    # take last column as y_t because there is our return
    y_t = np.array(data_set_scaled[:, -1])

    # with this method the last column y_t will be shifted by once back and get the shape: (pred_days=n_row, step_out)
    # X will be prepared for LSTM in shape: (length(data) -  number of backcandles, number of backcandles, number of features)
    for i in range(backcandles, data_set_scaled.shape[0] + 1 - step_out):
        X.append(data_set_scaled[i - backcandles:i, :-1])
        y.append(y_t[i:i + step_out])

    # make array for getting shape
    X = np.array(X)
    y = np.array(y)

    X = X.reshape(y.shape[0], backcandles, data_set_scaled.shape[1] - 1)
    '''print('X_last', X.shape, X[-test_d:, :, :])
    print('X_last 1', X[-1:, :, :].shape, X[-1:, :, :])
    print('y_after', y.shape, y[-test_d:])'''

    # only last row because it contains the last step_out values (step_out is a number: first iteration: 10, last 1)
    splitlimit = X.shape[0] - pred_days
    X_train, X_test = X[:splitlimit], X[splitlimit:]
    y_train, y_test = y[:splitlimit], y[splitlimit:]

    return X_train, X_test, y_train, y_test


def main():
    print('go')
    # data prepaaring -> got no close and so on on day t
    data = pd.read_csv('featureSelectionDataset_CharliTIs.csv', sep=',', header=0, index_col=0, parse_dates=True,
                       decimal=".")
    data.dropna(inplace=True)
    # print(data, data.Open, data.Return_Open, data.Target)
    # data_true = yf.download(tickers='^GSPC', start='2016-01-04', end='2023-01-07')
    # print('true', data_true)

    # to choose interval where we want to test
    data = data.iloc[2200:-2200:, :]
    data = data.drop(['Target'], axis=1)
    print('len data', len(data), data)
    data_or = data.copy()

    cut = 0
    if cut == 1:
        data = data.iloc[:, :-5]
        data['Open'] = data_or.Open

    # transformation
    # to decide if we want to predict raw prices or returns -> i think returns might work better cause of non-stationarity
    transformation = 1

    df_log = (np.log(data.Open))
    # data['Open_log'] = df_log
    df_log_diff = df_log - df_log.shift()
    data['Target'] = df_log_diff
    data.dropna(inplace=True)
    print(data)
    print('header', data.columns.values.tolist())

    # choose columns: all but target variable (its last column)
    liste = list(range(0, data.shape[1]))

    # what I choose at my test set: I make n_row predictions: In each prediction I predict step_out days at once
    n_row = 4  # 7

    # how many days i want to predict at once -> I do this n_row times
    step_out = 3

    # how many last days I include in my prediction
    backcandles = 8  # 10

    # days to predict: these are the actually different days I predict
    # because the algorithm works as follows:(y_t = true value on day_t, y_p_t = predicted value on day_t)

    # Example: n_rows=2 step_out=3
    # 1. iteration:
    # at day_t so for X_t I predict:
    # 1: inp: X_t, y_t ---predict-----> y_p_t+1, y_p_t+2, y_p_t+3
    # 2: inp: X_t+1, y_t+1 --predict--> y_p_t+2, y_p_t+3, y_p_t+4
    # this are step_out + n_row - 1 different days: 3+2-1=4 -> day_t+1 to day_t+4
    #

    # days to predict: these are the actually different days I predict: example above
    # works for step_out= 3 dont know for other values but should be
    test_days = step_out + n_row - 1

    # for plotting reasons
    test_or = data_or.Open.iloc[-(test_days):]
    # print('test_pr', test_or) right

    # just to check for mistakes -> easier to debug with this in prepare_multidata than with data_set_scaled
    # data_arr = np.asarray(data)

    # scaling
    scaling = 2
    if scaling == 2:
        scaler_r = RobustScaler()  # MinMaxScaler() #MinMaxScaler(feature_range=(-1, 1)) #StandardScaler()
        data_set_scaled = scaler_r.fit_transform(data)
    scaler_m = MinMaxScaler()  # MinMaxScaler(feature_range=(-1, 1))
    data_set_scaled = scaler_m.fit_transform(data_set_scaled)
    # print('data_set_scaled', data_set_scaled)

    # prepare data for lstm
    print('data preparation')
    X_train, X_test, y_train, y_test = prepare_multidata(data_set_scaled, backcandles, n_row, step_out)

    # just to look at values -> to check for mistakes
    '''print('data_set_scaled', data_set_scaled[-15:, -1])
    print('y_test', y_test.shape, y_test)
    print('X_test', X_test.shape, X_test)
    print('iter', i+1)'''

    # to see shape and values
    # print('y_test', y_test.shape, y_test)
    # print('X_test', X_test.shape, X_test)

    # train and predict
    # n_members -> how many predictors we wanted to use
    # ep = 15. batch=2
    n_members = 1
    epochs = 20  # 30 # 36
    batch_size = 2
    print('train')
    ensemble, y_pred_scale, = fit_ensemble(n_members, X_train, X_test, y_train, y_test, epochs,
                                           batch_size)
    # print('scaled y_pred', y_pred_scale)
    # print('scaled y_test', y_test)

    # inverse scaling
    if scaling == 2:
        y_pred_scale = inverse_scaling(data, y_pred_scale, scaler_m)
        y_test_m = np.zeros((y_test.shape[0], y_test.shape[1]))
        for i in range(y_test.shape[0]):
            yhat = y_test[i, :]
            yhat = yhat.reshape(-1, 1)
            y_t = np.tile(yhat, (1, data.shape[1]))
            y_t = scaler_m.inverse_transform(y_t)
            y_t = y_t[:, -1]
            y_test_m[i, :] = y_t
        y_test = y_test_m

    y_pred = inverse_scaling(data, y_pred_scale, scaler_r)

    y_tests = np.zeros((y_test.shape[0], y_test.shape[1]))
    for i in range(y_test.shape[0]):
        yhat = y_test[i, :]
        yhat = yhat.reshape(-1, 1)
        y_t = np.tile(yhat, (1, data.shape[1]))
        y_t = scaler_r.inverse_transform(y_t)
        y_t = y_t[:, -1]
        y_tests[i, :] = y_t
    # print('after inverse scaling y_test', y_test)

    # Calculating mean of ensembles
    y_mean = calculate_bounds(y_pred)

    # just to look at the values
    '''print('return mean pred', y_mean.shape, y_mean)
    print('test back tr return', y_tests.shape, y_tests)
    print('Target', data.Target.iloc[-test_days:])
    print('real', test_or)'''

    # inverse transformation: not possible to use for real script -> because of index, in real script we have no test_or
    # so for real script it might be better to use not a index or so
    if transformation == 1:
        y_mean_list = []
        y_test_list = []
        for k in range(y_mean.shape[0]):
            df_shift = df_log.shift().values[-test_days:]

            y_mean_pd = pd.DataFrame(y_mean[k, :], test_or.iloc[k:k + 1 * y_mean.shape[1]].index, columns=['Open'])
            y_mean_pd['Open'] = y_mean_pd['Open'] + df_shift[k:k + 1 * y_mean.shape[1]]
            # y_mean = (y_mean ** 2)
            y_mean_pd = np.exp(y_mean_pd)
            # print('pred', y_mean_pd)
            y_mean_pd = np.asarray(y_mean_pd)
            y_mean_list.append(y_mean_pd)

            y_test_pd = pd.DataFrame(y_tests[k, :], test_or.iloc[k:k + 1 * y_tests.shape[1]].index, columns=['Open'])
            y_test_pd['Open'] = y_test_pd['Open'] + df_shift[k:k + 1 * y_test.shape[1]]
            # y_mean = (y_mean ** 2)
            y_test_pd = np.exp(y_test_pd)
            # print('test', y_test_pd)
            y_test_pd = np.asarray(y_test_pd)
            y_test_list.append(y_test_pd)

    # error metrics
    print('MSE:', mean_squared_error(y_test_pd, y_mean_pd))
    print('MAPE:', mean_absolute_error(y_test_pd, y_mean_pd))
    print('MAE:', mean_absolute_percentage_error(y_test_pd, y_mean_pd))

    # one more value: I want to compare the current value_t to my predicted value_t+n and make decision on day_t
    # if I would just take data.Open[-(test_days):] the first value of this would be the first value I want to predict
    data_open = data_or.Open[-(test_days + 1):]

    # build decision rule: if open_p_t+3 > open_t -> buy on open_t
    decision = []
    # with the rule above you dont take decisions anymore  if open_t+3 is the last day:
    # therefore for the last 3 days, i compare open_t < open_p_t+2 and then open_t+1 < open_p_t+2
    # _p means predicted
    decision2 = []
    decision1 = []

    for q in range(len(y_mean_list)):
        y_pr = y_mean_list[q]
        # print('open',data_open[q])
        # print('pred', y_pr[2])

        if data_open[q] < y_pr[2]:
            decision2.append(+1)

        else:
            decision2.append(-1)

        if data_open[q] < y_pr[1]:
            decision1.append(+1)
        else:
            decision1.append(-1)

        if q == len(y_mean_list) - 1:

            if data_open[q + 1] < y_pr[2]:
                decision2.append(+1)
                decision1.append(+1)
            else:
                decision2.append(-1)
                decision1.append(-1)

            if data_open[q + 2] < y_pr[2]:
                decision2.append(+1)
                decision1.append(1)
            else:
                decision2.append(-1)
                decision1.append(-1)

    # build decision rule for real values
    real_decision = []
    # for comparing accuracy
    real_dec2 = []
    real_dec1 = []

    # here I make everyday prediction, just to compare
    for j in range(len(y_test_list)):
        y_true = y_test_list[j]
        # print('open',data_open[j])
        # print('true+1', y_true[0])

        # everyday prediction as benchmark comparison
        if data_open[j] < data_open[j + 1]:
            real_decision.append(+1)
        else:
            real_decision.append(-1)

        # just to compare predictions accuracy
        if data_open[j] < data_open[j + 3]:
            real_dec2.append(+1)
        else:
            real_dec2.append(-1)

        if data_open[j] < data_open[j + 2]:
            real_dec1.append(+1)
        else:
            real_dec1.append(-1)

        if j == len(y_test_list) - 1:

            if data_open[j + 1] < data_open[j + 3]:
                real_dec2.append(+1)
                real_dec1.append(+1)
            else:
                real_dec2.append(-1)
                real_dec1.append(-1)

            if data_open[j + 2] < data_open[j + 3]:
                real_dec2.append(+1)
                real_dec1.append(+1)
            else:
                real_dec2.append(-1)
                real_dec1.append(-1)

    print('----------')
    print('----------')
    print('real_dec2', real_dec2)
    print('dec2', decision2)
    print('----------')
    print('----------')
    print('ACC 2', accuracy_score(real_dec2, decision2))
    print('----------')
    print('----------')
    print('real_dec1', real_dec1)
    print('dec1', decision1)
    print('----------')
    print('----------')
    print('ACC 1', accuracy_score(real_dec1, decision1))
    print('----------')
    print('----------')

    # backtest it: because I make no loop: my predictions stops if y_pred[2] in decision function is the last value
    # therefore I  do nothing at the last 3 days -> have to fix this
    # before we made here also a mistake, mabye thats why our stratey wasnt that good?

    # PLS CHECK:
    # we predict the first value for day_t+1 and make the decision for day_t -> so the first value we have to include here is day_t
    # before the first value was day_t+1, but mabye iam wrong?

    # gain_pct, mpv, gain_bench = backtest_func(df=data_or.iloc[-test_days-1:], decision=decision)
    # print('this was for our prediction: Decision 1')
    # print(data_or.iloc[-test_days-1:])
    # backtest it
    print('this is for our prediction: Decision 2')
    print('Decision 2', decision2)
    gain_pct_prob, mpv_prob, gain_bench = backtest_func(df=data_or.iloc[-test_days - 1:], decision=decision2)
    print('----------')
    print('----------')
    print('this is for the real value 2')
    print('REAL', real_decision)
    gain_pct_best, mpv_best, gain_bench = backtest_func(df=data_or.iloc[-test_days - 1:], decision=real_dec2)
    print('----------')
    print('----------')
    print('this is for our prediction: Decision 1')
    print('Decision 1', decision1)
    gain_pct_prob, mpv_prob, gain_bench = backtest_func(df=data_or.iloc[-test_days - 1:], decision=decision1)
    print('----------')
    print('----------')
    print('this is for the real value 1')
    print('REAL', real_decision)
    gain_pct_best, mpv_best, gain_bench = backtest_func(df=data_or.iloc[-test_days - 1:], decision=real_dec1)
    print('----------')
    print('----------')


if __name__ == '__main__':
    main()