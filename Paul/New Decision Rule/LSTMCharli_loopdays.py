import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from Backtest_days import backtest_func
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional
from keras import initializers
import tensorflow as tf
import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, accuracy_score
from tensorflow.keras.optimizers import Adam, Adamax
from keras.callbacks import History
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
from keras import initializers


# from keras.layers.normalization import BatchNormalization

# tf.random.set_seed(20)
np.random.seed(10)
# 100 neurons ?
def build_model(n_inputs, n_features, n_outputs):
    opt = Adam(learning_rate=0.001)
    model = Sequential()
    model.add(LSTM(units=200, return_sequences=True,  bias_initializer=initializers.Constant(0.01), 
                   kernel_initializer='he_uniform', input_shape=(n_inputs, n_features)))
    model.add(Dropout(0.1))
    model.add(LSTM(units=200))
    model.add(Dropout(0.1))
    # model.add(Dense(32, kernel_initializer="uniform", activation='relu'))
    model.add(Dense(units=n_outputs, activation='linear'))
    model.compile(optimizer=opt, loss='mean_squared_error')
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
    model = build_model(n_inputs, n_features, n_outputs)
    
    # fit the model on the training dataset
    early_stopping = EarlyStopping(monitor="loss", patience=10, mode='auto', min_delta=0)
    model.fit(X_train, y_train, verbose=2, epochs=epochs, batch_size=batch_size, validation_split=0.15, callbacks=[early_stopping])
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
        #print('yhat', yhat.shape, yhat)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        print(f'MAPE:', mean_absolute_percentage_error(y_test, yhat))
        
        # store the model and prediction
        ensemble.append(model)
        y_pred.append(yhat)
        
    #print('y_pred_list', y_pred, len(y_pred))
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
       
    data_csv = pd.read_csv('featureSelectionDataset_CharliTIs.csv', sep=',', header=0, index_col=0, parse_dates=True, decimal=".")
    data_csv.dropna(inplace=True)
    data_csv = data_csv.iloc[-3000:, :]

    
    
    '''# what I choose at my test set: I make n_row predictions: In each prediction I predict step_out days at once
    n_row = 7

    # how many days i want to predict at once -> I do this n_row times
    step_out = 3

    # days to predict: these are the actually different days I predict
    test_days = step_out + n_row - 1'''

        
    # data_list is the time interval: where I cut the data
    data_list = [1500, 1200, 700, 500, 1300, 800]
    # number of backcandle
    backc = [15, 64]
    
    # look at these values for optimization
    acc_list = np.empty((len(data_list), len(backc)))
    gain_list = np.empty((len(data_list), len(backc)))
    mpv_list = np.empty((len(data_list), len(backc)))
    bench_list = np.empty((len(data_list), len(backc)))

    gain_list_prob = np.empty((len(data_list), len(backc)))
    mpv_list_prob = np.empty((len(data_list), len(backc)))

    gain_list_best = np.empty((len(data_list), len(backc)))
    mpv_list_best = np.empty((len(data_list), len(backc)))
    
    for c, i in enumerate(data_list):
        if c == 0:
            # first 1500
            data = data_csv.copy()
            data = data.iloc[:i, :]
        elif c == 1:
            # 800 in the middle
            data = data_csv.copy()
            data = data.iloc[i:-i, :]
        elif c == 2:
            # last 800
            data = data_csv.copy()
            data = data.iloc[-i:, :]
        elif c == 3:
            # first 800
            data = data_csv.copy()
            data = data.iloc[:i, :]
        elif c == 4:
            # smth in middle
            data = data_csv.copy()
            data = data.iloc[i:-i, :]
        elif c == 4:
            # 800 in the middle
            data = data_csv.copy()
            data = data.iloc[i:-i, :]
        
        # has to be done here, idk why
        data = data.drop(['Target'], axis=1)
        df_log = (np.log(data.Open))
        data['Open_log'] = df_log
        df_log_diff = df_log - df_log.shift()
        data['Target'] = df_log_diff
        data.dropna(inplace=True)
        
        
        # what I choose at my test set: I make n_row predictions: In each prediction I predict step_out days at once
        n_row = 8

        # how many days i want to predict at once -> I do this n_row times
        step_out = 3

        # days to predict: these are the actually different days I predict
        test_days = step_out + n_row - 1
        
        
        # choose columns: all but target variable (its last column)
        liste = list(range(0, data.shape[1]))

        # for plotting reasons
        test_or = data.Open.iloc[-(test_days):]
        data_or = data.copy()

        # just to check for mistakes -> easier to debug with this in prepare_multidata than with data_set_scaled
        #data_arr = np.asarray(data)
        
        #scaling
        scaling = 2
        if scaling == 2:
            scaler_r = RobustScaler()#MinMaxScaler() #MinMaxScaler(feature_range=(-1, 1)) #StandardScaler()
            data_set_scaled = scaler_r.fit_transform(data)
        scaler_m = MinMaxScaler()
        data_set_scaled = scaler_m.fit_transform(data_set_scaled)

        for b, b_c in enumerate(backc):
            
            # prepare data for lstm
            print('data preparation')
            X_train, X_test, y_train, y_test = prepare_multidata(data_set_scaled, b_c, n_row, step_out)

            #print('data_open', data.Open[-(test_days+1):])
            # just to look at values -> to check for mistakes
            '''print('data_set_scaled', data_set_scaled[-15:, -1])
            print('y_test', y_test.shape, y_test)
            print('X_test', X_test.shape, X_test)
            print('iter', i+1)'''

            # to see shape and values
            #print('y_test', y_test.shape, y_test)
            #print('X_test', X_test.shape, X_test)

            # train and predict
            # n_members -> how many predictors we wanted to use
            n_members = 1
            epochs = 20 # 36
            batch_size = 8
            print('train')
            ensemble, y_pred_scale, = fit_ensemble(n_members, X_train, X_test, y_train, y_test, epochs,
                                                            batch_size)
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
     


            # Calculating mean of ensembles 
            y_mean = calculate_bounds(y_pred)

            # just to look at the values
            '''print('return mean pred', y_mean.shape, y_mean)
            print('test back tr return', y_tests.shape, y_tests)
            print('Target', data.Target.iloc[-test_days:])
            print('real', test_or)'''

            # inverse transformation: not possible to use for real script -> because of index, in real script we have no test_or
            # so for real script it might be better to use not a index or so
            y_mean_list = []
            y_test_list = []
            for k in range(y_mean.shape[0]):
                df_shift = df_log.shift().values[-test_days:]

                y_mean_pd = pd.DataFrame(y_mean[k, :], test_or.iloc[k:k+1 * y_mean.shape[1]].index, columns=['Open'])
                y_mean_pd['Open'] = y_mean_pd['Open'] + df_shift[k:k+1 * y_mean.shape[1]]
                #y_mean = (y_mean ** 2)
                y_mean_pd = np.exp(y_mean_pd)
                print('pred', y_mean_pd)
                y_mean_pd = np.asarray(y_mean_pd)
                y_mean_list.append(y_mean_pd)

                y_test_pd = pd.DataFrame(y_tests[k, :], test_or.iloc[k:k+1 * y_tests.shape[1]].index, columns=['Open'])
                y_test_pd['Open'] = y_test_pd['Open'] + df_shift[k:k+1 * y_test.shape[1]]
                #y_mean = (y_mean ** 2)
                y_test_pd = np.exp(y_test_pd)
                print('test', y_test_pd)
                y_test_pd = np.asarray(y_test_pd)
                y_test_list.append(y_test_pd)


            #print('y_test trans', y_test)
            '''# plot of every predicted priced for each day
            plt.figure(figsize=(16, 8))
            plt.plot(test_or, color='black', label='Test', marker='o')
            #plt.plot(lower, color='green', label='lower')
            #plt.plot(upper, color='green', label='upper')
            plt.plot(y_mean, color='blue', label='Pred', marker='o')
            #plt.axhline(y=0, color='r', linestyle='-', label="Zero")
            plt.legend()
            plt.title(f'price in iteration: {i+1}')
            plt.show()

            # error metrics
            print('MSE:', mean_squared_error(y_test, y_mean))
            print('MAPE:', mean_absolute_error(y_test, y_mean))
            print('MAE:', mean_absolute_percentage_error(y_test, y_mean))

            y_predictions.append(np.asarray(y_mean))
            y_mean = np.asarray(y_mean).flatten()'''


            # build decision rule: if open_t+2 > open_t -> buy on open_t


            # one more value: I want to compare the current value_t to my predicted value_t+n and make decision on day_t
            # if I would just take data.Open[-(test_days):] the first value of this would be the first value I want to predict 
             # build decision rule: if open_p_t+3 > open_t -> buy on open_t
            data_open = data_or.Open[-(test_days+1):]
            decision = []
            # with the rule above you dont take decisions anymore  if open_t+3 is the last day:
            # therefore for the last 3 days, i compare open_t < open_p_t+2 and then open_t+1 < open_p_t+2
            # _p means predicted
            decision2 = []
            decision1 = []

            for q in range(len(y_mean_list)):
                y_pr = y_mean_list[q]
                #print('open',data_open[q])
                #print('pred', y_pr[2])

                if data_open[q] < y_pr[2]:
                    decision2.append(+1)

                else:
                    decision2.append(-1)

                if data_open[q] < y_pr[1]:
                    decision1.append(+1)
                else:
                    decision1.append(-1)

                if q == len(y_mean_list) - 1:

                    if data_open[q+1] < y_pr[2]:
                        decision2.append(+1)
                        decision1.append(+1)
                    else:
                        decision2.append(-1)
                        decision1.append(-1)

                    if data_open[q+2] < y_pr[2]:
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
                #print('open',data_open[j])
                #print('true+1', y_true[0])

                # everyday prediction as benchmark comparison
                if data_open[j] < data_open[j+1]:
                    real_decision.append(+1)
                else:
                    real_decision.append(-1)

                # just to compare predictions accuracy  
                if data_open[j] < data_open[j+3]:
                    real_dec2.append(+1)
                else:
                    real_dec2.append(-1)

                if data_open[j] < data_open[j+2]:
                    real_dec1.append(+1)
                else: 
                    real_dec1.append(-1)



                if j == len(y_test_list) - 1: 

                    if data_open[j+1] < data_open[j+3]:
                        real_dec2.append(+1)
                        real_dec1.append(+1)
                    else:
                        real_dec2.append(-1)
                        real_dec1.append(-1)

                    if data_open[j+2] < data_open[j+3]:
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

            #gain_pct, mpv, gain_bench = backtest_func(df=data_or.iloc[-test_days-1:], decision=decision)
            #print('this was for our prediction: Decision 1')
            #print(data_or.iloc[-test_days-1:])
            # backtest it
            print('this is for our prediction: Decision 2')
            print('Decision 2', decision2)
            gain_pct_prob2, mpv_prob2, gain_bench = backtest_func(df=data_or.iloc[-test_days-1:], decision=decision2)
            print('----------')
            print('----------')
            print('this is for the real value 2')
            print('REAL', real_decision)
            gain_pct_best2, mpv_best2, gain_bench = backtest_func(df=data_or.iloc[-test_days-1:], decision=real_dec2)          
            print('----------')
            print('----------')    
            print('this is for our prediction: Decision 1')
            print('Decision 1', decision1)                
            gain_pct_prob1, mpv_prob1, gain_bench = backtest_func(df=data_or.iloc[-test_days-1:], decision=decision1)
            print('----------')
            print('----------')
            print('this is for the real value 1')
            print('REAL', real_decision)
            gain_pct_best1, mpv_best1, gain_bench = backtest_func(df=data_or.iloc[-test_days-1:], decision=real_dec1)
            print('----------')
            print('----------')
            print('----------')
            print('----------')
            print(f'test_list{c}, {i} ;backcandles: {b_c}')
            print('----------')
            print('----------')
            
            
            # saves the gain and so on for different time intervals and backcandles:
            # shape: (time intervals, backcandles)
            # so to compare the backcandles: watch columns and see what performs the best
            #gain_list[c, b] = gain_pct
            #mpv_list[c, b] = mpv
            acc_list[c, b] = acc
            bench_list[c, b] = gain_bench
            
            gain_list_prob2[c, b] = gain_pct_prob2
            mpv_list_prob2[c, b] = mpv_prob2
            
            mpv_list_best2[c, b] = mpv_best2
            gain_list_best2[c, b] = gain_pct_best2
            
            gain_list_prob1[c, b] = gain_pct_prob1
            mpv_list_prob1[c, b] = mpv_prob1
            
            mpv_list_best1[c, b] = mpv_best1
            gain_list_best1[c, b] = gain_pct_best1

    print(f'acc: {acc_list} %')
    print('----------')
    print('----------')
    #print(f'Gain rule: {gain_list} %')
    print(f'Gain rule prob2: {gain_list_prob2} %')
    print(f'Gain best 2: {gain_list_best2} %')
    print(f'Gain rule prob1: {gain_list_prob1} %')
    print(f'Gain best 1: {gain_list_best1} %')
    print(f'Gain bench: {bench_list} %')
    print('----------')
    print('----------')
    print('MPV rule prob2', mpv_list_prob2)
    print('MPV best2', mpv_list_best2)
    print('MPV rule prob1', mpv_list_prob1)
    print('MPV best1', mpv_list_best1)
    #print(f'Gain sum of rule: {np.sum(gain_list, axis=0)}')
    print(f'Gain sum of prob2: {np.sum(gain_list_prob2, axis=0)}')
    print(f'Gain sum of prob1: {np.sum(gain_list_prob1, axis=0)}')
    print('----------')
    print('----------')
    #print(f'MPV mean of rule: {np.mean(mpv_list, axis=0)}')
    print(f'MPV mean of prob2: {np.mean(mpv_list_prob2, axis=0)}')
    print(f'MPV mean of prob1: {np.mean(mpv_list_prob1, axis=0)}')
    
    

if __name__ == '__main__':
    main()
