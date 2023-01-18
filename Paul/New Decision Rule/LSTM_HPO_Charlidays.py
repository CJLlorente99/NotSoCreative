import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Backtest_days import backtest_func
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import TimeDistributed
from skopt.space import Real, Categorical, Integer
from tensorflow.keras.optimizers import Adam, SGD, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from keras.callbacks import History
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
import numpy as np
from scipy.stats import uniform, randint
from sklearn.metrics import accuracy_score
from keras.layers import LeakyReLU
from keras.wrappers.scikit_learn import KerasRegressor
from skopt import BayesSearchCV
from scipy.stats import randint as sp_randint
from sklearn.utils.fixes import loguniform
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from keras import initializers
# tf.random.set_seed(20)
np.random.seed(10)


# (Bidirectional(LSTM(20,
'''def build_model(n_inputs, n_features):
    model = Sequential()
    model.add(LSTM(units=150, return_sequences=True, input_shape=(n_inputs, n_features)))
    model.add(Dropout(0.1))
    model.add(LSTM(units=150))
    model.add(Dropout(0.2))
    # model.add(Dense(32, kernel_initializer="uniform", activation='relu'))
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    # history = model.fit
    return model'''

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
    
    # reshape
    X = X.reshape(y.shape[0], backcandles, data_set_scaled.shape[1] - 1)

    # only last row because it contains the last step_out values (step_out is a number: first iteration: 10, last 1)
    splitlimit = X.shape[0] - pred_days
    X_train, X_test = X[:splitlimit], X[splitlimit:]
    y_train, y_test = y[:splitlimit], y[splitlimit:]

    return X_train, X_test, y_train, y_test

def inverse_scaling(data, y_t, scaler):
    y_preds = np.zeros((y_t.shape[0], y_t.shape[1]))
    # select row of prediction of ensemble j
    for i in range(y_t.shape[0]):
        yhat = y_t[i, :]
        yhat = yhat.reshape(-1, 1)
        y_pred = np.tile(yhat, (1, data.shape[1]))
        y_pred = scaler.inverse_transform(y_pred)
        y_pred = y_pred[:, -1]
        y_preds[i, :] = y_pred
    return y_preds


def main():
    data = pd.read_csv('featureSelectionDataset_CharliTIs.csv', sep=',', header=0, index_col=0, parse_dates=True, decimal=".")
    data.dropna(inplace=True)

    # to choose interval 
    data = data.iloc[-1200:, :]
    data = data.drop(['Target'], axis=1)
    # make return as target
    df_log = (np.log(data.Open))
    data['Open_log'] = df_log
    df_log_diff = df_log - df_log.shift()
    data['Target'] = df_log_diff
    data.dropna(inplace=True)
    print('real', data)
    data_or = data.copy()


    # choose columns: all but target variable (its last column)
    liste = list(range(0, data.shape[1]))

    # what I choose at my test set: I make n_row predictions: In each prediction I predict step_out days at once
    n_row = 7

    # how many days i want to predict at once -> I do this n_row times
    step_out = 3
    
    # how many last days I include in my prediction
    backcandles = 15 #10

    # days to predict: these are the actually different days I predict: example above
    # works for step_out= 3 dont know for other values but should be 
    test_days = step_out + n_row - 1

    # for plotting reasons
    test_or = data_or.Open.iloc[-(test_days):]
    
    # just to check for mistakes -> easier to debug with this in prepare_multidata than with data_set_scaled
    #data_arr = np.asarray(data)
    
    #scaling
    scaler_r = RobustScaler()#MinMaxScaler() #MinMaxScaler(feature_range=(-1, 1)) #StandardScaler()
    data_set_scaled = scaler_r.fit_transform(data)
    scaler_m = MinMaxScaler() #MinMaxScaler(feature_range=(-1, 1))
    data_set_scaled = scaler_m.fit_transform(data_set_scaled)
    #print('data_set_scaled', data_set_scaled)
    
    # prepare data for lstm
    print('data preparation')
    X_train, X_test, y_train, y_test = prepare_multidata(data_set_scaled, backcandles, n_row, step_out)
    
    n_inputs = X_train.shape[1]
    n_features = X_train.shape[2]
    n_outputs = y_train.shape[1]

    param_nn_rand = {
        'unit1': randint(125, 300),
        'unit2': randint(125, 300),
        #'activation': ['relu', 'sigmoid', 'tanh', 'elu'],
        'learning_rate': loguniform(0.0001, 0.04),
        #'layers2': [0, 1],
        #'dropout': uniform(0, 1),
        #'dropout_rate': uniform(0, 0.3),
        'nb_epoch': randint(15, 50),
        'batch_size': randint(5, 15),
        'kernel_initializer': ['he_normal', 'glorot_uniform']}
        #'normalization': uniform(0, 1),
        #'optimizerL': ['Adam','Adagrad', 'Adamax', 'RMSprop', 'Adadelta']}

    
    param_nn_bs = {
        'unit1': Integer(125, 300),
        'unit2': Integer(125, 300),
        'learning_rate': Real(0.0001, 0.01),
        #'layers1': Integer(0, 1),
        'nb_epoch': Integer(5, 30),
        'batch_size': Integer(2, 16), 
        'kernel_initializer': Categorical(['he_uniform', 'glorot_uniform'])}
        #'optimizerL': ['Adam', 'Adagrad', 'Adamax', 'Adadelta']}
         #'layers1': Integer(0, 1),
        #'activation': Categorical(['relu', 'sigmoid', 'tanh', 'elu']),
        #        'dropout': Real(0, 1),
        #'dropout_rate': Real(0, 0.3),
    
    print('build model')
    def build_LSTM(learning_rate=0.01, unit1=100, unit2=100, activation='relu', layers1=1, layers2=1, dropout=0.3, dropout_rate=0.23,
                   nb_epoch=20, batch_size=20, optimizerL='Adam', kernel_initializer='he_uniform'):
        
        optimizerD = {'Adam': Adam(learning_rate=learning_rate), 'SGD': SGD(learning_rate=learning_rate),
                      'Adadelta': Adadelta(learning_rate=learning_rate),
                      'Adagrad': Adagrad(learning_rate=learning_rate), 'Adamax': Adamax(learning_rate=learning_rate),
                      'Nadam': Nadam(learning_rate=learning_rate), 'Ftrl': Ftrl(learning_rate=learning_rate)}
        #      'RMSprop': RMSprop(learning_rate=learning_rate),
      
        opt = optimizerD[optimizerL]
        nn = Sequential()
        nn.add((LSTM(units=unit1, return_sequences=True, 
                     bias_initializer=initializers.Constant(0.01), kernel_initializer=kernel_initializer, 
                     input_shape=(n_inputs, n_features))))
        nn.add(Dropout(0.1))
        nn.add((LSTM(units=unit2)))
        nn.add(Dropout(0.1))
        #for i in range(layers2):
            #nn.add(Dense(32, activation='relu'))
        nn.add(Dense(n_outputs, activation='linear'))
        nn.compile(loss='mse', optimizer=opt, metrics=['mse'])
        early_stopping = EarlyStopping(monitor='loss', patience=6, mode='auto', min_delta=0)
        history = nn.fit(X_train, y_train, batch_size=batch_size,
                         epochs=nb_epoch, callbacks=[early_stopping], validation_split=0.15)
        return nn


    nn_reg = KerasRegressor(build_fn=build_LSTM, verbose=0)

    n_iter_search = 40

    # optimize
    print('opt')
    search_reg = RandomizedSearchCV(nn_reg, param_nn_rand, n_iter=n_iter_search, scoring='neg_mean_squared_error')
    #search_reg = BayesSearchCV(nn_reg, param_nn_bs, n_iter=n_iter_search, scoring='neg_mean_squared_error', cv=5)
    search_reg.fit(X_train, y_train)
    print(search_reg.best_params_)
    print(f'best score: {search_reg.best_score_}')
    
    # test prediction
    y_pred = search_reg.predict(X_test)
    
    # inverse scaling
    y_mean = inverse_scaling(data, y_pred, scaler_m)
    y_mean = inverse_scaling(data, y_mean, scaler_r)
    
    y_test = inverse_scaling(data, y_test, scaler_m)
    y_test = inverse_scaling(data, y_test, scaler_r)
    
    # inverse transformation: back from return to prices
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

        y_test_pd = pd.DataFrame(y_test[k, :], test_or.iloc[k:k+1 * y_test.shape[1]].index, columns=['Open'])
        y_test_pd['Open'] = y_test_pd['Open'] + df_shift[k:k+1 * y_test.shape[1]]
        #y_mean = (y_mean ** 2)
        y_test_pd = np.exp(y_test_pd)
        print('test', y_test_pd)
        y_test_pd = np.asarray(y_test_pd)
        y_test_list.append(y_test_pd)
    
    
        # one more value: I want to compare the current value_t to my predicted value_t+n and make decision on day_t
    # if I would just take data.Open[-(test_days):] the first value of this would be the first value I want to predict 
    data_open = data.Open[-(test_days+1):]
    # one more value: I want to compare the current value_t to my predicted value_t+n and make decision on day_t
     # build decision rule: if open_p_t+3 > open_t -> buy on open_t
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
    gain_pct_prob, mpv_prob, gain_bench = backtest_func(df=data_or.iloc[-test_days-1:], decision=decision2)
    print('----------')
    print('----------')
    print('this is for the real value 2')
    print('REAL', real_decision)
    gain_pct_best, mpv_best, gain_bench = backtest_func(df=data_or.iloc[-test_days-1:], decision=real_dec2)          
    print('----------')
    print('----------')    
    print('this is for our prediction: Decision 1')
    print('Decision 1', decision1)                
    gain_pct_prob, mpv_prob, gain_bench = backtest_func(df=data_or.iloc[-test_days-1:], decision=decision1)
    print('----------')
    print('----------')
    print('this is for the real value 1')
    print('REAL', real_decision)
    gain_pct_best, mpv_best, gain_bench = backtest_func(df=data_or.iloc[-test_days-1:], decision=real_dec1)
    print('----------')
    print('----------')
    
    
    

if __name__ == '__main__':
    main()
