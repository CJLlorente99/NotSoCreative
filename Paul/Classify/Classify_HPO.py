import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from Backtest_open_close import backtest_func
from sklearn.preprocessing import StandardScaler
from skopt.space import Real, Categorical, Integer
import numpy as np
from scipy.stats import uniform, randint
from sklearn.metrics import accuracy_score
from skopt import BayesSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
from sklearn.utils.fixes import loguniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# tf.random.set_seed(20)
np.random.seed(10)

def data_shift(X, window, inp):
    X_plc = X
    for i in range(window):
        X_shift_bw = X.shift(periods=(i + 1), fill_value=0)
        inp_bw = [x + f'_-{i + 1}' for x in inp]
        X_shift_bw.columns = inp_bw
        X_plc = pd.concat([X_plc, X_shift_bw], axis=1)

    return X_plc

def main():
    data = pd.read_csv("../featureSelectionDataset_Paul_Class_shift_forBackt_close_open.csv", sep=',', header=0,
                           index_col=0, parse_dates=True,
                           decimal=".")
    data_or = data.copy()
    # because open and close not important feat but i need it for backtesting
    data = data.drop(['Open', 'Close'], axis=1)
    y_target = np.asarray([1 if data.Target[i] > 0 else 0 for i in range(len(data))]).reshape(-1, 1)
    X = data.drop(['Target'], axis=1)
    # drop extra features
    X = X.drop(X.columns[10:], axis=1)

    # include t-window data points as additional features
    inp = X.columns.values.tolist()
    # window mabye 0 to 1,2,3
    window = 3
    X = data_shift(X, window, inp)
    # print(X)
    X = np.asarray(X)

    # scaling: try different scaling or no scaling
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)

    splitlimit = ((X.shape[0]) - 10)
    X_train, X_test = X[:splitlimit], X[splitlimit:]
    y_train, y_test = y_target[:splitlimit], y_target[splitlimit:]

    # parameter to optimize:
    rf_params_rand = {"bootstrap": [True, False],
                      "max_depth": randint(4, 14),
                      "max_features": ['auto', 'sqrt', 'log2'],
                      "min_samples_leaf": randint(2, 6),
                      "min_samples_split": randint(2, 6),
                      "n_estimators": randint(100, 1000),
                      "criterion": ['gini', 'entropy']}

    rf_params_bs = {"bootstrap": Categorical([True, False]),
                    "max_depth": Integer(4, 14),
                    "max_features": Categorical(['auto', 'sqrt', 'log2']),
                    "min_samples_leaf": Integer(2, 6),
                    "min_samples_split": Integer(2, 6),
                    "n_estimators": Integer(100, 1000),
                    "criterion": Categorical(['gini', 'entropy'])}

    # for xgb
    xgb_params_rand = {'max_depth': randint(3, 20),
                   'learning_rate': loguniform(0.01, 0.3),
                   'subsample': np.arange(0.5, 1.0, 0.1),
                   'colsample_bytree': np.arange(0.4, 1.0, 0.1),
                   'colsample_bylevel': np.arange(0.4, 1.0, 0.1),
                   'n_estimators': [100, 500, 1000]}

    xgb_params_bs = {'learning_rate': Real(0.01, 0.3),
                     'subsample': Real(0.5, 1.0),
                     'colsample_bytree': Real(0.4, 1.0),
                     'colsample_bylevel': Real(0.4, 1.0),
                     'max_depth': Integer(5, 17),
                     'n_estimators': [100, 500, 1000]}


    print('build model')
    n_iter_search = 30
    # n_estimator: if more or less better, you can try in your script out
    model = RandomForestClassifier(verbose=2)
    # model = xgb.XGBClassifier()

    # optimize: if it takes to long, reduce n_estimators or define it directly like RandomForestClassifier(
    # n_estimators=100, verbose=2) and delete it out of the params
    print('opt')
    # Bayes or Random Search
    # for xgb -> you have to change rf_params to xgb_params
    # search_reg = RandomizedSearchCV(model, rf_params_rand , n_iter=n_iter_search, scoring='accuracy')
    search_reg = BayesSearchCV(model, rf_params_bs, n_iter=n_iter_search, scoring='accuracy', cv=5)
    search_reg.fit(X_train, y_train)
    print('Best Params', search_reg.best_params_)
    print(f'best score: {search_reg.best_score_}')

    # test prediction
    y_pred = search_reg.predict(X_test)

    # compare decisions
    print('------')
    print('Accuracy:', accuracy_score(y_pred, y_test))
    print('Report:', classification_report(y_test, y_pred))
    print('------')
    gain_pct, mpv, gain_bench = backtest_func(df=data_or.iloc[-len(y_test):], decision=y_pred)
    print('this was for our prediction')
    print('------')
    gain_pct_best, mpv_best, gain_bench_best = backtest_func(df=data_or.iloc[-len(y_test):], decision=y_test)
    print('this was for the real value')


if __name__ == '__main__':
    main()
