import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score
from Backtest_open_close import backtest_func
from Backtest_open_close_best import backtest_func_best
from sklearn.metrics import classification_report
import xgboost as xgb


def backtest_class(data, y, model, start=200, step=40):
    predictions = []
    accuracy = []
    # scaler = StandardScaler()
    # data = scaler.fit_transform(data)
    # Loop over the dataset in increments
    for i in range(start, data.shape[0], step):
        # Split into train and test sets
        X_train = data[0:i, :]
        y_train = y[0:i, :]
        X_test = data[i:(i + step), :]
        y_test = y[i:(i + step), :]
        model.fit(X_train, y_train.ravel())

        # Make predictions
        y_pred = model.predict(X_test)
        predictions.append(y_pred)
        acc = accuracy_score(y_test, y_pred)
        accuracy.append(acc)

    return predictions, accuracy


def data_shift(X, window, inp):
    X_plc = X
    for i in range(window):
        X_shift_bw = X.shift(periods=(i + 1), fill_value=0)
        inp_bw = [x + f'_-{i + 1}' for x in inp]
        X_shift_bw.columns = inp_bw
        X_plc = pd.concat([X_plc, X_shift_bw], axis=1)

    return X_plc


def main():
    print('go')
    data_csv = pd.read_csv("featureSelectionDataset_Paul_Class_shift_forBackt_close_open.csv", sep=',', header=0,
                           index_col=0, parse_dates=True,
                           decimal=".")
    # try different data window
    test_list = [800, 500, 800, 1500, 200]

    acc_list = []
    gain_list = []
    mpv_list = []
    bench_list = []

    gain_list_best = []
    mpv_list_best = []

    for c, i in enumerate(test_list):
        if c == 0:
            # first 800
            data = data_csv.copy()
            data = data.iloc[:i, :]
        elif c == 1:
            # about 800 in the middle
            data = data_csv.copy()
            data = data.iloc[i:-i, :]
        elif c == 2:
            # last 800
            data = data_csv.copy()
            data = data.iloc[-i:, :]
        elif c == 3:
            # first 1500
            data = data_csv.copy()
            data = data.iloc[:i, :]
        elif c == 4:
            # about 800 in the middle
            data = data_csv.copy()
            data = data.iloc[i:-i, :]

        data_or = data.copy()
        # because open and close not important feat but i need it for backtesting
        data = data.drop(['Open', 'Close'], axis=1)
        y_target = np.asarray([1 if data.Target[i] > 0 else 0 for i in range(len(data))]).reshape(-1, 1)
        X = data.drop(['Target'], axis=1)

        # include t-window data points as additional features
        inp = X.columns.values.tolist()
        window = 0
        X = data_shift(X, window, inp)
        #print(X)
        X = np.asarray(X)

        # scaling: try different scaling or no scaling
        # scaler = StandardScaler()
        # X = scaler.fit_transform(X)

        splitlimit = ((X.shape[0]) - 10)
        X_train, X_test = X[:splitlimit], X[splitlimit:]
        y_train, y_test = y_target[:splitlimit], y_target[splitlimit:]

        # model = RandomForestClassifier(n_estimators=300, verbose=1)
        model = xgb.XGBClassifier()
        model.fit(np.asarray(X_train), y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print('Report:', classification_report(y_test, y_pred))
        print('ACC', acc)
        print('------')
        gain_pct, mpv, gain_bench = backtest_func(df=data_or.iloc[-len(y_test):], decision=y_pred)
        print('this was for our prediction')
        print('------')
        gain_pct_best, mpv_best = backtest_func_best(df=data_or.iloc[-len(y_test):], decision=y_test)
        print('this was for the real value')

        acc_list.append(acc)
        gain_list.append(gain_pct)
        mpv_list.append(mpv)
        bench_list.append(gain_bench)
        mpv_list_best.append(mpv_best)
        gain_list_best.append(gain_pct_best)

    print(f'acc: {acc_list} %')
    print('------')
    print(f'Gain rule: {gain_list} %')
    print(f'Gain best: {gain_list_best} %')
    print(f'Gain bench: {bench_list} %')
    print('------')
    print('MPV rule', mpv_list)
    print('MPV best', mpv_list_best)


if __name__ == '__main__':
    main()