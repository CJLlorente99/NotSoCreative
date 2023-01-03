import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score
import math
import yfinance as yf
# import pandas_ta as ta
# from Backtest import backtest_func
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional
import tensorflow as tf
import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from keras.callbacks import History
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

# not important
def backtest(data, y, model, start=200, step=40):
    predictions = []
    accuracy = []
    # Loop over the dataset in increments
    for i in range(start, data.shape[0], step):
        # Split into train and test sets
        X_train = data[0:i, :]
        y_train = y[0:i, :]
        X_test = data[i:(i + step), :]
        y_test = y[i:(i + step), :]
        
        #print('tr', train)
        #print('test', test)
        #print(train['TargetClass'])
        # Fit the random forest model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict_proba(X_test)
        predictions.append(y_pred)
        acc = accuracy_score(y_test, y_pred)
        accuracy.append(acc)
        
    
    return predictions, accuracy

def main():
    data = pd.read_csv('featureSelectionDataset600MixedShifted.csv', sep=',', header=0, index_col=0, parse_dates=True, decimal=".")
    # ["Return"]
    # for dataset Log300 use this:
    #data = data.drop(['LogReturn', 'Return', 'ReturnBefore', 'log(Open)', 'Class', 'LogReturnBefore'], axis=1)
    # for other use this
    data = data.drop(['LogReturn', 'Return', 'ReturnBefore', 'log(Open)', 'Class', 'LogReturnBefore', 'LogReturnBeforeClose'], axis=1)

    # Each row of the data frame has (for day t)
    # Close(t-1) | Open(t) | Return_close/Diff_close{Close(t-1) - Close(t-2)} | Return_open/Diff_open{Open(t) - Open(t-1)} | Return_intraday/Diff_intraday{Close(t-1) - Open(t-1)} |
    # Return_interday/Diff_interday{Open(t) - Close(t-1)}
    # ALL INDICATORS HAVE ALREADY BEEN SHIFTED AS NONE DEPENDS ON THE OPEN VALUE
    data['Close'] = data['Close'].shift(+1)
    data['log_Close'] = np.log(data['Close'])
    data["log_Open"] = np.log(data["Open"])
    data["Return_close"] = data["log_Close"] - data["log_Close"].shift(+1)
    data["Return_open"] = data["log_Open"] - data["log_Open"].shift(+1)
    data["Return_intraday"] = data["log_Close"] - data["log_Open"].shift(+1)
    data['Return_interday'] = data["log_Open"] - data["log_Close"]
    data['Diff_open'] = data["Open"] - data["Open"].shift()
    data['Diff_close'] = data['Close'] - data['Close'].shift()
    data['Diff_intraday'] = data['Close'] - data['Open'].shift(1)
    data['Diff_interday'] = data['Open'] - data['Close']
    data['Class_open'] = [1 if data.Return_open[i] > 0 else 0 for i in range(len(data))]
    data['Class_close'] = [1 if data.Return_close[i]>0 else 0 for i in range(len(data))]
    data['Class_intraday'] = [1 if data.Return_intraday[i]>0 else 0 for i in range(len(data))]
    data['Class_interday'] = [1 if data.Return_interday[i] > 0 else 0 for i in range(len(data))]
    yx = data["log_Open"].shift(-1) - data["log_Open"]
    yx.dropna(inplace=True)
    yx = yx[1:]
    print(yx)
    
    data.dropna(inplace=True)
    data = data[:-1]
    print(data)
    y = [1 if yx.iloc[i] > 0 else 0 for i in range(len(data))]
    
    
    #y_open = np.asarray([1 if data.Return_open[i]>0 else 0 for i in range(len(data))]).reshape(-1, 1)
    #y_close = np.asarray([1 if data.Return_close[i]>0 else 0 for i in range(len(data))]).reshape(-1, 1)
    #y_target = np.asarray([1 if data.Target[i]>0 else 0 for i in range(len(data))]).reshape(-1, 1)

    # Create correlation matrix
    corr_matrix = data.corr()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))
    to_drop = []
    for column in upper_tri.columns:
        if any(upper_tri[column] > 0.95):
            to_drop.append(column)
    data = data.drop(to_drop, axis=1)
    print(data)

    '''data = yf.download(tickers='^GSPC', start='2018-01-01', end='2022-12-28')
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
    data.dropna(inplace=True)
    #data["Return_target"] = data["log_Open"].shift(-1) - data["log_Open"]
    y = data["log_Open"].shift(-1) - data["log_Open"]
    y.dropna(inplace=True)
    data = data[:-1]'''

    #sc = StandardScaler()
    #y = sc.fit_transform(np.asarray(y).reshape(-1, 1))
    #X = sc.fit_transform(data)
    X = data
    splitlimit = (len(data) - 100)
    
    
    X_train, X_test = X[:splitlimit], X[splitlimit:]
    y_train, y_test = y[:splitlimit], y[splitlimit:]
    
    #model = RandomForestClassifier(n_estimators=100, verbose=2)
    model = DecisionTreeClassifier()
    #model = RandomForestRegressor(n_estimators=100, verbose=2)
    model.fit(np.asarray(X_train), y_train)  # , validation_split=0.3)
    y_pred = model.predict(X_test)
    
    feat_imp = model.feature_importances_
    #print('feat_imp:', feat_imp, feat_imp.shape)
    idx = np.argsort(feat_imp)
    idx = idx[::-1]
    feat_imp_sort = np.take_along_axis(feat_imp, idx, axis=0)
    X_sort = data.iloc[:, idx]
    k = 20
    X_sort = X_sort.iloc[:, :k]
    print(X_sort)
    feat_imp_k = feat_imp_sort[:k]
    print('sorted:', feat_imp_k)
    plt.barh(X_sort.columns, feat_imp_k)
    plt.title('feature importance')
    plt.show()
    
    corr_matrix = X_sort.corr(method='pearson')
    #print(corr_matrix)
    f, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', linewidth=0.4, annot_kws={'size': 10}, cmap='coolwarm', ax=ax)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()
    # X_sort['Open'] = data['Open']
    # X_sort['Target'] = data["log_Open"].shift(-1) - data["log_Open"]
    X_sort.to_csv("featureSelectionDataset_Paul_Class.csv")
    print(X_sort)


if __name__ == '__main__':
    main()
