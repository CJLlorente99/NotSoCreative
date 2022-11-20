import os
import warnings
import numpy as np
import pandas as pd
from pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import numpy as np
import pandas_datareader as web
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from statsmodels.tsa.stattools import acf, pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
warnings.filterwarnings('ignore')
rcParams['figure.figsize'] = 10, 6


#Test for staionarity
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
    # output for dft will give us without defining what the values are.
    #hence we manually write what values does it explains using a for loop
    output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
    for key, values in adft[4].items():
        output['critical value (%s)'%key] = values
    print(output)

def forecast_test(ARIMA_model, X_train, X_test, n_periods=51):
    # Forecast
    n_periods = len(X_test)
    fc_series, conf_int = ARIMA_model.predict(n_periods=n_periods, return_conf_int=True)

                                            #exogenous=forecast_df[['month_index']])

    lower_series = pd.Series(conf_int[:, 0], index=X_test.index)
    upper_series = pd.Series(conf_int[:, 1], index=X_test.index)
    # Plot
    plt.figure(figsize=(15, 7))
    plt.plot(X_train, label='training data')
    plt.plot(X_test, color='blue', label='Actual Stock Price')
    plt.plot(X_test.index, fc_series, color='orange', label='Predicted Stock Price')
    plt.fill_between(lower_series.index, lower_series, upper_series,
                     color='k', alpha=.10)
    plt.title("ARIMA/SARIMA - Forecast of S&p 500")
    plt.show()
    return fc_series
def forecast_exp_test(ARIMA_model, X_train, X_test, n_periods=51):
    # Forecast
    n_periods = len(X_test)
    # ARIMA_model.fit(X_train) -> do i need this
    fc_series, conf_int = ARIMA_model.predict(n_periods=n_periods, return_conf_int=True)

    lower_series = pd.Series(conf_int[:, 0], index=X_test.index)
    upper_series = pd.Series(conf_int[:, 1], index=X_test.index)
    # Plot
    plt.figure(figsize=(15, 7))
    plt.plot(np.exp(X_train), label='training data')
    plt.plot(np.exp(X_test), color='blue', label='Actual Stock Price')
    plt.plot(X_test.index, np.exp(fc_series), color='orange', label='Predicted Stock Price')
    plt.fill_between(lower_series.index, np.exp(lower_series), np.exp(upper_series),
                     color='k', alpha=.10)
    plt.title("ARIMA/SARIMA - Forecast of S&p 500")
    plt.show()
    return np.exp(fc_series)

def sarimax_forecast(SARIMAX_model, X_train, X_test, n_periods=51):
    # Forecast
    n_periods = len(X_test)
    fc_series, conf_int = SARIMAX_model.predict(n_periods=n_periods, return_conf_int=True)

                                            #exogenous=forecast_df[['month_index']])

    lower_series = pd.Series(conf_int[:, 0], index=X_test.index)
    upper_series = pd.Series(conf_int[:, 1], index=X_test.index)
    # Plot
    plt.figure(figsize=(15, 7))
    plt.plot(X_train, label='training data')
    plt.plot(X_test, color='blue', label='Actual Stock Price')
    plt.plot(X_test.index, fc_series, color='orange', label='Predicted Stock Price')
    plt.fill_between(lower_series.index, lower_series, upper_series,
                     color='k', alpha=.10)
    plt.title("SARIMAX - Forecast of S&p 500")
    plt.show()
    return fc_series



def main():
    start = dt.datetime(2018, 1, 1)
    end = dt.datetime(2022, 1, 1)
    ticker = '^GSPC'
    df = web.DataReader(ticker, 'yahoo', start, end)
    # Distribution of the dataset
    df_open = df['Open']
    '''df_open.plot(kind='kde')
    plt.title('Kernel Density Estimation')
    plt.ylabel('Density')
    plt.show()'''

    test_stationarity(df_open)

    # To separate the trend and the seasonality from a time series,
    # we can decompose the series using the following code.
    '''decomp = seasonal_decompose(df_open, model='multiplicative', period=30)
    plt.figure(figsize=(10, 8))
    decomp.plot()
    plt.show()'''

    df_log = np.log(df_open)
    decomposition = seasonal_decompose(df_log, model='multiplicative', period=30)
    # trend: represents long term change in time series
    trend = decomposition.trend
    # seasonal: a periodic pattern in the time series
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    plt.figure(figsize=(16, 7))
    fig = plt.figure(1)
    plt.subplot(411)
    plt.plot(df_log, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.show()

    # ACF and PACF
    # tests of stationarity
    # plots:
    '''df_log_diff = df_log - df_log.shift()
    lag_acf = acf(df_log_diff, nlags=20)
    lag_pacf = pacf(df_log_diff, nlags=20, method='ols')
    # Plot ACF:
    plt.subplot(121)
    plt.plot(lag_acf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96 / np.sqrt(len(df_log_diff)), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(df_log_diff)), linestyle='--', color='gray')
    plt.title('Autocorrelation Function')
    # Plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96 / np.sqrt(len(df_log_diff)), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(df_log_diff)), linestyle='--', color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()'''

    # to reduce magnitude
    '''df_log = np.log(df_open)
    moving_avg = df_log.rolling(12).mean()
    std_dev = df_log.rolling(12).std()
    plt.legend(loc='best')
    plt.title('Moving Average')
    plt.plot(std_dev, color="black", label="Standard Deviation")
    plt.plot(moving_avg, color="red", label="Mean")
    plt.legend()
    plt.show()'''

    #split data into train and training set
    train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]
    # train and test data are in log (more stationary): Also possible: to make it even more stationary
    '''df_log = np.log(df_open)
    moving_avg = pd.rolling_mean(df_log, 12)
    ts_log_moving_avg_diff = df_log - moving_avg
    ts_log_moving_avg_diff.dropna(inplace=True)
    train_data, test_data = ts_log_moving_avg_diff[3:int(len(df_log)*0.9)], ts_log_moving_avg_diff[int(len(df_log)*0.9):]'''
    # I also can give non stationary data:
    '''test_days = round(0.2 * df.shape[0])
    train_data = df_open.iloc[:-test_days]
    test_data = df_open.iloc[-test_days:]'''
    #train_data, test_data = df_open[3:int(len(df_open)*0.9)], df_open[int(len(df_open)*0.9):]

    '''plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.xlabel('Dates')
    plt.ylabel('Opening Prices')
    plt.plot(df_log, 'green', label='Train data')
    plt.plot(test_data, 'blue', label='Test data')
    plt.legend()
    plt.show()'''

    autoARIMA_model = auto_arima(train_data, start_p=0, start_q=0,
                                 test='adf',  # use adftest to find optimal 'd'
                                 max_p=3, max_q=3,  # maximum p and q
                                 m=1,  # frequency of series
                                 d=None,  # let model determine 'd'
                                 seasonal=False,  # No Seasonality
                                 start_P=0,
                                 D=0,
                                 trace=True,
                                 error_action='ignore',
                                 suppress_warnings=True,
                                 stepwise=True)

    SARIMA_model = auto_arima(train_data, start_p=1, start_q=1,
                                 test='adf',
                                 max_p=3, max_q=3,
                                 m=12,  # 12 is the frequncy of the cycle
                                 start_P=0,
                                 seasonal=True,  # set to seasonal
                                 d=None,
                                 D=1,  # order of the seasonal differencing
                                 trace=False,
                                 error_action='ignore',
                                 suppress_warnings=True,
                                 stepwise=True)

    SARIMAX_model = auto_arima(train_data,
                                  start_p=1, start_q=1,
                                  test='adf',
                                  max_p=3, max_q=3, m=12,
                                  start_P=0, seasonal=True,
                                  d=None, D=1,
                                  trace=False,
                                  error_action='ignore',
                                  suppress_warnings=True,
                                  stepwise=True)

    '''print(autoARIMA_model.summary())
    autoARIMA_model.plot_diagnostics(figsize=(15, 8))'''

    '''print(SARIMA_model.summary())
    SARIMA_model.plot_diagnostics(figsize=(15, 8))'''

    '''print(SARIMAX_model.summary())
    SARIMAX_model.plot_diagnostics(figsize=(15, 8))'''

    plt.show()

    # Modeling
    # Build Model
    test_pred_arima = forecast_exp_test(autoARIMA_model, train_data, test_data, n_periods=len(test_data))

    test_pred_sarima = forecast_exp_test(SARIMA_model, train_data, test_data, n_periods=len(test_data))

    test_pred_sarimax = sarimax_forecast(SARIMAX_model, train_data, test_data, n_periods=len(test_data))

    test_data = np.exp(test_data)
    mse_arima = mean_squared_error(test_data, test_pred_arima)
    print('MSE ARIMA: ' + str(mse_arima))
    mae_arima = mean_absolute_error(test_data, test_pred_arima)
    print('MAE ARIMA: ' + str(mae_arima))
    rmse_arima = math.sqrt(mean_squared_error(test_data, test_pred_arima))
    print('RMSE ARIMA: ' + str(rmse_arima))
    mape_arima = np.mean(np.abs(test_pred_arima - test_data) / np.abs(test_data))
    print('MAPE ARIMA: ' + str(mape_arima))

    mse_sarima = mean_squared_error(test_data, test_pred_sarima)
    print(f'MSE SARIMA:{mse_sarima}')
    mae_sarima = mean_absolute_error(test_data, test_pred_sarima)
    print('MAE SARIMA: ' + str(mae_sarima))
    rmse_sarima = math.sqrt(mean_squared_error(test_data, test_pred_sarima))
    print('RMSE SARIMA: ' + str(rmse_sarima))
    mape_sarima = np.mean(np.abs(test_pred_sarima - test_data) / np.abs(test_data))
    print('MAPE SARIMA: ' + str(mape_sarima))

    mse_sarimax = mean_squared_error(test_data, test_pred_sarimax)
    print(f'MSE SARIMAX:{mse_sarimax}')
    mae_sarimax = mean_absolute_error(test_data, test_pred_sarimax)
    print('MAE SARIMAX: ' + str(mae_sarimax))
    rmse_sarimax = math.sqrt(mean_squared_error(test_data, test_pred_sarimax))
    print('RMSE SARIMAX: ' + str(rmse_sarimax))
    mape_sarimax = np.mean(np.abs(test_pred_sarimax - test_data) / np.abs(test_data))
    print('MAPE SARIMAX: ' + str(mape_sarimax))

if __name__ == '__main__':
    main()


