import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader as web
import datetime as dt
import ta
import pandas_ta as tap
import strategy as strgy


def buy_stock(
    real_movement,
    signal,
    initial_money=10000,
    max_buy=1,
    max_sell=1,
):

    """
    real_movement = actual movement in the real world
    delay = how much interval you want to delay to change our decision from buy to sell, vice versa
    initial_state = 1 is buy, 0 is sell
    initial_money = 1000, ignore what kind of currency
    max_buy = max quantity for share to buy
    max_sell = max quantity for share to sell
    """

    starting_money = initial_money
    states_sell = []
    states_buy = []
    total_inv = 0
    total_sell = 0
    current_inventory = 0
    init_list = np.empty((real_movement.shape[0]))
    init_list[0] = initial_money
    gain_list_pct = np.empty((real_movement.shape[0]))
    gain_list = np.empty((real_movement.shape[0]))
    current_money = np.empty((real_movement.shape[0]))


    def buy(i, initial_money, total_inv, current_inventory):
        shares = initial_money // real_movement.iloc[i]
        if shares < 1:
            print(
                'day %d: total balances %f, not enough money to buy a unit price %f'
                % (i, initial_money, real_movement.iloc[i])
            )
        else:
            if shares > max_buy:
                buy_units = max_buy
            else:
                buy_units = shares
            total_inv += buy_units * real_movement.iloc[i]
            initial_money -= buy_units * real_movement.iloc[i]
            current_inventory += buy_units
            print(
                'day %d: buy %d units at price %f, total balance %f'
                % (i, buy_units, buy_units * real_movement.iloc[i], initial_money)
            )

            # starting money = initial money - money inv
        return initial_money, total_inv, current_inventory

    def sell(i, initial_money, total_sell, current_inventory):
        if current_inventory == 0:
            print('day %d: cannot sell anything, inventory 0' % (i))
        else:
            if current_inventory > max_sell:  # before >
                sell_units = max_sell
            else:
                sell_units = current_inventory
            current_inventory -= sell_units
            total_sell += sell_units * real_movement.iloc[i]
            initial_money += sell_units * real_movement.iloc[i]
            '''try:
                invest = (
                                 (real_movement[i] - real_movement[states_buy[-1]])
                                 / real_movement[states_buy[-1]]
                         ) * 100
            except:
                invest = 0'''
            # 'day %d, sell %d units at price %f, investment %f %%, total balance %f,'
            # % (i, sell_units, sell_units * real_movement[i], invest, initial_money)
            print(
                'day %d, sell %d units at price %f, total balance %f,'
                % (i, sell_units, sell_units * real_movement.iloc[i], initial_money)
            )
        return initial_money, total_sell, current_inventory


    for i in range(real_movement.shape[0]):  #- int(0.025 * len(real_movement))#len(df)
        state = signal.iloc[i]
        if state == 1:
            initial_money, total_inv, current_inventory = buy(i, initial_money, total_inv, current_inventory)
            states_buy.append(i)

        elif state == -1:
            initial_money, total_sell, current_inventory = sell(i, initial_money, total_sell, current_inventory)
            states_sell.append(i)
        #init_list[i] = initial_money
        current_money[i] = current_inventory * real_movement.iloc[i] + initial_money
        if i == 0:
            # da tag davor habe ich nicht in df frame: in echt später gegenüber initial money
            gain_list_pct[0] = 0 #init_list[i] - current_inventory * real_movement[i]
            gain_list[0] = 0 #init_list[i]
        else:
            gain_list_pct[i] = ((current_money[i] - current_money[i-1]) / current_money[i-1]) * 100
            gain_list[i] = current_money[i] - current_money[i-1]


    current_money_inv = current_inventory * real_movement.iloc[-1]
    gain_pct = ((initial_money - starting_money + current_money_inv) / starting_money) * 100
    total_gain = (initial_money - starting_money + current_money_inv)
    print(f'daily gain in pct: {gain_list_pct}')
    print(f'daily gain: {gain_list}')

    return states_buy, states_sell, total_gain, gain_pct, initial_money, total_inv, total_sell, current_money_inv




def main():
    # load data
    start = dt.datetime(2022, 11, 14)
    end = dt.datetime.today()
    ticker = '^GSPC'
    df = web.DataReader(ticker, 'yahoo', start, end)


    #df.to_csv(r'/Users/paulheller/PycharmProjects/DataScience/SavedData/export_dataframe.csv', header=True)
    #df = pd.read_csv(r'/Users/paulheller/PycharmProjects/DataScience/SavedData/export_dataframe.csv', index_col=0)

    # strategies to backtest
    df = strgy.Goldencross_dec(df, window_short=2, window_long=3)
    #df = strgy.RSI_SMA_dec(df, window_rsi=5, window_sma=20)
    #df = strgy.candlestickpattern(df, n1=2, n2=2, backCandles=45) # 22% mit close, 25% mit open
    #df = strgy.shootingstar(df)
    #df = strgy.mytarget(df, barsupfront=40)
    #df = strgy.scalping_rsi(df, window_ema=200, window_rsi=5)
    df.dropna(inplace=True)
    print(df)


    # backtest
    states_buy, states_sell, total_gain, gain_pct, money_notinv, total_inv, total_sell, current_money_inv = buy_stock(
        df.Open, df['Decision GC'])
    '''states_buy, states_sell, total_gain, gain_pct, money_notinv, total_inv, total_sell, current_money_inv = buy_stock(
        df.Open, df['Decision RSI/SMA'])'''
    '''states_buy, states_sell, total_gain, gain_pct, money_notinv, total_inv, total_sell, current_money_inv = buy_stock(
        df.Open, df['Decision MACD'])'''
    '''states_buy, states_sell, total_gain, gain_pct, money_notinv, total_inv, total_sell, current_money_inv = buy_stock(
        df.Open, df['Decision EMA'])'''
    '''states_buy, states_sell, total_gain, gain_pct, money_notinv, total_inv, total_sell, current_money_inv = buy_stock(
        df.Close, df['Trendcat'])'''
    '''states_buy, states_sell, total_gain, gain_pct, money_notinv, total_inv, total_sell, current_money_inv = buy_stock(
        df.Open, df['Dec_candle'])'''
    '''states_buy, states_sell, total_gain, gain_pct, money_notinv, total_inv, total_sell, current_money_inv = buy_stock(
        df.Open, df['Dec_RSI'])'''


    print(f'Total Gain: {total_gain}, Gain in PCT: {gain_pct} %')
    print(f'Total Investment: {total_inv}, Total sell {total_sell}')
    print(f'Money not invested: {money_notinv}, Money worth invested: {current_money_inv}')
    print(f'Total Money: {money_notinv + current_money_inv}')

    # Benchmarks
    df_open = df.Open
    gain_bench = ((df_open.iloc[-1] - df_open.iloc[0]) / df_open.iloc[0]) * 100
    print(f'Gain Benchmark: {gain_bench} %')

    plt.figure(figsize=(15, 5))
    plt.plot(df.Open, color='r', lw=2.)
    plt.plot(df.Open, '^', markersize=10, color='m', label='buying signal', markevery=states_buy)
    plt.plot(df.Open, 'v', markersize=10, color='k', label='selling signal', markevery=states_sell)
    #plt.plot(df[f'SMA{window_short}'], label=f'SMA{window_short}')
    #plt.plot(df[f'SMA{window_long}'], label=f'SMA{window_long}')
    plt.title(f'total gains: {(total_gain)}, total gain in pct {(gain_pct)} %')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()