import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader as web
import datetime as dt
import ta




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

    def buy(i, initial_money, total_inv, current_inventory):
        shares = initial_money // real_movement[i]
        if shares < 1:
            print(
                'day %d: total balances %f, not enough money to buy a unit price %f'
                % (i, initial_money, real_movement[i])
            )
        else:
            if shares > max_buy:
                buy_units = max_buy
            else:
                buy_units = shares
            total_inv += buy_units * real_movement[i]
            initial_money -= buy_units * real_movement[i]
            current_inventory += buy_units
            print(
                'day %d: buy %d units at price %f, total balance %f'
                % (i, buy_units, buy_units * real_movement[i], initial_money)
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
            total_sell += sell_units * real_movement[i]
            initial_money += sell_units * real_movement[i]
            try:
                invest = (
                                 (real_movement[i] - real_movement[states_buy[-1]])
                                 / real_movement[states_buy[-1]]
                         ) * 100
            except:
                invest = 0
            print(
                'day %d, sell %d units at price %f, investment %f %%, total balance %f,'
                % (i, sell_units, sell_units * real_movement[i], invest, initial_money)
            )
        return initial_money, total_sell, current_inventory


    for i in range(real_movement.shape[0]):  #- int(0.025 * len(real_movement))#len(df)
        state = signal[i]
        if state == 1:
            initial_money, total_inv, current_inventory = buy(i, initial_money, total_inv, current_inventory)
            states_buy.append(i)

        elif state == -1:
            initial_money, total_sell, current_inventory = sell(i, initial_money, total_sell, current_inventory)
            states_sell.append(i)

    current_money_inv = current_inventory * real_movement[-1]
    gain_pct = ((initial_money - starting_money + current_money_inv) / starting_money) * 100
    total_gain = (initial_money - starting_money + current_money_inv)


    return states_buy, states_sell, total_gain, gain_pct, initial_money, total_inv, total_sell, current_money_inv

def MACD_dec(df):
    # wrong
    df['MACD_diff'] = ta.trend.macd_diff(df.Open)
    df['Decision MACD'] = np.where((df.MACD_diff > 0) & (df.MACD_diff.shift(1) < 0), 1.0, 0.0)

    return df

def Goldencross_dec(df, window_short=20, window_long=50):
    #50, 200
    df[f'SMA{window_short}'] = ta.trend.sma_indicator(df.Open, window=window_short)
    df[f'SMA{window_long}'] = ta.trend.sma_indicator(df.Open, window=window_long)
    df['Signal'] = np.where(df[f'SMA{window_short}'] > df[f'SMA{window_long}'], 1.0, 0.0)
    df['Decision GC'] = df.Signal.diff()
    return df

def expMovingAve_dec(df, window_short=50, window_long=200):
    #50, 200
    df[f'EMA{window_short}'] = ta.trend.ema_indicator(df.Open, window=window_short)
    df[f'EMA{window_long}'] = ta.trend.ema_indicator(df.Open, window=window_long)
    df['Signal'] = np.where(df[f'EMA{window_short}'] > df[f'EMA{window_long}'], 1.0, 0.0)
    df['Decision EMA'] = df.Signal.diff()
    return df

def main():
    # load data
    start = dt.datetime(2018, 1, 1)
    end = dt.datetime(2022, 1, 1)
    ticker = '^GSPC'
    df = web.DataReader(ticker, 'yahoo', start, end)
    window_short = 50
    window_long = 200

    df = Goldencross_dec(df, window_short, window_long)

    #wrong
    df = MACD_dec(df)

    df = expMovingAve_dec(df, window_short, window_long)

    states_buy, states_sell, total_gain, gain_pct, money_notinv, total_inv, total_sell, current_money_inv = buy_stock(
        df.Open, df['Decision GC'])
    '''states_buy, states_sell, total_gain, gain_pct, money_notinv, total_inv, total_sell, current_money_inv = buy_stock(
        df.Open, df['Decision MACD'])'''
    '''states_buy, states_sell, total_gain, gain_pct, money_notinv, total_inv, total_sell, current_money_inv = buy_stock(
        df.Open, df['Decision EMA'])'''
    print(f'Total Gain: {total_gain}, Gain in PCT: {gain_pct} %')
    print(f'Total Investment: {total_inv}, Total sell {total_sell}')
    print(f'Money not invested: {money_notinv}, Money worth invested: {current_money_inv}')
    print(f'Total Money: {money_notinv + current_money_inv}')


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