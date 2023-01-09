import numpy as np
import matplotlib.pyplot as plt


def buy_stock(
        real_movement_open,
        real_movement_close,
        signal,
        initial_money=100000,
        max_buy=1,
        max_sell=1,
):
    def round_down(value, decimals):
        factor = 1 / (10 ** decimals)
        return (value // factor) * factor

    """
    real_movement = actual movement in the real world
    delay = how much interval you want to delay to change our decision from buy to sell, vice versa
    initial_state = 1 is buy, 0 is sell
    initial_money = 1000, ignore what kind of currency
    max_buy = max quantity for share to buy
    max_sell = max quantity for share to sell
    """

    starting_money = initial_money
    states_noth = []
    states_buy = []
    total_inv = 0
    total_sell = 0
    current_inventory = 0
    init_list = np.empty((real_movement_open.shape[0]))
    init_list[0] = initial_money
    gain_list_pct = np.empty((real_movement_open.shape[0]))
    gain_list = np.empty((real_movement_open.shape[0]))
    current_money = np.empty((real_movement_open.shape[0]))
    mpv_list = np.empty((real_movement_open.shape[0]))

    def buy(i, initial_money, total_inv, current_inventory):
        shares = round_down(initial_money / real_movement_open.iloc[i], 3)

        buy_units = shares
        total_inv += buy_units * real_movement_open.iloc[i]
        initial_money -= buy_units * real_movement_open.iloc[i]
        current_inventory += buy_units
        print('day %d: buy %d units at price %f, total balance %f'
              % (i + 1, buy_units, real_movement_open.iloc[i], initial_money))

        # starting money = initial money - money inv
        return initial_money, total_inv, current_inventory

    def sell(i, initial_money, total_sell, current_inventory):

        sell_units = current_inventory
        current_inventory -= sell_units
        total_sell += sell_units * real_movement_close.iloc[i]
        initial_money += sell_units * real_movement_close.iloc[i]

        print('day %d, sell %d units at price %f, total balance %f,'
              % (i + 1, sell_units, real_movement_close.iloc[i], initial_money))
        return initial_money, total_sell, current_inventory

    for i in range(real_movement_open.shape[0]):  # - int(0.025 * len(real_movement))#len(df)
        # state = signal.iloc[i]
        state = signal[i]
        if state == 1:
            initial_money, total_inv, current_inventory = buy(i, initial_money, total_inv, current_inventory)
            states_buy.append(i)
            initial_money, total_inv, current_inventory = sell(i, initial_money, total_inv, current_inventory)
        else:
            print(f'Did nothing on day {i + 1}!')
            states_noth.append(i)

        # init_list[i] = initial_money
        current_money[i] = current_inventory * real_movement_open.iloc[i] + initial_money
        if i == 0:
            # da tag davor habe ich nicht in df frame: in echt später gegenüber initial money
            gain_list_pct[0] = 0  # init_list[i] - current_inventory * real_movement[i]
            gain_list[0] = 0  # init_list[i]
        else:
            gain_list_pct[i] = ((current_money[i] - current_money[i - 1]) / current_money[i - 1]) * 100
            gain_list[i] = current_money[i] - current_money[i - 1]

        mpv_list[i] = current_money[i]

    current_money_inv = current_inventory * real_movement_open.iloc[-1]
    gain_pct = round(((initial_money - starting_money + current_money_inv) / starting_money) * 100, 4)
    total_gain = round((initial_money - starting_money + current_money_inv), 3)
    mpv = round(sum(mpv_list) / len(mpv_list), 3)
    # print(f'daily gain in pct: {gain_list_pct}')
    # print(f'daily gain: {gain_list}')

    return states_buy, states_noth, total_gain, gain_pct, initial_money, total_inv, total_sell, current_money_inv, mpv


def backtest_func(df, decision):
    df.dropna(inplace=True)

    # backtest
    states_buy, states_noth, total_gain, gain_pct, money_notinv, total_inv, total_sell, current_money_inv, mpv \
        = buy_stock(real_movement_open=df.Open, real_movement_close=df.Close, signal=decision)

    print(f'Total Gain: {total_gain}, Gain in PCT: {gain_pct} %')
    print(f'Total Investment: {total_inv}, Total sell {total_sell}')
    print(f'Money not invested: {money_notinv}, Money worth invested: {current_money_inv}')
    print(f'Total Money: {money_notinv + current_money_inv}')
    print(f'Mean Portfolio Value: {mpv}')

    # Benchmark

    gain_bench = ((df.Open.iloc[-1] - df.Open.iloc[0]) / df.Open.iloc[0]) * 100
    print(f'Gain Benchmark: {gain_bench} %')

    plt.figure(figsize=(15, 5))
    plt.plot(df.Open, color='r', lw=2., label='Open')
    plt.plot(df.Close, color='b', lw=2., label='Close')
    plt.plot(df.Open, '^', markersize=10, color='m', label='buying signal', markevery=states_buy)
    plt.plot(df.Open, 'v', markersize=10, color='k', label='did nothing', markevery=states_noth)
    plt.title(f'total gains: {(total_gain)}, total gain in pct {(gain_pct)} %')
    plt.legend()
    plt.show()

    return gain_pct, mpv, gain_bench