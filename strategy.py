import yfinance as yf
import pandas_datareader as web
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import ta
import numpy as np
import pandas_ta as tap



def MACD_dec(df):
    # wrong
    df['MACD_diff'] = ta.trend.macd_diff(df.Open)
    df['Decision_MACD'] = np.where((df.MACD_diff > 0) & (df.MACD_diff.shift(1) < 0), 1.0, 0.0)
    tap.rsi(df.open)

    return df

def Goldencross_dec(df, window_short=20, window_long=50):
    #50, 200
    df[f'SMA{window_short}'] = ta.trend.sma_indicator(df.Open, window=window_short)
    df[f'SMA{window_long}'] = ta.trend.sma_indicator(df.Open, window=window_long)
    df['Signal_GC'] = np.where(df[f'SMA{window_short}'] > df[f'SMA{window_long}'], 1.0, 0.0)
    df['Decision GC'] = df.Signal_GC.diff()
    return df

def expMovingAve_dec(df, window_short=50, window_long=200):
    #50, 200
    df[f'EMA{window_short}'] = ta.trend.ema_indicator(df.Open, window=window_short)
    df[f'EMA{window_long}'] = ta.trend.ema_indicator(df.Open, window=window_long)
    df['Signal_EMA'] = np.where(df[f'EMA{window_short}'] > df[f'EMA{window_long}'], 1.0, 0.0)
    df['Decision EMA'] = df.Signal_EMA_EMA.diff()
    return df

def RSI_SMA_dec(df, window_rsi=10, window_sma=200):
    df['RSI'] = ta.momentum.rsi(df.Open, window=window_rsi)
    df[f'SMA{window_sma}'] = ta.trend.sma_indicator(df.Open, window=window_sma)
    df['Decision RSI/SMA'] = np.where((df.Open > df[f'SMA{window_sma}']) & (df.RSI < 30), 1, 0)
    return df

def shootingstar(df):
    df['ATR'] = ta.volatility.average_true_range(high=df.High, low=df.Low, close=df.Close, window=14)
    df['RSI'] = ta.momentum.rsi(df.Close, window=10)
    df.dropna()

    def Revsignal1(df1):
        df.dropna()
        df.reset_index(drop=True, inplace=True)

        length = len(df1)
        high = list(df1['High'])
        low = list(df1['Low'])
        close = list(df1['Close'])
        open = list(df1['Open'])
        signal = [0] * length
        highdiff = [0] * length
        lowdiff = [0] * length
        bodydiff = [0] * length
        ratio1 = [0] * length
        ratio2 = [0] * length

        for row in range(0, length):

            highdiff[row] = high[row] - max(open[row], close[row])
            bodydiff[row] = abs(open[row] - close[row])
            if bodydiff[row] < 0.002:
                bodydiff[row] = 0.002
            lowdiff[row] = min(open[row], close[row]) - low[row]
            ratio1[row] = highdiff[row] / bodydiff[row]
            ratio2[row] = lowdiff[row] / bodydiff[row]

            # print(df.RSI[row])
            #  |
            # _|_
            # |__|
            # |
            #

            if (ratio1[row] > 2.5 and lowdiff[row] < 0.3 * highdiff[row] and bodydiff[row] > 0.03 and df.RSI[
                row] > 50 and df.RSI[row] < 70):
                signal[row] = -1

            # elif (ratio2[row-1]>2.5 and highdiff[row-1]<0.23*lowdiff[row-1] and bodydiff[row-1]>0.03 and bodydiff[row]>0.04 and close[row]>open[row] and close[row]>high[row-1] and df.RSI[row]<55 and df.RSI[row]>30):
            #    signal[row] = 2
            # _|_
            # |__|
            # |
            # |

            elif (ratio2[row] > 2.5 and highdiff[row] < 0.23 * lowdiff[row] and bodydiff[row] > 0.03 and df.RSI[
                row] < 55 and df.RSI[row] > 30):
                signal[row] = 1
        return signal

    df['Dec_ShoStar'] = Revsignal1(df)
    #df[df['signal1'] == 1].count()
    return df

# for shooting start
def mytarget(df1, barsupfront):
    length = len(df1)
    high = list(df1['High'])
    low = list(df1['Low'])
    close = list(df1['Close'])
    open = list(df1['Open'])
    datr = list(df1['ATR'])
    trendcat = [0] * length

    for line in range(0, length - barsupfront - 1):
        valueOpenLow = 0
        valueOpenHigh = 0

        highdiff = high[line] - max(open[line], close[line])
        bodydiff = abs(open[line] - close[line])

        pipdiff = datr[line] * 1.  # highdiff*1.3 #for SL 400*1e-3
        if pipdiff < 1.1:
            pipdiff = 1.1

        SLTPRatio = 2.  # pipdiff*Ratio gives TP

        for i in range(1, barsupfront + 1):
            value1 = close[line] - low[line + i]
            value2 = close[line] - high[line + i]
            valueOpenLow = max(value1, valueOpenLow)
            valueOpenHigh = min(value2, valueOpenHigh)

            if ((valueOpenLow >= (SLTPRatio * pipdiff)) and (-valueOpenHigh < pipdiff)):
                trendcat[line] = -1  # -1 downtrend
                break
            elif ((valueOpenLow < pipdiff)) and (-valueOpenHigh >= (SLTPRatio * pipdiff)):
                trendcat[line] = 1 # uptrend
                break
            else:
                trendcat[line] = 0  # no clear trend
    df1['Trendcat'] = trendcat
    return df1

### next strategy: Automated candlestick pattern
def candlestickpattern(df, n1=2, n2=2, backCandles=45):
    #n1 = 2
    #n2 = 2
    #backCandles = 45

    length = len(df)
    high = list(df['High'])
    low = list(df['Low'])
    close = list(df['Close'])
    open = list(df['Open'])
    bodydiff = [0] * length
    highdiff = [0] * length
    lowdiff = [0] * length
    ratio1 = [0] * length
    ratio2 = [0] * length
    signal = [0] * length

    def support(df1, l, n1, n2):  # n1 n2 before and after candle l
        for i in range(l - n1 + 1, l + 1):
            if (df1.Low[i] > df1.Low[i - 1]):
                return 0
        for i in range(l + 1, l + n2 + 1):
            if (df1.Low[i] < df1.Low[i - 1]):
                return 0
        return 1


    def resistance(df1, l, n1, n2):  # n1 n2 before and after candle l
        for i in range(l - n1 + 1, l + 1):
            if (df1.High[i] < df1.High[i - 1]):
                return 0
        for i in range(l + 1, l + n2 + 1):
            if (df1.High[i] > df1.High[i - 1]):
                return 0
        return 1


    def isEngulfing(l):
        row = l
        bodydiff[row] = abs(open[row] - close[row])
        if bodydiff[row] < 0.000001:
            bodydiff[row] = 0.000001

        bodydiffmin = 0.002
        if (bodydiff[row] > bodydiffmin and bodydiff[row - 1] > bodydiffmin and
                open[row - 1] < close[row - 1] and
                open[row] > close[row] and
                (open[row] - close[row - 1]) >= -0e-5 and close[row] < open[row - 1]):  # +0e-5 -5e-5
            return -1

        elif (bodydiff[row] > bodydiffmin and bodydiff[row - 1] > bodydiffmin and
              open[row - 1] > close[row - 1] and
              open[row] < close[row] and
              (open[row] - close[row - 1]) <= +0e-5 and close[row] > open[row - 1]):  # -0e-5 +5e-5
            return 1
        else:
            return 0


    def isStar(l):
        bodydiffmin = 0.0020
        row = l
        highdiff[row] = high[row] - max(open[row], close[row])
        lowdiff[row] = min(open[row], close[row]) - low[row]
        bodydiff[row] = abs(open[row] - close[row])
        if bodydiff[row] < 0.000001:
            bodydiff[row] = 0.000001
        ratio1[row] = highdiff[row] / bodydiff[row]
        ratio2[row] = lowdiff[row] / bodydiff[row]

        if (ratio1[row] > 1 and lowdiff[row] < 0.2 * highdiff[row] and bodydiff[
            row] > bodydiffmin):  # and open[row]>close[row]):
            return -1
        elif (ratio2[row] > 1 and highdiff[row] < 0.2 * lowdiff[row] and bodydiff[
            row] > bodydiffmin):  # and open[row]<close[row]):
            return 1
        else:
            return 0


    def closeResistance(l, levels, lim):
        if len(levels) == 0:
            return 0
        c1 = abs(df.High[l] - min(levels, key=lambda x: abs(x - df.High[l]))) <= lim
        c2 = abs(max(df.Open[l], df.Close[l]) - min(levels, key=lambda x: abs(x - df.High[l]))) <= lim
        c3 = min(df.Open[l], df.Close[l]) < min(levels, key=lambda x: abs(x - df.High[l]))
        c4 = df.Low[l] < min(levels, key=lambda x: abs(x - df.High[l]))
        if ((c1 or c2) and c3 and c4):
            return 1
        else:
            return 0


    def closeSupport(l, levels, lim):
        if len(levels) == 0:
            return 0
        c1 = abs(df.Low[l] - min(levels, key=lambda x: abs(x - df.Low[l]))) <= lim
        c2 = abs(min(df.Open[l], df.Close[l]) - min(levels, key=lambda x: abs(x - df.Low[l]))) <= lim
        c3 = max(df.Open[l], df.Close[l]) > min(levels, key=lambda x: abs(x - df.Low[l]))
        c4 = df.High[l] > min(levels, key=lambda x: abs(x - df.Low[l]))
        if ((c1 or c2) and c3 and c4):
            return 1
        else:
            return 0

    for row in range(backCandles, len(df) - n2):
        ss = []
        rr = []
        for subrow in range(row - backCandles + n1, row + 1):
            if support(df, subrow, n1, n2):
                ss.append(df.Low[subrow])
            if resistance(df, subrow, n1, n2):
                rr.append(df.High[subrow])
        # !!!! parameters
        if ((isEngulfing(row) == -1 or isStar(row) == -1) and closeResistance(row, rr, 150e-5)):  # and df.RSI[row]<30
            signal[row] = -1
        elif ((isEngulfing(row) == 1 or isStar(row) == 1) and closeSupport(row, ss, 150e-5)):  # and df.RSI[row]>70
            signal[row] = 1
        else:
            signal[row] = 0

    df['Dec_candle'] = signal

    return df

def scalping_rsi(df, window_ema=200, window_rsi=3):
    df[f"EMA200"] = ta.trend.ema_indicator(df.Open, window=window_ema)
    df[f"RSI"] = ta.momentum.rsi(df.Open, window=window_rsi)
    df['ATR'] = ta.volatility.average_true_range(high=df.High, low=df.Low, close=df.Close, window=14)

    emasignal = [0] * len(df)
    backcandles = 8

    for row in range(backcandles - 1, len(df)):
        upt = 1
        dnt = 1
        for i in range(row - backcandles, row + 1):
            if df.High[row] >= df.EMA200[row]:
                dnt = 0
            if df.Low[row] <= df.EMA200[row]:
                upt = 0
        if upt == 1 and dnt == 1:
            # print("!!!!! check trend loop !!!!")
            emasignal[row] = 3
        elif upt == 1:
            emasignal[row] = 1
        elif dnt == 1:
            emasignal[row] = -1

    df['EMAsignal'] = emasignal

    TotSignal = [0] * len(df)
    for row in range(0, len(df)):
        TotSignal[row] = 0
        if df.EMAsignal[row] == -1 and df.RSI[row] >= 90:
            TotSignal[row] = -1
        if df.EMAsignal[row] == 1 and df.RSI[row] <= 10:
            TotSignal[row] = 1

    df['Dec_RSI'] = TotSignal

    return df

