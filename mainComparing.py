import pandas as pd
from investorClass import Investor
from dataClass import DataManager, DataGetter
import datetime as dt
from rsi import normalRSI, RSIInvestorParams
from ma import simpleMovingAverage, SMAInvestorParams, GradientQuarter, exponentialMovingAverage, EMAInvestorParams, movingAverageConvergenceDivergence, MACDInvestorParams
from matplotlib import pyplot as plt
from pandas.tseries.offsets import CDay
from pandas.tseries.holiday import USFederalHolidayCalendar


figNum = 0


def main():
    global figNum
    # Create DataGetter instance
    dataGetter = DataGetter()

    # Run various experiments
    numExperiments = 10
    summaryResults = pd.DataFrame()
    advancedData = pd.DataFrame()
    for j in range(numExperiments):
        print(f'------------------------------------------\n'
              f'------------EXPERIMENT {j}------------------\n'
              f'------------------------------------------')
        initDate = pd.DatetimeIndex([dataGetter.today])
        # Load data
        df = dataGetter.getPastData()

        # Create data manager
        dataManager = DataManager()
        dataManager.pastStockValue = df.Open[-1]

        # Create investor RSI
        RSIwindow = 5
        rsiParams = RSIInvestorParams(70, 40, RSIwindow)
        investorRSI = Investor(10000, dataGetter.today, 2500, 10000, rsiParams=rsiParams)

        # Create investor SMA
        SMAwindow = 5
        sellParams = GradientQuarter(-10, 10, -30, 0)
        buyParams = GradientQuarter(-10, 10, 0, 30)
        smaParams = SMAInvestorParams(buyParams, sellParams, SMAwindow)
        investorSMA = Investor(10000, dataGetter.today, 2500, 10000, smaParams=smaParams)

        # Create investor EMA
        EMAwindow = 5
        sellParams = GradientQuarter(-10, 10, -30, 0)
        buyParams = GradientQuarter(-10, 10, 0, 30)
        emaParams = EMAInvestorParams(buyParams, sellParams, EMAwindow)
        investorEMA = Investor(10000, dataGetter.today, 2500, 10000, emaParams=emaParams)

        # Create investor MACD
        macdFastWindow = 12
        macdSlowWindow = 26
        macdParams = MACDInvestorParams(30, -30, macdFastWindow, macdSlowWindow, 9)
        investorMACD = Investor(10000, dataGetter.today, 2500, 10000, macdParams=macdParams)

        # Variables to store data
        auxRsi = pd.DataFrame()
        auxSma = pd.DataFrame()
        auxEma = pd.DataFrame()
        auxMacd = pd.DataFrame()
        auxLoop = pd.DataFrame()
        # Run for loop as if days passed
        for i in range(10):
            # print()
            todayData = dataGetter.getToday()
            df = dataGetter.getUntilToday()

            # Refresh data for today
            dataManager.date = todayData.index[0]
            dataManager.actualStockValue = todayData.Open.values[0]

            # Save data into df for record
            aux = pd.DataFrame({'nExperiment': [j], 'date': [dataGetter.today], 'stockValue': todayData.Open.values[0]})
            auxLoop = pd.concat([auxLoop, aux], ignore_index=True)

            # RSI try
            rsiResults = normalRSI(df.Open, RSIwindow)
            dataManager.rsi = rsiResults[-1]
            # print(f'RSI is {dataManager.rsi}')
            moneyToInvest, moneyToSell, investedMoney, nonInvestedMoney = investorRSI.broker(dataManager, 'rsi')
            aux = pd.DataFrame(
                {'rsi': [rsiResults[-1]], 'moneyToInvestRSI': [moneyToInvest], 'moneyToSellRSI': [moneyToSell],
                 'investedMoneyRSI': [investedMoney], 'nonInvestedMoneyRSI': [nonInvestedMoney]})
            auxRsi = pd.concat([auxRsi, aux], ignore_index=True)

            # SMA try
            smaResults = simpleMovingAverage(df.Open, SMAwindow)
            dataManager.sma = smaResults
            moneyToInvest, moneyToSell, investedMoney, nonInvestedMoney = investorSMA.broker(dataManager, 'sma')
            aux = pd.DataFrame(
                {'sma': [smaResults[-1]], 'moneyToInvestSMA': [moneyToInvest], 'moneyToSellSMA': [moneyToSell],
                 'investedMoneySMA': [investedMoney], 'nonInvestedMoneySMA': [nonInvestedMoney]})
            auxSma = pd.concat([auxSma, aux], ignore_index=True)

            # EMA try
            emaResults = exponentialMovingAverage(df.Open, EMAwindow)
            dataManager.ema = emaResults
            moneyToInvest, moneyToSell, investedMoney, nonInvestedMoney = investorEMA.broker(dataManager, 'ema')
            aux = pd.DataFrame(
                {'ema': [emaResults[-1]], 'moneyToInvestEMA': [moneyToInvest], 'moneyToSellEMA': [moneyToSell],
                 'investedMoneyEMA': [investedMoney], 'nonInvestedMoneyEMA': [nonInvestedMoney]})
            auxEma = pd.concat([auxEma, aux], ignore_index=True)

            # MACD try
            macdResults = movingAverageConvergenceDivergence(df.Open, macdSlowWindow, macdFastWindow)
            dataManager.macd = macdResults[-1]
            moneyToInvest, moneyToSell, investedMoney, nonInvestedMoney = investorMACD.broker(dataManager, 'macd')
            aux = pd.DataFrame(
                {'macd': [macdResults[-1]], 'moneyToInvestMACD': [moneyToInvest], 'moneyToSellMACD': [moneyToSell],
                 'investedMoneyMACD': [investedMoney], 'nonInvestedMoneyMACD': [nonInvestedMoney]})
            auxMacd = pd.concat([auxMacd, aux], ignore_index=True)

            # Refresh for next day
            dataManager.pastStockValue = todayData.Open.values[0]
            dataGetter.goNextDay()
        lastDate = pd.DatetimeIndex([(dataGetter.today - CDay(calendar=USFederalHolidayCalendar()))])
        # Reset day to have a different 2 weeks window
        dataGetter.today += CDay(5, calendar=USFederalHolidayCalendar())

        # Deal with experiment data
        aux = pd.concat([auxLoop, auxRsi, auxSma, auxEma, auxMacd], axis=1)
        advancedData = pd.concat([advancedData, aux])

        # Calculate summary results
        percentualGainRSI, meanPortfolioValueRSI = investorRSI.calculateMetrics()
        percentualGainSMA, meanPortfolioValueSMA = investorSMA.calculateMetrics()
        percentualGainEMA, meanPortfolioValueEMA = investorEMA.calculateMetrics()
        percentualGainMACD, meanPortfolioValueMACD = investorMACD.calculateMetrics()
        print("Percentual gain RSI {:.2f}%, mean portfolio value RSI {:.2f}$".format(percentualGainRSI,
                                                                                     meanPortfolioValueRSI))
        print("Percentual gain SMA {:.2f}%, mean portfolio value SMA {:.2f}$".format(percentualGainSMA,
                                                                                     meanPortfolioValueSMA))
        print("Percentual gain EMA {:.2f}%, mean portfolio value EMA {:.2f}$".format(percentualGainEMA,
                                                                                     meanPortfolioValueEMA))
        print("Percentual gain MACD {:.2f}%, mean portfolio value MACD {:.2f}$".format(percentualGainMACD,
                                                                                     meanPortfolioValueMACD))
        results = pd.DataFrame(
            {"initDate": [initDate.strftime("%d/%m/%Y")[0]], "lastDate": [lastDate.strftime("%d/%m/%Y")[0]],
             "percentageRSI": [percentualGainRSI], "percentageSMA": [percentualGainSMA],
             "percentageEMA": [percentualGainEMA], "percentageMACD": [percentualGainMACD],
             "meanPortfolioValueRSI": [meanPortfolioValueRSI], "meanPortfolioValueSMA": [meanPortfolioValueSMA],
             "meanPortfolioValueEMA": [meanPortfolioValueEMA], "meanPortfolioValueMACD": [meanPortfolioValueMACD]})
        summaryResults = pd.concat([summaryResults, results], ignore_index=True)

    now = dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    summaryResults.to_csv("data/" + now + ".csv", index_label="experiment")
    advancedData.to_csv("data/" + now + "_advancedData.csv", index_label="experiment")

    with open("data/" + now + ".txt", "w") as f:
        f.write(str(rsiParams) + "\n")
        f.write(str(smaParams) + "\n")
        f.write(str(emaParams) + "\n")
        f.write(str(macdParams))


if __name__ == '__main__':
    main()