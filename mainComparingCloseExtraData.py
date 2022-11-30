import pandas as pd
from investorClass import Investor
from dataClass import DataManager, DataGetter
import datetime as dt
from rsi import relativeStrengthIndex, plotRSIDecisionRules
from ma import simpleMovingAverage, exponentialMovingAverage, movingAverageConvergenceDivergence, plotSMADecisionRules, plotEMADecisionRules, plotMACDDecisionRules
from bb import bollingerBands, plotBBDecisionRules
from investorParamsClass import RSIInvestorParams, MAInvestorParams, MACDInvestorParams, BBInvestorParams, GradientQuarter
import plotly.graph_objects as go
from pandas.tseries.offsets import CDay
from pandas.tseries.holiday import USFederalHolidayCalendar


figNum = 0


def main():
    global figNum
    # Create DataGetter instance
    dataGetter = DataGetter()

    # Common data
    maxBuy = 2500
    maxSell = 10000

    # Run various experiments
    numExperiments = 2
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
        upperBound = 70
        lowerBound = 40
        rsiParams = RSIInvestorParams(upperBound, lowerBound, RSIwindow, maxBuy, maxSell)
        investorRSI = Investor(10000, dataGetter.today, rsiParams=rsiParams)

        # Create investor SMA
        SMAwindow = 5
        sellParams = GradientQuarter(-10, 10, -30, 0)
        buyParams = GradientQuarter(-10, 10, 0, 30)
        smaParams = MAInvestorParams(buyParams, sellParams, SMAwindow, maxBuy, maxSell)
        investorSMA = Investor(10000, dataGetter.today, smaParams=smaParams)

        # Create investor EMA
        EMAwindow = 5
        sellParams = GradientQuarter(-10, 10, -30, 0)
        buyParams = GradientQuarter(-10, 10, 0, 30)
        emaParams = MAInvestorParams(buyParams, sellParams, EMAwindow, maxBuy, maxSell)
        investorEMA = Investor(10000, dataGetter.today, emaParams=emaParams)

        # Create investor MACD
        macdFastWindow = 12
        macdSlowWindow = 26
        upperBound = 1
        lowerBound = -1
        macdParams = MACDInvestorParams(upperBound, lowerBound, macdFastWindow, macdSlowWindow, 9, maxBuy, maxSell)
        investorMACD = Investor(10000, dataGetter.today, macdParams=macdParams)

        # Create investor BB
        bbWindow = 5
        bbStdDev = 2
        lowerBound = 0.3
        upperBound = 0.7
        sellingSlope = 5000
        buyingSlope = 5000
        bbParams = BBInvestorParams(bbWindow, bbStdDev, lowerBound, upperBound, sellingSlope, buyingSlope, maxBuy, maxSell)
        investorBB = Investor(10000, dataGetter.today, bbParams=bbParams)

        # Variables to store data
        auxRsi = pd.DataFrame()
        auxSma = pd.DataFrame()
        auxEma = pd.DataFrame()
        auxMacd = pd.DataFrame()
        auxBb = pd.DataFrame()
        auxLoop = pd.DataFrame()
        # Run for loop as if days passed
        for i in range(10):
            # print()
            todayData = dataGetter.getToday()
            df = dataGetter.getUntilToday()

            # Add new predicted data into df
            # TODO

            # Refresh data for today
            dataManager.date = todayData.index[0]
            dataManager.actualStockValue = todayData.Open.values[0]

            # Save data into df for record
            # TODO add predicted value for the record
            aux = pd.DataFrame({'nExperiment': [j], 'date': [dataGetter.today], 'stockValueOpen': todayData.Open.values[0], 'stockValueClose': todayData.Close.values[0]})
            auxLoop = pd.concat([auxLoop, aux], ignore_index=True)

            # RSI try
            rsiResults = relativeStrengthIndex(df.Close, rsiParams)
            dataManager.rsi = rsiResults[-1]
            # print(f'RSI is {dataManager.rsi}')
            # TODO change Investor.broker behavior when the indicator is calculated with predicted data
            moneyToInvest, moneyToSell, investedMoney, nonInvestedMoney = investorRSI.broker(dataManager, 'rsi')
            aux = pd.DataFrame(
                {'rsi': [rsiResults[-1]], 'moneyToInvestRSI': [moneyToInvest], 'moneyToSellRSI': [moneyToSell],
                 'investedMoneyRSI': [investedMoney], 'nonInvestedMoneyRSI': [nonInvestedMoney]})
            auxRsi = pd.concat([auxRsi, aux], ignore_index=True)

            # SMA try
            smaResults = simpleMovingAverage(df.Close, smaParams)
            dataManager.sma = smaResults
            # TODO change Investor.broker behavior when the indicator is calculated with predicted data
            moneyToInvest, moneyToSell, investedMoney, nonInvestedMoney = investorSMA.broker(dataManager, 'sma')
            aux = pd.DataFrame(
                {'sma': [smaResults[-1]], 'moneyToInvestSMA': [moneyToInvest], 'moneyToSellSMA': [moneyToSell],
                 'investedMoneySMA': [investedMoney], 'nonInvestedMoneySMA': [nonInvestedMoney]})
            auxSma = pd.concat([auxSma, aux], ignore_index=True)

            # EMA try
            emaResults = exponentialMovingAverage(df.Close, emaParams)
            dataManager.ema = emaResults
            # TODO change Investor.broker behavior when the indicator is calculated with predicted data
            moneyToInvest, moneyToSell, investedMoney, nonInvestedMoney = investorEMA.broker(dataManager, 'ema')
            aux = pd.DataFrame(
                {'ema': [emaResults[-1]], 'moneyToInvestEMA': [moneyToInvest], 'moneyToSellEMA': [moneyToSell],
                 'investedMoneyEMA': [investedMoney], 'nonInvestedMoneyEMA': [nonInvestedMoney]})
            auxEma = pd.concat([auxEma, aux], ignore_index=True)

            # MACD try
            macdResults = movingAverageConvergenceDivergence(df.Close, macdParams)
            dataManager.macd = macdResults[-1]
            # TODO change Investor.broker behavior when the indicator is calculated with predicted data
            moneyToInvest, moneyToSell, investedMoney, nonInvestedMoney = investorMACD.broker(dataManager, 'macd')
            aux = pd.DataFrame(
                {'macd': [macdResults[-1]], 'moneyToInvestMACD': [moneyToInvest], 'moneyToSellMACD': [moneyToSell],
                 'investedMoneyMACD': [investedMoney], 'nonInvestedMoneyMACD': [nonInvestedMoney]})
            auxMacd = pd.concat([auxMacd, aux], ignore_index=True)

            # BB try
            bbResults = bollingerBands(df.Close, bbParams)
            dataManager.bb = bbResults[-1]
            # TODO change Investor.broker behavior when the indicator is calculated with predicted data
            moneyToInvest, moneyToSell, investedMoney, nonInvestedMoney = investorBB.broker(dataManager, 'bb')
            aux = pd.DataFrame(
                {'bb': [bbResults[-1]], 'moneyToInvestBB': [moneyToInvest], 'moneyToSellBB': [moneyToSell],
                 'investedMoneyBB': [investedMoney], 'nonInvestedMoneyBB': [nonInvestedMoney]})
            auxBb = pd.concat([auxBb, aux], ignore_index=True)

            # Refresh for next day
            dataManager.pastStockValue = todayData.Open.values[0]
            dataGetter.goNextDay()
        lastDate = pd.DatetimeIndex([(dataGetter.today - CDay(calendar=USFederalHolidayCalendar()))])
        # Reset day to have a different 2 weeks window
        dataGetter.today += CDay(5, calendar=USFederalHolidayCalendar())

        # Deal with experiment data
        aux = pd.concat([auxLoop, auxRsi, auxSma, auxEma, auxMacd, auxBb], axis=1)
        advancedData = pd.concat([advancedData, aux])

        # Calculate summary results
        percentualGainRSI, meanPortfolioValueRSI = investorRSI.calculateMetrics()
        percentualGainSMA, meanPortfolioValueSMA = investorSMA.calculateMetrics()
        percentualGainEMA, meanPortfolioValueEMA = investorEMA.calculateMetrics()
        percentualGainMACD, meanPortfolioValueMACD = investorMACD.calculateMetrics()
        percentualGainBB, meanPortfolioValueBB = investorBB.calculateMetrics()
        print("Percentual gain RSI {:.2f}%, mean portfolio value RSI {:.2f}$".format(percentualGainRSI,
                                                                                     meanPortfolioValueRSI))
        print("Percentual gain SMA {:.2f}%, mean portfolio value SMA {:.2f}$".format(percentualGainSMA,
                                                                                     meanPortfolioValueSMA))
        print("Percentual gain EMA {:.2f}%, mean portfolio value EMA {:.2f}$".format(percentualGainEMA,
                                                                                     meanPortfolioValueEMA))
        print("Percentual gain MACD {:.2f}%, mean portfolio value MACD {:.2f}$".format(percentualGainMACD,
                                                                                     meanPortfolioValueMACD))
        print("Percentual gain BB {:.2f}%, mean portfolio value BB {:.2f}$".format(percentualGainBB,
                                                                                       meanPortfolioValueBB))
        results = pd.DataFrame(
            {"initDate": [initDate.strftime("%d/%m/%Y")[0]], "lastDate": [lastDate.strftime("%d/%m/%Y")[0]],
             "percentageRSI": [percentualGainRSI], "percentageSMA": [percentualGainSMA],
             "percentageEMA": [percentualGainEMA], "percentageMACD": [percentualGainMACD],
             "meanPortfolioValueRSI": [meanPortfolioValueRSI], "meanPortfolioValueSMA": [meanPortfolioValueSMA],
             "meanPortfolioValueEMA": [meanPortfolioValueEMA], "meanPortfolioValueMACD": [meanPortfolioValueMACD]})
        summaryResults = pd.concat([summaryResults, results], ignore_index=True)

        # Plot the evolution per experiment
        investorRSI.plotEvolution(rsiResults, df, "RSI")
        investorSMA.plotEvolution(smaResults, df, "SMA")
        investorEMA.plotEvolution(emaResults, df, "EMA")
        investorMACD.plotEvolution(macdResults, df, "MACD")
        investorBB.plotEvolution(bbResults, df, "BB")

    now = dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    summaryResults.to_csv("data/" + now + ".csv", index_label="experiment")
    advancedData.to_csv("data/" + now + "_advancedData.csv", index_label="experiment")

    with open("data/" + now + ".txt", "w") as f:
        f.write(str(rsiParams) + "\n")
        f.write(str(smaParams) + "\n")
        f.write(str(emaParams) + "\n")
        f.write(str(macdParams) + "\n")
        f.write(str(bbParams))

    plotRSIDecisionRules(rsiParams)
    plotBBDecisionRules(bbParams)
    plotSMADecisionRules(smaParams)
    plotEMADecisionRules(emaParams)
    plotMACDDecisionRules(macdParams)

if __name__ == '__main__':
    main()