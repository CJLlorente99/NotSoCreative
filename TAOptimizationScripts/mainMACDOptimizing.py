import pandas as pd
from classes.investorClass import Investor
from classes.dataClass import DataManager, DataGetter
import datetime as dt
from TAIndicators.ma import movingAverageConvergenceDivergence
from classes.investorParamsClass import MACDInvestorParams, GradientQuarter
from pandas.tseries.offsets import CDay
from pandas.tseries.holiday import USFederalHolidayCalendar
import numpy as np


def main():
    # Create DataGetter instance
    dataGetter = DataGetter()

    # Trying to find data
    lowerBoundGradientSellArray = np.arange(-200, 0, 50)
    upperBoundGradientSellArray = np.arange(0, 200, 50)
    lowerBoundSecondGradientSellArray = np.arange(-200, 0, 50)
    upperSecondGradientSellArray = [0]
    sellNum = len(lowerBoundGradientSellArray) * len(upperBoundGradientSellArray) * len(
        lowerBoundSecondGradientSellArray) * len(upperSecondGradientSellArray)

    lowerBoundGradientBuyArray = np.arange(-200, 0, 50)
    upperBoundGradientBuyArray = np.arange(0, 200, 50)
    lowerBoundSecondGradientBuyArray = np.arange(0, 200, 50)
    upperSecondGradientBuyArray = [0]
    buyNum = len(lowerBoundGradientBuyArray) * len(upperBoundGradientBuyArray) * len(
        lowerBoundSecondGradientBuyArray) * len(upperSecondGradientBuyArray)

    fastWindowValues = np.arange(2, 8, 3)
    slowWindowValues = np.arange(6, 12, 3)
    signalValues = np.arange(5, 9, 2)
    windowNum = len(fastWindowValues)*len(slowWindowValues)*len(signalValues)

    aValues = np.arange(0.1, 1.3, 0.6)
    bValues = np.arange(0.5, 4.5, 2)
    tanNum = len(aValues)*len(bValues)

    maxSellValues = [10000]
    maxBuyValues = [10000]
    maxNum = len(maxSellValues)*len(maxBuyValues)

    # Run various experiments
    numExperiments = 3
    numDays = 10

    # Record data
    now = dt.datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
    name = "optimizationMACD" + now

    # Load the data for the experiments, so it doesn't have to be loaded every time
    initDates = {}
    dfPastData = {}
    listToday = {}
    listLastDates = {}
    dfUntilToday = {}
    dfTodayData = {}
    for j in range(numExperiments):
        print(f'numExperiment {j}')
        initDates[str(j)] = pd.DatetimeIndex([dataGetter.today])
        dfPastData[str(j)] = dataGetter.getPastData()
        listToday[str(j)] = dataGetter.today
        for i in range(numDays):
            print(f'numDay {i}')
            dfUntilToday[str(j)+str(i)] = dataGetter.getUntilToday()
            dfTodayData[str(j)+str(i)] = dataGetter.getToday()
            dataGetter.goNextDay()
        listLastDates[str(j)] = pd.DatetimeIndex([(dataGetter.today - CDay(calendar=USFederalHolidayCalendar()))])
        # Reset day to have a different dataset for next experiment
        dataGetter.today += CDay(100, calendar=USFederalHolidayCalendar())

    summaryResults = pd.DataFrame()
    optParams = []
    nOpt = 0
    nExp = 0
    nTotal = sellNum*buyNum*windowNum*tanNum*maxNum*numExperiments
    for lowerBoundGradientSell in lowerBoundGradientSellArray:
        for lowerBoundSecondGradientSell in lowerBoundSecondGradientSellArray:
            for upperBoundGradientSell in upperBoundGradientSellArray:
                for upperSecondGradientSell in upperSecondGradientSellArray:
                    for lowerBoundGradientBuy in lowerBoundGradientBuyArray:
                        for lowerBoundSecondGradientBuy in lowerBoundSecondGradientBuyArray:
                            for upperBoundGradientBuy in upperBoundGradientBuyArray:
                                for upperSecondGradientBuy in upperSecondGradientBuyArray:
                                    for slowWindow in slowWindowValues:
                                        for fastWindow in fastWindowValues:
                                            for signal in signalValues:
                                                for a in aValues:
                                                    for b in bValues:
                                                        for maxSell in maxSellValues:
                                                            for maxBuy in maxBuyValues:
                                                                for j in range(numExperiments):
                                                                    print(f'({nExp}/{nTotal})')
                                                                    nExp += 1

                                                                    initDate = initDates[str(j)]
                                                                    # Load data
                                                                    df = dfPastData[str(j)]

                                                                    # Create data manager
                                                                    dataManager = DataManager()
                                                                    dataManager.pastStockValue = df.Open[-1]

                                                                    # Create investor MACD
                                                                    sellGradient = GradientQuarter(lowerBoundGradientSell, upperBoundGradientSell, lowerBoundSecondGradientSell, upperSecondGradientSell)
                                                                    buyGradient = GradientQuarter(lowerBoundGradientBuy, upperBoundGradientBuy, lowerBoundSecondGradientBuy, upperSecondGradientBuy)
                                                                    macdParamsGrad = MACDInvestorParams(buyGradient,
                                                                                                    sellGradient,
                                                                                                    fastWindow,
                                                                                                    slowWindow, signal,
                                                                                                    maxBuy, maxSell, a,
                                                                                                    b, "grad")
                                                                    investorMACDGrad = Investor(10000,
                                                                                                listToday[str(j)],
                                                                                                macdParams=macdParamsGrad)
                                                                    macdParamsZero = MACDInvestorParams(buyGradient,
                                                                                                    sellGradient,
                                                                                                    fastWindow,
                                                                                                    slowWindow, signal,
                                                                                                    maxBuy, maxSell, a,
                                                                                                    b, "grad_crossZero")
                                                                    investorMACDGradZero = Investor(10000,
                                                                                                listToday[str(j)],
                                                                                                macdParams=macdParamsZero)
                                                                    macdParamsSignal = MACDInvestorParams(buyGradient,
                                                                                                    sellGradient,
                                                                                                    fastWindow,
                                                                                                    slowWindow, signal,
                                                                                                    maxBuy, maxSell, a,
                                                                                                    b, "grad_crossSignal")
                                                                    investorMACDGradSignal = Investor(10000,
                                                                                                listToday[str(j)],
                                                                                                macdParams=macdParamsSignal)

                                                                    # Run for loop as if days passed
                                                                    for i in range(numDays):
                                                                        todayData = dfTodayData[str(j)+str(i)]
                                                                        df = dfUntilToday[str(j)+str(i)]

                                                                        # Refresh data for today
                                                                        dataManager.date = todayData.index[0]
                                                                        dataManager.actualStockValue = todayData.Open.values[0]

                                                                        # MACD try
                                                                        macdResults = movingAverageConvergenceDivergence(df.Close, macdParamsGrad)
                                                                        dataManager.macd = macdResults
                                                                        investorMACDGrad.broker(dataManager, 'macd')
                                                                        investorMACDGradZero.broker(dataManager, 'macd')
                                                                        investorMACDGradSignal.broker(dataManager, 'macd')

                                                                        # Refresh for next day
                                                                        dataManager.pastStockValue = todayData.Open.values[0]
                                                                    lastDate = listLastDates[str(j)]

                                                                    # Calculate summary results
                                                                    percentualGainMACDGrad, meanPortfolioValueMACDGrad = investorMACDGrad.calculateMetrics()
                                                                    percentualGainMACDZero, meanPortfolioValueMACDZero = investorMACDGradZero.calculateMetrics()
                                                                    percentualGainMACDSignal, meanPortfolioValueMACDSignal = investorMACDGradSignal.calculateMetrics()
                                                                    # print("Percentual gain MACD {:.2f}%, mean portfolio value MACD {:.2f}$".format(percentualGainMACD,
                                                                    #                                                                              meanPortfolioValueMACD))
                                                                    results = pd.DataFrame(
                                                                        {"nOpt": [nOpt],
                                                                         "initDate": [initDate.strftime("%d/%m/%Y")[0]],
                                                                         "lastDate": [lastDate.strftime("%d/%m/%Y")[0]],
                                                                         "percentageMACDGrad": [percentualGainMACDGrad],
                                                                         "meanPortfolioValueMACDGrad": [
                                                                             meanPortfolioValueMACDGrad],
                                                                         "percentageMACDZero": [percentualGainMACDZero],
                                                                         "meanPortfolioValueMACDZero": [
                                                                             meanPortfolioValueMACDZero],
                                                                         "percentageMACDSignal": [percentualGainMACDSignal],
                                                                         "meanPortfolioValueMACDSignal": [
                                                                             meanPortfolioValueMACDSignal]})
                                                                    summaryResults = pd.concat([summaryResults, results], ignore_index=True)

                                                                optParams = np.append(optParams, (
                                                                            str(nOpt) + ":" + str(macdParamsGrad)))
                                                                optParams = np.append(optParams, (
                                                                            str(nOpt) + ":" + str(macdParamsZero)))
                                                                optParams = np.append(optParams, (
                                                                            str(nOpt) + ":" + str(macdParamsSignal)))

                                                                nOpt += 1

                                                        summaryResults.to_csv("data/" + name + ".csv", index_label="experiment", mode="a")
                                                        with open("data/" + name + ".txt", "a") as f:
                                                            for opt in optParams:
                                                                f.write(opt + "\n")
                                                            f.close()
                                                        summaryResults = pd.DataFrame(
                                                            columns=["nOpt", "initDate", "lastDate",
                                                                     "percentageMACDGrad", "meanPortfolioValueMACDGrad",
                                                                     "percentageMACDZero", "meanPortfolioValueMACDZero",
                                                                     "percentageMACDSignal",
                                                                     "meanPortfolioValueMACDSignal"])
                                                        optParams = []


if __name__ == '__main__':
    main()
