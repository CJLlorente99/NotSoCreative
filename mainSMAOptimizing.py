import pandas as pd
from investorClass import Investor
from dataClass import DataManager, DataGetter
import datetime as dt
from ma import simpleMovingAverage, SMAInvestorParams, GradientQuarter
from pandas.tseries.offsets import CDay
from pandas.tseries.holiday import USFederalHolidayCalendar
import numpy as np


figNum = 0


def main():
    global figNum
    # Create DataGetter instance
    dataGetter = DataGetter()

    # Trying to find data
    smaWindowValues = np.arange(2, 20, 4)
    lowerBoundGradientSellArray = np.arange(-50, 1, 10)
    upperBoundGradientSellArray = np.arange(1, 50, 10)
    lowerBoundSecondGradientSellArray = np.arange(-50, 5, 10)
    upperSecondGradientSellArray = np.arange(-5, 50, 10)
    lowerBoundGradientBuyArray = np.arange(-50, 1, 10)
    upperBoundGradientBuyArray = np.arange(1, 50, 10)
    lowerBoundSecondGradientBuyArray = np.arange(-5, 30, 10)
    upperSecondGradientBuyArray = np.arange(5, 50, 10)
    maxSellValues = np.arange(1000, 10000, 3000)
    maxBuyValues = np.arange(1000, 10000, 3000)

    # Run various experiments
    numExperiments = 10
    numDays = 10

    # Record data
    now = dt.datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
    name = "optimizationSMA" + now

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
            dfUntilToday[str(j) + str(i)] = dataGetter.getUntilToday()
            dfTodayData[str(j) + str(i)] = dataGetter.getToday()
            dataGetter.goNextDay()
        listLastDates[str(j)] = pd.DatetimeIndex([(dataGetter.today - CDay(calendar=USFederalHolidayCalendar()))])
        # Reset day to have a different dataset for next experiment
        dataGetter.today += CDay(10, calendar=USFederalHolidayCalendar())

    summaryResults = pd.DataFrame()
    optParams = []
    nOpt = 0
    nExp = 0
    nTotal = len(smaWindowValues) * len(lowerBoundGradientSellArray) * len(upperBoundGradientSellArray) * len(
        lowerBoundSecondGradientSellArray) * len(
        upperSecondGradientSellArray) * len(lowerBoundGradientBuyArray) * len(upperBoundGradientBuyArray) * len(
        lowerBoundSecondGradientBuyArray) * len(
        upperSecondGradientBuyArray) * len(maxSellValues) * len(maxBuyValues) * numExperiments
    for smaWindow in smaWindowValues:
        for lowerBoundGradientSell in lowerBoundGradientSellArray:
            for upperBoundGradientSell in upperBoundGradientSellArray:
                for lowerBoundSecondGradientSell in lowerBoundSecondGradientSellArray:
                    for upperSecondGradientSell in upperSecondGradientSellArray:
                        for lowerBoundGradientBuy in lowerBoundGradientBuyArray:
                            for upperBoundGradientBuy in upperBoundGradientBuyArray:
                                for lowerBoundSecondGradientBuy in lowerBoundSecondGradientBuyArray:
                                    for upperSecondGradientBuy in upperSecondGradientBuyArray:
                                        for maxSell in maxSellValues:
                                            for maxBuy in maxBuyValues:
                                                print(f'({nExp}/{nTotal})')
                                                nExp += 1

                                                initDate = initDates[str(j)]
                                                # Load data
                                                df = dfPastData[str(j)]

                                                # Create data manager
                                                dataManager = DataManager()
                                                dataManager.pastStockValue = df.Open[-1]

                                                # Create investor SMA
                                                sellParams = GradientQuarter(lowerBoundGradientSell, upperBoundGradientSell, lowerBoundSecondGradientSell, upperSecondGradientSell)
                                                buyParams = GradientQuarter(lowerBoundGradientBuy, upperBoundGradientBuy, lowerBoundSecondGradientBuy, upperSecondGradientBuy)
                                                smaParams = SMAInvestorParams(buyParams, sellParams, smaWindow)
                                                investorSMA = Investor(10000, listToday[str(j)], maxBuy, maxSell, smaParams=smaParams)

                                                # Run for loop as if days passed
                                                for i in range(10):
                                                    # print()
                                                    todayData = dfTodayData[str(j) + str(i)]
                                                    df = dfUntilToday[str(j) + str(i)]

                                                    # Refresh data for today
                                                    dataManager.date = todayData.index[0]
                                                    dataManager.actualStockValue = todayData.Open.values[0]

                                                    # SMA try
                                                    smaResults = simpleMovingAverage(df.Open, smaWindow)
                                                    dataManager.sma = smaResults
                                                    investorSMA.broker(dataManager, 'sma')

                                                    # Refresh for next day
                                                    dataManager.pastStockValue = todayData.Open.values[0]
                                                lastDate = listLastDates[str(j)]

                                                # Calculate summary results
                                                percentualGainSMA, meanPortfolioValueSMA = investorSMA.calculateMetrics()
                                                # print("Percentual gain SMA {:.2f}%, mean portfolio value SMA {:.2f}$".format(percentualGainSMA,
                                                #                                                                              meanPortfolioValueSMA))
                                                results = pd.DataFrame(
                                                    {"nOpt": [nOpt], "initDate": [initDate.strftime("%d/%m/%Y")[0]], "lastDate": [lastDate.strftime("%d/%m/%Y")[0]],
                                                     "percentageSMA": [percentualGainSMA],  "meanPortfolioValueSMA": [meanPortfolioValueSMA]})
                                                summaryResults = pd.concat([summaryResults, results], ignore_index=True)

                                            optParams = np.append(optParams, (
                                                        str(nOpt) + ":" + str(smaParams) + "," + str(
                                                    maxSell) + "," + str(maxBuy)))

                                            nOpt += 1

                                summaryResults.to_csv("data/" + name + ".csv", index_label="experiment", mode="a")
                                with open("data/" + name + ".txt", "a") as f:
                                    for opt in optParams:
                                        f.write(opt + "\n")
                                    f.close()
                                summaryResults = pd.DataFrame()
                                optParams = []


if __name__ == '__main__':
    main()