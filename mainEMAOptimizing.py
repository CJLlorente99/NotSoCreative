import pandas as pd
from investorClass import Investor
from dataClass import DataManager, DataGetter
import datetime as dt
from ma import exponentialMovingAverage
from investorParamsClass import MAInvestorParams, GradientQuarter
from pandas.tseries.offsets import CDay
from pandas.tseries.holiday import USFederalHolidayCalendar
import numpy as np


figNum = 0


def main():
    global figNum
    # Create DataGetter instance
    dataGetter = DataGetter()

    # Trying to find data
    emaWindowValues = np.arange(2, 20, 4)
    lowerBoundGradientSellArray = np.arange(-30, 1, 15)
    upperBoundGradientSellArray = np.arange(1, 30, 15)
    lowerBoundSecondGradientSellArray = np.arange(-30, 5, 15)
    upperSecondGradientSellArray = np.arange(-5, 30, 15)
    lowerBoundGradientBuyArray = np.arange(-30, 1, 15)
    upperBoundGradientBuyArray = np.arange(1, 30, 15)
    lowerBoundSecondGradientBuyArray = np.arange(-5, 30, 15)
    upperSecondGradientBuyArray = np.arange(5, 30, 15)
    maxSellValues = [5000, 7500, 10000]
    maxBuyValues = [2500, 5000]

    # Run various experiments
    numExperiments = 5
    numDays = 10

    # Record data
    now = dt.datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
    name = "optimizationEMA" + now

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
        dataGetter.today += CDay(100, calendar=USFederalHolidayCalendar())

    summaryResults = pd.DataFrame(columns=["nOpt", "initDate", "lastDate", "percentageEMA", "meanPortfolioValueEMA"])
    optParams = []
    nOpt = 0
    nExp = 0
    nTotal = len(emaWindowValues) * len(lowerBoundGradientSellArray) * len(upperBoundGradientSellArray) * len(
        lowerBoundSecondGradientSellArray) * len(
        upperSecondGradientSellArray) * len(lowerBoundGradientBuyArray) * len(upperBoundGradientBuyArray) * len(
        lowerBoundSecondGradientBuyArray) * len(
        upperSecondGradientBuyArray) * len(maxSellValues) * len(maxBuyValues) * numExperiments
    for smaWindow in emaWindowValues:
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
                                                for j in range(numExperiments):
                                                    print(f'({nExp}/{nTotal})')
                                                    nExp += 1

                                                    initDate = initDates[str(j)]
                                                    # Load data
                                                    df = dfPastData[str(j)]

                                                    # Create data manager
                                                    dataManager = DataManager()
                                                    dataManager.pastStockValue = df.Open[-1]

                                                    # Create investor EMA
                                                    sellParams = GradientQuarter(lowerBoundGradientSell, upperBoundGradientSell, lowerBoundSecondGradientSell, upperSecondGradientSell)
                                                    buyParams = GradientQuarter(lowerBoundGradientBuy, upperBoundGradientBuy, lowerBoundSecondGradientBuy, upperSecondGradientBuy)
                                                    emaParams = MAInvestorParams(buyParams, sellParams, smaWindow, maxBuy, maxSell)
                                                    investorEMA = Investor(10000, listToday[str(j)], emaParams=emaParams)

                                                    # Run for loop as if days passed
                                                    for i in range(10):
                                                        # print()
                                                        todayData = dfTodayData[str(j) + str(i)]
                                                        df = dfUntilToday[str(j) + str(i)]

                                                        # Refresh data for today
                                                        dataManager.date = todayData.index[0]
                                                        dataManager.actualStockValue = todayData.Open.values[0]

                                                        # EMA try
                                                        emaResults = exponentialMovingAverage(df.Open, emaParams)
                                                        dataManager.ema = emaResults
                                                        investorEMA.broker(dataManager, 'ema')

                                                        # Refresh for next day
                                                        dataManager.pastStockValue = todayData.Open.values[0]
                                                    lastDate = listLastDates[str(j)]

                                                    # Calculate summary results
                                                    percentualGainEMA, meanPortfolioValueEMA = investorEMA.calculateMetrics()
                                                    # print("Percentual gain EMA {:.2f}%, mean portfolio value EMA {:.2f}$".format(percentualGainEMA,
                                                    #                                                                              meanPortfolioValueEMA))
                                                    results = pd.DataFrame(
                                                        {"nOpt": [nOpt], "initDate": [initDate.strftime("%d/%m/%Y")[0]], "lastDate": [lastDate.strftime("%d/%m/%Y")[0]],
                                                         "percentageEMA": [percentualGainEMA],  "meanPortfolioValueEMA": [meanPortfolioValueEMA]})
                                                    summaryResults = pd.concat([summaryResults, results], ignore_index=True)

                                                optParams = np.append(optParams, (str(nOpt) + ":" + str(emaParams)))
                                                nOpt += 1

                                summaryResults.to_csv("data/" + name + ".csv", index_label="experiment", mode="a")
                                with open("data/" + name + ".txt", "a") as f:
                                    for opt in optParams:
                                        f.write(opt + "\n")
                                    f.close()
                                summaryResults = pd.DataFrame(columns=["nOpt", "initDate", "lastDate", "percentageEMA", "meanPortfolioValueEMA"])
                                optParams = []


if __name__ == '__main__':
    main()
