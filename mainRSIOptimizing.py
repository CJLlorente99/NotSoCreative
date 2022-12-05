import pandas as pd
from investorClass import Investor
from dataClass import DataManager, DataGetter
import datetime as dt
from rsi import relativeStrengthIndex
from investorParamsClass import RSIInvestorParams
from pandas.tseries.offsets import CDay
from pandas.tseries.holiday import USFederalHolidayCalendar
import numpy as np


def main():
    # Create DataGetter instance
    dataGetter = DataGetter()

    # Trying to find data
    rsiWindowValues = np.arange(3, 5, 1)
    upperBoundValues = np.arange(55, 65, 2)
    lowerBoundValues = np.arange(28, 32, 1)
    aValues = np.arange(0.2, 1, 0.2)
    bValues = np.arange(0.5, 2, 0.4)
    maxSellValues = [5000, 10000]
    maxBuyValues = [5000, 10000]

    # Run various experiments
    numExperiments = 5
    numDays = 10

    # Record data
    now = dt.datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
    name = "optimizationRSI" + now

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
        dataGetter.today += CDay(10, calendar=USFederalHolidayCalendar())

    summaryResults = pd.DataFrame()
    optParams = []
    nOpt = 0
    nExp = 0
    nTotal = len(rsiWindowValues)*len(upperBoundValues)*len(lowerBoundValues)*len(aValues)*len(bValues)*len(maxSellValues)*len(maxBuyValues)*numExperiments
    for rsiWindow in rsiWindowValues:
        for upperBound in upperBoundValues:
            for lowerBound in lowerBoundValues:
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

                                    # Create investor RSI
                                    rsiParams = RSIInvestorParams(upperBound, lowerBound, rsiWindow, maxBuy, maxSell, a, b)
                                    investorRSI = Investor(10000, listToday[str(j)], rsiParams=rsiParams)

                                    # Run for loop as if days passed
                                    for i in range(numDays):
                                        todayData = dfTodayData[str(j)+str(i)]
                                        df = dfUntilToday[str(j)+str(i)]

                                        # Refresh data for today
                                        dataManager.date = todayData.index[0]
                                        dataManager.actualStockValue = todayData.Open.values[0]

                                        # RSI try
                                        rsiResults = relativeStrengthIndex(df.Close, rsiParams)
                                        dataManager.rsi = rsiResults[-1]
                                        investorRSI.broker(dataManager, 'rsi')

                                        # Refresh for next day
                                        dataManager.pastStockValue = todayData.Open.values[0]
                                    lastDate = listLastDates[str(j)]

                                    # Calculate summary results
                                    percentualGainRSI, meanPortfolioValueRSI = investorRSI.calculateMetrics()
                                    # print("Percentual gain RSI {:.2f}%, mean portfolio value RSI {:.2f}$".format(percentualGainRSI,
                                    #                                                                              meanPortfolioValueRSI))
                                    results = pd.DataFrame(
                                        {"nOpt": [nOpt], "initDate": [initDate.strftime("%d/%m/%Y")[0]], "lastDate": [lastDate.strftime("%d/%m/%Y")[0]],
                                         "percentageRSI": [percentualGainRSI], "meanPortfolioValueRSI": [meanPortfolioValueRSI]})
                                    summaryResults = pd.concat([summaryResults, results], ignore_index=True)

                                optParams = np.append(optParams, (str(nOpt) + ":" + str(rsiParams)))

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
