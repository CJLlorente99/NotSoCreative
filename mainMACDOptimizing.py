import pandas as pd
from investorClass import Investor
from dataClass import DataManager, DataGetter
import datetime as dt
from ma import movingAverageConvergenceDivergence
from investorParamsClass import MACDInvestorParams, GradientQuarter
from pandas.tseries.offsets import CDay
from pandas.tseries.holiday import USFederalHolidayCalendar
import numpy as np


figNum = 0


def main():
    global figNum
    # Create DataGetter instance
    dataGetter = DataGetter()

    # Trying to find data
    fastWindowValues = np.arange(2, 16, 2)
    slowWindowValues = np.arange(2, 30, 2)
    upperBoundValues = np.arange(-1, 2, 0.1)
    lowerBoundValues = np.arange(-2, 1, 0.1)
    maxSellValues = [5000, 7500, 10000]
    maxBuyValues = [2500, 5000]

    # Run various experiments
    numExperiments = 5
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

    summaryResults = pd.DataFrame(columns=["nOpt", "initDate", "lastDate", "percentageMACD", "meanPortfolioValueMACD"])
    optParams = []
    nOpt = 0
    nExp = 0
    nTotal = len(slowWindowValues)*len(fastWindowValues)*len(upperBoundValues)*len(lowerBoundValues)*len(maxSellValues)*len(maxBuyValues)*numExperiments
    for slowWindow in slowWindowValues:
        for fastWindow in fastWindowValues:
            for upperBound in upperBoundValues:
                for lowerBound in lowerBoundValues:
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
                                macdParams = MACDInvestorParams(upperBound, lowerBound, fastWindow, slowWindow, 9, maxBuy, maxSell)
                                investorMACD = Investor(10000, listToday[str(j)], macdParams=macdParams)

                                # Run for loop as if days passed
                                for i in range(numDays):
                                    todayData = dfTodayData[str(j)+str(i)]
                                    df = dfUntilToday[str(j)+str(i)]

                                    # Refresh data for today
                                    dataManager.date = todayData.index[0]
                                    dataManager.actualStockValue = todayData.Open.values[0]

                                    # MACD try
                                    macdResults = movingAverageConvergenceDivergence(df.Open, macdParams)
                                    dataManager.macd = macdResults[-1]
                                    investorMACD.broker(dataManager, 'macd')

                                    # Refresh for next day
                                    dataManager.pastStockValue = todayData.Open.values[0]
                                lastDate = listLastDates[str(j)]

                                # Calculate summary results
                                percentualGainMACD, meanPortfolioValueMACD = investorMACD.calculateMetrics()
                                # print("Percentual gain MACD {:.2f}%, mean portfolio value MACD {:.2f}$".format(percentualGainMACD,
                                #                                                                              meanPortfolioValueMACD))
                                results = pd.DataFrame(
                                    {"nOpt": [nOpt], "initDate": [initDate.strftime("%d/%m/%Y")[0]], "lastDate": [lastDate.strftime("%d/%m/%Y")[0]],
                                     "percentageMACD": [percentualGainMACD], "meanPortfolioValueMACD": [meanPortfolioValueMACD]})
                                summaryResults = pd.concat([summaryResults, results], ignore_index=True)

                            optParams = np.append(optParams, (str(nOpt) + ":" + str(macdParams)))

                            nOpt += 1

                    summaryResults.to_csv("data/" + name + ".csv", index_label="experiment", mode="a")
                    with open("data/" + name + ".txt", "a") as f:
                        for opt in optParams:
                            f.write(opt + "\n")
                        f.close()
                    summaryResults = pd.DataFrame(columns=["nOpt", "initDate", "lastDate", "percentageMACD", "meanPortfolioValueMACD"])
                    optParams = []


if __name__ == '__main__':
    main()