import pandas as pd
from classes.investorClass import Investor
from classes.dataClass import DataManager, DataGetter
import datetime as dt
from TAIndicators.bb import bollingerBands
from classes.investorParamsClass import BBInvestorParams
from pandas.tseries.offsets import CDay
from pandas.tseries.holiday import USFederalHolidayCalendar
import numpy as np


def main():
    # Create DataGetter instance
    dataGetter = DataGetter()

    # Trying to find data
    windowValues = np.arange(9, 11, 1)
    stdDevValues = np.arange(1.5, 2.5, 0.5)
    upperBoundValues = np.arange(0.4, 1.2, 0.2)
    lowerBoundValues = np.arange(1.3, 2, 0.2)
    aValues = np.arange(1.5, 2.5, 0.3)
    bValues = np.arange(0.5, 1.5, 0.5)
    maxSellValues = [10000]
    maxBuyValues = [10000]

    # Run various experiments
    numExperiments = 5
    numDays = 10

    # Record data
    now = dt.datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
    name = "optimizationBB" + now

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

    summaryResults = pd.DataFrame(columns=["nOpt", "initDate", "lastDate", "percentageBB", "meanPortfolioValueBB"])
    optParams = []
    nOpt = 0
    nExp = 0
    nTotal = len(windowValues) * len(stdDevValues) * len(upperBoundValues) * len(lowerBoundValues) * len(
        aValues) * len(bValues) * len(
        maxSellValues) * len(maxBuyValues) * numExperiments
    for window in windowValues:
        for stdDev in stdDevValues:
            for upperBound in upperBoundValues:
                for lowerBound in lowerBoundValues:
                    for a in aValues:
                        for b in bValues:
                            for maxSell in maxSellValues:
                                for maxBuy in maxBuyValues:
                                    for j in range(numExperiments):
                                        print(f'({nExp+1}/{nTotal})')
                                        nExp += 1

                                        initDate = initDates[str(j)]
                                        # Load data
                                        df = dfPastData[str(j)]

                                        # Create data manager
                                        dataManager = DataManager()
                                        dataManager.pastStockValue = df.Open[-1]

                                        # Create investor BB
                                        bbParams = BBInvestorParams(window, stdDev, lowerBound, upperBound, maxBuy, maxSell, a, b)
                                        investorBB = Investor(10000, listToday[str(j)], bbParams=bbParams)

                                        # Run for loop as if days passed
                                        for i in range(numDays):
                                            todayData = dfTodayData[str(j) + str(i)]
                                            df = dfUntilToday[str(j) + str(i)]

                                            # Refresh data for today
                                            dataManager.date = todayData.index[0]
                                            dataManager.actualStockValue = todayData.Open.values[0]

                                            # BB try
                                            bbResults = bollingerBands(df.Close, bbParams)
                                            dataManager.bb = bbResults["pband"][-1]
                                            investorBB.broker(dataManager, 'bb')

                                            # Refresh for next day
                                            dataManager.pastStockValue = todayData.Open.values[0]
                                        lastDate = listLastDates[str(j)]

                                        # Calculate summary results
                                        percentualGainBB, meanPortfolioValueBB = investorBB.calculateMetrics()
                                        # print("Percentual gain BB {:.2f}%, mean portfolio value BB {:.2f}$".format(
                                        #     percentualGainBB,
                                        #     meanPortfolioValueBB))
                                        results = pd.DataFrame(
                                            {"nOpt": [nOpt], "initDate": [initDate.strftime("%d/%m/%Y")[0]],
                                             "lastDate": [lastDate.strftime("%d/%m/%Y")[0]],
                                             "percentageBB": [percentualGainBB],
                                             "meanPortfolioValueBB": [meanPortfolioValueBB]})
                                        summaryResults = pd.concat([summaryResults, results], ignore_index=True)

                                    optParams = np.append(optParams, (
                                                str(nOpt) + ":" + str(bbParams)))

                                    nOpt += 1

                summaryResults.to_csv("data/" + name + ".csv", index_label="experiment", mode="a")
                with open("data/" + name + ".txt", "a") as f:
                    for opt in optParams:
                        f.write(opt + "\n")
                    f.close()
                summaryResults = pd.DataFrame(columns=["nOpt", "initDate", "lastDate", "percentageBB", "meanPortfolioValueBB"])
                optParams = []


if __name__ == '__main__':
    main()
