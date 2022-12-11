import pandas as pd
from classes.dataClass import DataManager, DataGetter
from TAIndicators.rsi import relativeStrengthIndex, InvestorRSI
from TAIndicators.ma import movingAverageConvergenceDivergence, InvestorMACD
from TAIndicators.bb import bollingerBands, InvestorBB
from classes.investorParamsClass import RSIInvestorParams, MACDInvestorParams, BBInvestorParams, GradientQuarter
from pandas.tseries.offsets import CDay
from pandas.tseries.holiday import USFederalHolidayCalendar
from classes.testCriteriaClass import testCriteriaClass
import datetime as dt


def main():
    # Create DataGetter instance
    dataGetter = DataGetter()

    # Run various experiments
    numExperiments = 5
    nDays = 10
    advancedData = pd.DataFrame()
    dfTestCriteria = pd.DataFrame()
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
        RSIwindow = 3
        upperBound = 61
        lowerBound = 27.5
        maxBuy = 10000
        maxSell = 10000
        a = 1.1
        b = 2.4
        rsiParams = RSIInvestorParams(upperBound, lowerBound, RSIwindow, maxBuy, maxSell, a, b)
        investorRSI = InvestorRSI(10000, rsiParams)

        # Create investor MACD grad
        sellGradient = GradientQuarter(-50, 150, 0, 0)
        buyGradient = GradientQuarter(-200, 150, -150, 0)
        macdFastWindow = 2
        macdSlowWindow = 6
        signal = 7
        maxBuy = 10000
        maxSell = 10000
        a = 0.7
        b = 2.5
        macdParamsGrad = MACDInvestorParams(sellGradient, buyGradient, macdFastWindow, macdSlowWindow, signal, maxBuy,
                                        maxSell, a, b, "grad")
        investorMACDGrad = InvestorMACD(10000, macdParamsGrad)

        # Create investor MACD zero
        sellGradient = GradientQuarter(-50, 0, 150, 0)
        buyGradient = GradientQuarter(-100, 100, -200, 0)
        macdFastWindow = 2
        macdSlowWindow = 9
        signal = 7
        maxBuy = 10000
        maxSell = 10000
        a = 0.7
        b = 2.5
        macdParamsZero = MACDInvestorParams(sellGradient, buyGradient, macdFastWindow, macdSlowWindow, signal, maxBuy,
                                        maxSell, a, b, "grad_crossZero")
        investorMACDZero = InvestorMACD(10000, macdParamsZero)

        # Create investor MACD signal
        sellGradient = GradientQuarter(-150, 150, -200, 0)
        buyGradient = GradientQuarter(-200, 0, 100, 0)
        macdFastWindow = 2
        macdSlowWindow = 6
        signal = 5
        maxBuy = 10000
        maxSell = 10000
        a = 0.7
        b = 2.5
        macdParamsSignal = MACDInvestorParams(sellGradient, buyGradient, macdFastWindow, macdSlowWindow, signal, maxBuy,
                                        maxSell, a, b, "grad_crossSignal")
        investorMACDSignal = InvestorMACD(10000, macdParamsSignal)

        # Create investor BB
        bbWindow = 10
        bbStdDev = 1.5
        lowerBound = 1.9
        upperBound = 0.8
        maxBuy = 10000
        maxSell = 10000
        a = 2.4
        b = 0.5
        bbParams = BBInvestorParams(bbWindow, bbStdDev, lowerBound, upperBound, maxBuy, maxSell, a, b)
        investorBB = InvestorBB(10000, bbParams)

        # Variables to store data
        auxRsi = pd.DataFrame()
        auxMacdGrad = pd.DataFrame()
        auxMacdZero = pd.DataFrame()
        auxMacdSignal = pd.DataFrame()
        auxBb = pd.DataFrame()
        auxLoop = pd.DataFrame()

        # Run for loop as if days passed
        for i in range(nDays):
            # print()
            todayData = dataGetter.getToday()
            df = dataGetter.getUntilToday()

            # Refresh data for today
            dataManager.date = todayData.index[0]
            dataManager.actualStockValue = todayData.Open.values[0]

            # Save data into df for record
            aux = pd.DataFrame({'nExperiment': [j], 'date': [dataGetter.today], 'stockValueOpen': todayData.Open.values[0], 'stockValueClose': todayData.Close.values[0]})
            auxLoop = pd.concat([auxLoop, aux], ignore_index=True)

            # RSI try
            rsiResults = relativeStrengthIndex(df.Close, rsiParams)
            dataManager.rsi = rsiResults[-1]
            aux = investorRSI.broker(dataManager)
            auxRsi = pd.concat([auxRsi, aux], ignore_index=True)

            # MACD Grad try
            macdResults = movingAverageConvergenceDivergence(df.Close, macdParamsGrad)
            dataManager.macd = macdResults
            aux = investorMACDGrad.broker(dataManager)
            auxMacdGrad = pd.concat([auxMacdGrad, aux], ignore_index=True)

            # MACD Zero try
            macdResults = movingAverageConvergenceDivergence(df.Close, macdParamsZero)
            dataManager.macd = macdResults
            aux = investorMACDZero.broker(dataManager)
            auxMacdZero = pd.concat([auxMacdZero, aux], ignore_index=True)

            # MACD Signal try
            macdResults = movingAverageConvergenceDivergence(df.Close, macdParamsSignal)
            dataManager.macd = macdResults
            aux = investorMACDSignal.broker(dataManager)
            auxMacdSignal = pd.concat([auxMacdSignal, aux], ignore_index=True)

            # BB try
            bbResults = bollingerBands(df.Close, bbParams)
            dataManager.bb = bbResults["pband"][-1]
            aux = investorBB.broker(dataManager)
            auxBb = pd.concat([auxBb, aux], ignore_index=True)

            # Refresh for next day
            dataManager.pastStockValue = todayData.Open.values[0]
            dataGetter.goNextDay()
        # To compensate the last goNextDay()
        lastDate = pd.DatetimeIndex([(dataGetter.today - CDay(calendar=USFederalHolidayCalendar()))])
        # Reset day to have a different 2 weeks window
        dataGetter.today += CDay(50, calendar=USFederalHolidayCalendar())

        # Deal with experiment data
        aux = pd.concat([auxLoop, auxRsi, auxMacdGrad, auxMacdZero, auxMacdSignal, auxBb], axis=1)
        advancedData = pd.concat([advancedData, aux])

        # Calculate summary results
        testCriteriaRSI = pd.DataFrame(testCriteriaClass.calculateCriteria("rsi", investorRSI.record), index=[j])
        testCriteriaMACDGrad = pd.DataFrame(testCriteriaClass.calculateCriteria("macdGrad", investorMACDGrad.record), index=[j])
        testCriteriaMACDZero = pd.DataFrame(testCriteriaClass.calculateCriteria("macdZero", investorMACDZero.record), index=[j])
        testCriteriaMACDSignal = pd.DataFrame(testCriteriaClass.calculateCriteria("macdSignal", investorMACDSignal.record), index=[j])
        testCriteriaBB = pd.DataFrame(testCriteriaClass.calculateCriteria("bb", investorBB.record), index=[j])
        dfTestCriteriaAux = pd.concat(
            [testCriteriaRSI, testCriteriaMACDGrad, testCriteriaMACDZero, testCriteriaMACDSignal, testCriteriaBB])

        # Plot test criteria
        title = "Test criteria (" + initDate.strftime("%Y/%m/%d")[0] + "-" + lastDate.strftime("%Y/%m/%d")[0] + ")"
        testCriteriaClass.plotCriteria(dfTestCriteriaAux, title)

        dfTestCriteria = pd.concat([dfTestCriteria, dfTestCriteriaAux])

        # Plot the evolution per experiment
        # investorRSI.plotEvolution(rsiResults, df)
        # investorMACDGrad.plotEvolution(macdResults, df)
        # investorMACDZero.plotEvolution(macdResults, df)
        # investorMACDSignal.plotEvolution(macdResults, df)
        # investorBB.plotEvolution(bbResults, df)

    # Plot summary of test criteria
    result = testCriteriaClass.calculateCriteriaVariousExperiments(dfTestCriteria)
    title = "Summary of the test criteria"
    testCriteriaClass.plotCriteriaVariousExperiments(result, title)

    # Push the data into files for later inspection
    now = dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    dfTestCriteria.to_csv("data/" + now + ".csv", index_label="n")
    # advancedData.to_csv("data/" + now + "_advancedData.csv", index_label="n")
    #
    # with open("data/" + now + ".txt", "w") as f:
    #     f.write(str(rsiParams) + "\n")
    #     f.write(str(macdParamsGrad) + "\n")
    #     f.write(str(macdParamsZero) + "\n")
    #     f.write(str(macdParamsSignal) + "\n")
    #     f.write(str(bbParams))

    # Show the decision rules with the parameters used
    # investorRSI.plotDecisionRules()
    # investorBB.plotDecisionRules()
    # investorMACDGrad.plotDecisionRules()
    # investorMACDZero.plotDecisionRules()
    # investorMACDSignal.plotDecisionRules()


if __name__ == '__main__':
    main()
