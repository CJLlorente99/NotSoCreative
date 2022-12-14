import pandas as pd
from classes.dataClass import DataManager, DataGetter
from TAIndicators.rsi import relativeStrengthIndex, InvestorRSI
from TAIndicators.ma import movingAverageConvergenceDivergence, InvestorMACD
from TAIndicators.bb import bollingerBands, InvestorBB
from classes.investorParamsClass import RSIInvestorParams, MACDInvestorParams, BBInvestorParams, GradientQuarter
from Benchmarks.randomBenchmark import InvestorRandom
from Benchmarks.bia import InvestorBIA
from Benchmarks.wia import InvestorWIA
from Benchmarks.costAverage import InvestorCA
from Benchmarks.bah import InvestorBaH
from Benchmarks.idle import InvestorIdle
from pandas.tseries.offsets import CDay
from pandas.tseries.holiday import USFederalHolidayCalendar
from classes.testCriteriaClass import testCriteriaClass
import datetime as dt


def main():
    # Create DataGetter instance
    dataGetter = DataGetter()

    # Run various experiments
    numExperiments = 2
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

        # Create investor Random
        investorRandom = InvestorRandom(10000)

        # Create investor BIA
        investorBIA = InvestorBIA(10000)

        # Create investor WIA
        investorWIA = InvestorWIA(10000)

        # Create investor CA
        investorCA = InvestorCA(10000, 1100)

        # Create investor BaH
        investorBaH = InvestorBaH(10000)

        # Create investor Idle
        investorIdle = InvestorIdle(10000)

        # Variables to store data
        auxRsi = pd.DataFrame()
        auxMacdGrad = pd.DataFrame()
        auxMacdZero = pd.DataFrame()
        auxMacdSignal = pd.DataFrame()
        auxBb = pd.DataFrame()
        auxRandom = pd.DataFrame()
        auxBIA = pd.DataFrame()
        auxWIA = pd.DataFrame()
        auxCA = pd.DataFrame()
        auxBaH = pd.DataFrame()
        auxIdle = pd.DataFrame()
        auxLoop = pd.DataFrame()

        # Run for loop as if days passed
        for i in range(nDays):
            dataManager.nDay = i
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

            # Random try
            aux = investorRandom.broker(dataManager)
            auxRandom = pd.concat([auxRandom, aux], ignore_index=True)

            # BIA try
            dataManager.nextnextStockValueOpen = dataGetter.getNextNextDay().Open.values[0]
            dataManager.nextStockValueOpen = dataGetter.getNextDay().Open.values[0]
            aux = investorBIA.broker(dataManager)
            auxBIA = pd.concat([auxBIA, aux], ignore_index=True)

            # WIA try
            aux = investorWIA.broker(dataManager)
            auxWIA = pd.concat([auxWIA, aux], ignore_index=True)

            # CA try
            aux = investorCA.broker(dataManager)
            auxCA = pd.concat([auxCA, aux], ignore_index=True)

            # BaH try
            aux = investorBaH.broker(dataManager)
            auxBaH = pd.concat([auxBaH, aux], ignore_index=True)

            # Idle try
            aux = investorIdle.broker(dataManager)
            auxIdle = pd.concat([auxIdle, aux], ignore_index=True)

            # Refresh for next day
            dataManager.pastStockValue = todayData.Open.values[0]
            dataGetter.goNextDay()
        # To compensate the last goNextDay()
        lastDate = pd.DatetimeIndex([(dataGetter.today - CDay(calendar=USFederalHolidayCalendar()))])
        # Reset day to have a different 2 weeks window
        dataGetter.today += CDay(50, calendar=USFederalHolidayCalendar())

        # Deal with experiment data
        aux = pd.concat([auxLoop, auxRsi, auxMacdGrad, auxMacdZero, auxMacdSignal, auxBb, auxRandom], axis=1)
        advancedData = pd.concat([advancedData, aux])

        # Calculate summary results
        firstDate = investorRSI.record.index.values[0]
        letzteDate = investorRSI.record.index.values[-1]
        criteriaCalculator = testCriteriaClass(firstDate, letzteDate)
        testCriteriaRSI = pd.DataFrame(criteriaCalculator.calculateCriteria("rsi", investorRSI.record), index=[j])
        testCriteriaMACDGrad = pd.DataFrame(criteriaCalculator.calculateCriteria("macdGrad", investorMACDGrad.record), index=[j])
        testCriteriaMACDZero = pd.DataFrame(criteriaCalculator.calculateCriteria("macdZero", investorMACDZero.record), index=[j])
        testCriteriaMACDSignal = pd.DataFrame(criteriaCalculator.calculateCriteria("macdSignal", investorMACDSignal.record), index=[j])
        testCriteriaBB = pd.DataFrame(criteriaCalculator.calculateCriteria("bb", investorBB.record), index=[j])
        testCriteriaRandom = pd.DataFrame(criteriaCalculator.calculateCriteria("random", investorRandom.record), index=[j])
        testCriteriaBIA = pd.DataFrame(criteriaCalculator.calculateCriteria("bia", investorBIA.record),
                                          index=[j])
        testCriteriaWIA = pd.DataFrame(criteriaCalculator.calculateCriteria("wia", investorWIA.record),
                                          index=[j])
        testCriteriaCA = pd.DataFrame(criteriaCalculator.calculateCriteria("ca", investorCA.record),
                                       index=[j])
        testCriteriaBaH = pd.DataFrame(criteriaCalculator.calculateCriteria("bah", investorBaH.record),
                                      index=[j])
        testCriteriaIdle = pd.DataFrame(criteriaCalculator.calculateCriteria("idle", investorIdle.record),
                                       index=[j])
        dfTestCriteriaAux = pd.concat(
            [testCriteriaRSI, testCriteriaMACDGrad, testCriteriaMACDZero, testCriteriaMACDSignal, testCriteriaBB,
             testCriteriaRandom, testCriteriaBIA, testCriteriaWIA, testCriteriaCA, testCriteriaBaH, testCriteriaIdle])

        # Plot test criteria
        title = "Test criteria (" + initDate.strftime("%Y/%m/%d")[0] + "-" + lastDate.strftime("%Y/%m/%d")[0] + ")"
        criteriaCalculator.plotCriteria(dfTestCriteriaAux, title)

        dfTestCriteria = pd.concat([dfTestCriteria, dfTestCriteriaAux])

        # Plot the evolution per experiment
        # investorRSI.plotEvolution(rsiResults, df)
        # investorMACDGrad.plotEvolution(macdResults, df)
        # investorMACDZero.plotEvolution(macdResults, df)
        # investorMACDSignal.plotEvolution(macdResults, df)
        # investorBB.plotEvolution(bbResults, df)
        investorRandom.plotEvolution(None, df)
        investorBIA.plotEvolution(None, df)
        investorWIA.plotEvolution(None, df)
        investorCA.plotEvolution(None, df)
        investorBaH.plotEvolution(None, df)
        investorIdle.plotEvolution(None, df)

    # Plot summary of test criteria
    result = criteriaCalculator.calculateCriteriaVariousExperiments(dfTestCriteria)
    title = "Summary of the test criteria"
    criteriaCalculator.plotCriteriaVariousExperiments(result, title)

    # Push the data into files for later inspection
    # now = dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # dfTestCriteria.to_csv("data/" + now + ".csv", index_label="n")
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
