import datetime
import os.path
import pandas as pd
from classes.dataClass import DataGetter
from TAIndicators.rsi import InvestorRSI
from TAIndicators.ma import InvestorMACD
from TAIndicators.bb import InvestorBB
from DecisionFunction.investorDecisionTree import InvestorDecisionTree
from LSTM.investorLSTMThreshold import InvestorLSTMThreshold, InvestorLSTMProb
from LSTM.investorLSTMConfidence import InvestorLSTMConfidenceClass
from classes.investorParamsClass import RSIInvestorParams, MACDInvestorParams, BBInvestorParams, GradientQuarter, NNInvestorParams, DTInvestorParams, ADXInvestorParams, ADIInvestorParams, AroonInvestorParams, OBVInvestorParams, StochasticRSIInvestorParams, ATRInvestorParams, LSTMInvestorParams
from Benchmarks.randomBenchmark import InvestorRandom
from Benchmarks.bia import InvestorBIA
from Benchmarks.wia import InvestorWIA
from Benchmarks.costAverage import InvestorCA
from Benchmarks.bah import InvestorBaH
from Benchmarks.idle import InvestorIdle
from pandas.tseries.offsets import CDay
from pandas.tseries.holiday import USFederalHolidayCalendar
from experimentManager import ExperimentManager


def main():
    # Create DataGetter instance
    name = "Charli"
    # name = "Paul"
    # name = "Tobias"
    # name = "Sanchita"
    # name = "Rishabh"
    # name = "Kim"
    dataGetter = DataGetter(name=name)

    # Run various experiments
    numExperiments = 3
    nDays = 10
    dfTestCriteria = pd.DataFrame()

    for j in range(numExperiments):
        print(f'------------------------------------------\n'
              f'------------EXPERIMENT {j}------------------\n'
              f'------------------------------------------')
        experimentManager = ExperimentManager()

        initDate = pd.DatetimeIndex([dataGetter.today])
        # Load data
        df = dataGetter.getPastData()

        # Create investor RSI
        RSIwindow = 3
        upperBound = 61
        lowerBound = 27.5
        a = 1.1
        b = 2.4
        rsiParams = RSIInvestorParams(upperBound, lowerBound, RSIwindow, a, b)
        investorRSI = InvestorRSI(10000, rsiParams)
        experimentManager.addStrategy(investorRSI, "rsi", [experimentManager.createTIInput("rsi", rsiParams, "rsi", 1)], False)
        print("investorRSI created")

        # Create investor MACD grad
        sellGradient = GradientQuarter(-50, 150, 0, 0)
        buyGradient = GradientQuarter(-200, 150, -150, 0)
        macdFastWindow = 2
        macdSlowWindow = 6
        signal = 7
        a = 0.7
        b = 2.5
        macdParamsGrad = MACDInvestorParams(sellGradient, buyGradient, macdFastWindow, macdSlowWindow, signal,
                                            a, b, "grad")
        investorMACDGrad = InvestorMACD(10000, macdParamsGrad)
        experimentManager.addStrategy(investorMACDGrad, "macdGrad",
                                      [experimentManager.createTIInput("macd", macdParamsGrad, "macd", 5),
                                       experimentManager.createTIInput("macd", macdParamsGrad, "signal", 5)], False)
        print("investorMACDGrad created")

        # Create investor MACD zero
        sellGradient = GradientQuarter(-50, 0, 150, 0)
        buyGradient = GradientQuarter(-100, 100, -200, 0)
        macdFastWindow = 2
        macdSlowWindow = 9
        signal = 7
        a = 0.7
        b = 2.5
        macdParamsZero = MACDInvestorParams(sellGradient, buyGradient, macdFastWindow, macdSlowWindow, signal,
                                            a, b, "grad_crossZero")
        investorMACDZero = InvestorMACD(10000, macdParamsZero)
        experimentManager.addStrategy(investorMACDZero, "macdGradZero",
                                      [experimentManager.createTIInput("macd", macdParamsZero, "macd", 5),
                                       experimentManager.createTIInput("macd", macdParamsZero, "signal", 5)], False)
        print("investorMACDZero created")

        # Create investor MACD signal
        sellGradient = GradientQuarter(-150, 150, -200, 0)
        buyGradient = GradientQuarter(-200, 0, 100, 0)
        macdFastWindow = 2
        macdSlowWindow = 6
        signal = 5
        a = 0.7
        b = 2.5
        macdParamsSignal = MACDInvestorParams(sellGradient, buyGradient, macdFastWindow, macdSlowWindow, signal,
                                        a, b, "grad_crossSignal")
        investorMACDSignal = InvestorMACD(10000, macdParamsSignal)
        experimentManager.addStrategy(investorMACDSignal, "macdGradSignal",
                                      [experimentManager.createTIInput("macd", macdParamsSignal, "macd", 5),
                                       experimentManager.createTIInput("macd", macdParamsSignal, "signal", 5)], False)
        print("investorMACDSignal created")

        # Create investor BB
        bbWindow = 10
        bbStdDev = 1.5
        lowerBound = 1.9
        upperBound = 0.8
        a = 2.4
        b = 0.5
        bbParams = BBInvestorParams(bbWindow, bbStdDev, lowerBound, upperBound, a, b)
        investorBB = InvestorBB(10000, bbParams)
        experimentManager.addStrategy(investorBB, "bb", [experimentManager.createTIInput("bb", bbParams, "pband", 1)], False)
        print("investorBB created")

        # Create investor based on DT
        bbWindow = 10
        bbStdDev = 1.5
        lowerBound = 1.9
        upperBound = 0.8
        a = 2.4
        b = 0.5
        bbParams = BBInvestorParams(bbWindow, bbStdDev, lowerBound, upperBound, a, b)
        RSIwindow = 3
        upperBound = 61
        lowerBound = 27.5
        a = 1.1
        b = 2.4
        rsiParams = RSIInvestorParams(upperBound, lowerBound, RSIwindow, a, b)
        adxParams = ADXInvestorParams(14)
        aroonParams = AroonInvestorParams(25)
        atrParams = ATRInvestorParams(5)
        stochParams = StochasticRSIInvestorParams(14, 3, 3)
        file = "data/dt"
        dtParams = DTInvestorParams(file, ["rsirsi", "bbpband", "adiacc_dist_index", "adxadx", "aroonaroon_indicator"
                                           , "atraverage_true_range", "obvon_balance_volume", "stochrsistochrsi"])
        investorDT = InvestorDecisionTree(10000, dtParams)
        experimentManager.addStrategy(investorDT, "DT",
                                      [experimentManager.createTIInput("rsi", rsiParams, "rsi", 1),
                                       experimentManager.createTIInput("bb", bbParams, "pband", 1),
                                       experimentManager.createTIInput("adi", None, "acc_dist_index", 1),
                                       experimentManager.createTIInput("adx", adxParams, "adx", 1),
                                       experimentManager.createTIInput("aroon", aroonParams, "aroon_indicator", 1),
                                       experimentManager.createTIInput("atr", atrParams, "average_true_range", 1),
                                       experimentManager.createTIInput("obv", None, "on_balance_volume", 1),
                                       experimentManager.createTIInput("stochrsi", stochParams, "stochrsi", 1)], True)
        print("investorDT created")

        # Create investor based on LSTM Threshold
        file = "../data/modellstm.h5"
        lstmParams = LSTMInvestorParams(file, 0.05)
        investorLSTMThreshold = InvestorLSTMThreshold(10000, lstmParams)
        experimentManager.addStrategy(investorLSTMThreshold, "lstmThreshold", [experimentManager.createTIInput("lstm")], True)
        print("investorLSTMThreshold created")

        # Create investor based on LSTM Prob
        file = "../data/modellstm.h5"
        lstmParams = LSTMInvestorParams(file, 0.05)
        investorLSTMProb = InvestorLSTMProb(10000, lstmParams)
        experimentManager.addStrategy(investorLSTMProb, "lstmProb", [experimentManager.createTIInput("lstm")], True)
        print("investorLSTMProb created")

        # Create investor based on class voting (only sell and buy everything)
        investorLSTMConfidenceClass = InvestorLSTMConfidenceClass(10000, 2)
        experimentManager.addStrategy(investorLSTMConfidenceClass, "lstmConfidenceEvery",
                                      [experimentManager.createTIInput("df")], True)

        # Create investor Random
        investorRandom = InvestorRandom(10000)
        experimentManager.addStrategy(investorRandom, "random", plotEvolution=True)
        print("investorRandom created")

        # Create investor BIA
        investorBIA = InvestorBIA(10000)
        experimentManager.addStrategy(investorBIA, "bia", plotEvolution=True)
        print("investorBIA created")

        # Create investor WIA
        investorWIA = InvestorWIA(10000)
        experimentManager.addStrategy(investorWIA, "wia", plotEvolution=True)
        print("investorWIA created")

        # Create investor CA
        investorCA = InvestorCA(10000, 0.05)
        experimentManager.addStrategy(investorCA, "ca", plotEvolution=True)
        print("investorCA created")

        # Create investor BaH
        investorBaH = InvestorBaH(10000)
        experimentManager.addStrategy(investorBaH, "bah", plotEvolution=True)
        print("investorBaH created")

        # Create investor Idle
        investorIdle = InvestorIdle(10000)
        experimentManager.addStrategy(investorIdle, "idle", plotEvolution=True)
        print("investorIdle created")

        auxLoop = pd.DataFrame()
        # Run for loop as if days passed
        pastStockValue = df.Close[0]
        nDay = 1
        for i in range(nDays+1):
            todayData = dataGetter.getToday()
            df = dataGetter.getUntilToday()

            # Run morning operation
            experimentManager.runMorning(todayData, df, dataGetter.getNextNextDay(), dataGetter.getNextDay(), pastStockValue, j, nDay)
            nDay += 1
            # Save data into df for record
            dataTag = dataGetter.today.combine(dataGetter.today, datetime.time(9, 30))
            aux = pd.DataFrame({'nExperiment': [j], 'date': [dataTag], 'stockValueOpen': todayData.Open.values[0], 'stockValueClose': todayData.Close.values[0]})
            auxLoop = pd.concat([auxLoop, aux], ignore_index=True)


            # Run afternoon operation
            experimentManager.runAfternoon(todayData, df, dataGetter.getNextNextDay(), dataGetter.getNextDay(),
                                         pastStockValue, j, nDay)
            nDay += 1
            # Save data into df for record
            dataTag = dataGetter.today.combine(dataGetter.today, datetime.time(16, 00))
            aux = pd.DataFrame(
                {'nExperiment': [j], 'date': [dataTag], 'stockValueOpen': todayData.Open.values[0],
                 'stockValueClose': todayData.Close.values[0]})
            auxLoop = pd.concat([auxLoop, aux], ignore_index=True)

            # Refresh for next day
            pastStockValue = todayData.Close.values[0]
            dataGetter.goNextDay()
        # To compensate the last goNextDay()
        lastDate = pd.DatetimeIndex([(dataGetter.today - CDay(calendar=USFederalHolidayCalendar()))])
        # Reset day to have a different 2 weeks window
        dataGetter.today += CDay(10, calendar=USFederalHolidayCalendar())

        # Deal with experiment data
        aux = pd.concat([auxLoop, experimentManager.returnExpData()], axis=1)
        aux.to_csv("images/advancedData.csv", index_label="nExperiment", mode="a")

        # Calculate summary results
        firstDate = investorRSI.record.index.values[0]
        letzteDate = investorRSI.record.index.values[-1]

        dfTestCriteria = pd.concat([dfTestCriteria, experimentManager.criteriaCalculationAndPlotting(initDate, lastDate, firstDate, letzteDate, j)])

        # Plot the evolution per experiment
        experimentManager.plotEvolution(df)

    # Plot summary of test criteria
    experimentManager.summaryCriteriaCalculatorAndPlotting(dfTestCriteria)


if __name__ == '__main__':
    if not os.path.exists("images"):
        os.mkdir("images")
    main()
