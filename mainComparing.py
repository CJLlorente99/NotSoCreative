import pandas as pd
from classes.dataClass import DataGetter
from TAIndicators.rsi import InvestorRSI
from TAIndicators.ma import InvestorMACD
from TAIndicators.bb import InvestorBB
from DecisionFunction.investorNN import InvestorBBNN, InvestorBBRSINNClass, InvestorBBRSINN
from DecisionFunction.investorDecisionTree import InvestorDecisionTree
from LSTM.investorLSTMThreshold import InvestorLSTMThreshold, InvestorLSTMProb
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
        experimentManager.addStrategy(investorRSI, "rsi", [experimentManager.createTIInput("rsi", rsiParams, "rsi", 1)])
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
                                       experimentManager.createTIInput("macd", macdParamsGrad, "signal", 5)])
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
                                       experimentManager.createTIInput("macd", macdParamsZero, "signal", 5)])
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
                                       experimentManager.createTIInput("macd", macdParamsSignal, "signal", 5)])
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
        experimentManager.addStrategy(investorBB, "bb", [experimentManager.createTIInput("bb", bbParams, "pband", 1)])
        print("investorBB created")

        # Create investor BBNN
        bbWindow = 10
        bbStdDev = 1.5
        lowerBound = 1.9
        upperBound = 0.8
        a = 2.4
        b = 0.5
        bbParams = BBInvestorParams(bbWindow, bbStdDev, lowerBound, upperBound, a, b)
        file = "data/modelnn2BB2022_12_20_11_27_08.h5"
        nnParams = NNInvestorParams(file)
        investorBBNN = InvestorBBNN(10000, nnParams)
        experimentManager.addStrategy(investorBBNN, "bbnn", [experimentManager.createTIInput("bb", bbParams, "pband", 2)])
        print("investorBBNN created")

        # Create investor BB+RSI NN Class
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
        file = "data/modelnn2BB_2RSIClass2022_12_20_11_27_08.h5"
        nnParams = NNInvestorParams(file)
        investorBBRSINNClass = InvestorBBRSINNClass(10000, nnParams)
        experimentManager.addStrategy(investorBBRSINNClass, "bbRsiNNClass",
                                      [experimentManager.createTIInput("bb", bbParams, "pband", 2),
                                       experimentManager.createTIInput("rsi", rsiParams, "rsi", 2)])
        print("investorBBRSINNClass created")

        # Create investor BB+RSI NN
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
        file = "data/modelnn2BB_2RSI2022_12_20_11_27_08.h5"
        nnParams = NNInvestorParams(file)
        investorBBRSINN = InvestorBBRSINN(10000, nnParams)
        experimentManager.addStrategy(investorBBRSINN, "bbRsiNN",
                                      [experimentManager.createTIInput("bb", bbParams, "pband", 2),
                                       experimentManager.createTIInput("rsi", rsiParams, "rsi", 2)])
        print("investorBBRSINN created")

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
                                       experimentManager.createTIInput("stochrsi", stochParams, "stochrsi", 1)])
        print("investorDT created")

        # Create investor based on LSTM Threshold
        file = "../data/modellstm.h5"
        lstmParams = LSTMInvestorParams(file, 0.05)
        investorLSTMThreshold = InvestorLSTMThreshold(10000, lstmParams)
        experimentManager.addStrategy(investorLSTMThreshold, "lstmThreshold", [experimentManager.createTIInput("lstm")], False)
        print("investorLSTMThreshold created")

        # Create investor based on LSTM Prob
        file = "../data/modellstm.h5"
        lstmParams = LSTMInvestorParams(file, 0.05)
        investorLSTMProb = InvestorLSTMProb(10000, lstmParams)
        experimentManager.addStrategy(investorLSTMProb, "lstmProb", [experimentManager.createTIInput("lstm")], True)
        print("investorLSTMProb created")

        # Create investor Random
        investorRandom = InvestorRandom(10000)
        experimentManager.addStrategy(investorRandom, "random")
        print("investorRandom created")

        # Create investor BIA
        investorBIA = InvestorBIA(10000)
        experimentManager.addStrategy(investorBIA, "bia")
        print("investorBIA created")

        # Create investor WIA
        investorWIA = InvestorWIA(10000)
        experimentManager.addStrategy(investorWIA, "wia")
        print("investorWIA created")

        # Create investor CA
        investorCA = InvestorCA(10000, 0.1)
        experimentManager.addStrategy(investorCA, "ca")
        print("investorCA created")

        # Create investor BaH
        investorBaH = InvestorBaH(10000)
        experimentManager.addStrategy(investorBaH, "bah")
        print("investorBaH created")

        # Create investor Idle
        investorIdle = InvestorIdle(10000)
        experimentManager.addStrategy(investorIdle, "idle")
        print("investorIdle created")

        auxLoop = pd.DataFrame()
        # Run for loop as if days passed
        pastStockValue = df.Open[-1]
        for i in range(nDays+1):
            todayData = dataGetter.getToday()
            df = dataGetter.getUntilToday()

            experimentManager.runDay(todayData, df, dataGetter.getNextNextDay(), dataGetter.getNextDay(), pastStockValue, j, i)
            # Save data into df for record
            aux = pd.DataFrame({'nExperiment': [j], 'date': [dataGetter.today], 'stockValueOpen': todayData.Open.values[0], 'stockValueClose': todayData.Close.values[0]})
            auxLoop = pd.concat([auxLoop, aux], ignore_index=True)

            # Refresh for next day
            pastStockValue = todayData.Open.values[0]
            dataGetter.goNextDay()
        # To compensate the last goNextDay()
        lastDate = pd.DatetimeIndex([(dataGetter.today - CDay(calendar=USFederalHolidayCalendar()))])
        # Reset day to have a different 2 weeks window
        dataGetter.today += CDay(10, calendar=USFederalHolidayCalendar())

        # Deal with experiment data
        aux = pd.concat([auxLoop, experimentManager.returnExpData()], axis=1)
        advancedData = pd.concat([advancedData, aux])

        # Calculate summary results
        firstDate = investorRSI.record.index.values[0]
        letzteDate = investorRSI.record.index.values[-1]

        dfTestCriteria = pd.concat([dfTestCriteria, experimentManager.criteriaCalculationAndPlotting(initDate, lastDate, firstDate, letzteDate, j)])

        # Plot the evolution per experiment
        experimentManager.plotEvolution(df)

    # Plot summary of test criteria
    experimentManager.summaryCriteriaCalculatorAndPlotting(dfTestCriteria)


if __name__ == '__main__':
    main()
