import datetime
import os.path
import pandas as pd
from classes.dataClass import DataGetter
from TAIndicators.rsi import InvestorRSI
from TAIndicators.ma import InvestorMACD
from TAIndicators.bb import InvestorBB
from DecisionFunction.investorDecisionTree import InvestorDecisionTree
from RF_DT.investorRandomForestClassifier import InvestorRandomForestClassifier
from RF_DT.investorXGB import InvestorXGB
from RF_DT.investorXGBShift import InvestorXGBWindow
from RF_DT.investorXGBReduced import InvestorXGBReduced
from LSTM.investorLSTMEnsemble import InvestorLSTMEnsembleClass1, InvestorLSTMEnsembleClass2
from LSTM.investorLSTMWindowStandardScaler import InvestorLSTMWindowStandardScalerT1, InvestorLSTMWindowStandardScalerT2
from LSTM.investorLSTMWindowMinMaxScaler import InvestorLSTMWindowMinMaxT1, InvestorLSTMWindowMinMaxT2
from LSTM.investorLSTMWindowRobustMinMaxScaler import InvestorLSTMWindowRobustMinMaxT2, InvestorLSTMWindowRobustMinMaxT1
from classes.investorParamsClass import RSIInvestorParams, MACDInvestorParams, BBInvestorParams, GradientQuarter, NNInvestorParams, DTInvestorParams, ADXInvestorParams, ADIInvestorParams, AroonInvestorParams, OBVInvestorParams, StochasticRSIInvestorParams, ATRInvestorParams, LSTMInvestorParams
from Benchmarks.randomBenchmark import InvestorRandom
from Benchmarks.bia import InvestorBIA
from Benchmarks.wia import InvestorWIA
from Benchmarks.costAverage import InvestorCA
from Benchmarks.bah import InvestorBaH
from Benchmarks.idle import InvestorIdle
from pandas.tseries.offsets import CDay
from pandas.tseries.holiday import USFederalHolidayCalendar
from classes.experimentManager import ExperimentManager
import warnings


def main():
    # Create DataGetter instance
    dataGetter = DataGetter('2021-01-01', '2021-01-30')

    # Run various experiments
    numExperiments = 15
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

        # # Create investor RSI
        # RSIwindow = 3
        # upperBound = 61
        # lowerBound = 27.5
        # a = 1.1
        # b = 2.4
        # rsiParams = RSIInvestorParams(upperBound, lowerBound, RSIwindow, a, b)
        # investorRSI = InvestorRSI(10000, rsiParams)
        # experimentManager.addStrategy(investorRSI, "rsi", [experimentManager.createTIInput("rsi", rsiParams, "rsi", 1)], False)
        # print("investorRSI created")
        #
        # # Create investor MACD grad
        # sellGradient = GradientQuarter(-50, 150, 0, 0)
        # buyGradient = GradientQuarter(-200, 150, -150, 0)
        # macdFastWindow = 2
        # macdSlowWindow = 6
        # signal = 7
        # a = 0.7
        # b = 2.5
        # macdParamsGrad = MACDInvestorParams(sellGradient, buyGradient, macdFastWindow, macdSlowWindow, signal,
        #                                     a, b, "grad")
        # investorMACDGrad = InvestorMACD(10000, macdParamsGrad)
        # experimentManager.addStrategy(investorMACDGrad, "macdGrad",
        #                               [experimentManager.createTIInput("macd", macdParamsGrad, "macd", 5),
        #                                experimentManager.createTIInput("macd", macdParamsGrad, "signal", 5)], False)
        # print("investorMACDGrad created")
        #
        # # Create investor MACD zero
        # sellGradient = GradientQuarter(-50, 0, 150, 0)
        # buyGradient = GradientQuarter(-100, 100, -200, 0)
        # macdFastWindow = 2
        # macdSlowWindow = 9
        # signal = 7
        # a = 0.7
        # b = 2.5
        # macdParamsZero = MACDInvestorParams(sellGradient, buyGradient, macdFastWindow, macdSlowWindow, signal,
        #                                     a, b, "grad_crossZero")
        # investorMACDZero = InvestorMACD(10000, macdParamsZero)
        # experimentManager.addStrategy(investorMACDZero, "macdGradZero",
        #                               [experimentManager.createTIInput("macd", macdParamsZero, "macd", 5),
        #                                experimentManager.createTIInput("macd", macdParamsZero, "signal", 5)], False)
        # print("investorMACDZero created")
        #
        # # Create investor MACD signal
        # sellGradient = GradientQuarter(-150, 150, -200, 0)
        # buyGradient = GradientQuarter(-200, 0, 100, 0)
        # macdFastWindow = 2
        # macdSlowWindow = 6
        # signal = 5
        # a = 0.7
        # b = 2.5
        # macdParamsSignal = MACDInvestorParams(sellGradient, buyGradient, macdFastWindow, macdSlowWindow, signal,
        #                                 a, b, "grad_crossSignal")
        # investorMACDSignal = InvestorMACD(10000, macdParamsSignal)
        # experimentManager.addStrategy(investorMACDSignal, "macdGradSignal",
        #                               [experimentManager.createTIInput("macd", macdParamsSignal, "macd", 5),
        #                                experimentManager.createTIInput("macd", macdParamsSignal, "signal", 5)], False)
        # print("investorMACDSignal created")
        #
        # # Create investor BB
        # bbWindow = 10
        # bbStdDev = 1.5
        # lowerBound = 1.9
        # upperBound = 0.8
        # a = 2.4
        # b = 0.5
        # bbParams = BBInvestorParams(bbWindow, bbStdDev, lowerBound, upperBound, a, b)
        # investorBB = InvestorBB(10000, bbParams)
        # experimentManager.addStrategy(investorBB, "bb", [experimentManager.createTIInput("bb", bbParams, "pband", 1)], False)
        # print("investorBB created")

        # # Create investor based on Random Forest Classifier
        # investorRFClass = InvestorRandomForestClassifier(10000, 1)
        # experimentManager.addStrategy(investorRFClass, 'RFClass',
        #                               [experimentManager.createTIInput("df")], True)
        # print('investorRFClass created')
        #
        # # Create investor based on Random Forest Classifier
        # investorRFClass2 = InvestorRandomForestClassifier(10000, 2)
        # experimentManager.addStrategy(investorRFClass2, 'RFClass2',
        #                               [experimentManager.createTIInput("df")], True)
        # print('investorRFClass2 created')
        #
        # # Create investor based on XGB
        # investorXGB = InvestorXGB(10000)
        # experimentManager.addStrategy(investorXGB, 'XGB',
        #                               [experimentManager.createTIInput("df")], True)
        # print('investorXGB created')
        #
        # # Create investor based on XGB with window
        # investorXGBWindow = InvestorXGBWindow(10000, 3)
        # experimentManager.addStrategy(investorXGBWindow, 'XGBWindow',
        #                               [experimentManager.createTIInput("df")], True)
        # print('investorXGBWindow created')
        #
        # # Create investor based on XGB Reduced
        # investorXGBReduced = InvestorXGBReduced(10000, 3)
        # experimentManager.addStrategy(investorXGBReduced, 'XGBReduced',
        #                               [experimentManager.createTIInput("df")], True)
        # print('investorXGBReduced created')

        # # Create investor based on class voting (only sell and buy everything)
        # investorLSTMEmsemble1 = InvestorLSTMEnsembleClass1(10000, 5)
        # experimentManager.addStrategy(investorLSTMEmsemble1, "lstmEnsembleClass1",
        #                               [experimentManager.createTIInput("df")], True)
        #
        # # Create investor based on class voting (only sell and buy depending on prob)
        # investorLSTMEmsemble2 = InvestorLSTMEnsembleClass2(10000, 5)
        # experimentManager.addStrategy(investorLSTMEmsemble2, "lstmEnsembleClass2",
        #                               [experimentManager.createTIInput("df")], True)

        # # Create investor based on window forecasting (open_t - open_t+2)
        # investorLSTMWindowSCT1 = InvestorLSTMWindowStandardScalerT1(10000, 5)
        # experimentManager.addStrategy(investorLSTMWindowSCT1, "lstmWindowSCT1",
        #                               [experimentManager.createTIInput("df")], True)
        #
        # # Create investor based on window forecasting (open_t - open_t+3)
        # investorLSTMWindowSCT2 = InvestorLSTMWindowStandardScalerT2(10000, 5)
        # experimentManager.addStrategy(investorLSTMWindowSCT2, "lstmWindowSCT2",
        #                               [experimentManager.createTIInput("df")], True)

        # Create investor based on window forecasting (open_t - open_t+2)
        investorLSTMWindowMMT1 = InvestorLSTMWindowMinMaxT1(10000, 5)
        experimentManager.addStrategy(investorLSTMWindowMMT1, "lstmWindowMMT1",
                                      [experimentManager.createTIInput("df")], True)

        # Create investor based on window forecasting (open_t - open_t+3)
        investorLSTMWindowMMT2 = InvestorLSTMWindowMinMaxT2(10000, 5)
        experimentManager.addStrategy(investorLSTMWindowMMT2, "lstmWindowMMT2",
                                      [experimentManager.createTIInput("df")], True)

        # Create investor based on window forecasting (open_t - open_t+2)
        investorLSTMWindowRobMMT1 = InvestorLSTMWindowRobustMinMaxT1(10000, 5)
        experimentManager.addStrategy(investorLSTMWindowRobMMT1, "lstmWindowRobMMT1",
                                      [experimentManager.createTIInput("df")], True)

        # Create investor based on window forecasting (open_t - open_t+3)
        investorLSTMWindowRobMMT2 = InvestorLSTMWindowRobustMinMaxT2(10000, 5)
        experimentManager.addStrategy(investorLSTMWindowRobMMT2, "lstmWindowRobMMT2",
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
        nDay = 0
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
        # Prepare next first day
        dataGetter.today += CDay(1, calendar=USFederalHolidayCalendar())

        # Deal with experiment data
        aux = pd.concat([auxLoop, experimentManager.returnExpData()], axis=1)
        aux.to_csv("images/advancedData.csv", index_label="nExperiment", mode="a")

        # Calculate summary results
        firstDate = investorBaH.record.index.values[0]
        letzteDate = investorBaH.record.index.values[-1]

        dfTestCriteria = pd.concat([dfTestCriteria, experimentManager.criteriaCalculationAndPlotting(initDate, lastDate, firstDate, letzteDate, j)])

        # Plot the evolution per experiment
        experimentManager.plotEvolution(df)

    # Plot summary of test criteria
    experimentManager.summaryCriteriaCalculatorAndPlotting(dfTestCriteria)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    if not os.path.exists("images"):
        os.mkdir("images")
    main()
