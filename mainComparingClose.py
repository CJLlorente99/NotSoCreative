import pandas as pd
from classes.dataClass import DataManager, DataGetter
from TAIndicators.rsi import relativeStrengthIndex, InvestorRSI
from TAIndicators.ma import movingAverageConvergenceDivergence, InvestorMACD
from TAIndicators.bb import bollingerBands, InvestorBB
from TAIndicators.adi import accDistIndexIndicator
from TAIndicators.adx import averageDirectionalMovementIndex
from TAIndicators.aroon import aroon
from TAIndicators.atr import averageTrueRange
from TAIndicators.obv import on_balance_volume
from TAIndicators.stochasticRsi import stochasticRSI
from DecisionFunction.investorNN import InvestorBBNN, InvestorBBRSINNClass, InvestorBBRSINN
from DecisionFunction.investorDecisionTree import InvestorDecisionTree
from LSTM.investorLSTM import InvestorLSTM
from classes.investorParamsClass import RSIInvestorParams, MACDInvestorParams, BBInvestorParams, GradientQuarter, NNInvestorParams, DTInvestorParams, ADXInvestorParams, ADIInvestorParams, AroonInvestorParams, OBVInvestorParams, StochasticRSIInvestorParams, ATRInvestorParams, LSTMInvestorParams
from Benchmarks.randomBenchmark import InvestorRandom
from Benchmarks.bia import InvestorBIA
from Benchmarks.wia import InvestorWIA
from Benchmarks.costAverage import InvestorCA
from Benchmarks.bah import InvestorBaH
from Benchmarks.idle import InvestorIdle
from pandas.tseries.offsets import CDay
from pandas.tseries.holiday import USFederalHolidayCalendar
from classes.TestCriteriaClass import TestCriteriaClass
import datetime as dt
import numpy as np
import ta


def main():
    # Create DataGetter instance
    dataGetter = DataGetter()

    # Run various experiments
    numExperiments = 1
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
        a = 1.1
        b = 2.4
        rsiParams = RSIInvestorParams(upperBound, lowerBound, RSIwindow, a, b)
        investorRSI = InvestorRSI(10000, rsiParams)
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
        print("investorBB created")

        # Create investor BBNN
        file = "data/modelnn2BB2022_12_20_11_27_08.h5"
        nnParams = NNInvestorParams(file)
        investorBBNN = InvestorBBNN(10000, nnParams)
        print("investorBBNN created")

        # Create investor BB+RSI NN Class
        file = "data/modelnn2BB_2RSIClass2022_12_20_11_27_08.h5"
        nnParams = NNInvestorParams(file)
        investorBBRSINNClass = InvestorBBRSINNClass(10000, nnParams)
        print("investorBBRSINNClass created")

        # Create investor BB+RSI NN
        file = "data/modelnn2BB_2RSI2022_12_20_11_27_08.h5"
        nnParams = NNInvestorParams(file)
        investorBBRSINN = InvestorBBRSINN(10000, nnParams)
        print("investorBBRSINN created")

        # Create investor based on DT
        adxParams = ADXInvestorParams(14)
        aroonParams = AroonInvestorParams(25)
        atrParams = ATRInvestorParams(5)
        stochParams = StochasticRSIInvestorParams(14, 3, 3)
        file = "data/dt"
        dtParams = DTInvestorParams(file)
        investorDT = InvestorDecisionTree(10000, dtParams)
        print("investorDT created")

        # Create investor based on LSTM
        file = "../data/modellstm.h5"
        lstmParams = LSTMInvestorParams(file, 0.05)
        investorLSTM = InvestorLSTM(10000, lstmParams)
        print("investorLSTM created")

        # Create investor Random
        investorRandom = InvestorRandom(10000)
        print("investorRandom created")

        # Create investor BIA
        investorBIA = InvestorBIA(10000)
        print("investorBIA created")

        # Create investor WIA
        investorWIA = InvestorWIA(10000)
        print("investorWIA created")

        # Create investor CA
        investorCA = InvestorCA(10000, 0.1)
        print("investorCA created")

        # Create investor BaH
        investorBaH = InvestorBaH(10000)
        print("investorBaH created")

        # Create investor Idle
        investorIdle = InvestorIdle(10000)
        print("investorIdle created")

        # Variables to store data
        auxRsi = pd.DataFrame()
        auxMacdGrad = pd.DataFrame()
        auxMacdZero = pd.DataFrame()
        auxMacdSignal = pd.DataFrame()
        auxBb = pd.DataFrame()
        auxBbnn = pd.DataFrame()
        auxBbRsiClassnn = pd.DataFrame()
        auxBbRsinn = pd.DataFrame()
        auxDt = pd.DataFrame()
        auxLstm = pd.DataFrame()
        auxRandom = pd.DataFrame()
        auxBIA = pd.DataFrame()
        auxWIA = pd.DataFrame()
        auxCA = pd.DataFrame()
        auxBaH = pd.DataFrame()
        auxIdle = pd.DataFrame()
        auxLoop = pd.DataFrame()

        lstmValues = pd.DataFrame()
        # Run for loop as if days passed
        for i in range(nDays+1):
            dataManager.nDay = i

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
            print(f"Experiment {j} Day {i} RSI Completed")

            # MACD Grad try
            macdResults = movingAverageConvergenceDivergence(df.Close, macdParamsGrad)
            dataManager.macd = macdResults
            aux = investorMACDGrad.broker(dataManager)
            auxMacdGrad = pd.concat([auxMacdGrad, aux], ignore_index=True)
            print(f"Experiment {j} Day {i} MACD Grad Completed")

            # MACD Zero try
            macdResults = movingAverageConvergenceDivergence(df.Close, macdParamsZero)
            dataManager.macd = macdResults
            aux = investorMACDZero.broker(dataManager)
            auxMacdZero = pd.concat([auxMacdZero, aux], ignore_index=True)
            print(f"Experiment {j} Day {i} MACD Zero Completed")

            # MACD Signal try
            macdResults = movingAverageConvergenceDivergence(df.Close, macdParamsSignal)
            dataManager.macd = macdResults
            aux = investorMACDSignal.broker(dataManager)
            auxMacdSignal = pd.concat([auxMacdSignal, aux], ignore_index=True)
            print(f"Experiment {j} Day {i} MACD Signal Completed")

            # BB try
            bbResults = bollingerBands(df.Close, bbParams)
            dataManager.bb = bbResults["pband"][-2:]
            aux = investorBB.broker(dataManager)
            auxBb = pd.concat([auxBb, aux], ignore_index=True)
            print(f"Experiment {j} Day {i} BB Completed")

            # BBNN try
            bbResults = bollingerBands(df.Close, bbParams)
            dataManager.bb = bbResults["pband"][-2:]
            auxBbnn = investorBBNN.broker(dataManager)
            auxBbnn = pd.concat([auxBbnn, aux], ignore_index=True)
            print(f"Experiment {j} Day {i} BBNN Completed")

            # BB+RSI NN Try Class
            bbResults = bollingerBands(df.Close, bbParams)
            rsiResults = relativeStrengthIndex(df.Close, rsiParams)
            dataManager.bb = bbResults["pband"][-2:]
            dataManager.rsi = rsiResults[-2:]
            auxBbRsiClassnn = investorBBRSINNClass.broker(dataManager)
            auxBbRsiClassnn = pd.concat([auxBbRsiClassnn, aux], ignore_index=True)
            print(f"Experiment {j} Day {i} BBRSINNClass Completed")

            # BB+RSI NN Try
            bbResults = bollingerBands(df.Close, bbParams)
            rsiResults = relativeStrengthIndex(df.Close, rsiParams)
            dataManager.bb = bbResults["pband"][-2:]
            dataManager.rsi = rsiResults [-2:]
            auxBbRsinn = investorBBRSINN.broker(dataManager)
            auxBbRsinn = pd.concat([auxBbRsinn, aux], ignore_index=True)
            print(f"Experiment {j} Day {i} BBRSINN Completed")

            # Dt Try
            adiResults = accDistIndexIndicator(df.High, df.Low, df.Close, df.Volume)["acc_dist_index"]
            adxResults = averageDirectionalMovementIndex(df.High, df.Low, df.Close, adxParams)["adx"]
            aroonResults = aroon(df.Close, aroonParams)["aroon_indicator"]
            atrResults = averageTrueRange(df.High, df.Low, df.Close, atrParams)["average_true_range"]
            obvResults = on_balance_volume(df.Close, df.Volume)["on_balance_volume"]
            stochRsiResults = stochasticRSI(df.Close, stochParams)["stochrsi"]
            bbResults = bollingerBands(df.Close, bbParams)["pband"]
            rsiResults = relativeStrengthIndex(df.Close, rsiParams)
            dataManager.dt["adi"] = adiResults.values[-1]
            dataManager.dt["adx"] = adxResults.values[-1]
            dataManager.dt["aroon" ]= aroonResults.values[-1]
            dataManager.dt["atr"] = atrResults.values[-1]
            dataManager.dt["obv"] = obvResults.values[-1]
            dataManager.dt["stochrsi"] = stochRsiResults.values[-1]
            dataManager.dt["bb"] = bbResults.values[-1]
            dataManager.dt["rsi"] = rsiResults.values[-1]
            dataManager.dt["aggregated"] = [np.asarray([rsiResults.values[-1], bbResults.values[-1], adiResults.values[-1], adxResults.values[-1], aroonResults.values[-1],
                        atrResults.values[-1], obvResults.values[-1], stochRsiResults.values[-1]]).transpose()]
            auxDt = investorDT.broker(dataManager)
            auxDt = pd.concat([auxDt, aux], ignore_index=True)
            print(f"Experiment {j} Day {i} DT Completed")

            # LSTM try
            yClass = investorLSTM.model.trainAndPredictClassification(df)
            dataManager.lstm = {"return": investorLSTM.model.trainAndPredict(df)[0],
                                "prob0": yClass[:, 0],
                                "prob1": yClass[:, 1]}
            lstmValues = pd.concat([lstmValues, pd.DataFrame(dataManager.lstm)], ignore_index=True)
            auxLstm = investorLSTM.broker(dataManager)
            auxLstm = pd.concat([auxLstm, aux], ignore_index=True)
            print(f"Experiment {j} Day {i} LSTM Completed")

            # Random try
            aux = investorRandom.broker(dataManager)
            auxRandom = pd.concat([auxRandom, aux], ignore_index=True)
            print(f"Experiment {j} Day {i} Random Completed")

            # BIA try
            dataManager.nextnextStockValueOpen = dataGetter.getNextNextDay().Open.values[0]
            dataManager.nextStockValueOpen = dataGetter.getNextDay().Open.values[0]
            aux = investorBIA.broker(dataManager)
            auxBIA = pd.concat([auxBIA, aux], ignore_index=True)
            print(f"Experiment {j} Day {i} BIA Completed")

            # WIA try
            aux = investorWIA.broker(dataManager)
            auxWIA = pd.concat([auxWIA, aux], ignore_index=True)
            print(f"Experiment {j} Day {i} WIA Completed")

            # CA try
            aux = investorCA.broker(dataManager)
            auxCA = pd.concat([auxCA, aux], ignore_index=True)
            print(f"Experiment {j} Day {i} CA Completed")

            # BaH try
            aux = investorBaH.broker(dataManager)
            auxBaH = pd.concat([auxBaH, aux], ignore_index=True)
            print(f"Experiment {j} Day {i} BaH Completed")

            # Idle try
            aux = investorIdle.broker(dataManager)
            auxIdle = pd.concat([auxIdle, aux], ignore_index=True)
            print(f"Experiment {j} Day {i} Idle Completed")

            # Refresh for next day
            dataManager.pastStockValue = todayData.Open.values[0]
            dataGetter.goNextDay()
        # To compensate the last goNextDay()
        lastDate = pd.DatetimeIndex([(dataGetter.today - CDay(calendar=USFederalHolidayCalendar()))])
        # Reset day to have a different 2 weeks window
        dataGetter.today += CDay(10, calendar=USFederalHolidayCalendar())

        # Deal with experiment data
        aux = pd.concat([auxLoop, auxRsi, auxMacdGrad, auxMacdZero, auxMacdSignal, auxBb, auxBbnn, auxBbRsiClassnn, auxBbRsinn,
                         auxDt, auxLstm, auxRandom, auxBIA, auxWIA, auxCA, auxBaH, auxIdle], axis=1)
        advancedData = pd.concat([advancedData, aux])

        # Calculate summary results
        firstDate = investorRSI.record.index.values[0]
        letzteDate = investorRSI.record.index.values[-1]
        criteriaCalculator = TestCriteriaClass(firstDate, letzteDate)
        testCriteriaRSI = pd.DataFrame(criteriaCalculator.calculateCriteria("rsi", investorRSI.record), index=[j])
        testCriteriaMACDGrad = pd.DataFrame(criteriaCalculator.calculateCriteria("macdGrad", investorMACDGrad.record), index=[j])
        testCriteriaMACDZero = pd.DataFrame(criteriaCalculator.calculateCriteria("macdZero", investorMACDZero.record), index=[j])
        testCriteriaMACDSignal = pd.DataFrame(criteriaCalculator.calculateCriteria("macdSignal", investorMACDSignal.record), index=[j])
        testCriteriaBB = pd.DataFrame(criteriaCalculator.calculateCriteria("bb", investorBB.record), index=[j])
        testCriteriaBBNN = pd.DataFrame(criteriaCalculator.calculateCriteria("bbnn", investorBBNN.record), index=[j])
        testCriteriaBBRSINNClass = pd.DataFrame(
            criteriaCalculator.calculateCriteria("BBRSINNClass", investorBBRSINNClass.record), index=[j])
        testCriteriaBBRSINN = pd.DataFrame(
            criteriaCalculator.calculateCriteria("BBRSINNC", investorBBRSINN.record), index=[j])
        testCriteriaDT = pd.DataFrame(
            criteriaCalculator.calculateCriteria("DT", investorDT.record), index=[j])
        testCriteriaLSTM = pd.DataFrame(
            criteriaCalculator.calculateCriteria("LSTM", investorLSTM.record), index=[j])
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
            [testCriteriaRSI, testCriteriaMACDGrad, testCriteriaMACDZero, testCriteriaMACDSignal, testCriteriaBB, testCriteriaBBNN,
             testCriteriaRandom, testCriteriaBIA, testCriteriaWIA, testCriteriaCA, testCriteriaBaH, testCriteriaIdle,
             testCriteriaBBRSINNClass, testCriteriaBBRSINN, testCriteriaDT, testCriteriaLSTM])

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
        # investorBBNN.plotEvolution(bbResults, df)
        # investorBBRSINNClass.plotEvolution({"bb":bbResults, "rsi":rsiResults}, df)
        # investorBBRSINN.plotEvolution({"bb": bbResults, "rsi": rsiResults}, df)
        investorDT.plotEvolution({"bb": bbResults, "rsi": rsiResults, "adi": adiResults, "adx":adxResults, "aroon":aroonResults
                                  , "atr": atrResults, "obv": obvResults, "stochRsi":stochRsiResults}, df)
        investorLSTM.plotEvolution(lstmValues, df)
        # investorRandom.plotEvolution(None, df)
        # investorBIA.plotEvolution(None, df)
        # investorWIA.plotEvolution(None, df)
        # investorCA.plotEvolution(None, df)
        # investorBaH.plotEvolution(None, df)
        # investorIdle.plotEvolution(None, df)

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
