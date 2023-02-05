import datetime
import os.path
import pandas as pd
from classes.dataClass import DataGetter
from LSTM.investorBiLSTMWindowRobustMinMaxScalerLegacy import InvestorBiLSTMWindowRobustMinMaxT1T2Legacy
from LSTM.investorLSTMWindowRobustMinMaxScalerLegacy import InvestorLSTMWindowRobustMinMaxT2Legacy
from RF_DT.investorRandomForestClassifier import InvestorRandomForestClassifier
from RF_DT.investorXGB import InvestorXGB
from TAStrategies.bb import InvestorBB
from TAStrategies.macd import InvestorMACD
from Benchmarks.randomBenchmark import InvestorRandom
from Benchmarks.bia import InvestorBIA
from Benchmarks.wia import InvestorWIA
from Benchmarks.costAverage import InvestorCA
from Benchmarks.bah import InvestorBaH
from Benchmarks.idle import InvestorIdle
from pandas.tseries.offsets import CDay
from pandas.tseries.holiday import USFederalHolidayCalendar
from classes.experimentManager import ExperimentManager

def main():
    # Create DataGetter instance
    dataGetter = DataGetter('2020-05-01', '2020-05-30')

    # Run various experiments
    numExperiments = 30
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

        """
        MACD
        """

        investorMACD = InvestorMACD(10000)
        experimentManager.addStrategy(investorMACD, "macd",
                                      [experimentManager.createTIInput("df")], False)

        """
		BB
		"""

        investorBB = InvestorBB(10000)
        experimentManager.addStrategy(investorBB, "bb",
                                      [experimentManager.createTIInput("df")], False)

        """
        BiLSTM RobMinMax Legacy
        """

        # Create investor based on window forecasting (open_t - open_t+2)
        investorBiLSTMWindowRobMMT1T2Legacy = InvestorBiLSTMWindowRobustMinMaxT1T2Legacy(10000, 1)
        experimentManager.addStrategy(investorBiLSTMWindowRobMMT1T2Legacy, "bilstmWindowRobMMT1T2Legacy",
                                      [experimentManager.createTIInput("df")], False)

        """
        LSTM RobMinMax Legacy
        """

        # Create investor based on window forecasting (open_t - open_t+3)
        investorLSTMWindowRobMMT2Legacy = InvestorLSTMWindowRobustMinMaxT2Legacy(10000, 1)
        experimentManager.addStrategy(investorLSTMWindowRobMMT2Legacy, "lstmWindowRobMMT2Legacy",
                                      [experimentManager.createTIInput("df")], False)

        """
        Random Forest
        """

        investorRandomForest = InvestorRandomForestClassifier(10000)
        experimentManager.addStrategy(investorRandomForest, "randomForest",
                                      [experimentManager.createTIInput("df")], False)

        """
        XGBoost
        """

        investorXGBoost = InvestorXGB(10000)
        experimentManager.addStrategy(investorXGBoost, "xgBoost",
                                      [experimentManager.createTIInput("df")], False)


        # Create investor Random
        investorRandom = InvestorRandom(10000)
        experimentManager.addStrategy(investorRandom, "random", plotEvolution=False)
        print("investorRandom created")

        # Create investor BIA
        investorBIA = InvestorBIA(10000)
        experimentManager.addStrategy(investorBIA, "bia", plotEvolution=False)
        print("investorBIA created")

        # Create investor WIA
        investorWIA = InvestorWIA(10000)
        experimentManager.addStrategy(investorWIA, "wia", plotEvolution=False)
        print("investorWIA created")

        # Create investor CA
        investorCA = InvestorCA(10000, 0.1)
        experimentManager.addStrategy(investorCA, "ca", plotEvolution=False)
        print("investorCA created")

        # Create investor BaH
        investorBaH = InvestorBaH(10000)
        experimentManager.addStrategy(investorBaH, "bah", plotEvolution=False)
        print("investorBaH created")

        # Create investor Idle
        investorIdle = InvestorIdle(10000)
        experimentManager.addStrategy(investorIdle, "idle", plotEvolution=False)
        print("investorIdle created")

        auxLoop = pd.DataFrame()
        # Run for loop as if days passed
        pastStockValue = df.Close[0]
        nDay = 0
        for i in range(nDays):
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
        dataGetter.today += CDay(0, calendar=USFederalHolidayCalendar())

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
    if not os.path.exists("images"):
        os.mkdir("images")
    main()
