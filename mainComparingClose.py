import pandas as pd
from classes.investorClass import Investor
from classes.dataClass import DataManager, DataGetter
from TAIndicators.rsi import relativeStrengthIndex, plotRSIDecisionRules
from TAIndicators.ma import movingAverageConvergenceDivergence, plotMACDDecisionRules
from TAIndicators.bb import bollingerBands, plotBBDecisionRules
from classes.investorParamsClass import RSIInvestorParams, MACDInvestorParams, BBInvestorParams, GradientQuarter
from pandas.tseries.offsets import CDay
from pandas.tseries.holiday import USFederalHolidayCalendar


def main():
    # Create DataGetter instance
    dataGetter = DataGetter()

    # Common data
    maxBuy = 10000
    maxSell = 10000

    # Run various experiments
    numExperiments = 1
    nDays = 10
    summaryResults = pd.DataFrame()
    advancedData = pd.DataFrame()
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
        rsiParams = RSIInvestorParams(upperBound, lowerBound, RSIwindow, maxBuy, maxSell, a, b)
        investorRSI = Investor(10000, dataGetter.today, rsiParams=rsiParams)

        # Create investor MACD grad
        sellGradient = GradientQuarter(-200, 50, -50, 0)
        buyGradient = GradientQuarter(-100, 50, 0, 0)
        macdFastWindow = 2
        macdSlowWindow = 6
        signal = 7
        a = 0.7
        b = 2.5
        macdParamsGrad = MACDInvestorParams(sellGradient, buyGradient, macdFastWindow, macdSlowWindow, signal, maxBuy,
                                        maxSell, a, b, "grad")
        investorMACDGrad = Investor(10000, dataGetter.today, macdParams=macdParamsGrad)

        # Create investor MACD zero
        sellGradient = GradientQuarter(-50, 100, -100, 0)
        buyGradient = GradientQuarter(-100, 0, 50, 0)
        macdFastWindow = 2
        macdSlowWindow = 9
        signal = 7
        a = 0.7
        b = 0.5
        macdParamsZero = MACDInvestorParams(sellGradient, buyGradient, macdFastWindow, macdSlowWindow, signal, maxBuy,
                                        maxSell, a, b, "grad_crossZero")
        investorMACDZero = Investor(10000, dataGetter.today, macdParams=macdParamsZero)

        # Create investor MACD signal
        sellGradient = GradientQuarter(-150, 150, -200, 0)
        buyGradient = GradientQuarter(-200, 0, 100, 0)
        macdFastWindow = 2
        macdSlowWindow = 6
        signal = 5
        a = 0.7
        b = 2.5
        macdParamsSignal = MACDInvestorParams(sellGradient, buyGradient, macdFastWindow, macdSlowWindow, signal, maxBuy,
                                        maxSell, a, b, "grad_crossSignal")
        investorMACDSignal = Investor(10000, dataGetter.today, macdParams=macdParamsSignal)

        # Create investor BB
        bbWindow = 10
        bbStdDev = 1.5
        lowerBound = 1.9
        upperBound = 0.8
        a = 2.4
        b = 0.5
        bbParams = BBInvestorParams(bbWindow, bbStdDev, lowerBound, upperBound, maxBuy, maxSell, a, b)
        investorBB = Investor(10000, dataGetter.today, bbParams=bbParams)

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
            # print(f'RSI is {dataManager.rsi}')
            moneyToInvest, moneyToSell, investedMoney, nonInvestedMoney = investorRSI.broker(dataManager, 'rsi')
            aux = pd.DataFrame(
                {'rsi': [rsiResults[-1]], 'moneyToInvestRSI': [moneyToInvest], 'moneyToSellRSI': [moneyToSell],
                 'investedMoneyRSI': [investedMoney], 'nonInvestedMoneyRSI': [nonInvestedMoney]})
            auxRsi = pd.concat([auxRsi, aux], ignore_index=True)

            # MACD Grad try
            macdResults = movingAverageConvergenceDivergence(df.Close, macdParamsGrad)
            dataManager.macd = macdResults
            moneyToInvest, moneyToSell, investedMoney, nonInvestedMoney = investorMACDGrad.broker(dataManager, 'macd')
            aux = pd.DataFrame(
                {'macdGrad': [macdResults["macd"][-1]], 'moneyToInvestMACDGrad': [moneyToInvest], 'moneyToSellMACDGrad': [moneyToSell],
                 'investedMoneyMACDGrad': [investedMoney], 'nonInvestedMoneyMACDGrad': [nonInvestedMoney]})
            auxMacdGrad = pd.concat([auxMacdGrad, aux], ignore_index=True)

            # MACD Zero try
            macdResults = movingAverageConvergenceDivergence(df.Close, macdParamsZero)
            dataManager.macd = macdResults
            moneyToInvest, moneyToSell, investedMoney, nonInvestedMoney = investorMACDZero.broker(dataManager, 'macd')
            aux = pd.DataFrame(
                {'macdZero': [macdResults["macd"][-1]], 'moneyToInvestMACDZero': [moneyToInvest], 'moneyToSellMACDZero': [moneyToSell],
                 'investedMoneyMACDZero': [investedMoney], 'nonInvestedMoneyMACDZero': [nonInvestedMoney]})
            auxMacdZero = pd.concat([auxMacdZero, aux], ignore_index=True)

            # MACD Signal try
            macdResults = movingAverageConvergenceDivergence(df.Close, macdParamsSignal)
            dataManager.macd = macdResults
            moneyToInvest, moneyToSell, investedMoney, nonInvestedMoney = investorMACDSignal.broker(dataManager, 'macd')
            aux = pd.DataFrame(
                {'macdSignal': [macdResults["macd"][-1]], 'moneyToInvestMACDSignal': [moneyToInvest], 'moneyToSellMACDSignal': [moneyToSell],
                 'investedMoneyMACDSignal': [investedMoney], 'nonInvestedMoneyMACDSignal': [nonInvestedMoney]})
            auxMacdSignal = pd.concat([auxMacdSignal, aux], ignore_index=True)

            # BB try
            bbResults = bollingerBands(df.Close, bbParams)
            dataManager.bb = bbResults["pband"][-1]
            moneyToInvest, moneyToSell, investedMoney, nonInvestedMoney = investorBB.broker(dataManager, 'bb')
            aux = pd.DataFrame(
                {'bb': [bbResults["pband"][-1]], 'moneyToInvestBB': [moneyToInvest], 'moneyToSellBB': [moneyToSell],
                 'investedMoneyBB': [investedMoney], 'nonInvestedMoneyBB': [nonInvestedMoney]})
            auxBb = pd.concat([auxBb, aux], ignore_index=True)

            # Refresh for next day
            dataManager.pastStockValue = todayData.Open.values[0]
            dataGetter.goNextDay()
        lastDate = pd.DatetimeIndex([(dataGetter.today - CDay(calendar=USFederalHolidayCalendar()))])
        # Reset day to have a different 2 weeks window
        dataGetter.today += CDay(200, calendar=USFederalHolidayCalendar())

        # Deal with experiment data
        aux = pd.concat([auxLoop, auxRsi, auxMacdGrad, auxMacdZero, auxMacdSignal, auxBb], axis=1)
        advancedData = pd.concat([advancedData, aux])

        # Calculate summary results
        percentualGainRSI, meanPortfolioValueRSI = investorRSI.calculateMetrics()
        percentualGainMACDGrad, meanPortfolioValueMACDGrad = investorMACDGrad.calculateMetrics()
        percentualGainMACDZero, meanPortfolioValueMACDZero = investorMACDZero.calculateMetrics()
        percentualGainMACDSignal, meanPortfolioValueMACDSignal = investorMACDSignal.calculateMetrics()
        percentualGainBB, meanPortfolioValueBB = investorBB.calculateMetrics()

        # Show final percentual gain and mean portfolio value per experiment
        print("Percentual gain RSI {:.2f}%, mean portfolio value RSI {:.2f}$".format(percentualGainRSI,
                                                                                     meanPortfolioValueRSI))
        print("Percentual gain MACD Grad {:.2f}%, mean portfolio value MACD Grad {:.2f}$".format(percentualGainMACDGrad,
                                                                                     meanPortfolioValueMACDGrad))
        print("Percentual gain MACD Zero {:.2f}%, mean portfolio value MACD Zero {:.2f}$".format(percentualGainMACDZero,
                                                                                                 meanPortfolioValueMACDZero))
        print("Percentual gain MACD Signal {:.2f}%, mean portfolio value MACD Signal {:.2f}$".format(percentualGainMACDSignal,
                                                                                                 meanPortfolioValueMACDSignal))
        print("Percentual gain BB {:.2f}%, mean portfolio value BB {:.2f}$".format(percentualGainBB,
                                                                                       meanPortfolioValueBB))
        results = pd.DataFrame(
            {"initDate": [initDate.strftime("%d/%m/%Y")[0]], "lastDate": [lastDate.strftime("%d/%m/%Y")[0]],
             "percentageRSI": [percentualGainRSI], "percentageMACDGrad": [percentualGainMACDGrad],
             "percentageMACDZero": [percentualGainMACDZero],
             "percentageMACDSignal": [percentualGainMACDSignal], "percentageBB": [percentualGainBB],
             "meanPortfolioValueRSI": [meanPortfolioValueRSI],
             "meanPortfolioValueMACDGrad": [meanPortfolioValueMACDGrad],
             "meanPortfolioValueMACDZero": [meanPortfolioValueMACDZero],
             "meanPortfolioValueMACDSignal": [meanPortfolioValueMACDSignal],
             "meanPortfolioValueBB": [meanPortfolioValueBB]})
        summaryResults = pd.concat([summaryResults, results], ignore_index=True)

        # Plot the evolution per experiment
        investorRSI.plotEvolution(rsiResults, df, "RSI")
        investorMACDGrad.plotEvolution(macdResults, df, "MACD (Grad Method)")
        investorMACDZero.plotEvolution(macdResults, df, "MACD (Crossover Zero Method)")
        investorMACDSignal.plotEvolution(macdResults, df, "MACD (Crossover Signal)")
        investorBB.plotEvolution(bbResults, df, "BB")

    # Push the data into files for later inspection
    # now = dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # summaryResults.to_csv("data/" + now + ".csv", index_label="experiment")
    # advancedData.to_csv("data/" + now + "_advancedData.csv", index_label="experiment")

    # with open("data/" + now + ".txt", "w") as f:
    #     f.write(str(rsiParams) + "\n")
    #     # f.write(str(smaParams) + "\n")
    #     # f.write(str(emaParams) + "\n")
    #     f.write(str(macdParamsGrad) + "\n")
    #     f.write(str(macdParamsZero) + "\n")
    #     f.write(str(macdParamsSignal) + "\n")
    #     f.write(str(bbParams))

    # Show the decision rules with the parameters used
    plotRSIDecisionRules(rsiParams)
    plotBBDecisionRules(bbParams)
    plotMACDDecisionRules(macdParamsGrad)
    plotMACDDecisionRules(macdParamsZero)
    plotMACDDecisionRules(macdParamsSignal)


if __name__ == '__main__':
    main()
