import pandas as pd
from classes.dataClass import DataManager, DataGetter
from TAIndicators.rsi import relativeStrengthIndex, InvestorRSI
from TAIndicators.ma import movingAverageConvergenceDivergence, InvestorMACD
from TAIndicators.bb import bollingerBands, InvestorBB
from Benchmarks.bia import InvestorBIA
from classes.investorParamsClass import RSIInvestorParams, MACDInvestorParams, BBInvestorParams, GradientQuarter
from pandas.tseries.offsets import CDay
from pandas.tseries.holiday import USFederalHolidayCalendar
import pandas_datareader as web
from pandas.tseries.offsets import CDay
from pandas.tseries.holiday import USFederalHolidayCalendar



def main():
	nDays = 10000

	# Get data
	data = web.DataReader('^GSPC', 'yahoo',
					  pd.to_datetime("today").date() - CDay(nDays+60+2, calendar=USFederalHolidayCalendar()),
					  pd.to_datetime("today").date())
	today = pd.to_datetime("today").date() - CDay(nDays+2, calendar=USFederalHolidayCalendar())
	today = data.index.get_loc(today)

	# Create DataManager instance
	dataManager = DataManager()

	# Load data
	df = data.iloc[today - 1]

	# Create investor RSI
	RSIwindow = 3
	upperBound = 61
	lowerBound = 27.5
	maxBuy = 10000
	maxSell = 10000
	a = 1.1
	b = 2.4
	rsiParams = RSIInvestorParams(upperBound, lowerBound, RSIwindow, maxBuy, maxSell, a, b)

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

	# Create investor BIA
	investorBIA = InvestorBIA(10000)
	dataManager.pastStockValue = df.Open

	# Results
	results = pd.DataFrame()

	# Run for loop as if days passed
	for i in range(nDays):
		print(f"{i}/{nDays-1}")

		# Refresh data for today
		todayData = data.iloc[today]
		df = data.iloc[today - 60: today]
		dataManager.date = todayData.index[0]
		dataManager.actualStockValue = todayData.Open

		result = {}

		# Inputs of NN
		result["rsiResults"] = [relativeStrengthIndex(df.Close, rsiParams).values[-1]]
		# result["macdResultsGrad"] = movingAverageConvergenceDivergence(df.Close, macdParamsGrad)
		# result["macdResultsZero"] = movingAverageConvergenceDivergence(df.Close, macdParamsZero)
		# result["macdResultsSignal"] = movingAverageConvergenceDivergence(df.Close, macdParamsSignal)
		result["bbResults"] = [bollingerBands(df.Close, bbParams)["pband"].values[-1]]

		# Ideal output
		dataManager.nextnextStockValueOpen = data.iloc[today + 2].Open
		dataManager.nextStockValueOpen = data.iloc[today + 1].Open
		aux = investorBIA.broker(dataManager)
		moneyToInvest, moneyToSell = aux["moneyToInvestBIA"].values[0], aux["moneyToSellBIA"].values[0]
		result["output"] = [moneyToInvest - moneyToSell]

		results = pd.concat([results, pd.DataFrame(result)], ignore_index=True)

		# Refresh for next day
		today += 1

	print(results)
	results.to_csv("../data/optimizationTrainingSet.csv", index_label="n", mode="a")


if __name__ == '__main__':
	main()
