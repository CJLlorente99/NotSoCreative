import math
import numpy as np
import pandas as pd
from classes.investorClass import Investor
from TAIndicators.adi import accDistIndexIndicator
from TAIndicators.adx import averageDirectionalMovementIndex
from TAIndicators.aroon import aroon
from TAIndicators.atr import averageTrueRange
from TAIndicators.bb import bollingerBands
from TAIndicators.ma import movingAverageConvergenceDivergence
from TAIndicators.obv import on_balance_volume
from TAIndicators.rsi import relativeStrengthIndex
from TAIndicators.stochasticRsi import stochasticRSI
from classes.TestCriteriaClass import  TestCriteriaClass
import datetime
class ExperimentManager:
	def __init__(self):
		self.strategies = []  # Entries have an Investor, a name and an ordered list of inputs.
		self.advancedData = pd.DataFrame()
		self.criteriaCalculator = None

	def addStrategy(self, investor, name, listOrderedInputs=[], plotEvolution=False):
		entry = {"investor": investor, "name": name, "listOrderedInputs": listOrderedInputs, "expData": pd.DataFrame()
				 , "plotEvolution": plotEvolution}
		self.strategies.append(entry)

	def runMorning(self, todayData, data: pd.DataFrame, nextNextData, nextData, pastOpen, expNum, numDay):
		df = data.copy()
		df['Low'] = df['Low'].shift()
		df['High'] = df['High'].shift()
		df['Close'] = df['Close'].shift()
		df['Adj Close'] = df['Adj Close'].shift()
		df['Volume'] = df['Volume'].shift()
		df.dropna(inplace=True)

		dataManager = {}

		dataManager["nDay"] = numDay
		dataManager["pastStockValue"] = pastOpen
		dataManager["date"] = todayData.index[0].combine(todayData.index[0], datetime.time(9, 30))
		dataManager["actualStockValue"] = todayData.Open.values[0]
		dataManager["nextStockValue"] = nextData.Open.values[0]

		for strategy in self.strategies:
			investor = strategy["investor"]
			name = strategy["name"]
			listOrderedInputs = strategy["listOrderedInputs"]  # Containing dicts with (name, params, key, numValues)
			for inp in listOrderedInputs:
				inpName = inp["name"]
				inpParams = inp["params"]
				inpKey = inp["key"]
				inpNumValues = inp["numValues"]
				tag = inpName + (inpKey if inpKey else "")
				if inpName == "adi":
					dataManager[tag] = accDistIndexIndicator(df.High, df.Low, df.Close, df.Volume, inpParams)[inpKey].values[-inpNumValues:]
				elif inpName == "adx":
					dataManager[tag] = averageDirectionalMovementIndex(df.High, df.Low, df.Close, inpParams)[inpKey].values[-inpNumValues:]
				elif inpName == "aroon":
					dataManager[tag] = aroon(df.Close, inpParams)[inpKey].values[-inpNumValues:]
				elif inpName == "atr":
					dataManager[tag] = averageTrueRange(df.High, df.Low, df.Close, inpParams)[inpKey].values[-inpNumValues:]
				elif inpName == "bb":
					dataManager[tag] = bollingerBands(df.Close, inpParams)[inpKey].values[-inpNumValues:]
				elif inpName == "macd":
					dataManager[tag] = movingAverageConvergenceDivergence(df.Close, inpParams)[inpKey]
				elif inpName == "obv":
					dataManager[tag] = on_balance_volume(df.Close, df.Volume, inpParams)[inpKey].values[-inpNumValues:]
				elif inpName == "rsi":
					dataManager[tag] = relativeStrengthIndex(df.Close, inpParams)[inpKey].values[-inpNumValues:]
				elif inpName == "stochrsi":
					dataManager[tag] = stochasticRSI(df.Close, inpParams)[inpKey].values[-inpNumValues:]
				elif inpName == 'df':
					dataManager['df'] = df

			aux = investor.brokerMorning(dataManager)
			strategy["expData"] = pd.concat([strategy["expData"], aux], ignore_index=True)

			print(f"Experiment {expNum} Day {math.floor(numDay/2)} Morning {name} Completed")

	def runAfternoon(self, todayData, df, nextNextData, nextData, pastOpen, expNum, numDay):
		dataManager = {}

		dataManager["nDay"] = numDay
		dataManager["pastStockValue"] = pastOpen
		dataManager["date"] = todayData.index[0].combine(todayData.index[0], datetime.time(16, 00))
		dataManager["actualStockValue"] = todayData.Close.values[0]
		dataManager["nextStockValue"] = nextData.Open.values[0]

		for strategy in self.strategies:
			investor = strategy["investor"]
			name = strategy["name"]
			listOrderedInputs = strategy["listOrderedInputs"]  # Containing dicts with (name, params, key, numValues)
			for inp in listOrderedInputs:
				inpName = inp["name"]
				inpParams = inp["params"]
				inpKey = inp["key"]
				inpNumValues = inp["numValues"]
				tag = inpName + (inpKey if inpKey else "")
				if inpName == "adi":
					dataManager[tag] = accDistIndexIndicator(df.High, df.Low, df.Close, df.Volume, inpParams)[inpKey].values[-inpNumValues:]
				elif inpName == "adx":
					dataManager[tag] = averageDirectionalMovementIndex(df.High, df.Low, df.Close, inpParams)[inpKey].values[-inpNumValues:]
				elif inpName == "aroon":
					dataManager[tag] = aroon(df.Close, inpParams)[inpKey].values[-inpNumValues:]
				elif inpName == "atr":
					dataManager[tag] = averageTrueRange(df.High, df.Low, df.Close, inpParams)[inpKey].values[-inpNumValues:]
				elif inpName == "bb":
					dataManager[tag] = bollingerBands(df.Close, inpParams)[inpKey].values[-inpNumValues:]
				elif inpName == "macd":
					dataManager[tag] = movingAverageConvergenceDivergence(df.Close, inpParams)[inpKey]
				elif inpName == "obv":
					dataManager[tag] = on_balance_volume(df.Close, df.Volume, inpParams)[inpKey].values[-inpNumValues:]
				elif inpName == "rsi":
					dataManager[tag] = relativeStrengthIndex(df.Close, inpParams)[inpKey].values[-inpNumValues:]
				elif inpName == "stochrsi":
					dataManager[tag] = stochasticRSI(df.Close, inpParams)[inpKey].values[-inpNumValues:]
				elif inpName == 'df':
					dataManager[tag] = df

			aux = investor.brokerAfternoon(dataManager)
			strategy["expData"] = pd.concat([strategy["expData"], aux], ignore_index=True)

			print(f"Experiment {expNum} Day {math.floor(numDay/2)} Afternoon {name} Completed")

	def returnExpData(self):
		aux = pd.DataFrame()
		for strategy in self.strategies:
			aux = pd.concat([aux, strategy["expData"]], axis=1)
		return aux

	def criteriaCalculationAndPlotting(self, initDate, lastDate, firstDate, letzteDate, numExp):
		self.criteriaCalculator = TestCriteriaClass(firstDate, letzteDate)
		aux = pd.DataFrame()

		for strategy in self.strategies:
			name = strategy["name"]
			investor = strategy["investor"]
			aux = pd.concat([aux, pd.DataFrame(self.criteriaCalculator.calculateCriteria(name, investor.record), index=[numExp])])

		title = "TestCriteria(" + initDate.strftime("%Y_%m_%d")[0] + "-" + lastDate.strftime("%Y_%m_%d")[0] + ")"
		self.criteriaCalculator.plotCriteria(aux, title)

		return aux

	def plotEvolution(self, df):
		df_new = pd.DataFrame(np.repeat(df.values,2, axis=0), index=np.repeat(df.index.values, 2, axis=0), columns=df.columns)
		df_new['Open'] = df_new['Open'].shift(-1)
		df_new = df_new[:-1]
		for strategy in self.strategies:
			if strategy["plotEvolution"]:
				strategy["investor"].plotEvolution(strategy["expData"], df_new)

	def summaryCriteriaCalculatorAndPlotting(self, dfTestCriteria):
		result = self.criteriaCalculator.calculateCriteriaVariousExperiments(dfTestCriteria)
		self.criteriaCalculator.plotCriteriaVariousExperiments(result, "SummaryTestCriteria")
		return result

	@staticmethod
	def createTIInput(name, params=None, key=None, numValues=None):
		return {"name": name, "params": params, "key": key, "numValues": numValues}

