import pandas as pd
from investorClass import Investor
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

class ExperimentManager:
	def __init__(self):
		self.strategies = []  # Entries have an Investor, a name and an ordered list of inputs.
		self.advancedData = pd.DataFrame()
		self.criteriaCalculator = None

	def addStrategy(self, investor, name, listOrderedInputs=[], plotEvolution=False):
		entry = {"investor": investor, "name": name, "listOrderedInputs": listOrderedInputs, "expData": pd.DataFrame()
				 , "plotEvolution": plotEvolution}
		self.strategies.append(entry)

	def runDay(self, todayData, df, nextNextData, nextData, pastOpen, expNum, numDay):
		dataManager = {}
		dataManager["nDay"] = numDay
		dataManager["pastStockValue"] = pastOpen
		dataManager["date"] = todayData.index[0]
		dataManager["actualStockValue"] = todayData.Open.values[0]
		dataManager["nextnextStockValueOpen"] = nextNextData.Open.values[0]
		dataManager["nextStockValueOpen"] = nextData.Open.values[0]

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
				elif inpName == "lstm":
					returnPred = investor.model.trainAndPredict(df)
					returnClass = investor.model.trainAndPredictClassification(df)
					dataManager[tag] = {"return": returnPred[0], "prob0": returnClass[:, 0], "prob1": returnClass[:, 1]}
				elif inpName == "lstmConfidence":
					returnPred = investor.trainAndPredict(df)
					dataManager[tag] = returnPred

			aux = investor.broker(dataManager)
			strategy["expData"] = pd.concat([strategy["expData"], aux], ignore_index=True)

			print(f"Experiment {expNum} Day {numDay} {name} Completed")

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

		title = "Test criteria (" + initDate.strftime("%Y/%m/%d")[0] + "-" + lastDate.strftime("%Y/%m/%d")[0] + ")"
		self.criteriaCalculator.plotCriteria(aux, title)

		return aux

	def plotEvolution(self, df):
		for strategy in self.strategies:
			if strategy["plotEvolution"]:
				strategy["investor"].plotEvolution(strategy["expData"], df)

	def summaryCriteriaCalculatorAndPlotting(self, dfTestCriteria):
		result = self.criteriaCalculator.calculateCriteriaVariousExperiments(dfTestCriteria)
		self.criteriaCalculator.plotCriteriaVariousExperiments(result, "Summary of the test criteria")
		return result

	@staticmethod
	def createTIInput(name, params=None, key=None, numValues=None):
		return {"name": name, "params": params, "key": key, "numValues": numValues}

