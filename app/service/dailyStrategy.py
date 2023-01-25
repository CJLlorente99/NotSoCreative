import pandas as pd
from abc import ABC, abstractmethod
from jsonManagement.inversionStrategyJSONAPI import Strategy
import numpy as np


class DailyStrategy(ABC):
	"""
	Investor is an abstract class that is inherited and expanded by all the different strategies' classes (benchmarks and
	our own strategies)
	"""
	def __init__(self, record: pd.DataFrame, strategyDefinition: Strategy):
		"""
		Initialization function for the Investor class
		:param record: Available info in the CSV
		:param strategyDefinition: Available info coming from json
		"""
		self.name = strategyDefinition.name
		self.initialInvestment = strategyDefinition.initialMoney
		self.description = strategyDefinition.description
		self.record = record

		self.perToInvest = 0
		if len(record) == 0:
			self.investedMoney = 0
			self.nonInvestedMoney = self.initialInvestment
		else:
			self.investedMoney = record['MoneyInvested'].values[-1]
			self.nonInvestedMoney = record['MoneyNotInvested'].values[-1]


	"""
	CONCRETE METHODS
	"""

	def broker(self, operation, inputs) -> pd.DataFrame:
		"""
		Function that takes decisions on buy/sell/hold based on today's value and predicted value for tomorrow
		:param data: Decision data based on the type of indicator
		:return dataFrame with the data relevant to the actual strategy used and actions taken out that day
		"""
		# Morning -> Apply logic (intraday return prediction) and update portfolio value
		if operation == 'Morning':
			moneyInvested = self.brokerMorning(inputs)
			testCriteriaEntry = self.calculateTestCriteriaMorning()
		# Afternoon -> Apply logic (interday return prediction) and update portfolio value
		elif operation == 'Afternoon':
			moneyInvested = self.brokerAfternoon(inputs)
			testCriteriaEntry = self.calculateTestCriteriaAfternoon()

		aux = pd.DataFrame({'investorStrategy': self.name, 'MoneyInvested': self.investedMoney, 'MoneyNotInvested': self.nonInvestedMoney,
							 'MoneyInvestedToday': moneyInvested, 'PerInvestToday': self.perToInvest,
							 'TotalPortfolioValue': self.investedMoney + self.nonInvestedMoney}, index=[0])
		return pd.concat([aux, testCriteriaEntry], axis=1)

		# return pd.DataFrame({'investorStrategy': self.name, 'MoneyInvested': self.investedMoney, 'MoneyNotInvested': self.nonInvestedMoney,
		# 					 'MoneyInvestedToday': moneyInvested, 'PerInvestToday': self.perToInvest,
		# 					 'TotalPortfolioValue': self.investedMoney + self.nonInvestedMoney}, index=[0])

	def brokerMorning(self, inputs: pd.DataFrame):
		"""
		1) Update portfolio value as Open_t/Close_t-1
		2) Operation (buy and sell according to last perToSell and perToBuy)
		3) Prediction for next action
		:param inputs:
		:return:
		"""
		# Update investedMoney value (both open and close are already shifted appropriately)
		self.investedMoney *= inputs["Open"][-1] / inputs["Close"][-1]

		# Broker operations for next day
		self.possiblyOperationMorning(inputs)

		# Broker operation for today
		moneyInvested = self.__investAndSellToday()

		return moneyInvested

	def brokerAfternoon(self, inputs: pd.DataFrame):
		"""
		1) Update portfolio value as Close_t/Open_t
		2) Operation (buy and sell according to last perToSell and perToBuy)
		3) Prediction for next action
		:param inputs:
		:return:
		"""
		# Update investedMoney value
		self.investedMoney *= inputs["Close"][-1] / inputs["Open"][-1]

		# Broker operations for today
		self.possiblyOperationAfternoon(inputs)

		# Broker operation for today
		moneyInvested = self.__investAndSellToday()

		return moneyInvested

	def __investAndSellToday(self) -> float:
		"""
		This function performs the operation given by signals established the day before
		:return float representing the money that has been finally bought (positive) or sold (negative)
		"""
		# Calculate the money bought and sold depending on the actual nonInvested and Invested.
		moneyInvested = 0
		if self.perToInvest > 0:  # We buy
			if self.nonInvestedMoney == 0:
				self.perToInvest = 0
			moneyInvested = self.perToInvest * self.nonInvestedMoney
			self.investedMoney += self.perToInvest * self.nonInvestedMoney
			self.nonInvestedMoney -= self.perToInvest * self.nonInvestedMoney

		if self.perToInvest < 0:  # We sell
			if self.investedMoney == 0:
				self.perToInvest = 0
			moneyInvested = self.perToInvest * self.investedMoney
			self.nonInvestedMoney += -self.perToInvest * self.investedMoney
			self.investedMoney -= -self.perToInvest * self.investedMoney

		return moneyInvested

	def calculateTestCriteriaMorning(self):
		if len(self.record) != 0:
			pvs = np.append(self.record['TotalPortfolioValue'][:-1].iloc[::-2].values, self.nonInvestedMoney + self.investedMoney)
			# MPV
			mpv = pvs.mean()

			# StdPV
			std = pvs.std()

			# Max
			maxPV = pvs.max()

			# Min
			minPV = pvs.min()

			# Absolute Gain
			absGain = pvs[-1] - pvs[0]

			# Per Gain
			perGain = (pvs[-1] - pvs[0]) / pvs[0] * 100

			# Max gain
			maxGain = 0

			# Min gain
			minGain = 0

			return pd.DataFrame({'MPV': mpv, 'StdPV': std, 'maxPV': maxPV, 'minPV': minPV, 'absGain': absGain,
								 'perGain': perGain, 'maxGain': maxGain, 'minGain': minGain}, index=[0])

		else:
			# MPV
			mpv = self.initialInvestment

			# StdPV
			std = 0

			# Max
			maxPV = self.initialInvestment

			# Min
			minPV = self.initialInvestment

			# Absolute Gain
			absGain = 0

			# Per Gain
			perGain = 0

			# Max gain
			maxGain = 0

			# Min gain
			minGain = 0

			return pd.DataFrame({'MPV': mpv, 'StdPV': std, 'maxPV': maxPV, 'minPV': minPV, 'absGain': absGain,
								 'perGain': perGain, 'maxGain': maxGain, 'minGain': minGain}, index=[0])

	def calculateTestCriteriaAfternoon(self):
		if len(self.record) != 0:
			pvs = np.append(self.record['TotalPortfolioValue'].iloc[::-2].values,
							self.nonInvestedMoney + self.investedMoney)
			# MPV
			mpv = pvs.mean()

			# StdPV
			std = pvs.std()

			# Max
			maxPV = pvs.max()

			# Min
			minPV = pvs.min()

			# Absolute Gain
			absGain = pvs[-1] - pvs[0]

			# Per Gain
			perGain = (pvs[-1] - pvs[0]) / pvs[0] * 100

			# Max gain
			maxGain = 0

			# Min gain
			minGain = 0

			return pd.DataFrame({'MPV': mpv, 'StdPV': std, 'maxPV': maxPV, 'minPV': minPV, 'absGain': absGain,
								 'perGain': perGain, 'maxGain': maxGain, 'minGain': minGain}, index=[0])

		else:
			# MPV
			mpv = self.initialInvestment

			# StdPV
			std = 0

			# Max
			maxPV = self.initialInvestment

			# Min
			minPV = self.initialInvestment

			# Absolute Gain
			absGain = 0

			# Per Gain
			perGain = 0

			# Max gain
			maxGain = 0

			# Min gain
			minGain = 0

			return pd.DataFrame({'MPV': mpv, 'StdPV': std, 'maxPV': maxPV, 'minPV': minPV, 'absGain': absGain,
								 'perGain': perGain, 'maxGain': maxGain, 'minGain': minGain}, index=[0])

	"""
	ABSTRACT METHODS
	"""

	@abstractmethod
	def possiblyOperationMorning(self, data):
		"""
		Function prototype that calls the buy function and updates the investment values
		:param data: Decision data based on the type of indicator
		"""
		pass

	@abstractmethod
	def possiblyOperationAfternoon(self, data):
		"""
		Function prototype that calls the sell function and updates the investment values
		:param data: Decision data based on the type of indicator
		"""
		pass