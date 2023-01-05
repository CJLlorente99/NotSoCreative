import pandas as pd
from abc import ABC, abstractmethod
from jsonManagement.inversionStrategyJSONAPI import Strategy


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
		# Afternoon -> Apply logic (interday return prediction) and update portfolio value
		elif operation == 'Afternoon':
			moneyInvested = self.brokerAfternoon(inputs)

		return pd.DataFrame({'investorStrategy': self.name, 'MoneyInvested': self.investedMoney, 'MoneyNotInvested': self.nonInvestedMoney,
							 'MoneyInvestedToday': moneyInvested, 'PerInvestToday': self.perToInvest,
							 'TotalPortfolioValue': self.investedMoney + self.nonInvestedMoney}, index=[0])

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
		if self.perToInvest >= 0:  # We buy
			moneyInvested = self.perToInvest * self.nonInvestedMoney
			self.investedMoney += self.perToInvest * self.nonInvestedMoney
			self.nonInvestedMoney -= self.perToInvest * self.nonInvestedMoney

		if self.perToInvest < 0:  # We sell
			moneyInvested = self.perToInvest * self.investedMoney
			self.nonInvestedMoney += -self.perToInvest * self.investedMoney
			self.investedMoney -= -self.perToInvest * self.investedMoney

		return moneyInvested

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