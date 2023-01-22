from dailyStrategy import DailyStrategy
import pandas as pd
from jsonManagement.inversionStrategyJSONAPI import Strategy
import math


class BaH(DailyStrategy):
	def __init__(self, record: pd.DataFrame, strategyDefinition: Strategy):
		super().__init__(record, strategyDefinition)
		if len(record) != 0:
			self.nDay = math.floor(len(record)/2)
		else:
			self.nDay = 0

	def possiblyOperationMorning(self, data):
		self.perToInvest = 0
		if self.nDay == 0 or self.nDay == 1:
			self.perToInvest = 1

	def possiblyOperationAfternoon(self, data):
		self.perToInvest = 0