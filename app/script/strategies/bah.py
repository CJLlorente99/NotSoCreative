from dailyStrategy import DailyStrategy
import pandas as pd
from jsonManagement.inversionStrategyJSONAPI import Strategy


class BaH(DailyStrategy):
	def __init__(self, record: pd.DataFrame, strategyDefinition: Strategy):
		super().__init__(record, strategyDefinition)
		if len(record) != 0:
			self.nDay = record['nDay'].values[-1]
		else:
			self.nDay = 0

	def possiblyOperationMorning(self, data):
		self.perToInvest = 0
		if self.nDay == 0:
			self.perToInvest = 1

	def possiblyOperationAfternoon(self, data):
		self.perToInvest = 0
		if self.nDay == 0:
			self.perToInvest = 1