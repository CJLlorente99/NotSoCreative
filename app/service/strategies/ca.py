import math

from dailyStrategy import DailyStrategy
import pandas as pd
from jsonManagement.inversionStrategyJSONAPI import Strategy

class CA(DailyStrategy):
	def __init__(self, record: pd.DataFrame, strategyDefinition: Strategy):
		super().__init__(record, strategyDefinition)
		if len(record) != 0:
			self.nDay = math.floor(len(record)/2)
		else:
			self.nDay = 0
		self.dailyWindow = 0.05
	def possiblyOperationMorning(self, data):
		self.perToInvest = 0
		if 1 - self.nDay * self.dailyWindow > 0:
			self.perToInvest = self.dailyWindow / (1 - self.nDay * self.dailyWindow)

	def possiblyOperationAfternoon(self, data):
		self.perToInvest = 0
		# if 1 - self.nDay * self.dailyWindow > 0:
		# 	self.perToInvest = self.dailyWindow / (1 - self.nDay * self.dailyWindow)