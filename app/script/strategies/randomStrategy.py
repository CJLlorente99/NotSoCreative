from app.script.dailyStrategy import DailyStrategy
import pandas as pd
from inversionStrategyJSONAPI import Strategy
from numpy import random


class Random(DailyStrategy):
	def possiblyOperationMorning(self, data):
		self.perToInvest = random.uniform(-1, 1)

	def possiblyOperationAfternoon(self, data):
		self.perToInvest = random.uniform(-1, 1)
