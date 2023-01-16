from dailyStrategy import DailyStrategy
from numpy import random


class Random(DailyStrategy):
	def possiblyOperationMorning(self, data):
		self.perToInvest = random.uniform(-1, 1)

	def possiblyOperationAfternoon(self, data):
		# self.perToInvest = random.uniform(-1, 1)
		self.perToInvest = 0
