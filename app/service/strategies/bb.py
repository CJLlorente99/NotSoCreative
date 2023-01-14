from dailyStrategy import DailyStrategy
import math


class BB(DailyStrategy):

	def possiblyOperationMorning(self, data):
		bb = data['bb_w10_stdDev1.5'][-1]
		if bb > 1.9:
			self.perToInvest = math.tanh(2.4 * (bb - 1.8) ** 0.5)
		elif bb > 0.8:
			self.perToInvest = math.tanh(2.4 * (bb - 0.8) ** 0.5)
		else:
			self.perToInvest = 0

	def possiblyOperationAfternoon(self, data):
		bb = data['bb_w10_stdDev1.5'][-1]
		if bb > 1.9:
			self.perToInvest = math.tanh(2.4 * (bb - 1.8) ** 0.5)
		elif bb > 0.8:
			self.perToInvest = math.tanh(2.4 * (bb - 0.8) ** 0.5)
		else:
			self.perToInvest = 0


