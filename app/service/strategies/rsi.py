from dailyStrategy import DailyStrategy
import math


class RSI(DailyStrategy):

	def possiblyOperationMorning(self, data):
		if data['rsi_w3'][-1] < 27.5:  # Buy linearly then with factor f
			self.perToInvest = math.tanh(1.1 * (27.5 - data['rsi_w3'][-1]) ** 2.4)
		elif data['rsi_w3'][-1] > 61:  # Buy linearly then with factor f
			self.perToInvest = math.tanh(1.1 * (data['rsi_w3'][-1] - 61) ** 2.4)
		else:
			self.perToInvest = 0

	def possiblyOperationAfternoon(self, data):
		if data['rsi_w3'][-1] < 27.5:  # Buy linearly then with factor f
			self.perToInvest = math.tanh(1.1 * (27.5 - data['rsi_w3'][-1]) ** 2.4)
		elif data['rsi_w3'][-1] > 61:  # Buy linearly then with factor f
			self.perToInvest = math.tanh(1.1 * (data['rsi_w3'][-1] - 61) ** 2.4)
		else:
			self.perToInvest = 0