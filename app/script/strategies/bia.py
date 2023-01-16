from dailyStrategy import DailyStrategy


class BIA(DailyStrategy):
	def possiblyOperationMorning(self, data):
		self.investedMoney /= data["Open"][-1] / data["Close"][-1]
		self.investedMoney *= data["Close"][-1] / data["Open"][-2]

		if data['Open'].values[-1] > data['Open'].values[-2]:
			self.perToInvest = 1
		elif data['Open'].values[-1] < data['Open'].values[-2]:
			self.perToInvest = -1
		else:
			self.perToInvest = 0

	def possiblyOperationAfternoon(self, data):
		self.perToInvest = 0
		self.investedMoney /= data["Close"][-1] / data["Open"][-1]
		self.investedMoney *= data["Open"][-1] / data["Close"][-2]

		# if data['Close'].values[-1] > data['Open'].values[-1]:
		# 	self.perToInvest = 1
		# elif data['Close'].values[-1] < data['Open'].values[-1]:
		# 	self.perToInvest = -1
		# else:
		# 	self.perToInvest = 0
