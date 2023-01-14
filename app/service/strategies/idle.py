from dailyStrategy import DailyStrategy


class Idle(DailyStrategy):
	def possiblyOperationMorning(self, data):
		self.perToInvest = 0

	def possiblyOperationAfternoon(self, data) :
		self.perToInvest = 0
