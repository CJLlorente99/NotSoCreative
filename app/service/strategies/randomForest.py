from dailyStrategy import DailyStrategy
from jsonManagement.inversionStrategyJSONAPI import Strategy
import pandas as pd
import joblib

class RandomForestStrategy(DailyStrategy):

	def __init__(self, record: pd.DataFrame, strategyDefinition: Strategy, fileName):
		super().__init__(record, strategyDefinition)
		self.model = joblib.load(fileName)

	def possiblyOperationMorning(self, data):
		self.perToInvest = self.model.predict([data.drop(['Open', 'Close', 'High', 'Low'], axis=1).to_numpy()[-1]])[0]

	def possiblyOperationAfternoon(self, data):
		self.perToInvest = -1