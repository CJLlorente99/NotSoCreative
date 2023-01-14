from dailyStrategy import DailyStrategy
import pandas as pd
from jsonManagement.inversionStrategyJSONAPI import Strategy
import xgboost as xgb
import numpy as np

class XGBStrategy(DailyStrategy):

	def __init__(self, record: pd.DataFrame, strategyDefinition: Strategy, fileName):
		super().__init__(record, strategyDefinition)
		self.model = xgb.XGBClassifier()
		self.model.load_model(fileName)
	def possiblyOperationMorning(self, data):
		res = prepareData(data.drop(['Open', 'Close', 'High', 'Low'], axis=1))
		self.perToInvest = self.model.predict([res[-1]])[0]

	def possiblyOperationAfternoon(self, data):
		self.perToInvest = -1

def prepareData(data: pd.DataFrame):
	# include t-window data points as additional features
	inp = data.columns.values.tolist()
	# window mabye 0 to 1,2,3
	X = data_shift(data, 3, inp)
	# print(X)
	X = np.asarray(X)
	return X

def data_shift(X, window, inp):
	X_plc = X
	for i in range(window):
		X_shift_bw = X.shift(periods=(i + 1), fill_value=0)
		inp_bw = [x + f'_-{i + 1}' for x in inp]
		X_shift_bw.columns = inp_bw
		X_plc = pd.concat([X_plc, X_shift_bw], axis=1)

	return X_plc