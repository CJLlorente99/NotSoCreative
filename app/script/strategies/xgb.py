from dailyStrategy import DailyStrategy
import pandas as pd
from jsonManagement.inversionStrategyJSONAPI import Strategy
import xgboost as xgb
import numpy as np

class XGBStrategy(DailyStrategy):

	def __init__(self, record: pd.DataFrame, strategyDefinition: Strategy, var):
		super().__init__(record, strategyDefinition)
		self.var = var

	def possiblyOperationMorning(self, data):
		if self.var == 1:
			model = self.createAndTrainModel1(data)
		elif self.var == 2:
			model = self.createAndTrainModel2(data)
		res = prepareData(data.drop(['Open', 'Close', 'High', 'Low'], axis=1))
		self.perToInvest = model.predict([res[-1]])[0]

	def possiblyOperationAfternoon(self, data):
		self.perToInvest = 0

	def createAndTrainModel1(self, df: pd.DataFrame):
		data = df.copy().drop(['Close', 'High', 'Low'], axis=1)

		# Intraday return (target)
		data['Target'] = data['Open'].shift(-1) - data['Open']
		data = data.drop(['Open'], axis=1)
		data = data[1:-1]  # last is the input for prediction
		y_target = np.asarray([1 if data.Target[i] > 0 else 0 for i in range(len(data))]).reshape(-1, 1)

		X = data.drop(['Target'], axis=1)

		# include t-window data points as additional features
		inp = X.columns.values.tolist()
		# window mabye 0 to 1,2,3
		X = data_shift(X, 3, inp)
		X = np.asarray(X)

		model = xgb.XGBClassifier(colsample_bylevel=0.6452280156999572, colsample_bytree=0.9581223733932949,
								  learning_rate=0.06266659029259186,
								  max_depth=14, n_estimators=1000, subsample=1)
		model.fit(X, y_target)

		return model

	def createAndTrainModel2(self, df: pd.DataFrame):
		data = df.copy().drop(['Close', 'High', 'Low'], axis=1)

		# Intraday return (target)
		data['Target'] = data['Open'].shift(-1) - data['Open']
		data = data.drop(['Open'], axis=1)
		data = data[1:-1]  # last is the input for prediction
		y_target = np.asarray([1 if data.Target[i] > 0 else 0 for i in range(len(data))]).reshape(-1, 1)

		X = data.drop(['Target'], axis=1)

		# include t-window data points as additional features
		inp = X.columns.values.tolist()
		# window mabye 0 to 1,2,3
		X = data_shift(X, 3, inp)
		X = np.asarray(X)

		model = xgb.XGBClassifier(colsample_bylevel=0.4, colsample_bytree=1, learning_rate=0.1331430678429723,
							  max_depth=17, n_estimators=500, subsample=0.5041453030448557)
		model.fit(X, y_target)

		return model

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