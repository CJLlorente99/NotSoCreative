from dailyStrategy import DailyStrategy
from jsonManagement.inversionStrategyJSONAPI import Strategy
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class RandomForestStrategy(DailyStrategy):

	def possiblyOperationMorning(self, data):
		model = self.createAndTrainModel(data)
		self.perToInvest = model.predict([data.drop(['Open', 'Close', 'High', 'Low'], axis=1).to_numpy()[-1]])[0]

	def possiblyOperationAfternoon(self, data):
		self.perToInvest = 0

	def createAndTrainModel(self, df: pd.DataFrame):
		data = df.copy().drop(['Close', 'High', 'Low'], axis=1)

		# Intraday return (target)
		data['Target'] = data['Open'].shift(-1) - data['Open']
		data = data.drop(['Open'], axis=1)
		data = data[1:-1]  # last is the input for prediction (1 has nan)
		y_target = np.asarray([1 if data.Target[i] > 0 else 0 for i in range(len(data))]).reshape(-1, 1)

		X = data.drop(['Target'], axis=1)

		# include t-window data points as additional features
		inp = X.columns.values.tolist()
		# window mabye 0 to 1,2,3
		window = 0
		X = data_shift(X, window, inp)
		X = np.asarray(X)

		model = RandomForestClassifier(bootstrap=False, criterion='entropy', max_depth=4, max_features='sqrt',
									   min_samples_leaf=3,
									   min_samples_split=4, n_estimators=108)
		model.fit(X, y_target.reshape((y_target.shape[0],)))

		return model

def data_shift(X, window, inp):
	X_plc = X
	for i in range(window):
		X_shift_bw = X.shift(periods=(i + 1), fill_value=0)
		inp_bw = [x + f'_-{i + 1}' for x in inp]
		X_shift_bw.columns = inp_bw
		X_plc = pd.concat([X_plc, X_shift_bw], axis=1)

	return X_plc