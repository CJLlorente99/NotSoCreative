import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler


def data_shift(X, window, inp):
	X_plc = X
	for i in range(window):
		X_shift_bw = X.shift(periods=(i + 1), fill_value=0)
		inp_bw = [x + f'_-{i + 1}' for x in inp]
		X_shift_bw.columns = inp_bw
		X_plc = pd.concat([X_plc, X_shift_bw], axis=1)

	return X_plc

def main():
	data = pd.read_csv("../Paul/featureSelectionDataset_Paul_Class_shift_forBackt_close_open.csv", sep=',', header=0,
						   index_col=0, parse_dates=True,
						   decimal=".")
	# because open and close not important feat but i need it for backtesting
	data = data.drop(['Open', 'Close'], axis=1)
	y_target = np.asarray([1 if data.Target[i] > 0 else 0 for i in range(len(data))]).reshape(-1, 1)
	X = data.drop(['Target'], axis=1)

	# include t-window data points as additional features
	inp = X.columns.values.tolist()
	# window mabye 0 to 1,2,3
	window = 3
	X = data_shift(X, window, inp)
	# print(X)
	X = np.asarray(X)

	# scaling: try different scaling or no scaling
	# scaler = StandardScaler()
	# X = scaler.fit_transform(X)

	model = xgb.XGBClassifier(colsample_bylevel=0.4, colsample_bytree=1, learning_rate=0.1331430678429723,
							  max_depth=17, n_estimators=500, subsample=0.5041453030448557)
	model.fit(X, y_target)

	model.save_model('../data/xgb_model(2).json')

if __name__ == '__main__':
	main()