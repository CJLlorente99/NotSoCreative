import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

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
	window = 0
	X = data_shift(X, window, inp)
	# print(X)
	X = np.asarray(X)

	# scaling: try different scaling or no scaling
	# scaler = StandardScaler()
	# X = scaler.fit_transform(X)

	model = RandomForestClassifier(bootstrap=False, criterion='entropy', max_depth=4, max_features='auto', min_samples_leaf=6,
								   min_samples_split=2, n_estimators=753)
	model.fit(X, y_target)
	joblib.dump(model, "../data/random_forest_class.joblib")

def data_shift(X, window, inp):
	X_plc = X
	for i in range(window):
		X_shift_bw = X.shift(periods=(i + 1), fill_value=0)
		inp_bw = [x + f'_-{i + 1}' for x in inp]
		X_shift_bw.columns = inp_bw
		X_plc = pd.concat([X_plc, X_shift_bw], axis=1)

	return X_plc

if __name__ == '__main__':
	main()