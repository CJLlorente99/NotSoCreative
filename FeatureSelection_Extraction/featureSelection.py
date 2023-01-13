import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

def main():
	data = pd.read_csv('featureSelectionDataset.csv', sep=',', header=0, index_col=0, parse_dates=True, decimal=".")

	# Each row of the data frame has (for day t)
	# ALL INDICATORS HAVE ALREADY BEEN SHIFTED AS NONE DEPENDS ON THE OPEN VALUE
	data['log_Close'] = np.log(data['Close'])
	data["log_Open"] = np.log(data["Open"])
	data["Return_close"] = data["log_Close"] - data["log_Close"].shift(+1)
	data["Return_open"] = data["log_Open"] - data["log_Open"].shift(+1)
	data["Return_intraday"] = data["log_Close"] - data["log_Open"].shift(+1)
	data['Return_interday'] = data["log_Open"] - data["log_Close"]
	data['Diff_open'] = data["Open"] - data["Open"].shift()
	data['Diff_close'] = data['Close'] - data['Close'].shift()
	data['Diff_intraday'] = data['Close'] - data['Open'].shift(1)
	data['Diff_interday'] = data['Open'] - data['Close']
	yx = (data["log_Open"].shift(-1) - data["log_Open"]).copy()
	yx.dropna(inplace=True)
	yx = yx[1:]

	data.dropna(inplace=True)
	data = data[:-1]
	y = [1 if yx.iloc[i] > 0 else 0 for i in range(len(data))]

	print(f'Features before correlation matrix {len(data.columns)}')

	# Create correlation matrix
	corr_matrix = data.corr(method='pearson')
	upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
	to_drop = []
	for column in upper_tri.columns:
		if any(upper_tri[column] > 0.95):
			to_drop.append(column)
	data = data.drop(to_drop, axis=1)

	print(f'Features after correlation matrix {len(data.columns)}')

	X = data
	splitlimit = (len(data) - 100)

	X_train, X_test = X[:splitlimit], X[splitlimit:]
	y_train, y_test = y[:splitlimit], y[splitlimit:]

	model = DecisionTreeRegressor()
	model.fit(np.asarray(X_train), y_train)  # , validation_split=0.3)
	y_pred = model.predict(X_test)

	feat_imp = model.feature_importances_
	idx = np.argsort(feat_imp)
	idx = idx[::-1]
	feat_imp_sort = np.take_along_axis(feat_imp, idx, axis=0)
	X_sort = data.iloc[:, idx]
	k = 14
	X_sort = X_sort.iloc[:, :k]
	feat_imp_k = feat_imp_sort[:k]
	print('sorted:', feat_imp_k)
	fig = plt.figure()
	plt.barh(X_sort.columns, feat_imp_k)
	plt.title('feature importance')
	plt.tight_layout()
	plt.show()
	fig.savefig('importance.png', transparent=True)

	corr_matrix = X_sort.corr(method='pearson')
	f, ax = plt.subplots(figsize=(16, 8))
	sns.heatmap(corr_matrix, annot=True, fmt='.2f', linewidth=0.4, annot_kws={'size': 10}, cmap='coolwarm', ax=ax)
	plt.tight_layout()
	plt.xticks(fontsize=10)
	plt.yticks(fontsize=10)
	plt.show()
	f.savefig('corr.png', transparent=True)

if __name__ == '__main__':
	main()
