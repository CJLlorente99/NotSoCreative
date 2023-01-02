from keras.layers import Dropout, LSTM, Dense
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from classes.investorClass import Investor
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from TAIndicators.atr import averageTrueRange
from TAIndicators.stochasticRsi import stochasticRSI
from TAIndicators.ma import movingAverageConvergenceDivergence
from TAIndicators.adx import averageDirectionalMovementIndex
from TAIndicators.rsi import relativeStrengthIndex
from TAIndicators.bb import bollingerBands
from TAIndicators.aroon import aroon
from classes.investorParamsClass import ATRInvestorParams, ADXInvestorParams, StochasticRSIInvestorParams, MACDInvestorParams, RSIInvestorParams, BBInvestorParams, AroonInvestorParams
from keras.backend import clear_session

class InvestorLSTMConfidenceClass(Investor):
	def __init__(self, initialInvestment=10000, n_members=10):
		super().__init__(initialInvestment)
		self.n_members = n_members

	def returnBrokerUpdate(self, moneyInvestedToday, moneySoldToday, data):
		aux = pd.DataFrame()
		for i in range(self.n_members):
			aux = pd.concat([aux, pd.DataFrame({"voter" + str(i): data["lstmConfidence"][i]})], axis=1)

		return pd.concat([aux, pd.DataFrame(
			{'moneyToInvestLSTM': moneyInvestedToday, 'moneyToSellLSTM': moneySoldToday,
			 'investedMoneyLSTM': self.investedMoney, 'nonInvestedMoneyLSTM': self.nonInvestedMoney}, index=[0])], axis=1)

	def possiblyInvestTomorrow(self, data) -> float:
		self.perToInvest = 0
		if sum(data["lstmConfidence"])/self.n_members > 0.5:
			self.perToInvest = 1

	def possiblySellTomorrow(self, data) -> float:
		self.perToSell = 0
		if sum(data["lstmConfidence"]) / self.n_members < 0.5:
			self.perToSell = 1

	def plotEvolution(self, expData, stockMarketData, recordPredictedValue=None):
		"""
		Function that plots the actual status of the investor investment as well as the decisions that have been made
		:param indicatorData: Data belonging to the indicator used to take decisions
		:param stockMarketData: df with the stock market data
		:param recordPredictedValue: Predicted data dataframe
		"""
		self.record = self.record.iloc[1:]
		# Plot indicating the evolution of the total value and contain (moneyInvested and moneyNotInvested)
		fig = go.Figure()
		fig.add_trace(go.Scatter(name="Money Invested", x=self.record.index, y=self.record["moneyInvested"], stackgroup="one"))
		fig.add_trace(go.Scatter(name="Money Not Invested", x=self.record.index, y=self.record["moneyNotInvested"], stackgroup="one"))
		fig.add_trace(go.Scatter(name="Total Value", x=self.record.index, y=self.record["totalValue"]))
		fig.update_layout(
			title="Evolution of Porfolio using LSTM Confidence (" + self.record.index[0].strftime(
				"%d/%m/%Y") + "-" +
				  self.record.index[-1].strftime("%d/%m/%Y") + ")", xaxis_title="Date",
			yaxis_title="Value [$]", hovermode='x unified')
		fig.write_image("images/EvolutionPorfolioLSTMConfidence(" + self.record.index[0].strftime(
				"%d_%m_%Y") + "-" +
				  self.record.index[-1].strftime("%d_%m_%Y") + ").png",scale=6, width=1080, height=1080)
		# fig.show()

		# Plot indicating the value of the indicator, the value of the stock market and the decisions made
		fig = make_subplots(rows=2, cols=2, specs=[[{"secondary_y": True, "colspan": 2}, None], [{"secondary_y": False}, {"secondary_y": False}]])

		fig.add_trace(go.Scatter(name="Stock Market Value Open", x=self.record.index,
								 y=stockMarketData.Open[-len(self.record.index):]), row=1, col=1, secondary_y=False)
		fig.add_trace(go.Scatter(name="Stock Market Value Close", x=self.record.index, visible='legendonly',
								 y=stockMarketData.Close[-len(self.record.index):]), row=1, col=1, secondary_y=False)
		fig.add_trace(go.Bar(name="Money Invested Today", x=self.record.index, y=self.record["moneyInvestedToday"], marker_color="green"), row=2, col=1, secondary_y=False)
		fig.add_trace(go.Bar(name="Money Sold Today", x=self.record.index, y=-self.record["moneySoldToday"], marker_color="red"), row=2, col=1, secondary_y=False)

		for i in range(self.n_members):
			fig.add_trace(go.Bar(name="Voter " + str(i), x=self.record.index, y=expData["voter"+str(i)]), row=2, col=2)
		fig.update_xaxes(title_text="Date", row=1, col=1)
		fig.update_xaxes(title_text="Date", row=2, col=1)
		fig.update_layout(
			title="Decision making under LSTM Confidence (" + self.record.index[0].strftime("%d/%m/%Y") + "-" +
				  self.record.index[-1].strftime("%d/%m/%Y") + ")", hovermode='x unified')
		fig.write_image("images/DecisionMakingLSTMConfidence(" + self.record.index[0].strftime("%d_%m_%Y") + "-" +
				  self.record.index[-1].strftime("%d_%m_%Y") + ").png",scale=6, width=1080, height=1080)
		# fig.show()

	def trainAndPredict(self, dataUntilToday: pd.DataFrame):
		data = pd.DataFrame()

		# Add indicators
		# Return_close
		data['log_Close'] = np.log(dataUntilToday['Close'])
		data["Return_close"] = data["log_Close"] - data["log_Close"].shift(+1)

		# Diff_close
		data['Diff_close'] = dataUntilToday['Close'] - dataUntilToday['Close'].shift()

		# stochRsi_stochrsi_w2_s126_s24
		params = StochasticRSIInvestorParams(2, 26, 4)
		data['stochRsi_stochrsi_w2_s126_s24'] = stochasticRSI(dataUntilToday['Close'], params)['stochrsi']

		# stochRsi_stochrsi_w2_s137_s234
		params = StochasticRSIInvestorParams(2, 37, 34)
		data['stochRsi_stochrsi_w2_s137_s234'] = stochasticRSI(dataUntilToday['Close'], params)['stochrsi']

		# rsi_w1
		params = RSIInvestorParams(1)
		data['rsi_w1'] = relativeStrengthIndex(dataUntilToday['Close'], params)

		# stochRsi_stochrsi_w2_s114_s236
		params = StochasticRSIInvestorParams(2, 14, 36)
		data['stochRsi_stochrsi_w2_s114_s236'] = stochasticRSI(dataUntilToday['Close'], params)['stochrsi']

		# stochRsi_stochrsi_w3_s12_s222
		params = StochasticRSIInvestorParams(3, 2, 22)
		data['stochRsi_stochrsi_w3_s12_s222'] = stochasticRSI(dataUntilToday['Close'], params)['stochrsi']

		# stochRsi_stochrsi_w2_s13_s218
		params = StochasticRSIInvestorParams(2, 3, 18)
		data['stochRsi_stochrsi_w2_s13_s218'] = stochasticRSI(dataUntilToday['Close'], params)['stochrsi']

		# stochRsi_stochrsi_w2_s115_s236
		params = StochasticRSIInvestorParams(2, 15, 36)
		data['stochRsi_stochrsi_w2_s115_s236'] = stochasticRSI(dataUntilToday['Close'], params)['stochrsi']

		# stochRsi_stochrsi_w3_s149_s21
		params = StochasticRSIInvestorParams(3, 49, 1)
		data['stochRsi_stochrsi_w3_s149_s21'] = stochasticRSI(dataUntilToday['Close'], params)['stochrsi']

		# stochRsi_stochrsi_w3_s129_s238
		params = StochasticRSIInvestorParams(3, 29, 38)
		data['stochRsi_stochrsi_w3_s129_s238'] = stochasticRSI(dataUntilToday['Close'], params)['stochrsi']

		# rsi_w2
		params = RSIInvestorParams(2)
		data['rsi_w2'] = relativeStrengthIndex(dataUntilToday['Close'], params)

		# bb_pband_w4_stdDev3.3223760630638397
		params = BBInvestorParams(4, 3.32)
		data['bb_pband_w4_stdDev3.3223760630638397'] = bollingerBands(dataUntilToday['Close'], params)

		# stochRsi_stochrsi_w4_s123_s245
		params = StochasticRSIInvestorParams(4, 23, 45)
		data['stochRsi_stochrsi_w4_s123_s245'] = stochasticRSI(dataUntilToday['Close'], params)['stochrsi']

		# macd_difffW1_sW5_signal12
		params = MACDInvestorParams(1, 5, 12)
		data['macd_difffW1_sW5_signal12'] = movingAverageConvergenceDivergence(dataUntilToday['Close'], params)

		# bb_pband_w4_stdDev0.7177075201973817
		params = BBInvestorParams(4, 0.72)
		data['bb_pband_w4_stdDev0.7177075201973817'] = bollingerBands(dataUntilToday['Close'], params)

		# Diff_open
		data['Diff_open'] = dataUntilToday["Open"] - dataUntilToday["Open"].shift()

		# Return_open
		data['log_Open'] = np.log(dataUntilToday['Open'])
		data["Return_open"] = data["log_Open"] - data["log_Open"].shift(+1)

		# aroon_up_w4
		params = AroonInvestorParams(4)
		data['aroon_up_w4'] = aroon(dataUntilToday['Close'], params)

		# Class_close
		y_close = np.asarray([1 if data.Return_close[i] > 0 else 0 for i in range(len(data))]).reshape(-1, 1)

		# target
		data["Return_target"] = data["log_Open"].shift(-1) - data["log_Open"]
		y_target = np.asarray([1 if data.Return_target[i] > 0 else 0 for i in range(len(data))]).reshape(-1, 1)

		# Drop based on correlation matrix
		data = data.drop(['log_Open', 'log_Close', 'Return_target', 'stochRsi_stochrsi_w2_s114_s236', 'stochRsi_stochrsi_w2_s126_s24',
                      'stochRsi_stochrsi_w2_s137_s234', 'stochRsi_stochrsi_w3_s149_s21', 'stochRsi_stochrsi_w3_s12_s222',
                     'stochRsi_stochrsi_w2_s115_s236'], axis=1)
		data.dropna(inplace=True)

		# scale data, and then add the class as feature input
		scaler = StandardScaler()
		data_set_scaled = scaler.fit_transform(data)
		data_set_scaled = np.concatenate((data_set_scaled, y_close[1:]), axis=1)
		data_set_scaled = np.concatenate((data_set_scaled, y_target[1:]), axis=1)

		# choose how many look back days
		backcandles = 30

		# choose columns: all but target variable (its last column)
		liste = list(range(0, data.shape[1] - 1))

		# split data into train test sets
		pred_days = 1

		# prepare data for lstm
		X_train, X_test, y_train, y_test = prepare_data(data_set_scaled, backcandles, liste, pred_days)

		# train and predict
		# n_members -> how many predictors we wanted to use
		epochs = 45
		batch_size = 10
		ensemble, y_pred, accuracy = fit_ensemble(self.n_members, X_train, X_test, y_train, y_test, epochs, batch_size)
		clear_session()
		return y_pred

class InvestorLSTMConfidenceClassProb(Investor):
	def __init__(self, initialInvestment=10000, n_members=10):
		super().__init__(initialInvestment)
		self.n_members = n_members

	def returnBrokerUpdate(self, moneyInvestedToday, moneySoldToday, data):
		aux = pd.DataFrame()
		for i in range(self.n_members):
			aux = pd.concat([aux, pd.DataFrame({"voter" + str(i): data["lstmConfidence"][i]})], axis=1)

		return pd.concat([aux, pd.DataFrame(
			{'moneyToInvestLSTM': moneyInvestedToday, 'moneyToSellLSTM': moneySoldToday,
			 'investedMoneyLSTM': self.investedMoney, 'nonInvestedMoneyLSTM': self.nonInvestedMoney}, index=[0])], axis=1)

	def possiblyInvestTomorrow(self, data) -> float:
		self.perToInvest = 0
		if sum(data["lstmConfidence"])/self.n_members > 0.5:
			self.perToInvest = (sum(data["lstmConfidence"])/self.n_members - 0.5)*2

	def possiblySellTomorrow(self, data) -> float:
		self.perToSell = 0
		if sum(data["lstmConfidence"]) / self.n_members < 0.5:
			self.perToSell = (sum(data["lstmConfidence"]) / self.n_members - 0.5) * 2

	def plotEvolution(self, expData, stockMarketData, recordPredictedValue=None):
		"""
		Function that plots the actual status of the investor investment as well as the decisions that have been made
		:param indicatorData: Data belonging to the indicator used to take decisions
		:param stockMarketData: df with the stock market data
		:param recordPredictedValue: Predicted data dataframe
		"""
		self.record = self.record.iloc[1:]
		# Plot indicating the evolution of the total value and contain (moneyInvested and moneyNotInvested)
		fig = go.Figure()
		fig.add_trace(go.Scatter(name="Money Invested", x=self.record.index, y=self.record["moneyInvested"], stackgroup="one"))
		fig.add_trace(go.Scatter(name="Money Not Invested", x=self.record.index, y=self.record["moneyNotInvested"], stackgroup="one"))
		fig.add_trace(go.Scatter(name="Total Value", x=self.record.index, y=self.record["totalValue"]))
		fig.update_layout(
			title="Evolution of Porfolio using LSTM Confidence (" + self.record.index[0].strftime(
				"%d/%m/%Y") + "-" +
				  self.record.index[-1].strftime("%d/%m/%Y") + ")", xaxis_title="Date",
			yaxis_title="Value [$]", hovermode='x unified')
		fig.write_image("images/EvolutionPorfolioLSTMConfidence(" + self.record.index[0].strftime(
				"%d_%m_%Y") + "-" +
				  self.record.index[-1].strftime("%d_%m_%Y") + ").png",scale=6, width=1080, height=1080)
		# fig.show()

		# Plot indicating the value of the indicator, the value of the stock market and the decisions made
		fig = make_subplots(rows=2, cols=2, specs=[[{"secondary_y": True, "colspan": 2}, None], [{"secondary_y": False}, {"secondary_y": False}]])

		fig.add_trace(go.Scatter(name="Stock Market Value Open", x=self.record.index,
								 y=stockMarketData.Open[-len(self.record.index):]), row=1, col=1, secondary_y=False)
		fig.add_trace(go.Scatter(name="Stock Market Value Close", x=self.record.index, visible='legendonly',
								 y=stockMarketData.Close[-len(self.record.index):]), row=1, col=1, secondary_y=False)
		fig.add_trace(go.Bar(name="Money Invested Today", x=self.record.index, y=self.record["moneyInvestedToday"], marker_color="green"), row=2, col=1, secondary_y=False)
		fig.add_trace(go.Bar(name="Money Sold Today", x=self.record.index, y=-self.record["moneySoldToday"], marker_color="red"), row=2, col=1, secondary_y=False)

		for i in range(self.n_members):
			fig.add_trace(go.Bar(name="Voter " + str(i), x=self.record.index, y=expData["voter"+str(i)]), row=2, col=2)
		fig.update_xaxes(title_text="Date", row=1, col=1)
		fig.update_xaxes(title_text="Date", row=2, col=1)
		fig.update_layout(
			title="Decision making under LSTM Confidence (" + self.record.index[0].strftime("%d/%m/%Y") + "-" +
				  self.record.index[-1].strftime("%d/%m/%Y") + ")", hovermode='x unified')
		fig.write_image("images/DecisionMakingLSTMConfidence(" + self.record.index[0].strftime("%d_%m_%Y") + "-" +
				  self.record.index[-1].strftime("%d_%m_%Y") + ").png",scale=6, width=1080, height=1080)
		# fig.show()

	def trainAndPredict(self, dataUntilToday: pd.DataFrame):
		data = pd.DataFrame()

		# Add indicators
		# Return_close
		data['log_Close'] = np.log(dataUntilToday['Close'])
		data["Return_close"] = data["log_Close"] - data["log_Close"].shift(+1)

		# Diff_close
		data['Diff_close'] = dataUntilToday['Close'] - dataUntilToday['Close'].shift()

		# stochRsi_stochrsi_w2_s126_s24
		params = StochasticRSIInvestorParams(2, 26, 4)
		data['stochRsi_stochrsi_w2_s126_s24'] = stochasticRSI(dataUntilToday['Close'], params)['stochrsi']

		# stochRsi_stochrsi_w2_s137_s234
		params = StochasticRSIInvestorParams(2, 37, 34)
		data['stochRsi_stochrsi_w2_s137_s234'] = stochasticRSI(dataUntilToday['Close'], params)['stochrsi']

		# rsi_w1
		params = RSIInvestorParams(1)
		data['rsi_w1'] = relativeStrengthIndex(dataUntilToday['Close'], params)

		# stochRsi_stochrsi_w2_s114_s236
		params = StochasticRSIInvestorParams(2, 14, 36)
		data['stochRsi_stochrsi_w2_s114_s236'] = stochasticRSI(dataUntilToday['Close'], params)['stochrsi']

		# stochRsi_stochrsi_w3_s12_s222
		params = StochasticRSIInvestorParams(3, 2, 22)
		data['stochRsi_stochrsi_w3_s12_s222'] = stochasticRSI(dataUntilToday['Close'], params)['stochrsi']

		# stochRsi_stochrsi_w2_s13_s218
		params = StochasticRSIInvestorParams(2, 3, 18)
		data['stochRsi_stochrsi_w2_s13_s218'] = stochasticRSI(dataUntilToday['Close'], params)['stochrsi']

		# stochRsi_stochrsi_w2_s115_s236
		params = StochasticRSIInvestorParams(2, 15, 36)
		data['stochRsi_stochrsi_w2_s115_s236'] = stochasticRSI(dataUntilToday['Close'], params)['stochrsi']

		# stochRsi_stochrsi_w3_s149_s21
		params = StochasticRSIInvestorParams(3, 49, 1)
		data['stochRsi_stochrsi_w3_s149_s21'] = stochasticRSI(dataUntilToday['Close'], params)['stochrsi']

		# stochRsi_stochrsi_w3_s129_s238
		params = StochasticRSIInvestorParams(3, 29, 38)
		data['stochRsi_stochrsi_w3_s129_s238'] = stochasticRSI(dataUntilToday['Close'], params)['stochrsi']

		# rsi_w2
		params = RSIInvestorParams(2)
		data['rsi_w2'] = relativeStrengthIndex(dataUntilToday['Close'], params)

		# bb_pband_w4_stdDev3.3223760630638397
		params = BBInvestorParams(4, 3.32)
		data['bb_pband_w4_stdDev3.3223760630638397'] = bollingerBands(dataUntilToday['Close'], params)

		# stochRsi_stochrsi_w4_s123_s245
		params = StochasticRSIInvestorParams(4, 23, 45)
		data['stochRsi_stochrsi_w4_s123_s245'] = stochasticRSI(dataUntilToday['Close'], params)['stochrsi']

		# macd_difffW1_sW5_signal12
		params = MACDInvestorParams(1, 5, 12)
		data['macd_difffW1_sW5_signal12'] = movingAverageConvergenceDivergence(dataUntilToday['Close'], params)

		# bb_pband_w4_stdDev0.7177075201973817
		params = BBInvestorParams(4, 0.72)
		data['bb_pband_w4_stdDev0.7177075201973817'] = bollingerBands(dataUntilToday['Close'], params)

		# Diff_open
		data['Diff_open'] = dataUntilToday["Open"] - dataUntilToday["Open"].shift()

		# Return_open
		data['log_Open'] = np.log(dataUntilToday['Open'])
		data["Return_open"] = data["log_Open"] - data["log_Open"].shift(+1)

		# aroon_up_w4
		params = AroonInvestorParams(4)
		data['aroon_up_w4'] = aroon(dataUntilToday['Close'], params)

		# Class_close
		y_close = np.asarray([1 if data.Return_close[i] > 0 else 0 for i in range(len(data))]).reshape(-1, 1)

		# target
		data["Return_target"] = data["log_Open"].shift(-1) - data["log_Open"]
		y_target = np.asarray([1 if data.Return_target[i] > 0 else 0 for i in range(len(data))]).reshape(-1, 1)

		# Drop based on correlation matrix
		data = data.drop(['log_Open', 'log_Close', 'Return_target', 'stochRsi_stochrsi_w2_s114_s236',
						  'stochRsi_stochrsi_w2_s126_s24',
						  'stochRsi_stochrsi_w2_s137_s234', 'stochRsi_stochrsi_w3_s149_s21',
						  'stochRsi_stochrsi_w3_s12_s222',
						  'stochRsi_stochrsi_w2_s115_s236'], axis=1)
		data.dropna(inplace=True)

		# scale data, and then add the class as feature input
		scaler = StandardScaler()
		data_set_scaled = scaler.fit_transform(data)
		data_set_scaled = np.concatenate((data_set_scaled, y_close[1:]), axis=1)
		data_set_scaled = np.concatenate((data_set_scaled, y_target[1:]), axis=1)

		# choose how many look back days
		backcandles = 30

		# choose columns: all but target variable (its last column)
		liste = list(range(0, data.shape[1] - 1))

		# split data into train test sets
		pred_days = 1

		# prepare data for lstm
		X_train, X_test, y_train, y_test = prepare_data(data_set_scaled, backcandles, liste, pred_days)

		# train and predict
		# n_members -> how many predictors we wanted to use
		epochs = 45
		batch_size = 10
		ensemble, y_pred, accuracy = fit_ensemble(self.n_members, X_train, X_test, y_train, y_test, epochs, batch_size)
		clear_session()
		return y_pred

def class_LSTM(n_inputs, n_features):
	model = Sequential()
	model.add(LSTM(units=200, return_sequences=True, input_shape=(n_inputs, n_features)))
	model.add(Dropout(0.01))
	model.add(LSTM(100))
	model.add(Dropout(0.1))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def prepare_data(data_set_scaled, backcandles, liste, pred_days):
	X = []
	for j in range(len(liste)):  # data_set_scaled[0].size):#2 columns are target not X
		X.append([])
		for i in range(backcandles, data_set_scaled.shape[0]):  # backcandles+2
			X[j].append(data_set_scaled[i - backcandles:i, liste[j]])

	# move axis from 0 to position 2
	X = np.moveaxis(X, [0], [2])

	X, yi = np.array(X), np.array(data_set_scaled[backcandles:, -1])
	y = np.reshape(yi, (len(yi), 1))
	splitlimit = X.shape[0] - pred_days

	X_train, X_test = X[:splitlimit], X[splitlimit:]
	y_train, y_test = y[:splitlimit], y[splitlimit:]

	return X_train, X_test, y_train, y_test


def fit_model(X_train, y_train, epochs, batch_size):
	# define neural network model
	n_inputs = X_train.shape[1]
	n_features = X_train.shape[2]
	model = class_LSTM(n_inputs, n_features)
	# model.fit(X_train, y_train, batch_size=10, epochs=93)
	# fit the model on the training dataset
	early_stopping = EarlyStopping(monitor="accuracy", patience=10, mode='auto', min_delta=0)
	model.fit(X_train, y_train, verbose=2, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])
	return model


def fit_ensemble(n_members, X_train, X_test, y_train, y_test, epochs, batch_size):
	ensemble = list()
	y_pred = np.empty((n_members, y_test.shape[0]))
	accuracy = []
	for i in range(n_members):
		# define and fit the model on the training set
		model = fit_model(X_train, y_train, epochs, batch_size)
		# evaluate model on the test set
		yhat = model.predict(X_test, verbose=2)
		y_pred[i, :] = yhat.flatten()

		# turn probability into class
		yhat[yhat > 0.5] = 1
		yhat[yhat <= 0.5] = 0

		# accuracy
		acc = accuracy_score(y_test, yhat)
		accuracy.append(acc)
		# store the model and prediction and accuracy
		ensemble.append(model)

	return ensemble, y_pred, accuracy