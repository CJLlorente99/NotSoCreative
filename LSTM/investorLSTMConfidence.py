from keras.layers import Dropout, LSTM, Dense
from keras.optimizers import Adamax
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras import initializers
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

	def returnBrokerUpdate(self, moneyInvestedToday, data):
		aux = pd.DataFrame()
		for i in range(self.n_members):
			aux = pd.concat([aux, pd.DataFrame({"voter" + str(i): data["lstmConfidence"][i]}, index=[0])], axis=1)

		return pd.concat([aux, pd.DataFrame(
			{'moneyToInvestLSTMConfClass': moneyInvestedToday,
			 'investedMoneyLSTMConfClass': self.investedMoney, 'nonInvestedMoneyLSTMConfClass': self.nonInvestedMoney}, index=[0])], axis=1)

	def possiblyInvestMorning(self, data):
		calculatedData = self.calculateInputsMorning(data['df'])
		self.perToInvest, data['lstmConfidence'] = self.getPredictionLSTM(calculatedData)

	def possiblyInvestAfternoon(self, data):
		data['lstmConfidence'] = np.zeros(self.n_members)
		self.perToInvest = -1  # Always sell all

	def plotEvolution(self, expData, stockMarketData, recordPredictedValue=None):
		"""
		Function that plots the actual status of the investor investment as well as the decisions that have been made
		:param indicatorData: Data belonging to the indicator used to take decisions
		:param stockMarketData: df with the stock market data
		:param recordPredictedValue: Predicted data dataframe
		"""
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
		fig.add_trace(go.Scatter(name="Stock Market Value Close", x=self.record.index,
								 y=stockMarketData.Close[-len(self.record.index):]), row=1, col=1, secondary_y=False)
		fig.add_trace(go.Bar(name="Money Invested Today", x=self.record.index, y=self.record["moneyInvestedToday"]), row=2, col=1, secondary_y=False)

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

	def calculateInputsMorning(self, df: pd.DataFrame):
		data = df.copy()
		data['Open'] = data['Open'].shift(-1)
		data = data[:-1]
		res = pd.DataFrame()

		# Return_interday
		res['Return_interday'] = np.log(data['Open']) - np.log(data['Close'])

		# bb_pband_w3_stdDev1.774447792366109
		params = BBInvestorParams(3, 1.775)
		res['bb_pband_w3_stdDev1.774447792366109'] = bollingerBands(data['Close'], params)['pband']

		# Return_open
		res['Return_open'] = np.log(data['Open']) - np.log(data['Open'].shift())

		# adx_pos_w6
		params = ADXInvestorParams(6)
		res['adx_pos_w6'] = averageDirectionalMovementIndex(data['High'], data['Low'], data['Close'], params)['adx_pos']

		# adx_pos_w42
		params = ADXInvestorParams(42)
		res['adx_pos_w42'] = averageDirectionalMovementIndex(data['High'], data['Low'], data['Close'], params)['adx_pos']

		# Volume
		res['Volume'] = data['Volume']

		# adx_neg_w1
		params = ADXInvestorParams(1)
		res['adx_neg_w1'] = averageDirectionalMovementIndex(data['High'], data['Low'], data['Close'], params)['adx_neg']

		# Return_intraday
		res['Return_intraday'] = np.log(data['Close']) - np.log(data['Open'])

		# stochRsi_k_w47_s143_s212
		params = StochasticRSIInvestorParams(47, 43, 12)
		res['stochRsi_k_w47_s143_s212'] = stochasticRSI(data['Close'], params)['k']

		# stochRsi_d_w9_s144_s246
		params = StochasticRSIInvestorParams(9, 44, 46)
		res['stochRsi_d_w9_s144_s246'] = stochasticRSI(data['Close'], params)['d']

		# stochRsi_d_w4_s16_s233
		params = StochasticRSIInvestorParams(4, 6, 33)
		res['stochRsi_d_w4_s16_s233'] = stochasticRSI(data['Close'], params)['d']

		# adx_w10
		params = ADXInvestorParams(10)
		res['adx_w10'] = averageDirectionalMovementIndex(data['High'], data['Low'], data['Close'], params)['adx']

		# bb_pband_w7_stdDev1.4065306043590475
		params = BBInvestorParams(7, 1.407)
		res['bb_pband_w7_stdDev1.4065306043590475'] = bollingerBands(data['Close'], params)['pband']

		# bb_pband_w13_stdDev1.7961852973078898
		params = BBInvestorParams(13, 1.796)
		res['bb_pband_w13_stdDev1.7961852973078898'] = bollingerBands(data['Close'], params)['pband']

		# adx_w18
		params = ADXInvestorParams(18)
		res['adx_w18'] = averageDirectionalMovementIndex(data['High'], data['Low'], data['Close'], params)['adx']

		# stochRsi_k_w4_s16_s233
		params = StochasticRSIInvestorParams(4, 6, 33)
		res['stochRsi_k_w4_s16_s233'] = stochasticRSI(data['Close'], params)['k']

		# adx_neg_w25
		params = ADXInvestorParams(25)
		res['adx_neg_w25'] = averageDirectionalMovementIndex(data['High'], data['Low'], data['Close'], params)['adx_neg']

		# stochRsi_d_w12_s125_s25
		params = StochasticRSIInvestorParams(12, 25, 5)
		res['stochRsi_d_w12_s125_s25'] = stochasticRSI(data['Close'], params)['d']

		# macd_difffW5_sW39_signal14
		params = MACDInvestorParams(5, 39, 14)
		res['macd_difffW5_sW39_signal14'] = movingAverageConvergenceDivergence(data['Close'], params)['diff']

		# stochRsi_k_w29_s18_s219
		params = StochasticRSIInvestorParams(29, 8, 19)
		res['stochRsi_k_w29_s18_s219'] = stochasticRSI(data['Close'], params)['k']

		return res

	def getPredictionLSTM(self, data):

		data['Target'] = data['Return_intraday'].shift(-1)
		y_target = np.asarray([1 if data.Target[i] > 0 else 0 for i in range(len(data))]).reshape(-1, 1)
		data = data.drop(['Target'], axis=1)

		scaler = StandardScaler()
		data_set_scaled = scaler.fit_transform(data)
		data_set_scaled = np.concatenate((data_set_scaled[1:], y_target[1:]), axis=1)

		backcandles = 40

		# choose columns: all but target variable (its last column)
		liste = list(range(0, data.shape[1] - 1))

		# print(data.iloc[:, liste])

		# split data into train test sets
		pred_days = 1

		# prepare data for lstm
		X_train, X_test, y_train, y_test = prepare_data(data_set_scaled, backcandles, liste, pred_days)

		# train and predict
		# n_members -> how many predictors we wanted to use
		n_members = self.n_members
		epochs = 35
		batch_size = 8
		ensemble, y_pred, prob = fit_ensemble(n_members, X_train, X_test, y_train, y_test, epochs, batch_size)

		clear_session()
		# majority vote, probs for amount
		return majority_vote(y_pred)[0], y_pred[:, 0]

class InvestorLSTMConfidenceProb(Investor):
	def __init__(self, initialInvestment=10000, n_members=10):
		super().__init__(initialInvestment)
		self.n_members = n_members

	def returnBrokerUpdate(self, moneyInvestedToday, data):
		aux = pd.DataFrame()
		for i in range(self.n_members):
			aux = pd.concat([aux, pd.DataFrame({"voter" + str(i): data["lstmConfidence"][i]}, index=[0])], axis=1)

		return pd.concat([aux, pd.DataFrame(
			{'moneyToInvestLSTMConfProb': moneyInvestedToday,
			 'investedMoneyLSTMConfProb': self.investedMoney, 'nonInvestedMoneyLSTMConfProb': self.nonInvestedMoney}, index=[0])], axis=1)

	def possiblyInvestMorning(self, data):
		calculatedData = self.calculateInputsMorning(data['df'])
		self.perToInvest, data['lstmConfidence'] = self.getPredictionLSTM(calculatedData)

	def possiblyInvestAfternoon(self, data):
		data['lstmConfidence'] = np.zeros(self.n_members)
		self.perToInvest = -1  # Always sell all

	def plotEvolution(self, expData, stockMarketData, recordPredictedValue=None):
		"""
		Function that plots the actual status of the investor investment as well as the decisions that have been made
		:param indicatorData: Data belonging to the indicator used to take decisions
		:param stockMarketData: df with the stock market data
		:param recordPredictedValue: Predicted data dataframe
		"""
		# Plot indicating the evolution of the total value and contain (moneyInvested and moneyNotInvested)
		fig = go.Figure()
		fig.add_trace(go.Scatter(name="Money Invested", x=self.record.index, y=self.record["moneyInvested"], stackgroup="one"))
		fig.add_trace(go.Scatter(name="Money Not Invested", x=self.record.index, y=self.record["moneyNotInvested"], stackgroup="one"))
		fig.add_trace(go.Scatter(name="Total Value", x=self.record.index, y=self.record["totalValue"]))
		fig.update_layout(
			title="Evolution of Porfolio using LSTM Confidence Prob (" + self.record.index[0].strftime(
				"%d/%m/%Y") + "-" +
				  self.record.index[-1].strftime("%d/%m/%Y") + ")", xaxis_title="Date",
			yaxis_title="Value [$]", hovermode='x unified')
		fig.write_image("images/EvolutionPorfolioLSTMConfidenceProb(" + self.record.index[0].strftime(
				"%d_%m_%Y") + "-" +
				  self.record.index[-1].strftime("%d_%m_%Y") + ").png",scale=6, width=1080, height=1080)
		# fig.show()

		# Plot indicating the value of the indicator, the value of the stock market and the decisions made
		fig = make_subplots(rows=2, cols=2, specs=[[{"secondary_y": True, "colspan": 2}, None], [{"secondary_y": False}, {"secondary_y": False}]])

		fig.add_trace(go.Scatter(name="Stock Market Value Open", x=self.record.index,
								 y=stockMarketData.Open[-len(self.record.index):]), row=1, col=1, secondary_y=False)
		fig.add_trace(go.Scatter(name="Stock Market Value Close", x=self.record.index,
								 y=stockMarketData.Close[-len(self.record.index):]), row=1, col=1, secondary_y=False)
		fig.add_trace(go.Bar(name="Money Invested Today", x=self.record.index, y=self.record["moneyInvestedToday"]), row=2, col=1, secondary_y=False)

		for i in range(self.n_members):
			fig.add_trace(go.Bar(name="Voter " + str(i), x=self.record.index, y=expData["voter"+str(i)]), row=2, col=2)
		fig.update_xaxes(title_text="Date", row=1, col=1)
		fig.update_xaxes(title_text="Date", row=2, col=1)
		fig.update_layout(
			title="Decision making under LSTM Confidence Prob (" + self.record.index[0].strftime("%d/%m/%Y") + "-" +
				  self.record.index[-1].strftime("%d/%m/%Y") + ")", hovermode='x unified')
		fig.write_image("images/DecisionMakingLSTMConfidenceProb(" + self.record.index[0].strftime("%d_%m_%Y") + "-" +
				  self.record.index[-1].strftime("%d_%m_%Y") + ").png",scale=6, width=1080, height=1080)
		# fig.show()

	def calculateInputsMorning(self, df: pd.DataFrame):
		data = df.copy()
		data['Open'] = data['Open'].shift(-1)
		data = data[:-1]
		res = pd.DataFrame()

		# Return_interday
		res['Return_interday'] = np.log(data['Open']) - np.log(data['Close'])

		# bb_pband_w3_stdDev1.774447792366109
		params = BBInvestorParams(3, 1.775)
		res['bb_pband_w3_stdDev1.774447792366109'] = bollingerBands(data['Close'], params)['pband']

		# Return_open
		res['Return_open'] = np.log(data['Open']) - np.log(data['Open'].shift())

		# adx_pos_w6
		params = ADXInvestorParams(6)
		res['adx_pos_w6'] = averageDirectionalMovementIndex(data['High'], data['Low'], data['Close'], params)['adx_pos']

		# adx_pos_w42
		params = ADXInvestorParams(42)
		res['adx_pos_w42'] = averageDirectionalMovementIndex(data['High'], data['Low'], data['Close'], params)['adx_pos']

		# Volume
		res['Volume'] = data['Volume']

		# adx_neg_w1
		params = ADXInvestorParams(1)
		res['adx_neg_w1'] = averageDirectionalMovementIndex(data['High'], data['Low'], data['Close'], params)['adx_neg']

		# Return_intraday
		res['Return_intraday'] = np.log(data['Close']) - np.log(data['Open'])

		# stochRsi_k_w47_s143_s212
		params = StochasticRSIInvestorParams(47, 43, 12)
		res['stochRsi_k_w47_s143_s212'] = stochasticRSI(data['Close'], params)['k']

		# stochRsi_d_w9_s144_s246
		params = StochasticRSIInvestorParams(9, 44, 46)
		res['stochRsi_d_w9_s144_s246'] = stochasticRSI(data['Close'], params)['d']

		# stochRsi_d_w4_s16_s233
		params = StochasticRSIInvestorParams(4, 6, 33)
		res['stochRsi_d_w4_s16_s233'] = stochasticRSI(data['Close'], params)['d']

		# adx_w10
		params = ADXInvestorParams(10)
		res['adx_w10'] = averageDirectionalMovementIndex(data['High'], data['Low'], data['Close'], params)['adx']

		# bb_pband_w7_stdDev1.4065306043590475
		params = BBInvestorParams(7, 1.407)
		res['bb_pband_w7_stdDev1.4065306043590475'] = bollingerBands(data['Close'], params)['pband']

		# bb_pband_w13_stdDev1.7961852973078898
		params = BBInvestorParams(13, 1.796)
		res['bb_pband_w13_stdDev1.7961852973078898'] = bollingerBands(data['Close'], params)['pband']

		# adx_w18
		params = ADXInvestorParams(18)
		res['adx_w18'] = averageDirectionalMovementIndex(data['High'], data['Low'], data['Close'], params)['adx']

		# stochRsi_k_w4_s16_s233
		params = StochasticRSIInvestorParams(4, 6, 33)
		res['stochRsi_k_w4_s16_s233'] = stochasticRSI(data['Close'], params)['k']

		# adx_neg_w25
		params = ADXInvestorParams(25)
		res['adx_neg_w25'] = averageDirectionalMovementIndex(data['High'], data['Low'], data['Close'], params)['adx_neg']

		# stochRsi_d_w12_s125_s25
		params = StochasticRSIInvestorParams(12, 25, 5)
		res['stochRsi_d_w12_s125_s25'] = stochasticRSI(data['Close'], params)['d']

		# macd_difffW5_sW39_signal14
		params = MACDInvestorParams(5, 39, 14)
		res['macd_difffW5_sW39_signal14'] = movingAverageConvergenceDivergence(data['Close'], params)['diff']

		# stochRsi_k_w29_s18_s219
		params = StochasticRSIInvestorParams(29, 8, 19)
		res['stochRsi_k_w29_s18_s219'] = stochasticRSI(data['Close'], params)['k']

		return res

	def getPredictionLSTM(self, data):

		data['Target'] = data['Return_intraday'].shift(-1)
		y_target = np.asarray([1 if data.Target[i] > 0 else 0 for i in range(len(data))]).reshape(-1, 1)
		data = data.drop(['Target'], axis=1)

		scaler = StandardScaler()
		data_set_scaled = scaler.fit_transform(data)
		data_set_scaled = np.concatenate((data_set_scaled[1:], y_target[1:]), axis=1)

		backcandles = 30

		# choose columns: all but target variable (its last column)
		liste = list(range(0, data.shape[1] - 1))

		# print(data.iloc[:, liste])

		# split data into train test sets
		pred_days = 1

		# prepare data for lstm
		X_train, X_test, y_train, y_test = prepare_data(data_set_scaled, backcandles, liste, pred_days)

		# train and predict
		# n_members -> how many predictors we wanted to use
		n_members = self.n_members
		epochs = 32
		batch_size = 8
		ensemble, y_pred, prob = fit_ensemble(n_members, X_train, X_test, y_train, y_test, epochs, batch_size)

		clear_session()
		# majority vote, probs for amount
		return prob.flatten().mean(), prob.flatten()

def class_LSTM(n_inputs, n_features):
	model = Sequential()
	model.add(LSTM(units=197, return_sequences=True, bias_initializer=initializers.Constant(0.1), input_shape=(n_inputs, n_features)))
	model.add(Dropout(0.1))
	model.add(LSTM(178))
	model.add(Dropout(0.1))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
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
	# fit the model on the training dataset
	early_stopping = EarlyStopping(monitor="accuracy", patience=10, mode='auto', min_delta=0)
	model.fit(X_train, y_train, verbose=2, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])
	return model

def fit_ensemble(n_members, X_train, X_test, y_train, y_test, epochs, batch_size):
	ensemble = list()
	y_pred = np.empty((n_members, y_test.shape[0]))
	prob = np.empty((n_members, y_test.shape[0]))
	for i in range(n_members):
		# define and fit the model on the training set
		model = fit_model(X_train, y_train, epochs, batch_size)
		# evaluate model on the test set
		yhat = model.predict(X_test, verbose=2)
		prob[i, :] = yhat.flatten()

		yhat[yhat > 0.5] = 1
		yhat[yhat <= 0.5] = 0

		# store the model and prediction
		ensemble.append(model)
		y_pred[i, :] = yhat.flatten()
	return ensemble, y_pred, prob

def majority_vote(yhat):
	# if majority is 1 -> signal = 1 -> buy
	y_mean = []
	probs = []
	for i in range(yhat.shape[1]):
		y_10 = yhat[:, i]
		probs.append(np.sum(y_10))
		n_one = np.count_nonzero(y_10 == 1)
		length =  round(y_10.shape[0] * 0.5)
		if n_one > length:
			y_mean.append(1)
		else:
			y_mean.append(-1)
	#error here
	#probs_mean = probs / len(probs)
	return y_mean #, probs_mean
