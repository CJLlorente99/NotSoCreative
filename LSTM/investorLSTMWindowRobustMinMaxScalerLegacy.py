from classes.investorClass import Investor
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from keras.models import Sequential
from sklearn.metrics import mean_absolute_error
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, LSTM
import numpy as np
from keras import initializers
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from TAIndicators.adx import averageDirectionalMovementIndex
from TAIndicators.bb import bollingerBands
from TAIndicators.rsi import relativeStrengthIndex
from TAIndicators.stochasticRsi import stochasticRSI
from TAIndicators.ma import movingAverageConvergenceDivergence
from classes.investorParamsClass import ADXInvestorParams, BBInvestorParams, StochasticRSIInvestorParams, RSIInvestorParams, MACDInvestorParams

modelMinMaxScaler = [None, None, None, None, None]

class InvestorLSTMWindowRobustMinMaxT2Legacy (Investor):

	def __init__(self, initialInvestment=10000, n_members=10):
		super().__init__(initialInvestment)
		self.n_members = n_members

	def returnBrokerUpdate(self, moneyInvestedToday, data) -> pd.DataFrame:
		return pd.DataFrame(
			{'moneyToInvestLSTMWindowRobustMinMaxT2Legacy': moneyInvestedToday,
			 'investedMoneyLSTMWindowRobustMinMaxT2Legacy': self.investedMoney,
			 'nonInvestedMoneyLSTMWindowRobustMinMaxT2Legacy': self.nonInvestedMoney}, index=[0])

	def possiblyInvestMorning(self, data):
		res = self.calculatePrediction(data['df'])
		if self.nonInvestedMoney == 0:
			x = 0
		else:
			x = 5000 / self.nonInvestedMoney
		if res >= 0:
			if x > 1:
				self.perToInvest = 1
			else:
				self.perToInvest = x
		else:
			self.perToInvest = res

	def possiblyInvestAfternoon(self, data):
		global modelMinMaxScaler
		self.perToInvest = 0
		modelMinMaxScaler = [None, None, None, None, None]

	def plotEvolution(self, expData, stockMarketData, recordPredictedValue=None):
		"""
				Function that plots the actual status of the investor investment as well as the decisions that have been made
				:param indicatorData: Data belonging to the indicator used to take decisions
				:param stockMarketData: df with the stock market data
				:param recordPredictedValue: Predicted data dataframe
				"""
		# Plot indicating the evolution of the total value and contain (moneyInvested and moneyNotInvested)
		fig = go.Figure()
		fig.add_trace(
			go.Scatter(name="Money Invested", x=self.record.index, y=self.record["moneyInvested"], stackgroup="one"))
		fig.add_trace(go.Scatter(name="Money Not Invested", x=self.record.index, y=self.record["moneyNotInvested"],
								 stackgroup="one"))
		fig.add_trace(go.Scatter(name="Total Value", x=self.record.index, y=self.record["totalValue"]))
		fig.update_layout(
			title="Evolution of Porfolio using LSTM Window Rob MM T2 Legacy (" + self.record.index[0].strftime(
				"%d/%m/%Y") + "-" +
				  self.record.index[-1].strftime("%d/%m/%Y") + ")", xaxis_title="Date",
			yaxis_title="Value [$]", hovermode='x unified')
		fig.write_image("images/EvolutionPorfolioLSTMWindowRobMMT2Legacy(" + self.record.index[0].strftime(
			"%d_%m_%Y") + "-" +
						self.record.index[-1].strftime("%d_%m_%Y") + ").png", scale=6, width=1080, height=1080)
		# fig.show()

		# Plot indicating the value of the indicator, the value of the stock market and the decisions made
		fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": True}],
												   [{"secondary_y": False}]])

		fig.add_trace(go.Scatter(name="Stock Market Value Open", x=self.record.index,
								 y=stockMarketData.Open[-len(self.record.index):]), row=1, col=1, secondary_y=False)
		fig.add_trace(go.Bar(name="Money Invested Today", x=self.record.index, y=self.record["moneyInvestedToday"]),
					  row=2, col=1, secondary_y=False)

		fig.update_xaxes(title_text="Date", row=1, col=1)
		fig.update_xaxes(title_text="Date", row=2, col=1)
		fig.update_layout(
			title="Decision making under LSTM Window Rob MM T2 Legacy(" + self.record.index[0].strftime("%d/%m/%Y") + "-" +
				  self.record.index[-1].strftime("%d/%m/%Y") + ")", hovermode='x unified')
		fig.write_image("images/DecisionMakingLSTMWindowRobMMT2Legacy(" + self.record.index[0].strftime("%d_%m_%Y") + "-" +
						self.record.index[-1].strftime("%d_%m_%Y") + ").png", scale=6, width=1080, height=1080)

	# fig.show()

	def calculatePrediction(self, data):
		# data is already shifted when it comes here
		res = pd.DataFrame()

		# transformation
		# to decide if we want to predict raw prices or returns -> i think returns might work better cause of non-stationarity
		transformation = 1
		if transformation == 0:
			res['Target'] = data.Open

		elif transformation == 1:

			df_log = np.sqrt(np.log(data.Open))
			df_log_diff = df_log - df_log.shift()
			res['Return_Open'] = df_log_diff
			res['Target'] = df_log_diff
			res.dropna(inplace=True)

		# Return_outre
		res['Return_outra'] = np.log(data.Open) - np.log(data.Close)

		# Volume
		res['Volume'] = data.Volume

		# adx_pos_w50
		params = ADXInvestorParams(50)
		res['adx_pos_w50'] = averageDirectionalMovementIndex(data.High, data.Low, data.Close, params)['adx_pos']

		# bb_lband_w31_stdDev3.5483563609690094
		params = BBInvestorParams(31, 3.5483563609690094)
		res['bb_lband_w31_stdDev3.5483563609690094'] = bollingerBands(data.Close, params)['lband']

		# Return_Open
		# in transformation

		# Diff_open
		res['Diff_open'] = data.Open - data.Open.shift()

		# Diff_outra
		res['Diff_outra'] = data.Open - data.Close

		# rsi_w44
		params = RSIInvestorParams(None, None, 44)
		res['rsi_w44'] = relativeStrengthIndex(data.Close, params)['rsi']

		# adx_neg_w14
		params = ADXInvestorParams(14)
		res['adx_neg_w14'] = averageDirectionalMovementIndex(data.High, data.Low, data.Close, params)['adx_neg']

		# stochRsi_d_w4_s131_s27
		params = StochasticRSIInvestorParams(4, 31, 27)
		res['stochRsi_d_w4_s131_s27'] = stochasticRSI(data.Close, params)['d']

		# adx_w50
		params = ADXInvestorParams(50)
		res['adx_w50'] = averageDirectionalMovementIndex(data.High, data.Low, data.Close, params)['adx']

		# Return_intra
		res['Return_intra'] = np.log(data.Close) - np.log(data.Open.shift())

		# macd_difffW12_sW44_signal3
		params = MACDInvestorParams(None, None, 12, 44, 3)
		res['macd_difffW12_sW44_signal3'] = movingAverageConvergenceDivergence(data.Close, params)['diff']

		# stochRsi_d_w50_s110_s218
		params = StochasticRSIInvestorParams(50, 10, 18)
		res['stochRsi_d_w50_s110_s218'] = stochasticRSI(data.Close, params)['d']

		# stochRsi_d_w3_s132_s247
		params = StochasticRSIInvestorParams(3, 32, 47)
		res['stochRsi_d_w50_s110_s218'] = stochasticRSI(data.Close, params)['d']

		# macd_fW1_sW41_signal34
		params = MACDInvestorParams(None, None, 1, 41, 34)
		res['macd_fW1_sW41_signal34'] = movingAverageConvergenceDivergence(data.Close, params)['macd']

		res.dropna(inplace=True)

		# choose columns: all but target variable (its last column)
		liste = list(range(0, data.shape[1]))

		# only for
		pred_days = 3

		# how many days i want to predict
		step_out = 3

		# how many last days i include in my prediction
		backcandles = 16  # 10

		# to save decision and predictions
		y_predictions = []

		# days to predict
		test_days = step_out

		# scale data with robust
		scaler_r = RobustScaler()
		data_set_scaled = scaler_r.fit_transform(res)
		# scale data
		scaler_m = MinMaxScaler()
		data_set_scaled = scaler_m.fit_transform(data_set_scaled)

		# prepare data for lstm
		data_set_scaled = np.vstack([data_set_scaled, np.zeros((test_days, data_set_scaled.shape[1]))])
		X_train, X_test, y_train, y_test = prepare_multidata(data_set_scaled, backcandles, pred_days, test_days)

		n_members = self.n_members
		epochs = 35 #
		batch_size = 4
		ensemble, y_pred_scale, = fit_ensemble(n_members, X_train, X_test, y_train, y_test, epochs,
											   batch_size)

		# inverse scaling
		y_pred_scale = inverse_scaling(res, y_pred_scale, scaler_m)

		y_pred = inverse_scaling(res, y_pred_scale, scaler_r)

		# bounds, mean -> further I only use mean
		lower, y_mean, upper = calculate_bounds(y_pred)

		# inverse transformation: not possible to use for real script -> because of index, in real script we have no test_or
		# so for real script it might be better to use not a index or so
		if transformation == 1:
			y_mean = pd.DataFrame(y_mean, range(len(y_mean)), columns=['Open'])
			y_mean['Open'] = y_mean['Open'] + df_log.shift().values[-test_days:]
			y_mean = (y_mean ** 2)
			y_mean = np.exp(y_mean)

		y_predictions.append(np.asarray(y_mean))
		y_mean = np.asarray(y_mean).flatten()

		# build decision rule: if open_t+2 > open_t -> buy on open_t
		if data.Open.values[-1] < y_mean[2]:
			return 1
		else:
			return -1

class InvestorLSTMWindowRobustMinMaxT1Legacy (Investor):

	def __init__(self, initialInvestment=10000, n_members=10):
		super().__init__(initialInvestment)
		self.n_members = n_members

	def returnBrokerUpdate(self, moneyInvestedToday, data) -> pd.DataFrame:
		return pd.DataFrame(
			{'moneyToInvestLSTMWindowRobustMinMaxT1Legacy': moneyInvestedToday,
			 'investedMoneyLSTMWindowRobustMinMaxT1Legacy': self.investedMoney,
			 'nonInvestedMoneyLSTMWindowRobustMinMaxT1Legacy': self.nonInvestedMoney}, index=[0])

	def possiblyInvestMorning(self, data):
		res = self.calculatePrediction(data['df'])
		if self.nonInvestedMoney == 0:
			x = 0
		else:
			x = 5000 / self.nonInvestedMoney
		if res >= 0:
			if x > 1:
				self.perToInvest = 1
			else:
				self.perToInvest = x
		else:
			self.perToInvest = res

	def possiblyInvestAfternoon(self, data):
		global modelMinMaxScaler
		self.perToInvest = 0
		modelMinMaxScaler = [None, None, None, None, None]

	def plotEvolution(self, expData, stockMarketData, recordPredictedValue=None):
		"""
				Function that plots the actual status of the investor investment as well as the decisions that have been made
				:param indicatorData: Data belonging to the indicator used to take decisions
				:param stockMarketData: df with the stock market data
				:param recordPredictedValue: Predicted data dataframe
				"""
		# Plot indicating the evolution of the total value and contain (moneyInvested and moneyNotInvested)
		fig = go.Figure()
		fig.add_trace(
			go.Scatter(name="Money Invested", x=self.record.index, y=self.record["moneyInvested"], stackgroup="one"))
		fig.add_trace(go.Scatter(name="Money Not Invested", x=self.record.index, y=self.record["moneyNotInvested"],
								 stackgroup="one"))
		fig.add_trace(go.Scatter(name="Total Value", x=self.record.index, y=self.record["totalValue"]))
		fig.update_layout(
			title="Evolution of Porfolio using LSTM Window Rob MM T1 Legacy(" + self.record.index[0].strftime(
				"%d/%m/%Y") + "-" +
				  self.record.index[-1].strftime("%d/%m/%Y") + ")", xaxis_title="Date",
			yaxis_title="Value [$]", hovermode='x unified')
		fig.write_image("images/EvolutionPorfolioLSTMWindowRobMMT1Legacy(" + self.record.index[0].strftime(
			"%d_%m_%Y") + "-" +
						self.record.index[-1].strftime("%d_%m_%Y") + ").png", scale=6, width=1080, height=1080)
		# fig.show()

		# Plot indicating the value of the indicator, the value of the stock market and the decisions made
		fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": True}],
												   [{"secondary_y": False}]])

		fig.add_trace(go.Scatter(name="Stock Market Value Open", x=self.record.index,
								 y=stockMarketData.Open[-len(self.record.index):]), row=1, col=1, secondary_y=False)
		fig.add_trace(go.Bar(name="Money Invested Today", x=self.record.index, y=self.record["moneyInvestedToday"]),
					  row=2, col=1, secondary_y=False)

		fig.update_xaxes(title_text="Date", row=1, col=1)
		fig.update_xaxes(title_text="Date", row=2, col=1)
		fig.update_layout(
			title="Decision making under LSTM Window Rob MM T1 Legacy(" + self.record.index[0].strftime("%d/%m/%Y") + "-" +
				  self.record.index[-1].strftime("%d/%m/%Y") + ")", hovermode='x unified')
		fig.write_image("images/DecisionMakingLSTMWindowRobMMT1Legacy(" + self.record.index[0].strftime("%d_%m_%Y") + "-" +
						self.record.index[-1].strftime("%d_%m_%Y") + ").png", scale=6, width=1080, height=1080)

	# fig.show()

	def calculatePrediction(self, data):
		# data is already shifted when it comes here
		res = pd.DataFrame()

		# transformation
		# to decide if we want to predict raw prices or returns -> i think returns might work better cause of non-stationarity
		transformation = 1
		if transformation == 0:
			res['Target'] = data.Open

		elif transformation == 1:

			df_log = np.sqrt(np.log(data.Open))
			df_log_diff = df_log - df_log.shift()
			res['Return_Open'] = df_log_diff
			res['Target'] = df_log_diff
			res.dropna(inplace=True)

		# Return_outre
		res['Return_outra'] = np.log(data.Open) - np.log(data.Close)

		# Volume
		res['Volume'] = data.Volume

		# adx_pos_w50
		params = ADXInvestorParams(50)
		res['adx_pos_w50'] = averageDirectionalMovementIndex(data.High, data.Low, data.Close, params)['adx_pos']

		# bb_lband_w31_stdDev3.5483563609690094
		params = BBInvestorParams(31, 3.5483563609690094)
		res['bb_lband_w31_stdDev3.5483563609690094'] = bollingerBands(data.Close, params)['lband']

		# Return_Open
		# in transformation

		# Diff_open
		res['Diff_open'] = data.Open - data.Open.shift()

		# Diff_outra
		res['Diff_outra'] = data.Open - data.Close

		# rsi_w44
		params = RSIInvestorParams(None, None, 44)
		res['rsi_w44'] = relativeStrengthIndex(data.Close, params)['rsi']

		# adx_neg_w14
		params = ADXInvestorParams(14)
		res['adx_neg_w14'] = averageDirectionalMovementIndex(data.High, data.Low, data.Close, params)['adx_neg']

		# stochRsi_d_w4_s131_s27
		params = StochasticRSIInvestorParams(4, 31, 27)
		res['stochRsi_d_w4_s131_s27'] = stochasticRSI(data.Close, params)['d']

		# adx_w50
		params = ADXInvestorParams(50)
		res['adx_w50'] = averageDirectionalMovementIndex(data.High, data.Low, data.Close, params)['adx']

		# Return_intra
		res['Return_intra'] = np.log(data.Close) - np.log(data.Open.shift())

		# macd_difffW12_sW44_signal3
		params = MACDInvestorParams(None, None, 12, 44, 3)
		res['macd_difffW12_sW44_signal3'] = movingAverageConvergenceDivergence(data.Close, params)['diff']

		# stochRsi_d_w50_s110_s218
		params = StochasticRSIInvestorParams(50, 10, 18)
		res['stochRsi_d_w50_s110_s218'] = stochasticRSI(data.Close, params)['d']

		# stochRsi_d_w3_s132_s247
		params = StochasticRSIInvestorParams(3, 32, 47)
		res['stochRsi_d_w50_s110_s218'] = stochasticRSI(data.Close, params)['d']

		# macd_fW1_sW41_signal34
		params = MACDInvestorParams(None, None, 1, 41, 34)
		res['macd_fW1_sW41_signal34'] = movingAverageConvergenceDivergence(data.Close, params)['macd']

		res.dropna(inplace=True)

		# choose columns: all but target variable (its last column)
		liste = list(range(0, data.shape[1]))

		# only for
		pred_days = 3

		# how many days i want to predict
		step_out = 3

		# how many last days i include in my prediction
		backcandles = 16  # 10

		# to save decision and predictions
		y_predictions = []

		# days to predict
		test_days = step_out

		# scale data with robust
		scaler_r = RobustScaler()
		data_set_scaled = scaler_r.fit_transform(res)
		# scale data
		scaler_m = MinMaxScaler()
		data_set_scaled = scaler_m.fit_transform(data_set_scaled)

		# prepare data for lstm
		data_set_scaled = np.vstack([data_set_scaled, np.zeros((test_days, data_set_scaled.shape[1]))])
		X_train, X_test, y_train, y_test = prepare_multidata(data_set_scaled, backcandles, pred_days, test_days)

		n_members = self.n_members
		epochs = 35  #
		batch_size = 4
		ensemble, y_pred_scale = fit_ensemble(n_members, X_train, X_test, y_train, y_test, epochs,
											   batch_size)

		# inverse scaling
		y_pred_scale = inverse_scaling(res, y_pred_scale, scaler_m)

		y_pred = inverse_scaling(res, y_pred_scale, scaler_r)

		# bounds, mean -> further I only use mean
		lower, y_mean, upper = calculate_bounds(y_pred)

		# inverse transformation: not possible to use for real script -> because of index, in real script we have no test_or
		# so for real script it might be better to use not a index or so
		if transformation == 1:
			y_mean = pd.DataFrame(y_mean, range(len(y_mean)), columns=['Open'])
			y_mean['Open'] = y_mean['Open'] + df_log.shift().values[-test_days:]
			y_mean = (y_mean ** 2)
			y_mean = np.exp(y_mean)

		y_predictions.append(np.asarray(y_mean))
		y_mean = np.asarray(y_mean).flatten()

		# build decision rule: if open_t+2 > open_t -> buy on open_t
		if data.Open.values[-1] < y_mean[1]:
			return 1
		else:
			return -1

class InvestorLSTMWindowRobustMinMaxT1T2Legacy (Investor):

	def __init__(self, initialInvestment=10000, n_members=10):
		super().__init__(initialInvestment)
		self.n_members = n_members

	def returnBrokerUpdate(self, moneyInvestedToday, data) -> pd.DataFrame:
		return pd.DataFrame(
			{'moneyToInvestLSTMWindowRobustMinMaxT1T2Legacy': moneyInvestedToday,
			 'investedMoneyLSTMWindowRobustMinMaxT1T2Legacy': self.investedMoney,
			 'nonInvestedMoneyLSTMWindowRobustMinMaxT1T2Legacy': self.nonInvestedMoney}, index=[0])

	def possiblyInvestMorning(self, data):
		self.perToInvest = self.calculatePrediction(data['df'])

	def possiblyInvestAfternoon(self, data):
		global modelMinMaxScaler
		self.perToInvest = 0
		modelMinMaxScaler = [None, None, None, None, None]

	def plotEvolution(self, expData, stockMarketData, recordPredictedValue=None):
		"""
				Function that plots the actual status of the investor investment as well as the decisions that have been made
				:param indicatorData: Data belonging to the indicator used to take decisions
				:param stockMarketData: df with the stock market data
				:param recordPredictedValue: Predicted data dataframe
				"""
		# Plot indicating the evolution of the total value and contain (moneyInvested and moneyNotInvested)
		fig = go.Figure()
		fig.add_trace(
			go.Scatter(name="Money Invested", x=self.record.index, y=self.record["moneyInvested"], stackgroup="one"))
		fig.add_trace(go.Scatter(name="Money Not Invested", x=self.record.index, y=self.record["moneyNotInvested"],
								 stackgroup="one"))
		fig.add_trace(go.Scatter(name="Total Value", x=self.record.index, y=self.record["totalValue"]))
		fig.update_layout(
			title="Evolution of Porfolio using LSTM Window Rob MM T1T2 Legacy(" + self.record.index[0].strftime(
				"%d/%m/%Y") + "-" +
				  self.record.index[-1].strftime("%d/%m/%Y") + ")", xaxis_title="Date",
			yaxis_title="Value [$]", hovermode='x unified')
		fig.write_image("images/EvolutionPorfolioLSTMWindowRobMMT1T2Legacy(" + self.record.index[0].strftime(
			"%d_%m_%Y") + "-" +
						self.record.index[-1].strftime("%d_%m_%Y") + ").png", scale=6, width=1080, height=1080)
		# fig.show()

		# Plot indicating the value of the indicator, the value of the stock market and the decisions made
		fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": True}],
												   [{"secondary_y": False}]])

		fig.add_trace(go.Scatter(name="Stock Market Value Open", x=self.record.index,
								 y=stockMarketData.Open[-len(self.record.index):]), row=1, col=1, secondary_y=False)
		fig.add_trace(go.Bar(name="Money Invested Today", x=self.record.index, y=self.record["moneyInvestedToday"]),
					  row=2, col=1, secondary_y=False)

		fig.update_xaxes(title_text="Date", row=1, col=1)
		fig.update_xaxes(title_text="Date", row=2, col=1)
		fig.update_layout(
			title="Decision making under LSTM Window Rob MM T1T2 Legacy(" + self.record.index[0].strftime("%d/%m/%Y") + "-" +
				  self.record.index[-1].strftime("%d/%m/%Y") + ")", hovermode='x unified')
		fig.write_image("images/DecisionMakingLSTMWindowRobMMT1T2Legacy(" + self.record.index[0].strftime("%d_%m_%Y") + "-" +
						self.record.index[-1].strftime("%d_%m_%Y") + ").png", scale=6, width=1080, height=1080)

	# fig.show()

	def calculatePrediction(self, data):
		# data is already shifted when it comes here
		res = pd.DataFrame()

		# transformation
		# to decide if we want to predict raw prices or returns -> i think returns might work better cause of non-stationarity
		transformation = 1
		if transformation == 0:
			res['Target'] = data.Open

		elif transformation == 1:

			df_log = np.sqrt(np.log(data.Open))
			df_log_diff = df_log - df_log.shift()
			res['Return_Open'] = df_log_diff
			res['Target'] = df_log_diff
			res.dropna(inplace=True)

		# Return_outre
		res['Return_outra'] = np.log(data.Open) - np.log(data.Close)

		# Volume
		res['Volume'] = data.Volume

		# adx_pos_w50
		params = ADXInvestorParams(50)
		res['adx_pos_w50'] = averageDirectionalMovementIndex(data.High, data.Low, data.Close, params)['adx_pos']

		# bb_lband_w31_stdDev3.5483563609690094
		params = BBInvestorParams(31, 3.5483563609690094)
		res['bb_lband_w31_stdDev3.5483563609690094'] = bollingerBands(data.Close, params)['lband']

		# Return_Open
		# in transformation

		# Diff_open
		res['Diff_open'] = data.Open - data.Open.shift()

		# Diff_outra
		res['Diff_outra'] = data.Open - data.Close

		# rsi_w44
		params = RSIInvestorParams(None, None, 44)
		res['rsi_w44'] = relativeStrengthIndex(data.Close, params)['rsi']

		# adx_neg_w14
		params = ADXInvestorParams(14)
		res['adx_neg_w14'] = averageDirectionalMovementIndex(data.High, data.Low, data.Close, params)['adx_neg']

		# stochRsi_d_w4_s131_s27
		params = StochasticRSIInvestorParams(4, 31, 27)
		res['stochRsi_d_w4_s131_s27'] = stochasticRSI(data.Close, params)['d']

		# adx_w50
		params = ADXInvestorParams(50)
		res['adx_w50'] = averageDirectionalMovementIndex(data.High, data.Low, data.Close, params)['adx']

		# Return_intra
		res['Return_intra'] = np.log(data.Close) - np.log(data.Open.shift())

		# macd_difffW12_sW44_signal3
		params = MACDInvestorParams(None, None, 12, 44, 3)
		res['macd_difffW12_sW44_signal3'] = movingAverageConvergenceDivergence(data.Close, params)['diff']

		# stochRsi_d_w50_s110_s218
		params = StochasticRSIInvestorParams(50, 10, 18)
		res['stochRsi_d_w50_s110_s218'] = stochasticRSI(data.Close, params)['d']

		# stochRsi_d_w3_s132_s247
		params = StochasticRSIInvestorParams(3, 32, 47)
		res['stochRsi_d_w50_s110_s218'] = stochasticRSI(data.Close, params)['d']

		# macd_fW1_sW41_signal34
		params = MACDInvestorParams(None, None, 1, 41, 34)
		res['macd_fW1_sW41_signal34'] = movingAverageConvergenceDivergence(data.Close, params)['macd']

		res.dropna(inplace=True)

		# choose columns: all but target variable (its last column)
		liste = list(range(0, data.shape[1]))

		# only for
		pred_days = 3

		# how many days i want to predict
		step_out = 3

		# how many last days i include in my prediction
		backcandles = 16  # 10

		# to save decision and predictions
		y_predictions = []

		# days to predict
		test_days = step_out

		# scale data with robust
		scaler_r = RobustScaler()
		data_set_scaled = scaler_r.fit_transform(res)
		# scale data
		scaler_m = MinMaxScaler()
		data_set_scaled = scaler_m.fit_transform(data_set_scaled)

		# prepare data for lstm
		data_set_scaled = np.vstack([data_set_scaled, np.zeros((test_days, data_set_scaled.shape[1]))])
		X_train, X_test, y_train, y_test = prepare_multidata(data_set_scaled, backcandles, pred_days, test_days)

		n_members = self.n_members
		epochs = 35  #
		batch_size = 4
		ensemble, y_pred_scale = fit_ensemble(n_members, X_train, X_test, y_train, y_test, epochs,
											   batch_size)

		# inverse scaling
		y_pred_scale = inverse_scaling(res, y_pred_scale, scaler_m)

		y_pred = inverse_scaling(res, y_pred_scale, scaler_r)

		# bounds, mean -> further I only use mean
		lower, y_mean, upper = calculate_bounds(y_pred)

		# inverse transformation: not possible to use for real script -> because of index, in real script we have no test_or
		# so for real script it might be better to use not a index or so
		if transformation == 1:
			y_mean = pd.DataFrame(y_mean, range(len(y_mean)), columns=['Open'])
			y_mean['Open'] = y_mean['Open'] + df_log.shift().values[-test_days:]
			y_mean = (y_mean ** 2)
			y_mean = np.exp(y_mean)

		y_predictions.append(np.asarray(y_mean))
		y_mean = np.asarray(y_mean).flatten()

		# build decision rule
		if data.Open.values[-1] < y_mean[1] < y_mean[2]:
			return 1
		elif data.Open.values[-1] < y_mean[2] < y_mean[1]:
			return 0.5
		elif y_mean[1] < data.Open.values[-1] < y_mean[2]:
			return -0.5
		elif y_mean[2] < data.Open.values[-1] and y_mean[1] < data.Open.values[-1]:
			return -1
		elif y_mean[2] < data.Open.values[-1] < y_mean[1]:
			return 0
		else:
			return -1

def build_model(n_inputs, n_features, n_outputs):
	opt = Adam(learning_rate=0.000149)
	model = Sequential()
	model.add(LSTM(units=225, return_sequences=True,  bias_initializer=initializers.Constant(0.01),
				   kernel_initializer='he_uniform', input_shape=(n_inputs, n_features)))
	model.add(Dropout(0.1))
	model.add(LSTM(units=250))
	model.add(Dropout(0.1))
	# model.add(Dense(32, kernel_initializer="uniform", activation='relu'))
	model.add(Dense(units=n_outputs, activation='linear'))
	model.compile(optimizer=opt, loss='mean_squared_error')
	# history = model.fit
	return model

def fit_model(X_train, y_train, epochs, batch_size):
	# define neural network model
	n_inputs = X_train.shape[1]
	n_features = X_train.shape[2]
	n_outputs = y_train.shape[1]
	model = build_model(n_inputs, n_features, n_outputs)
	# model.fit(X_train, y_train, batch_size=10, epochs=93)
	# fit the model on the training dataset
	early_stopping = EarlyStopping(monitor="loss", patience=10, mode='auto', min_delta=0)
	model.fit(X_train, y_train, verbose=2, epochs=epochs, batch_size=batch_size, validation_split=0.15, callbacks=[early_stopping])
	return model


def fit_ensemble(n_members, X_train, X_test, y_train, y_test, epochs, batch_size):
	global modelMinMaxScaler
	ensemble = list()
	y_pred = np.empty((n_members, y_test.shape[1]))

	for i in range(n_members):
		# define and fit the model on the training set
		if not modelMinMaxScaler[i]:
			print('Creating model MM')
			model = fit_model(X_train, y_train, epochs, batch_size)
			modelMinMaxScaler[i] = model
		else:
			print('Already created model MM')
			model = modelMinMaxScaler[i]
		# evaluate model on the test set
		yhat = model.predict(X_test, verbose=2)
		mae = mean_absolute_error(y_test, yhat)
		# store the model and prediction
		ensemble.append(model)
		y_pred[i, :] = yhat.flatten()
	return ensemble, y_pred


def inverse_scaling(data, yhat, scaler):
	y_preds = np.empty((yhat.shape[0], yhat.shape[1]))
	for i in range(yhat.shape[0]):
		y_pred = np.tile(yhat[i, :].reshape(-1, 1), (1, data.shape[1]))
		y_pred = scaler.inverse_transform(y_pred)
		y_pred = y_pred[:, -1]
		y_preds[i, :] = y_pred
	return y_preds



def calculate_bounds(yhat):
	lower = []
	upper = []
	y_mean = []
	for i in range(yhat.shape[1]):
		interval = 1.96 * yhat[:, i].std()
		y_mean.append(yhat[:, i].mean())
		lower.append(yhat[:, i].mean() - interval)
		upper.append(yhat[:, i].mean() + interval)
	return lower, y_mean, upper


def prepare_multidata(data_set_scaled, backcandles, pred_days, step_out):
	# preparing data
	X = list()
	y = list()
	y_t = np.array(data_set_scaled[:, -1])

	for i in range(backcandles, data_set_scaled.shape[0] + 1 - step_out):
		X.append(data_set_scaled[i - backcandles:i, :-1])
		y.append(y_t[i:i + step_out])

	# move axis from 0 to position 2
	X = np.array(X)
	y = np.array(y)

	X = X.reshape(y.shape[0], backcandles, data_set_scaled.shape[1] - 1)

	# only last row because it contains the last step_out values (step_out is a number: first iteration: 10, last 1)
	splitlimit = X.shape[0] - 1
	X_train, X_test = X[:splitlimit], X[splitlimit:]
	y_train, y_test = y[:splitlimit], y[splitlimit:]

	return X_train, X_test, y_train, y_test
