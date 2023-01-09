import pandas as pd
from TAIndicators.atr import averageTrueRange
from TAIndicators.stochasticRsi import stochasticRSI
from TAIndicators.ma import movingAverageConvergenceDivergence
from TAIndicators.adx import averageDirectionalMovementIndex
from TAIndicators.rsi import relativeStrengthIndex
from TAIndicators.bb import bollingerBands
from TAIndicators.aroon import aroon
from classes.investorParamsClass import ATRInvestorParams, ADXInvestorParams, StochasticRSIInvestorParams, MACDInvestorParams, RSIInvestorParams, BBInvestorParams, AroonInvestorParams
from classes.investorClass import Investor
import joblib
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class InvestorRandomForestClassifier(Investor):
	def __init__(self, initialInvestment, fileName):
		super().__init__(initialInvestment)
		self.model = joblib.load(fileName)

	def returnBrokerUpdate(self, moneyInvestedToday, data) -> pd.DataFrame:
		return pd.DataFrame(
			{'moneyToInvestRFClass': moneyInvestedToday,
			 'investedMoneyRFClass': self.investedMoney, 'nonInvestedMoneyRFClass': self.nonInvestedMoney},
			index=[0])

	def possiblyInvestMorning(self, data):
		res = self.calculateInputsMorning(data['df'])
		self.perToInvest = self.model.predict([res.to_numpy()[-1]])[0]

	def possiblyInvestAfternoon(self, data):
		self.perToInvest = -1

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
			title="Evolution of Porfolio using RF Class (" + self.record.index[0].strftime(
				"%d/%m/%Y") + "-" +
				  self.record.index[-1].strftime("%d/%m/%Y") + ")", xaxis_title="Date",
			yaxis_title="Value [$]", hovermode='x unified')
		fig.write_image("images/EvolutionPorfolioRFClass(" + self.record.index[0].strftime(
			"%d_%m_%Y") + "-" +
						self.record.index[-1].strftime("%d_%m_%Y") + ").png", scale=6, width=1080, height=1080)
		fig.show()

		# Plot indicating the value of the indicator, the value of the stock market and the decisions made
		fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": True}],
												   [{"secondary_y": False}]])

		fig.add_trace(go.Scatter(name="Stock Market Value Open", x=self.record.index,
								 y=stockMarketData.Open[-len(self.record.index):]), row=1, col=1, secondary_y=False)
		fig.add_trace(go.Scatter(name="Stock Market Value Close", x=self.record.index,
								 y=stockMarketData.Close[-len(self.record.index):]), row=1, col=1, secondary_y=False)
		fig.add_trace(go.Bar(name="Money Invested Today", x=self.record.index, y=self.record["moneyInvestedToday"]),
					  row=2, col=1, secondary_y=False)

		fig.update_xaxes(title_text="Date", row=1, col=1)
		fig.update_xaxes(title_text="Date", row=2, col=1)
		fig.update_layout(
			title="Decision making under RFClass (" + self.record.index[0].strftime("%d/%m/%Y") + "-" +
				  self.record.index[-1].strftime("%d/%m/%Y") + ")", hovermode='x unified')
		fig.write_image("images/DecisionMakingRFClass(" + self.record.index[0].strftime("%d_%m_%Y") + "-" +
						self.record.index[-1].strftime("%d_%m_%Y") + ").png", scale=6, width=1080, height=1080)
		fig.show()

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