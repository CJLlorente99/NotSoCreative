from classes.investorClass import Investor
from classes.investorParamsClass import LSTMInvestorParams
from LSTM.LSTMClass import LSTMClass
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


class InvestorLSTMThreshold(Investor):
	def __init__(self, initialInvestment=10000, lstmParams: LSTMInvestorParams = None):
		super().__init__(initialInvestment)
		self.lstmParams = lstmParams
		self.model = LSTMClass()

	def returnBrokerUpdate(self, moneyInvestedToday, data):
		return pd.DataFrame(
			{"lstmReturn": data["lstm"]["return"], "lstmProb0": data["lstm"]["prob0"], "lstmProb1": data["lstm"]["prob1"],
			 'moneyToInvestLSTM': moneyInvestedToday,
			 'investedMoneyLSTM': self.investedMoney, 'nonInvestedMoneyLSTM': self.nonInvestedMoney}, index=[0])

	def possiblyInvestMorning(self, data):
		"""
		Function that calls the buy function and updates the investment values
		:param data: Decision data based on the type of indicator
		"""
		self.perToInvest = 0
		if data["lstm"]["return"][0] > self.lstmParams.threshold:
			self.perToInvest = 1
		elif data["lstm"]["return"][0] < -self.lstmParams.threshold:
			self.perToInvest = -1

	def possiblyInvestAfternoon(self, data):
		"""
		Function that calls the sell function and updates the investment values
		:param data: Decision data based on the type of indicator
		"""
		self.perToInvest = 0
		if data["lstm"]["return"][0] > self.lstmParams.threshold:
			self.perToInvest = 1
		elif data["lstm"]["return"][0] < -self.lstmParams.threshold:
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
		fig.add_trace(go.Scatter(name="Money Invested", x=self.record.index, y=self.record["moneyInvested"], stackgroup="one"))
		fig.add_trace(go.Scatter(name="Money Not Invested", x=self.record.index, y=self.record["moneyNotInvested"], stackgroup="one"))
		fig.add_trace(go.Scatter(name="Total Value", x=self.record.index, y=self.record["totalValue"]))
		fig.update_layout(
			title="Evolution of Porfolio using LSTM Threshold (" + self.record.index[0].strftime(
				"%d/%m/%Y") + "-" +
				  self.record.index[-1].strftime("%d/%m/%Y") + ")", xaxis_title="Date",
			yaxis_title="Value [$]", hovermode='x unified')
		fig.write_image("images/EvolutionPorfolioLSTMThreshold(" + self.record.index[0].strftime(
				"%d_%m_%Y") + "-" +
				  self.record.index[-1].strftime("%d_%m_%Y") + ").png",scale=6, width=1080, height=1080)
		# fig.show()

		# Plot indicating the value of the indicator, the value of the stock market and the decisions made
		fig = make_subplots(rows=2, cols=2, specs=[[{"secondary_y": True, "colspan": 2}, None], [{"secondary_y": False}, {"secondary_y": False}]])

		fig.add_trace(go.Scatter(name="lstmPredictedReturn", x=self.record.index,
								 y=expData["lstmReturn"][-len(self.record.index):]), row=1, col=1,
					  secondary_y=True)
		realReturn = np.log(stockMarketData.Open.shift(-1)[-len(self.record.index):]) - np.log(stockMarketData.Open[-len(self.record.index):])
		fig.add_trace(go.Scatter(name="realReturn", x=self.record.index, y=realReturn[-len(self.record.index):]), row=1, col=1, secondary_y=True)
		fig.add_trace(go.Scatter(name="Stock Market Value Open", x=self.record.index,
								 y=stockMarketData.Open[-len(self.record.index):]), row=1, col=1, secondary_y=False)
		fig.add_trace(go.Scatter(name="Stock Market Value Close", x=self.record.index, visible='legendonly',
								 y=stockMarketData.Close[-len(self.record.index):]), row=1, col=1, secondary_y=False)
		fig.add_trace(go.Bar(name="Money Invested Today", x=self.record.index, y=self.record["moneyInvestedToday"]), row=2, col=1, secondary_y=False)

		fig.add_trace(go.Bar(name="Prob Sell", x=self.record.index, y=-expData["lstmProb0"][-len(self.record.index):],
							 marker_color="red"), row=2, col=2)
		fig.add_trace(
			go.Bar(name="Prob buy", x=self.record.index, y=expData["lstmProb1"][-len(self.record.index):], marker_color="green"),
			row=2, col=2)
		fig.update_xaxes(title_text="Date", row=1, col=1)
		fig.update_xaxes(title_text="Date", row=2, col=1)
		fig.update_layout(
			title="Decision making under LSTM Threshold (" + self.record.index[0].strftime("%d/%m/%Y") + "-" +
				  self.record.index[-1].strftime("%d/%m/%Y") + ")", hovermode='x unified')
		fig.write_image("images/DecisionMakingLSTMThreshold(" + self.record.index[0].strftime("%d_%m_%Y") + "-" +
				  self.record.index[-1].strftime("%d_%m_%Y") + ").png",scale=6, width=1080, height=1080)
		# fig.show()


class InvestorLSTMProb(Investor):
	def __init__(self, initialInvestment=10000, lstmParams: LSTMInvestorParams = None):
		super().__init__(initialInvestment)
		self.lstmParams = lstmParams
		self.model = LSTMClass()

	def returnBrokerUpdate(self, moneyInvestedToday, data):
		return pd.DataFrame(
			{"lstmReturn": data["lstm"]["return"], "lstmProb0": data["lstm"]["prob0"], "lstmProb1": data["lstm"]["prob1"],
			 'moneyToInvestLSTM': moneyInvestedToday,
			 'investedMoneyLSTM': self.investedMoney, 'nonInvestedMoneyLSTM': self.nonInvestedMoney}, index=[0])

	def possiblyInvestMorning(self, data):
		"""
		Function that calls the buy function and updates the investment values
		:param data: Decision data based on the type of indicator
		"""
		self.perToInvest = 0
		if data["lstm"]["return"][0] > 0:
			self.perToInvest = data["lstm"]["prob1"][0]
		elif data["lstm"]["return"][0] < 0:
			self.perToInvest = data["lstm"]["prob0"][0]

	def possiblyInvestAfternoon(self, data):
		"""
		Function that calls the sell function and updates the investment values
		:param data: Decision data based on the type of indicator
		"""
		self.perToInvest = 0
		if data["lstm"]["return"][0] > 0:
			self.perToInvest = data["lstm"]["prob1"][0]
		elif data["lstm"]["return"][0] < 0:
			self.perToInvest = data["lstm"]["prob0"][0]

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
			title="Evolution of Porfolio using LSTM Prob (" + self.record.index[0].strftime(
				"%d/%m/%Y") + "-" +
				  self.record.index[-1].strftime("%d/%m/%Y") + ")", xaxis_title="Date",
			yaxis_title="Value [$]", hovermode='x unified')
		fig.write_image("images/EvolutionPorfolioLSTMProb(" + self.record.index[0].strftime(
				"%d_%m_%Y") + "-" +
				  self.record.index[-1].strftime("%d_%m_%Y") + ").png",scale=6, width=1080, height=1080)
		# fig.show()

		# Plot indicating the value of the indicator, the value of the stock market and the decisions made
		fig = make_subplots(rows=2, cols=2, specs=[[{"secondary_y": True, "colspan": 2}, None], [{"secondary_y": False}, {"secondary_y": False}]])

		fig.add_trace(go.Scatter(name="lstmPredictedReturn", x=self.record.index,
								 y=expData["lstmReturn"][-len(self.record.index):]), row=1, col=1,
					  secondary_y=True)
		realReturn = np.log(stockMarketData.Open[-len(self.record.index):].shift(-1)) - np.log(stockMarketData.Open[-len(self.record.index):])
		fig.add_trace(go.Scatter(name="realReturn", x=self.record.index, y=realReturn[-len(self.record.index):]), row=1, col=1, secondary_y=True)
		fig.add_trace(go.Scatter(name="Stock Market Value Open", x=self.record.index,
								 y=stockMarketData.Open[-len(self.record.index):]), row=1, col=1, secondary_y=False)
		fig.add_trace(go.Scatter(name="Stock Market Value Close", x=self.record.index, visible='legendonly',
								 y=stockMarketData.Close[-len(self.record.index):]), row=1, col=1, secondary_y=False)
		fig.add_trace(go.Bar(name="Money Invested Today", x=self.record.index, y=self.record["moneyInvestedToday"]), row=2, col=1, secondary_y=False)

		fig.add_trace(go.Bar(name="Prob Sell", x=self.record.index, y=-expData["lstmProb0"][-len(self.record.index):],
							 marker_color="red"), row=2, col=2)
		fig.add_trace(
			go.Bar(name="Prob buy", x=self.record.index, y=expData["lstmProb1"][-len(self.record.index):], marker_color="green"),
			row=2, col=2)
		fig.update_xaxes(title_text="Date", row=1, col=1)
		fig.update_xaxes(title_text="Date", row=2, col=1)
		fig.update_layout(
			title="Decision making under LSTM Prob (" + self.record.index[0].strftime("%d/%m/%Y") + "-" +
				  self.record.index[-1].strftime("%d/%m/%Y") + ")", hovermode='x unified')
		fig.write_image("images/DecisionMakingLSTMProb(" + self.record.index[0].strftime("%d_%m_%Y") + "-" +
				  self.record.index[-1].strftime("%d_%m_%Y") + ").png",scale=6, width=1080, height=1080)
		# fig.show()