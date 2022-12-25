from classes.investorClass import Investor
from classes.investorParamsClass import LSTMInvestorParams
from LSTM.LSTMClass import LSTMClass
from classes.dataClass import DataManager
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


class InvestorLSTM(Investor):
	def __init__(self, initialInvestment=10000, lstmParams: LSTMInvestorParams = None):
		super().__init__(initialInvestment)
		self.lstmParams = lstmParams
		self.model = LSTMClass()

	def returnBrokerUpdate(self, moneyInvestedToday, moneySoldToday, data: DataManager):
		return pd.DataFrame(
			{"lstmReturn": [data.lstm["return"]], "lstmProb0": [data.lstm["prob0"]], "lstmProb1": [data.lstm["prob1"]],
			 'moneyToInvestLSTM': [moneyInvestedToday], 'moneyToSellLSTM': [moneySoldToday],
			 'investedMoneyLSTM': [self.investedMoney], 'nonInvestedMoneyLSTM': [self.nonInvestedMoney]})

	def possiblyInvestTomorrow(self, data: DataManager):
		"""
		Function that calls the buy function and updates the investment values
		:param data: Decision data based on the type of indicator
		"""
		self.perToInvest = self.buyPrediction(data.lstm["return"][0])

	def possiblySellTomorrow(self, data: DataManager):
		"""
		Function that calls the sell function and updates the investment values
		:param data: Decision data based on the type of indicator
		"""
		self.perToSell = self.sellPrediction(data.lstm["return"][0])

	def buyPrediction(self, data):
		"""
		Function that returns the money to be invested
		:param data: predicted return value by LSTM
		:return:
		"""
		if data > self.lstmParams.threshold:
			return 1
		return 0

	def sellPrediction(self, data):
		"""
		Function that returns the money to be sold
		:param data: predicted return value by LSTM
		:return:
		"""
		if data < -self.lstmParams.threshold:
			return 1
		return 0

	def plotEvolution(self, indicatorData, stockMarketData, recordPredictedValue=None):
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
			title="Evolution of Porfolio using LSTM (" + self.record.index[0].strftime(
				"%d/%m/%Y") + "-" +
				  self.record.index[-1].strftime("%d/%m/%Y") + ")", xaxis_title="Date",
			yaxis_title="Value [$]", hovermode='x unified')
		fig.show()

		# Plot indicating the value of the indicator, the value of the stock market and the decisions made
		fig = make_subplots(rows=2, cols=2, specs=[[{"secondary_y": True, "colspan": 2}, None], [{"secondary_y": False}, {"secondary_y": False}]])

		fig.add_trace(go.Scatter(name="lstmPredictedReturn", x=self.record.index,
								 y=indicatorData["return"][-len(self.record.index):]), row=1, col=1,
					  secondary_y=True)
		realReturn = np.log(stockMarketData.Open[-len(self.record.index):].shift(+1)) - np.log(stockMarketData.Open[-len(self.record.index):])
		fig.add_trace(go.Scatter(name="realReturn", x=self.record.index, y=realReturn[-len(self.record.index):]), row=1, col=1, secondary_y=True)
		fig.add_trace(go.Scatter(name="Stock Market Value Open", x=self.record.index,
								 y=stockMarketData.Open[-len(self.record.index):]), row=1, col=1, secondary_y=False)
		fig.add_trace(go.Scatter(name="Stock Market Value Close", x=self.record.index, visible='legendonly',
								 y=stockMarketData.Close[-len(self.record.index):]), row=1, col=1, secondary_y=False)
		fig.add_trace(go.Bar(name="Money Invested Today", x=self.record.index, y=self.record["moneyInvestedToday"], marker_color="green"), row=2, col=1, secondary_y=False)
		fig.add_trace(go.Bar(name="Money Sold Today", x=self.record.index, y=-self.record["moneySoldToday"], marker_color="red"), row=2, col=1, secondary_y=False)

		fig.add_trace(go.Bar(name="Prob Sell", x=self.record.index, y=-indicatorData["prob0"][-len(self.record.index):],
							 marker_color="red"), row=2, col=2)
		fig.add_trace(
			go.Bar(name="Prob buy", x=self.record.index, y=indicatorData["prob1"][-len(self.record.index):], marker_color="green"),
			row=2, col=2)
		fig.update_xaxes(title_text="Date", row=1, col=1)
		fig.update_xaxes(title_text="Date", row=2, col=1)
		fig.update_layout(
			title="Decision making under LSTM (" + self.record.index[0].strftime("%d/%m/%Y") + "-" +
				  self.record.index[-1].strftime("%d/%m/%Y") + ")", hovermode='x unified')
		fig.show()