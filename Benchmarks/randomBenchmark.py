import pandas as pd
from dataClass import DataManager
from classes.investorClass import Investor
from numpy import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots

"""
This strategy randomly invests or sells a percentage of the available or invested money.
"""
class InvestorRandom(Investor):

	def __init__(self, initialInvestment=10000):
		super().__init__(initialInvestment)
		self.rand = 0

	def returnBrokerUpdate(self, moneyInvestedToday, moneySoldToday, data):
		return pd.DataFrame(
			{'moneyToInvestRandom': [moneyInvestedToday], 'moneyToSellRandom': [moneySoldToday],
			 'investedMoneyRandom': [self.investedMoney], 'nonInvestedMoneyRandom': [self.nonInvestedMoney]})

	def possiblyInvestTomorrow(self, data: DataManager):
		"""
		Function prototype that calls the buy function and updates the investment values
		:param data: Decision data based on the type of indicator
		"""
		if self.rand > 0:
			self.moneyToInvest = self.rand * self.nonInvestedMoney
		else:
			self.moneyToInvest = 0

	def possiblySellTomorrow(self, data: DataManager):
		"""
		Function prototype that calls the sell function and updates the investment values
		:param data: Decision data based on the type of indicator
		"""
		self.rand = random.uniform(-1, 1)

		if self.rand < 0:
			self.moneyToSell = -self.rand * self.investedMoney
		else:
			self.moneyToSell = 0

	def plotEvolution(self, indicatorData, stockMarketData, recordPredictedValue=None):
		# Plot indicating the evolution of the total value and contain (moneyInvested and moneyNotInvested)
		fig = go.Figure()
		fig.add_trace(go.Scatter(name="Money Invested", x=self.record.index, y=self.record["moneyInvested"], stackgroup="one"))
		fig.add_trace(go.Scatter(name="Money Not Invested", x=self.record.index, y=self.record["moneyNotInvested"], stackgroup="one"))
		fig.add_trace(go.Scatter(name="Total Value", x=self.record.index, y=self.record["totalValue"]))
		fig.update_layout(
			title="Evolution of Porfolio using Random (" + self.record.index[0].strftime(
				"%d/%m/%Y") + "-" +
				  self.record.index[-1].strftime("%d/%m/%Y") + ")", xaxis_title="Date",
			yaxis_title="Value [$]", hovermode='x unified')
		fig.show()

		# Plot indicating the value of the indicator, the value of the stock market and the decisions made
		fig = go.Figure()
		fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": True}], [{"secondary_y": False}]])
		if recordPredictedValue is not None:
			fig.add_trace(go.Scatter(name="Predicted Stock Market Value Close", x=recordPredictedValue.index,
									 y=recordPredictedValue[0]), row=1, col=1,
						  secondary_y=False)

		fig.add_trace(go.Scatter(name="Stock Market Value Open", x=self.record.index,
								 y=stockMarketData.Open[-len(self.record.index):]), row=1, col=1, secondary_y=False)
		fig.add_trace(go.Scatter(name="Stock Market Value Close", x=self.record.index,
								 y=stockMarketData.Close[-len(self.record.index):]), row=1, col=1, secondary_y=False)
		fig.add_trace(go.Bar(name="Money Invested Today", x=self.record.index, y=self.record["moneyInvestedToday"], marker_color="green"), row=2, col=1)
		fig.add_trace(go.Bar(name="Money Sold Today", x=self.record.index, y=-self.record["moneySoldToday"], marker_color="red"), row=2, col=1)
		fig.update_xaxes(title_text="Date", row=1, col=1)
		fig.update_xaxes(title_text="Date", row=2, col=1)
		fig.update_layout(
			title="Decision making under Random (" + self.record.index[0].strftime("%d/%m/%Y") + "-" +
				  self.record.index[-1].strftime("%d/%m/%Y") + ")", hovermode='x unified')
		fig.show()
