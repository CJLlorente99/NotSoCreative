from classes.investorClass import Investor
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

"""
This strategy represents the worse approach. This is, depending on the future, if the stock market is going to increase,
all will be sold. If market is going to decrease, all will be bought.
"""
class InvestorWIA(Investor):
	def __init__(self, initialInvestment=10000):
		super().__init__(initialInvestment)

	def returnBrokerUpdate(self, moneyInvestedToday, moneySoldToday, data):
		return pd.DataFrame(
			{'nextStockValue': [data["nextStockValueOpen"]], 'actualStockValue': [data["actualStockValue"]],
			 'moneyToInvestWIA': [moneyInvestedToday], 'moneyToSellWIA': [moneySoldToday],
			 'investedMoneyWIA': [self.investedMoney], 'nonInvestedMoneyWIA': [self.nonInvestedMoney]})

	def possiblyInvestTomorrow(self, data):
		"""
		Function that calls the buy function and updates the investment values
		:param data: Decision data based on the type of indicator
		"""
		self.perToInvest = self.buyPredictionWIA(data["nextStockValueOpen"], data["nextnextStockValueOpen"], data["actualStockValue"])

	def possiblySellTomorrow(self, data):
		"""
		Function that calls the sell function and updates the investment values
		:param data: Decision data based on the type of indicator
		"""
		self.perToSell = self.sellPredictionWIA(data["nextStockValueOpen"], data["nextnextStockValueOpen"], data["actualStockValue"])

	def buyPredictionWIA(self, nextStockValueOpen, nextnextStockValueOpen, actualStockValueOpen):
		if nextnextStockValueOpen < nextStockValueOpen:
			return 1
		else:
			return 0

	def sellPredictionWIA(self, nextStockValueOpen, nextnextStockValueOpen, actualStockValueOpen):
		if nextnextStockValueOpen > nextStockValueOpen:
			return 1
		else:
			return 0

	def plotEvolution(self, expData, stockMarketData, recordPredictedValue=None):
		"""
		Function that plots the actual status of the investor investment as well as the decisions that have been made
		:param stockMarketData: df with the stock market data
		:param recordPredictedValue: Predicted data dataframe
		"""
		self.record = self.record.iloc[1:]
		# Plot indicating the evolution of the total value and contain (moneyInvested and moneyNotInvested)
		fig = go.Figure()
		fig.add_trace(
			go.Scatter(name="Money Invested", x=self.record.index, y=self.record["moneyInvested"], stackgroup="one"))
		fig.add_trace(go.Scatter(name="Money Not Invested", x=self.record.index, y=self.record["moneyNotInvested"],
								 stackgroup="one"))
		fig.add_trace(go.Scatter(name="Total Value", x=self.record.index, y=self.record["totalValue"]))
		fig.update_layout(
			title="Evolution of Porfolio using WIA (" + self.record.index[0].strftime(
				"%d/%m/%Y") + "-" +
				  self.record.index[-1].strftime("%d/%m/%Y") + ")", xaxis_title="Date",
			yaxis_title="Value [$]", hovermode='x unified')
		fig.write_image("images/EvolutionPorfolioWIA(" + self.record.index[0].strftime(
				"%d_%m_%Y") + "-" +
				  self.record.index[-1].strftime("%d_%m_%Y") + ").png",scale=6, width=1080, height=1080)
		# fig.show()

		# Plot indicating the value of the indicator, the value of the stock market and the decisions made
		fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": True}], [{"secondary_y": False}]])
		if recordPredictedValue is not None:
			fig.add_trace(go.Scatter(name="Predicted Stock Market Value Close", x=recordPredictedValue.index,
									 y=recordPredictedValue[0]), row=1, col=1,
						  secondary_y=False)

		fig.add_trace(go.Scatter(name="Stock Market Value Open", x=self.record.index,
								 y=stockMarketData.Open[-len(self.record.index):]), row=1, col=1, secondary_y=False)
		fig.add_trace(go.Scatter(name="Stock Market Value Close", x=self.record.index,
								 y=stockMarketData.Close[-len(self.record.index):]), row=1, col=1, secondary_y=False)
		fig.add_trace(go.Bar(name="Money Invested Today", x=self.record.index, y=self.record["moneyInvestedToday"],
							 marker_color="green"), row=2, col=1)
		fig.add_trace(
			go.Bar(name="Money Sold Today", x=self.record.index, y=-self.record["moneySoldToday"], marker_color="red"),
			row=2, col=1)
		fig.update_xaxes(title_text="Date", row=1, col=1)
		fig.update_xaxes(title_text="Date", row=2, col=1)
		fig.update_layout(
			title="Decision making under WIA (" + self.record.index[0].strftime("%d/%m/%Y") + "-" +
				  self.record.index[-1].strftime("%d/%m/%Y") + ")", hovermode='x unified')
		fig.write_image("images/DecisionMakingWIA(" + self.record.index[0].strftime("%d_%m_%Y") + "-" +
				  self.record.index[-1].strftime("%d_%m_%Y") + ").png",scale=6, width=1080, height=1080)
		# fig.show()