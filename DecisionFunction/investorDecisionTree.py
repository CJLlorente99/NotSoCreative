from classes.investorClass import Investor
from classes.investorParamsClass import DTInvestorParams
from DecisionFunction.decisionFunctionTree import DecisionFunctionTree
from classes.dataClass import DataManager
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class InvestorDecisionTree(Investor):
	def __init__(self, initialInvestment=10000, decisionTreeParams: DTInvestorParams = None):
		super().__init__(initialInvestment)
		self.dtParams = decisionTreeParams
		self.model = DecisionFunctionTree()
		self.model.load(self.dtParams.filename)

	def returnBrokerUpdate(self, moneyInvestedToday, moneySoldToday, data: DataManager):
		aux = pd.DataFrame()
		for key in data.dt:
			if key == "aggregated":
				continue
			aux = pd.concat([aux, pd.DataFrame({key: [data.dt[key]]})], axis=1)

		return pd.concat([aux,pd.DataFrame(
			{'moneyToInvestDT': [moneyInvestedToday], 'moneyToSellDT': [moneySoldToday],
			 'investedMoneyDT': [self.investedMoney], 'nonInvestedMoneyDT': [self.nonInvestedMoney]})], axis=1)

	def possiblyInvestTomorrow(self, data: DataManager):
		"""
		Function that calls the buy function and updates the investment values
		:param data: Decision data based on the type of indicator
		"""
		self.perToInvest = self.buyPrediction(data.dt)

	def possiblySellTomorrow(self, data: DataManager):
		"""
		Function that calls the sell function and updates the investment values
		:param data: Decision data based on the type of indicator
		"""
		self.perToSell = self.sellPrediction(data.dt)

	def buyPrediction(self, data):
		"""
		Function that returns the money to be invested
		:param bb: bollinger_pband() value
		:return:
		"""
		inputs = data["aggregated"]
		y = self.model.predict(inputs)
		if y > 0.5:
			return (y - 0.5) *  2
		return 0

	def sellPrediction(self, data):
		"""
		Function that returns the money to be sold
		:param bb: bollinger_pband() value
		:return:
		"""
		inputs = data["aggregated"]
		y = self.model.predict(inputs)
		if y < 0.5:
			return -(y - 0.5) *  2
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
			title="Evolution of Porfolio using DT (" + self.record.index[0].strftime(
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

		for name in indicatorData:
			fig.add_trace(go.Scatter(name=name, x=self.record.index,
									 y=indicatorData[name][-len(self.record.index):]), row=1, col=1,
						  secondary_y=True)
		fig.add_trace(go.Scatter(name="Stock Market Value Open", x=self.record.index,
								 y=stockMarketData.Open[-len(self.record.index):]), row=1, col=1, secondary_y=False)
		fig.add_trace(go.Scatter(name="Stock Market Value Close", x=self.record.index,
								 y=stockMarketData.Close[-len(self.record.index):]), row=1, col=1, secondary_y=False)
		fig.add_trace(go.Bar(name="Money Invested Today", x=self.record.index, y=self.record["moneyInvestedToday"], marker_color="green"), row=2, col=1)
		fig.add_trace(go.Bar(name="Money Sold Today", x=self.record.index, y=-self.record["moneySoldToday"], marker_color="red"), row=2, col=1)
		fig.update_xaxes(title_text="Date", row=1, col=1)
		fig.update_xaxes(title_text="Date", row=2, col=1)
		fig.update_layout(
			title="Decision making under DT (" + self.record.index[0].strftime("%d/%m/%Y") + "-" +
				  self.record.index[-1].strftime("%d/%m/%Y") + ")", hovermode='x unified')
		fig.show()
