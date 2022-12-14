import numpy as np
from classes.investorClass import Investor
from classes.investorParamsClass import DTInvestorParams
from DecisionFunction.decisionFunctionTree import DecisionFunctionTree
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class InvestorDecisionTree(Investor):
	def __init__(self, initialInvestment=10000, decisionTreeParams: DTInvestorParams = None):
		super().__init__(initialInvestment)
		self.dtParams = decisionTreeParams
		self.model = DecisionFunctionTree()
		self.model.load(self.dtParams.filename)

	def returnBrokerUpdate(self, moneyInvestedToday, data):
		aux = pd.DataFrame()
		for indicator in self.dtParams.orderedListArguments:
			aux = pd.concat([aux, pd.DataFrame({indicator: data[indicator]}, index=[0])], axis=1)

		return pd.concat([aux, pd.DataFrame(
			{'moneyToInvestDT': moneyInvestedToday,
			 'investedMoneyDT': self.investedMoney, 'nonInvestedMoneyDT': self.nonInvestedMoney}, index=[0])], axis=1)

	def possiblyInvestMorning(self, data):
		"""
		Function that calls the buy function and updates the investment values
		:param data: Decision data based on the type of indicator
		"""
		agg = []
		for indicator in self.dtParams.orderedListArguments:
			agg.append(data[indicator])
		agg = np.asarray(agg).transpose()
		y = self.model.predict(agg)
		self.perToInvest = (y - 0.5) * 2

	def possiblyInvestAfternoon(self, data):
		"""
		Function that calls the sell function and updates the investment values
		:param data: Decision data based on the type of indicator
		"""
		agg = []
		for indicator in self.dtParams.orderedListArguments:
			agg.append(data[indicator])
		agg = np.asarray(agg).transpose()
		y = self.model.predict(agg)
		self.perToInvest = (y - 0.5) * 2
		# self.perToInvest = 0


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
			title="Evolution of Porfolio using DT (" + self.record.index[0].strftime(
				"%d/%m/%Y") + "-" +
				  self.record.index[-1].strftime("%d/%m/%Y") + ")", xaxis_title="Date",
			yaxis_title="Value [$]", hovermode='x unified')
		fig.write_image("images/EvolutionPorfolioDT(" + self.record.index[0].strftime(
				"%d_%m_%Y") + "-" +
				  self.record.index[-1].strftime("%d_%m_%Y") + ").png",scale=6, width=1080, height=1080)
		# fig.show()

		# Plot indicating the value of the indicator, the value of the stock market and the decisions made
		fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": True}], [{"secondary_y": False}]])

		for indicator in self.dtParams.orderedListArguments:
			fig.add_trace(go.Scatter(name=indicator, x=self.record.index,
									 y=expData[indicator][-len(self.record.index):], visible='legendonly'), row=1, col=1,
						  secondary_y=True)
		fig.add_trace(go.Scatter(name="Stock Market Value Open", x=self.record.index,
								 y=stockMarketData.Open[-len(self.record.index):]), row=1, col=1, secondary_y=False)
		fig.add_trace(go.Scatter(name="Stock Market Value Close", x=self.record.index, visible='legendonly',
								 y=stockMarketData.Close[-len(self.record.index):]), row=1, col=1, secondary_y=False)
		fig.add_trace(go.Bar(name="Money Invested Today", x=self.record.index, y=self.record["moneyInvestedToday"]), row=2, col=1)
		fig.update_xaxes(title_text="Date", row=1, col=1)
		fig.update_xaxes(title_text="Date", row=2, col=1)
		fig.update_layout(
			title="Decision making under DT (" + self.record.index[0].strftime("%d/%m/%Y") + "-" +
				  self.record.index[-1].strftime("%d/%m/%Y") + ")", hovermode='x unified')
		fig.write_image("images/DecisionMakingDT(" + self.record.index[0].strftime("%d_%m_%Y") + "-" +
				  self.record.index[-1].strftime("%d_%m_%Y") + ").png",scale=6, width=1080, height=1080)
		# fig.show()
