from classes.investorParamsClass import NNInvestorParams
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from classes.investorClass import Investor
import pandas as pd
from DecisionFunction.decisionFunctionNN import NNDecisionFunction


class InvestorBBNN(Investor):
	def __init__(self, initialInvestment=10000, nnParams: NNInvestorParams = None):
		super().__init__(initialInvestment)
		self.nnParams = nnParams
		self.model = NNDecisionFunction()
		self.model.load(nnParams.file)

	def returnBrokerUpdate(self, moneyInvestedToday, data):
		return pd.DataFrame(
			{'BBNN': data["bbpband"][-1], 'moneyToInvestBBNN': moneyInvestedToday,
			 'investedMoneyBBNN':self.investedMoney, 'nonInvestedMoneyBBNN': self.nonInvestedMoney}, index=[0])

	def possiblyInvestMorning(self, data):
		"""
		Function that calls the buy function and updates the investment values
		:param data: Decision data based on the type of indicator
		"""
		bb = data["bbpband"]
		inputs = [bb[-1], bb[-2]]
		inputs = np.asarray(inputs)
		y = self.model.predict([inputs.tolist()])[0]
		self.perToInvest = (y - 0.5) * 2

	def possiblyInvestAfternoon(self, data):
		"""
		Function that calls the sell function and updates the investment values
		:param data: Decision data based on the type of indicator
		"""
		bb = data["bbpband"]
		inputs = [bb[-1], bb[-2]]
		inputs = np.asarray(inputs)
		y = self.model.predict([inputs.tolist()])[0]
		self.perToInvest = (y - 0.5) * 2

	def plotEvolution(self, expData, stockMarketData, recordPredictedValue=None):
		"""
		Function that plots the actual status of the investor investment as well as the decisions that have been made
		:param expData: Data belonging to the indicator used to take decisions
		:param stockMarketData: df with the stock market data
		:param recordPredictedValue: Predicted data dataframe
		"""
		# Plot indicating the evolution of the total value and contain (moneyInvested and moneyNotInvested)
		fig = go.Figure()
		fig.add_trace(go.Scatter(name="Money Invested", x=self.record.index, y=self.record["moneyInvested"], stackgroup="one"))
		fig.add_trace(go.Scatter(name="Money Not Invested", x=self.record.index, y=self.record["moneyNotInvested"], stackgroup="one"))
		fig.add_trace(go.Scatter(name="Total Value", x=self.record.index, y=self.record["totalValue"]))
		fig.update_layout(
			title="Evolution of Porfolio using BBNN (" + self.record.index[0].strftime(
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

		fig.add_trace(go.Scatter(name="BB PBand", x=self.record.index,
								 y=expData["BBNN"][-len(self.record.index):]), row=1, col=1,
					  secondary_y=True)
		fig.add_trace(go.Scatter(name="Stock Market Value Open", x=self.record.index,
								 y=stockMarketData.Open[-len(self.record.index):]), row=1, col=1, secondary_y=False)
		fig.add_trace(go.Scatter(name="Stock Market Value Close", x=self.record.index,
								 y=stockMarketData.Close[-len(self.record.index):]), row=1, col=1, secondary_y=False)
		fig.add_trace(go.Bar(name="Money Invested Today", x=self.record.index, y=self.record["moneyInvestedToday"]), row=2, col=1)
		fig.update_xaxes(title_text="Date", row=1, col=1)
		fig.update_xaxes(title_text="Date", row=2, col=1)
		fig.update_layout(
			title="Decision making under BBNN (" + self.record.index[0].strftime("%d/%m/%Y") + "-" +
				  self.record.index[-1].strftime("%d/%m/%Y") + ")", hovermode='x unified')
		fig.show()

class InvestorBBRSINNClass(Investor):
	def __init__(self, initialInvestment=10000, nnParams: NNInvestorParams = None):
		super().__init__(initialInvestment)
		self.nnParams = nnParams
		self.model = NNDecisionFunction()
		self.model.load(nnParams.file)

	def returnBrokerUpdate(self, moneyInvestedToday, data):
		return pd.DataFrame(
			{'BBRSINNClassBB': data["bbpband"][-1], 'BBRSINNClassRSI': data["rsirsi"][-1], 'moneyToInvestBBRSINNClass': moneyInvestedToday,
			 'investedMoneyBBRSINNClass': self.investedMoney, 'nonInvestedMoneyBBRSINNClass': self.nonInvestedMoney}, index=[0])

	def possiblyInvestMorning(self, data):
		"""
		Function that calls the buy function and updates the investment values
		:param data: Decision data based on the type of indicator
		"""
		bb = data["bbpband"]
		rsi= data["rsirsi"]
		self.perToInvest = 0
		inputs = [bb[-1], bb[-2], rsi[-1], rsi[-2]]
		inputs = np.asarray(inputs)
		y = self.model.predict([inputs.tolist()])
		if y[:, 1] > y[:, 0]:
			self.perToInvest = (y[:, 1] - 0.5) * 2
		if y[:, 1] < y[:, 0]:
			self.perToInvest = -(y[:, 0] - 0.5) * 2


	def possiblyInvestAfternoon(self, data):
		"""
		Function that calls the sell function and updates the investment values
		:param data: Decision data based on the type of indicator
		"""
		bb = data["bbpband"]
		rsi = data["rsirsi"]
		self.perToInvest = 0
		inputs = [bb[-1], bb[-2], rsi[-1], rsi[-2]]
		inputs = np.asarray(inputs)
		y = self.model.predict([inputs.tolist()])
		if y[:, 1] > y[:, 0]:
			self.perToInvest = (y[:, 1] - 0.5) * 2
		if y[:, 1] < y[:, 0]:
			self.perToInvest = -(y[:, 0] - 0.5) * 2


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
			title="Evolution of Porfolio using BBRSINNClass (" + self.record.index[0].strftime(
				"%d/%m/%Y") + "-" +
				  self.record.index[-1].strftime("%d/%m/%Y") + ")", xaxis_title="Date",
			yaxis_title="Value [$]", hovermode='x unified')
		fig.show()

		# Plot indicating the value of the indicator, the value of the stock market and the decisions made
		fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": True}], [{"secondary_y": False}]])
		if recordPredictedValue is not None:
			fig.add_trace(go.Scatter(name="Predicted Stock Market Value Close", x=recordPredictedValue.index,
									 y=recordPredictedValue[0]), row=1, col=1,
						  secondary_y=False)

		fig.add_trace(go.Scatter(name="RSI", x=self.record.index,
								 y=expData["BBRSINNClassRSI"][-len(self.record.index):]), row=1, col=1,
					  secondary_y=True)
		fig.add_trace(go.Scatter(name="BB PBand", x=self.record.index,
								 y=expData["BBRSINNClassBB"][-len(self.record.index):]), row=1, col=1,
					  secondary_y=True)
		fig.add_trace(go.Scatter(name="Stock Market Value Open", x=self.record.index,
								 y=stockMarketData.Open[-len(self.record.index):]), row=1, col=1, secondary_y=False)
		fig.add_trace(go.Scatter(name="Stock Market Value Close", x=self.record.index,
								 y=stockMarketData.Close[-len(self.record.index):]), row=1, col=1, secondary_y=False)
		fig.add_trace(go.Bar(name="Money Invested Today", x=self.record.index, y=self.record["moneyInvestedToday"]), row=2, col=1)
		fig.update_xaxes(title_text="Date", row=1, col=1)
		fig.update_xaxes(title_text="Date", row=2, col=1)
		fig.update_layout(
			title="Decision making under BBRSINNClass (" + self.record.index[0].strftime("%d/%m/%Y") + "-" +
				  self.record.index[-1].strftime("%d/%m/%Y") + ")", hovermode='x unified')
		fig.show()

class InvestorBBRSINN(Investor):
	def __init__(self, initialInvestment=10000, nnParams: NNInvestorParams = None):
		super().__init__(initialInvestment)
		self.nnParams = nnParams
		self.model = NNDecisionFunction()
		self.model.load(nnParams.file)

	def returnBrokerUpdate(self, moneyInvestedToday, data):
		return pd.DataFrame(
			{'BBRSINNBB': data["bbpband"][-1], 'BBRSINNRSI': data["rsirsi"][-1], 'moneyToInvestBBRSINN': moneyInvestedToday,
			 'investedMoneyBBRSINN': self.investedMoney, 'nonInvestedMoneyBBRSINN': self.nonInvestedMoney}, index=[0])

	def possiblyInvestMorning(self, data):
		"""
		Function that calls the buy function and updates the investment values
		:param data: Decision data based on the type of indicator
		"""
		bb = data["bbpband"]
		rsi = data["rsirsi"]
		inputs = [bb[-1], bb[-2], rsi[-1], rsi[-2]]
		inputs = np.asarray(inputs)
		y = self.model.predict([inputs.tolist()])[0]
		self.perToInvest = (y - 0.5) * 2

	def possiblyInvestAfternoon(self, data):
		"""
		Function that calls the sell function and updates the investment values
		:param data: Decision data based on the type of indicator
		"""
		bb = data["bbpband"]
		rsi = data["rsirsi"]
		inputs = [bb[-1], bb[-2], rsi[-1], rsi[-2]]
		inputs = np.asarray(inputs)
		y = self.model.predict([inputs.tolist()])[0]
		self.perToInvest = (y - 0.5) * 2

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
			title="Evolution of Porfolio using BBRSINN (" + self.record.index[0].strftime(
				"%d/%m/%Y") + "-" +
				  self.record.index[-1].strftime("%d/%m/%Y") + ")", xaxis_title="Date",
			yaxis_title="Value [$]", hovermode='x unified')
		fig.show()

		# Plot indicating the value of the indicator, the value of the stock market and the decisions made
		fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": True}], [{"secondary_y": False}]])
		if recordPredictedValue is not None:
			fig.add_trace(go.Scatter(name="Predicted Stock Market Value Close", x=recordPredictedValue.index,
									 y=recordPredictedValue[0]), row=1, col=1,
						  secondary_y=False)

		fig.add_trace(go.Scatter(name="RSI", x=self.record.index,
								 y=expData["BBRSINNRSI"][-len(self.record.index):]), row=1, col=1,
					  secondary_y=True)
		fig.add_trace(go.Scatter(name="BB PBand", x=self.record.index,
								 y=expData["BBRSINNBB"][-len(self.record.index):]), row=1, col=1,
					  secondary_y=True)
		fig.add_trace(go.Scatter(name="Stock Market Value Open", x=self.record.index,
								 y=stockMarketData.Open[-len(self.record.index):]), row=1, col=1, secondary_y=False)
		fig.add_trace(go.Scatter(name="Stock Market Value Close", x=self.record.index,
								 y=stockMarketData.Close[-len(self.record.index):]), row=1, col=1, secondary_y=False)
		fig.add_trace(go.Bar(name="Money Invested Today", x=self.record.index, y=self.record["moneyInvestedToday"]), row=2, col=1)
		fig.update_xaxes(title_text="Date", row=1, col=1)
		fig.update_xaxes(title_text="Date", row=2, col=1)
		fig.update_layout(
			title="Decision making under BBRSINN (" + self.record.index[0].strftime("%d/%m/%Y") + "-" +
				  self.record.index[-1].strftime("%d/%m/%Y") + ")", hovermode='x unified')
		fig.show()