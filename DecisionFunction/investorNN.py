from classes.investorParamsClass import NNInvestorParams
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from classes.investorClass import Investor
from classes.dataClass import DataManager
import pandas as pd
from DecisionFunction.decisionFunctionNN import NNDecisionFunction


class InvestorBBNN(Investor):
	def __init__(self, initialInvestment=10000, nnParams: NNInvestorParams = None):
		super().__init__(initialInvestment)
		self.nnParams = nnParams
		self.model = NNDecisionFunction()
		self.model.load(nnParams.file)

	def returnBrokerUpdate(self, moneyInvestedToday, moneySoldToday, data):
		return pd.DataFrame(
			{'BBNN': [data.bb[-1]], 'moneyToInvestBBNN': [moneyInvestedToday], 'moneyToSellBBNN': [moneySoldToday],
			 'investedMoneyBBNN': [self.investedMoney], 'nonInvestedMoneyBBNN': [self.nonInvestedMoney]})

	def possiblyInvestTomorrow(self, data: DataManager):
		"""
		Function that calls the buy function and updates the investment values
		:param data: Decision data based on the type of indicator
		"""
		self.perToInvest = self.buyPredictionBB(data.bb)

	def possiblySellTomorrow(self, data: DataManager):
		"""
		Function that calls the sell function and updates the investment values
		:param data: Decision data based on the type of indicator
		"""
		self.perToSell = self.sellPredictionBB(data.bb)

	def buyPredictionBB(self, bb):
		"""
		Function that returns the money to be invested
		:param bb: bollinger_pband() value
		:return:
		"""
		inputs = [bb[-1], bb[-2]]
		inputs = np.asarray(inputs)
		y = self.model.predict([inputs.tolist()])[0]
		if y > 0.5:
			return (y - 0.5) *  2
		return 0

	def sellPredictionBB(self, bb):
		"""
		Function that returns the money to be sold
		:param bb: bollinger_pband() value
		:return:
		"""
		inputs = [bb[-1], bb[-2]]
		inputs = np.asarray(inputs)
		y = self.model.predict([inputs.tolist()])[0]
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
								 y=indicatorData["pband"][-len(self.record.index):]), row=1, col=1,
					  secondary_y=True)
		fig.add_trace(go.Scatter(name="BB HBand", x=self.record.index,
								 y=indicatorData["hband"][-len(self.record.index):], line = dict(color='black', width=2, dash='dash')), row=1, col=1,
					  secondary_y=False)
		fig.add_trace(go.Scatter(name="BB LBand", x=self.record.index,
								 y=indicatorData["lband"][-len(self.record.index):], line = dict(color='black', width=2, dash='dash')), row=1, col=1,
					  secondary_y=False)
		fig.add_trace(go.Scatter(name="BB MAvg", x=self.record.index,
								 y=indicatorData["mavg"][-len(self.record.index):], line = dict(color='black', width=2, dash='dot')), row=1, col=1,
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
			title="Decision making under BBNN (" + self.record.index[0].strftime("%d/%m/%Y") + "-" +
				  self.record.index[-1].strftime("%d/%m/%Y") + ")", hovermode='x unified')
		fig.show()

class InvestorBBRSINNClass(Investor):
	def __init__(self, initialInvestment=10000, nnParams: NNInvestorParams = None):
		super().__init__(initialInvestment)
		self.nnParams = nnParams
		self.model = NNDecisionFunction()
		self.model.load(nnParams.file)

	def returnBrokerUpdate(self, moneyInvestedToday, moneySoldToday, data):
		return pd.DataFrame(
			{'BBRSINNClassBB': [data.bb[-1]], 'BBRSINNClassRSI': [data.rsi[-1]], 'moneyToInvestBBRSINNClass': [moneyInvestedToday], 'moneyToSellBBRSINNClass': [moneySoldToday],
			 'investedMoneyBBRSINNClass': [self.investedMoney], 'nonInvestedMoneyBBRSINNClass': [self.nonInvestedMoney]})

	def possiblyInvestTomorrow(self, data: DataManager):
		"""
		Function that calls the buy function and updates the investment values
		:param data: Decision data based on the type of indicator
		"""
		self.perToInvest = self.buyPrediction(data.bb, data.rsi)

	def possiblySellTomorrow(self, data: DataManager):
		"""
		Function that calls the sell function and updates the investment values
		:param data: Decision data based on the type of indicator
		"""
		self.perToSell = self.sellPrediction(data.bb, data.rsi)

	def buyPrediction(self, bb, rsi):
		"""
		Function that returns the money to be invested
		:param bb: bollinger_pband() value
		:return:
		"""
		inputs = [bb[-1], bb[-2], rsi[-1], rsi[-2]]
		inputs = np.asarray(inputs)
		y = self.model.predict([inputs.tolist()])
		if y[:, 1] > y[:, 0]:
			return (y[:, 1] - 0.5) * 2
		return 0

	def sellPrediction(self, bb, rsi):
		"""
		Function that returns the money to be sold
		:param bb: bollinger_pband() value
		:return:
		"""
		inputs = [bb[-1], bb[-2], rsi[-1], rsi[-2]]
		inputs = np.asarray(inputs)
		y = self.model.predict([inputs.tolist()])
		if y[:, 1] < y[:, 0]:
			return (y[:, 0] - 0.5) * 2
		return 0

	def plotEvolution(self, indicatorData, stockMarketData, recordPredictedValue=None):
		"""
		Function that plots the actual status of the investor investment as well as the decisions that have been made
		:param indicatorData: Data belonging to the indicator used to take decisions
		:param stockMarketData: df with the stock market data
		:param recordPredictedValue: Predicted data dataframe
		"""
		bbData = indicatorData["bb"]
		rsiData = indicatorData["rsi"]
		self.record = self.record.iloc[1:]
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
								 y=rsiData["rsi"][-len(self.record.index):]), row=1, col=1,
					  secondary_y=True)
		fig.add_trace(go.Scatter(name="BB PBand", x=self.record.index,
								 y=bbData["pband"][-len(self.record.index):]), row=1, col=1,
					  secondary_y=True)
		fig.add_trace(go.Scatter(name="BB HBand", x=self.record.index,
								 y=bbData["hband"][-len(self.record.index):], line = dict(color='black', width=2, dash='dash')), row=1, col=1,
					  secondary_y=False)
		fig.add_trace(go.Scatter(name="BB LBand", x=self.record.index,
								 y=bbData["lband"][-len(self.record.index):], line = dict(color='black', width=2, dash='dash')), row=1, col=1,
					  secondary_y=False)
		fig.add_trace(go.Scatter(name="BB MAvg", x=self.record.index,
								 y=bbData["mavg"][-len(self.record.index):], line = dict(color='black', width=2, dash='dot')), row=1, col=1,
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
			title="Decision making under BBRSINNClass (" + self.record.index[0].strftime("%d/%m/%Y") + "-" +
				  self.record.index[-1].strftime("%d/%m/%Y") + ")", hovermode='x unified')
		fig.show()

class InvestorBBRSINN(Investor):
	def __init__(self, initialInvestment=10000, nnParams: NNInvestorParams = None):
		super().__init__(initialInvestment)
		self.nnParams = nnParams
		self.model = NNDecisionFunction()
		self.model.load(nnParams.file)

	def returnBrokerUpdate(self, moneyInvestedToday, moneySoldToday, data):
		return pd.DataFrame(
			{'BBRSINNBB': [data.bb[-1]], 'BBRSINNRSI': [data.rsi[-1]], 'moneyToInvestBBRSINN': [moneyInvestedToday], 'moneyToSellBBRSINN': [moneySoldToday],
			 'investedMoneyBBRSINN': [self.investedMoney], 'nonInvestedMoneyBBRSINN': [self.nonInvestedMoney]})

	def possiblyInvestTomorrow(self, data: DataManager):
		"""
		Function that calls the buy function and updates the investment values
		:param data: Decision data based on the type of indicator
		"""
		self.perToInvest = self.buyPrediction(data.bb, data.rsi)

	def possiblySellTomorrow(self, data: DataManager):
		"""
		Function that calls the sell function and updates the investment values
		:param data: Decision data based on the type of indicator
		"""
		self.perToSell = self.sellPrediction(data.bb, data.rsi)

	def buyPrediction(self, bb, rsi):
		"""
		Function that returns the money to be invested
		:param bb: bollinger_pband() value
		:return:
		"""
		inputs = [bb[-1], bb[-2], rsi[-1], rsi[-2]]
		inputs = np.asarray(inputs)
		y = self.model.predict([inputs.tolist()])[0]
		if y > 0.5:
			return (y - 0.5) * 2
		return 0

	def sellPrediction(self, bb, rsi):
		"""
		Function that returns the money to be sold
		:param bb: bollinger_pband() value
		:return:
		"""
		inputs = [bb[-1], bb[-2], rsi[-1], rsi[-2]]
		inputs = np.asarray(inputs)
		y = self.model.predict([inputs.tolist()])[0]
		if y < 0.5:
			return (y - 0.5) * 2
		return 0

	def plotEvolution(self, indicatorData, stockMarketData, recordPredictedValue=None):
		"""
		Function that plots the actual status of the investor investment as well as the decisions that have been made
		:param indicatorData: Data belonging to the indicator used to take decisions
		:param stockMarketData: df with the stock market data
		:param recordPredictedValue: Predicted data dataframe
		"""
		bbData = indicatorData["bb"]
		rsiData = indicatorData["rsi"]
		self.record = self.record.iloc[1:]
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
		fig = go.Figure()
		fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": True}], [{"secondary_y": False}]])
		if recordPredictedValue is not None:
			fig.add_trace(go.Scatter(name="Predicted Stock Market Value Close", x=recordPredictedValue.index,
									 y=recordPredictedValue[0]), row=1, col=1,
						  secondary_y=False)

		fig.add_trace(go.Scatter(name="RSI", x=self.record.index,
								 y=rsiData["rsi"][-len(self.record.index):]), row=1, col=1,
					  secondary_y=True)
		fig.add_trace(go.Scatter(name="BB PBand", x=self.record.index,
								 y=bbData["pband"][-len(self.record.index):]), row=1, col=1,
					  secondary_y=True)
		fig.add_trace(go.Scatter(name="BB HBand", x=self.record.index,
								 y=bbData["hband"][-len(self.record.index):], line = dict(color='black', width=2, dash='dash')), row=1, col=1,
					  secondary_y=False)
		fig.add_trace(go.Scatter(name="BB LBand", x=self.record.index,
								 y=bbData["lband"][-len(self.record.index):], line = dict(color='black', width=2, dash='dash')), row=1, col=1,
					  secondary_y=False)
		fig.add_trace(go.Scatter(name="BB MAvg", x=self.record.index,
								 y=bbData["mavg"][-len(self.record.index):], line = dict(color='black', width=2, dash='dot')), row=1, col=1,
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
			title="Decision making under BBRSINN (" + self.record.index[0].strftime("%d/%m/%Y") + "-" +
				  self.record.index[-1].strftime("%d/%m/%Y") + ")", hovermode='x unified')
		fig.show()