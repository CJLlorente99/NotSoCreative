import pandas as pd
from classes.investorClass import Investor
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from TAIndicators.ma import movingAverageConvergenceDivergence
from classes.investorParamsClass import MACDInvestorParams


class InvestorMACD(Investor):

	def returnBrokerUpdate(self, moneyInvestedToday, data) -> pd.DataFrame:
		return pd.DataFrame(
			{'moneyToInvestMACD': moneyInvestedToday,
			 'investedMoneyMACD': self.investedMoney, 'nonInvestedMoneyMACD': self.nonInvestedMoney},
			index=[0])

	def possiblyInvestMorning(self, data):
		params = MACDInvestorParams(None, None, 12, 26, 9, None, None, None)
		ma = movingAverageConvergenceDivergence(data['df']['Close'], params)
		if ma['diff'][-1] * ma['diff'][-2] < 0:
			if ma['signal'][-2] < 0:
				self.perToInvest = 1
			else:
				self.perToInvest = -1
		else:
			self.perToInvest = 0


	def possiblyInvestAfternoon(self, data):
		self.perToInvest = 0

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
			title="Evolution of Porfolio using MACD (" + self.record.index[0].strftime(
				"%d/%m/%Y") + "-" +
				  self.record.index[-1].strftime("%d/%m/%Y") + ")", xaxis_title="Date",
			yaxis_title="Value [$]", hovermode='x unified')
		fig.write_image("images/EvolutionPorfolioMACD(" + self.record.index[0].strftime(
			"%d_%m_%Y") + "-" +
						self.record.index[-1].strftime("%d_%m_%Y") + ").png", scale=6, width=1080, height=1080)
		# fig.show()

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
			title="Decision making under MACD (" + self.record.index[0].strftime("%d/%m/%Y") + "-" +
				  self.record.index[-1].strftime("%d/%m/%Y") + ")", hovermode='x unified')
		fig.write_image("images/DecisionMakingMACD(" + self.record.index[0].strftime("%d_%m_%Y") + "-" +
						self.record.index[-1].strftime("%d_%m_%Y") + ").png", scale=6, width=1080, height=1080)
		# fig.show()


