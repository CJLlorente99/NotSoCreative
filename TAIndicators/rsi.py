import numpy as np
import ta
from classes.investorParamsClass import RSIInvestorParams
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
from classes.investorClass import Investor
import pandas as pd


class InvestorRSI(Investor):
    def __init__(self, initialInvestment=10000, rsiParams=None):
        super().__init__(initialInvestment)
        self.rsiParams = rsiParams

    def returnBrokerUpdate(self, moneyInvestedToday, data):
        return pd.DataFrame(
                {'rsi': data["rsirsi"], 'moneyToInvestRSI': moneyInvestedToday,
                 'investedMoneyRSI': self.investedMoney, 'nonInvestedMoneyRSI': self.nonInvestedMoney}, index=[0])

    def possiblyInvestMorning(self, data):
        """
        Function that calls the buy function and updates the investment values
        :param data: Decision data based on the type of indicator
        """
        rsi = data["rsirsi"]

        params = self.rsiParams

        self.perToInvest = 0
        if rsi < params.lowerBound:  # Buy linearly then with factor f
            self.perToInvest = math.tanh(params.a * (params.lowerBound - rsi) ** params.b)
        elif rsi > params.upperBound:  # Buy linearly then with factor f
            self.perToInvest = -math.tanh(params.a * (rsi - params.upperBound) ** params.b)

    def possiblyInvestAfternoon(self, data):
        """
        Function that calls the sell function and updates the investment values
        :param data: Decision data based on the type of indicator
        """
        rsi = data["rsirsi"]

        params = self.rsiParams

        self.perToInvest = 0
        # if rsi < params.lowerBound:  # Buy linearly then with factor f
        #     self.perToInvest = math.tanh(params.a * (params.lowerBound - rsi) ** params.b)
        # elif rsi > params.upperBound:  # Buy linearly then with factor f
        #     self.perToInvest = -math.tanh(params.a * (rsi - params.upperBound) ** params.b)

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
            title="Evolution of Porfolio using RSI (" + self.record.index[0].strftime(
                "%d/%m/%Y") + "-" +
                  self.record.index[-1].strftime("%d/%m/%Y") + ")", xaxis_title="Date",
            yaxis_title="Value [$]", hovermode='x unified')
        fig.write_image("images/EvolutionPorfolioRSI(" + self.record.index[0].strftime(
                "%d_%m_%Y") + "-" +
                  self.record.index[-1].strftime("%d_%m_%Y") + ").png",scale=6, width=1080, height=1080)
        # fig.show()

        # Plot indicating the value of the indicator, the value of the stock market and the decisions made
        fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": True}], [{"secondary_y": False}]])
        if recordPredictedValue is not None:
            fig.add_trace(go.Scatter(name="Predicted Stock Market Value Close", x=recordPredictedValue.index,
                                     y=recordPredictedValue[0]), row=1, col=1,
                          secondary_y=False)
        fig.add_trace(
            go.Scatter(name="RSI", x=self.record.index, y=expData["rsi"][-len(self.record.index):]), row=1,
            col=1, secondary_y=True)
        fig.add_trace(go.Scatter(name="Stock Market Value Open", x=self.record.index,
                                 y=stockMarketData.Open[-len(self.record.index):]), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(name="Stock Market Value Close", x=self.record.index,
                                 y=stockMarketData.Close[-len(self.record.index):]), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Bar(name="Money Invested Today", x=self.record.index, y=self.record["moneyInvestedToday"]), row=2, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_layout(
            title="Decision making under RSI (" + self.record.index[0].strftime("%d/%m/%Y") + "-" +
                  self.record.index[-1].strftime("%d/%m/%Y") + ")", hovermode='x unified')
        fig.write_image("images/DecisionMakingRSI(" + self.record.index[0].strftime("%d_%m_%Y") + "-" +
                  self.record.index[-1].strftime("%d_%m_%Y") + ").png",scale=6, width=1080, height=1080)
        # fig.show()


def relativeStrengthIndex(close, params: RSIInvestorParams):
    """
    Function that calculates the RSI values
    :param close:
    :param params: RSI parameters
    :return dict with the following keys ["rsi"]
    """
    rsi = ta.momentum.RSIIndicator(close, params.window, True)
    return {"rsi": rsi.rsi()}
