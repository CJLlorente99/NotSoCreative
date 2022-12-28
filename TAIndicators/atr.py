import math
from ta.volatility import AverageTrueRange
from classes.investorParamsClass import ATRInvestorParams
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from classes.investorClass import Investor
import pandas as pd


class InvestorMACD(Investor):
    def __init__(self, initialInvestment=10000, atrParams=None):
        super().__init__(initialInvestment)
        self.atrParams = atrParams

    def returnBrokerUpdate(self, moneyInvestedToday, moneySoldToday, data):
        return pd.DataFrame(
            {'atr': [data.atr["average_true_range"][-1]], 'moneyToInvestATR': [moneyInvestedToday],
             'moneyToSellATR': [moneySoldToday], 'investedMoneyATR': [self.investedMoney],
             'nonInvestedMoneyATR': [self.nonInvestedMoney]})

    def possiblyInvestTomorrow(self, data):
        """
        Function that calls the buy function and updates the investment values
        :param data: Decision data based on the type of indicator
        """
        self.perToInvest = self.buyPredictionATR(data.atr)

    def possiblySellTomorrow(self, data):
        """
        Function that calls the sell function and updates the investment values
        :param data: Decision data based on the type of indicator
        """
        self.perToSell = self.sellPredictionATR(data.atr)

    def buyPredictionATR(self, atr):
        """
        Function that is used to predict next day buying behavior
        :param adx: Dict with the values of the adx
        """
        params = self.atrParams
        # Unpackage macdDict
        average_true_range = atr["average_true_range"]

        return 0

    def sellPredictionATR(self, atr):
        """
        Function that is used to predict next day selling behavior
        :param adx: Dict with the values of the adx
        """
        params = self.atrParams
        # Unpackage macdDict
        average_true_range = atr["average_true_range"]

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
            title="Evolution of Porfolio using ATR " + self.macdParams.type + " (" + self.record.index[0].strftime(
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
        fig.add_trace(go.Scatter(name="ATR " + self.macdParams.type, x=self.record.index,
                                 y=indicatorData["average_true_range"][-len(self.record.index):]), row=1, col=1,
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
            title="Decision making under ATR " + self.macdParams.type + " (" + self.record.index[0].strftime("%d/%m/%Y") + "-" +
                  self.record.index[-1].strftime("%d/%m/%Y") + ")", hovermode='x unified')
        fig.show()


def averageTrueRange(high, low, close, params: ATRInvestorParams):
    """
    Function that returns the values for ATR indicator
    :param high: Market high value
    :param low: Market low value
    :param close: Market close value
    :param params: Parameters to be used for the indicator calculation (window)
    :return: dict with the following keys ["average_true_range"]
    """
    obv = AverageTrueRange(high, low, close, params.window, True)
    return {"average_true_range" : obv.average_true_range()}




