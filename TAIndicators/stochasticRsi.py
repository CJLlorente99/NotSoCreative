import ta
from classes.investorParamsClass import StochasticRSIInvestorParams
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
from classes.investorClass import Investor
import pandas as pd


class InvestorRSI(Investor):
    def __init__(self, initialInvestment=10000, stochasticRSIparams=None):
        super().__init__(initialInvestment)
        self.stochasticRSIparams = stochasticRSIparams

    def returnBrokerUpdate(self, moneyInvestedToday, moneySoldToday, data):
        return pd.DataFrame(
                {'stochrsi': [data["stochrsistochrsi"]], 'moneyToInvestStochRSI': [moneyInvestedToday], 'moneyToSellStochRSI': [moneySoldToday],
                 'investedMoneyStochRSI': [self.investedMoney], 'nonInvestedMoneyStochRSI': [self.nonInvestedMoney]})

    def possiblyInvestTomorrow(self, data):
        """
        Function that calls the buy function and updates the investment values
        :param data: Decision data based on the type of indicator
        """
        self.perToInvest = self.buyPredictionStochRSI(data["stochrsistochrsi"])

    def possiblySellTomorrow(self, data):
        """
        Function that calls the sell function and updates the investment values
        :param data: Decision data based on the type of indicator
        """
        self.perToSell = self.sellPredictionStochRSI(data["stochrsistochrsi"])

    def buyPredictionStochRSI(self, stochrsi):
        """
        Function that represents the buying behavior
        :param stochrsi: stochrsi value for today
        """
        params = self.stochasticRSIparams

        if stochrsi < params.lowerBound:  # Buy linearly then with factor f
            return math.tanh(params.a * (params.lowerBound - stochrsi) ** params.b)
        else:
            return 0

    def sellPredictionStochRSI(self, stochrsi):
        """
        Function that represents the selling behavior
        :param stochrsi: stochrsi value for today
        """
        params = self.stochasticRSIparams
        if stochrsi > params.upperBound:  # Buy linearly then with factor f
            return math.tanh(params.a * (stochrsi - params.upperBound) ** params.b)
        else:
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
            title="Evolution of Porfolio using StochRSI (" + self.record.index[0].strftime(
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
        fig.add_trace(
            go.Scatter(name="StochRSI", x=self.record.index, y=indicatorData["stochrsi"][-len(self.record.index):]), row=1,
            col=1, secondary_y=True)
        fig.add_trace(
            go.Scatter(name="KStochRSI", x=self.record.index, y=indicatorData["k"][-len(self.record.index):]),
            row=1,
            col=1, secondary_y=True)
        fig.add_trace(
            go.Scatter(name="DStochRSI", x=self.record.index, y=indicatorData["d"][-len(self.record.index):]),
            row=1,
            col=1, secondary_y=True)
        fig.add_trace(go.Scatter(name="Stock Market Value Open", x=self.record.index,
                                 y=stockMarketData.Open[-len(self.record.index):]), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(name="Stock Market Value Close", x=self.record.index,
                                 y=stockMarketData.Close[-len(self.record.index):]), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Bar(name="Money Invested Today", x=self.record.index, y=self.record["moneyInvestedToday"], marker_color="green"), row=2, col=1)
        fig.add_trace(go.Bar(name="Money Sold Today", x=self.record.index, y=-self.record["moneySoldToday"], marker_color="red"), row=2, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_layout(
            title="Decision making under StochRSI (" + self.record.index[0].strftime("%d/%m/%Y") + "-" +
                  self.record.index[-1].strftime("%d/%m/%Y") + ")", hovermode='x unified')
        fig.show()


def stochasticRSI(close, params: StochasticRSIInvestorParams):
    """
    Function that calculates the stochastic RSI values
    :param close: Market close value
    :param params: StochRSI parameters
    :return dict with the following keys ["stochrsi", "k", "d"]
    """
    stochRsi = ta.momentum.StochRSIIndicator(close, params.window, params.smooth1, params.smooth2, True)
    return {"stochrsi": stochRsi.stochrsi(), "k": stochRsi.stochrsi_k(), "d": stochRsi.stochrsi_d()}
