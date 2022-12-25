import math
from ta.trend import MACD
from classes.investorParamsClass import MACDInvestorParams
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from classes.investorClass import Investor
from classes.dataClass import DataManager
import pandas as pd


class InvestorMACD(Investor):
    def __init__(self, initialInvestment=10000, macdParams=None):
        super().__init__(initialInvestment)
        self.macdParams = macdParams

    def returnBrokerUpdate(self, moneyInvestedToday, moneySoldToday, data):
        return pd.DataFrame(
            {'macd' + self.macdParams.type: [data.macd["macd"][-1]],
             'moneyToInvestMACD' + self.macdParams.type: [moneyInvestedToday],
             'moneyToSellMACD' + self.macdParams.type: [moneySoldToday],
             'investedMoneyMACD' + self.macdParams.type: [self.investedMoney],
             'nonInvestedMoneyMACD' + self.macdParams.type: [self.nonInvestedMoney]})

    def possiblyInvestTomorrow(self, data: DataManager):
        """
        Function that calls the buy function and updates the investment values
        :param data: Decision data based on the type of indicator
        """
        firstGradient, secondGradient, self.perToInvest = self.buyPredictionMACD(data.macd)

    def possiblySellTomorrow(self, data: DataManager):
        """
        Function that calls the sell function and updates the investment values
        :param data: Decision data based on the type of indicator
        """
        self.perToSell = self.sellPredictionMACD(data.macd)

    def buyPredictionMACD(self, macdDict):
        """
        Function that is used to predict next day buying behavior
        :param macdDict: Dict with the values of the MACD (signal and macd)
        :param params: MACD params
        """
        params = self.macdParams
        # Unpackage macdDict
        macd = macdDict["macd"]
        signal = macdDict["signal"]
        type = params.type

        # Calculate gradients of the macd value
        firstGradient = np.gradient(macd.values)
        secondGradient = np.gradient(firstGradient)

        # Calculate gradient of the signal
        firstGradientSignal = np.gradient(signal.values)

        # Depending on the strategy, act accordingly
        if type == "grad":
            if params.buyGradients.lowerBoundGradient < firstGradient[
                -1] < params.buyGradients.upperBoundGradient and params.buyGradients.lowBoundSquareGradient < \
                    secondGradient[-1]:
                return firstGradient, secondGradient, math.tanh(
                    params.a * (secondGradient[-1] - params.buyGradients.lowBoundSquareGradient) ** params.b)
            return firstGradient, secondGradient, 0
        elif type == "grad_crossZero":
            if macd.values[-2] < 0 < macd.values[-1]:
                return firstGradient, secondGradient, math.tanh(params.a * firstGradient[-1] ** params.b)
            return firstGradient, secondGradient, 0
        elif type == "grad_crossSignal":
            if (signal.values[-2] - macd.values[-2]) > 0 > (signal.values[-1] - macd.values[-1]):
                return firstGradient, secondGradient, math.tanh(params.a * (firstGradient[-1] - firstGradientSignal[-1]) ** params.b)
            return firstGradient, secondGradient, 0

    def sellPredictionMACD(self, macdDict):
        """
        Function that is used to predict next day selling behavior
        :param macdDict: Dict with the values of the MACD (signal and macd)
        :param params: MACD params
        """
        params = self.macdParams
        # Unpackage macdDict
        macd = macdDict["macd"]
        signal = macdDict["signal"]
        type = params.type

        # Calculate gradients of the macd value
        firstGradient = np.gradient(macd.values)
        secondGradient = np.gradient(firstGradient)

        # Calculate gradient of the signal
        firstGradientSignal = np.gradient(signal.values)

        # Depending on the strategy, act accordingly
        if type == "grad":
            if params.sellGradients.lowerBoundGradient < firstGradient[
                -1] < params.sellGradients.upperBoundGradient and params.sellGradients.lowBoundSquareGradient > \
                    secondGradient[-1]:
                return math.tanh(
                    params.a * (params.sellGradients.lowBoundSquareGradient - secondGradient[-1]) ** params.b)
            else:
                return 0
        elif type == "grad_crossZero":
            if macd.values[-2] > 0 > macd.values[-1]:
                return math.tanh(params.a * (-firstGradient[-1]) ** params.b)
            return 0
        elif type == "grad_crossSignal":
            if (signal.values[-2] - macd.values[-2]) < 0 < (signal.values[-1] - macd.values[-1]):
                return math.tanh(params.a * (firstGradientSignal[-1] - firstGradient[-1]) ** params.b)
            return 0

    def plotDecisionRules(self):
        """
        Function that plots the decision rule used
        :param params: MACD params
        """
        params = self.macdParams
        testMACDdata = pd.Series(np.random.normal(0, 1, 30))
        buyPoints = []
        sellPoints = []
        for i in range(len(testMACDdata) - params.fastWindow):
            testMACD = movingAverageConvergenceDivergence(testMACDdata[0:i + 2], params)
            firstGradient, secondGradient, buyPoint = self.buyPredictionMACD(testMACD)
            buyPoints = np.append(buyPoints, buyPoint)
            sellPoints = np.append(sellPoints, self.sellPredictionMACD(testMACD))

        x = np.arange(len(testMACDdata))

        fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": True}], [{"secondary_y": True}]])
        fig.add_trace(go.Scatter(name="Stock data", x=x, y=testMACDdata.values), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(name="MACDValues", x=x[3:], y=testMACD["macd"].values), row=1, col=1,
                      secondary_y=False)
        fig.add_trace(go.Scatter(name="BuyPoints", x=x[4:], y=buyPoints, fill='tozeroy'), row=1, col=1,
                      secondary_y=True)
        fig.add_trace(go.Scatter(name="SellPoints", x=x[4:], y=-sellPoints, fill='tozeroy'), row=1, col=1,
                      secondary_y=True)
        fig.add_trace(go.Scatter(name="FirstGradient", x=x[3:], y=firstGradient), row=2, col=1, secondary_y=False)
        if params.type == "grad":
            fig.add_trace(go.Scatter(name="SecondGradient", x=x[3:], y=secondGradient), row=2, col=1, secondary_y=True)
            fig.update_layout(title="Decision Rules for MACD indicator (Grad)", xaxis={"title": "x"},
                              yaxis={"title": "Sell/Buy/Hold [$]"}, hovermode='x unified')
        elif params.type == "grad_crossZero":
            fig.update_layout(title="Decision Rules for MACD indicator (Grad+CrossZero)", xaxis={"title": "x"},
                              yaxis={"title": "Sell/Buy/Hold [$]"}, hovermode='x unified')
        elif params.type == "grad_crossSignal":
            fig.add_trace(go.Scatter(name="SignalValues", x=x[3:], y=testMACD["signal"].values), row=1, col=1,
                          secondary_y=False)
            fig.update_layout(title="Decision Rules for MACD indicator (Grad+CrossSignal)", xaxis={"title": "x"},
                              yaxis={"title": "Sell/Buy/Hold [$]"}, hovermode='x unified')
        fig.show()

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
            title="Evolution of Porfolio using MACD " + self.macdParams.type + " (" + self.record.index[0].strftime(
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
        fig.add_trace(go.Scatter(name="MACD " + self.macdParams.type, x=self.record.index,
                                 y=indicatorData["macd"][-len(self.record.index):]), row=1, col=1,
                      secondary_y=True)
        fig.add_trace(go.Scatter(name="MACD " + self.macdParams.type + " Signal", x=self.record.index,
                                 y=indicatorData["signal"][-len(self.record.index):]), row=1, col=1,
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
            title="Decision making under MACD " + self.macdParams.type + " (" + self.record.index[0].strftime("%d/%m/%Y") + "-" +
                  self.record.index[-1].strftime("%d/%m/%Y") + ")", hovermode='x unified')
        fig.show()


def movingAverageConvergenceDivergence(close, params: MACDInvestorParams):
    """
    Function that calculates the different returns of the MACD indicator
    :param close: Close market values
    :param params: Parameters to be used for the indicator calculation (fastWindow, slowWindow, signal)
    :return: dict with the following keys ["macd", "signal"]
    """
    macd = MACD(close, params.fastWindow, params.slowWindow, params.signal, True)
    return {"macd" : macd.macd(), "signal" : macd.macd_signal()}




