import math
from ta.trend import MACD, EMAIndicator
from classes.investorParamsClass import MACDInvestorParams, MAInvestorParams
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from classes.investorClass import Investor
import pandas as pd


class InvestorMACD(Investor):
    def __init__(self, initialInvestment=10000, macdParams=None):
        super().__init__(initialInvestment)
        self.macdParams = macdParams

    def returnBrokerUpdate(self, moneyInvestedToday, data):
        if self.macdParams.type == "grad":
            return pd.DataFrame(
                {'macd' + self.macdParams.type + "macd": data["macdmacd"][-1],
                 'moneyToInvestMACD' + self.macdParams.type: moneyInvestedToday,
                 'investedMoneyMACD' + self.macdParams.type: self.investedMoney,
                 'nonInvestedMoneyMACD' + self.macdParams.type: self.nonInvestedMoney}, index=[0])
        else:
            return pd.DataFrame(
                {'macd' + self.macdParams.type + "macd": data["macdmacd"][-1],
                 'macd' + self.macdParams.type + "signal": data["macdsignal"][-1],
                 'moneyToInvestMACD' + self.macdParams.type: moneyInvestedToday,
                 'investedMoneyMACD' + self.macdParams.type: self.investedMoney,
                 'nonInvestedMoneyMACD' + self.macdParams.type: self.nonInvestedMoney}, index=[0])

    def possiblyInvestMorning(self, data):
        """
        Function that calls the buy function and updates the investment values
        :param data: Decision data based on the type of indicator
        """
        macd = data["macdmacd"]
        signal = data["macdsignal"]

        params = self.macdParams
        type = params.type

        # Calculate gradients of the macd value
        firstGradient = np.gradient(macd.values)
        secondGradient = np.gradient(firstGradient)

        # Calculate gradient of the signal
        firstGradientSignal = np.gradient(signal.values)

        # Depending on the strategy, act accordingly
        self.perToInvest = 0
        if type == "grad":
            if params.buyGradients.lowerBoundGradient < firstGradient[
                -1] < params.buyGradients.upperBoundGradient and params.buyGradients.lowBoundSquareGradient < \
                    secondGradient[-1]:
                self.perToInvest = math.tanh(
                    params.a * (secondGradient[-1] - params.buyGradients.lowBoundSquareGradient) ** params.b)
            elif params.sellGradients.lowerBoundGradient < firstGradient[
                -1] < params.sellGradients.upperBoundGradient and params.sellGradients.lowBoundSquareGradient > \
                    secondGradient[-1]:
                self.perToInvest = -math.tanh(
                    params.a * (params.sellGradients.lowBoundSquareGradient - secondGradient[-1]) ** params.b)
        elif type == "grad_crossZero":
            if macd.values[-2] < 0 < macd.values[-1]:
                self.perToInvest = math.tanh(params.a * firstGradient[-1] ** params.b)
            elif macd.values[-2] > 0 > macd.values[-1]:
                self.perToInvest = -math.tanh(params.a * (-firstGradient[-1]) ** params.b)
        elif type == "grad_crossSignal":
            if (signal.values[-2] - macd.values[-2]) > 0 > (signal.values[-1] - macd.values[-1]):
                self.perToInvest = math.tanh(
                    params.a * (firstGradient[-1] - firstGradientSignal[-1]) ** params.b)
            elif (signal.values[-2] - macd.values[-2]) < 0 < (signal.values[-1] - macd.values[-1]):
                self.perToInvest = -math.tanh(params.a * (firstGradientSignal[-1] - firstGradient[-1]) ** params.b)

    def possiblyInvestAfternoon(self, data):
        """
        Function that calls the sell function and updates the investment values
        :param data: Decision data based on the type of indicator
        """
        macd = data["macdmacd"]
        signal = data["macdsignal"]

        params = self.macdParams
        type = params.type

        # Calculate gradients of the macd value
        firstGradient = np.gradient(macd.values)
        secondGradient = np.gradient(firstGradient)

        # Calculate gradient of the signal
        firstGradientSignal = np.gradient(signal.values)

        # Depending on the strategy, act accordingly
        self.perToInvest = 0
        # if type == "grad":
        #     if params.buyGradients.lowerBoundGradient < firstGradient[
        #         -1] < params.buyGradients.upperBoundGradient and params.buyGradients.lowBoundSquareGradient < \
        #             secondGradient[-1]:
        #         self.perToInvest = math.tanh(
        #             params.a * (secondGradient[-1] - params.buyGradients.lowBoundSquareGradient) ** params.b)
        #     elif params.sellGradients.lowerBoundGradient < firstGradient[
        #         -1] < params.sellGradients.upperBoundGradient and params.sellGradients.lowBoundSquareGradient > \
        #             secondGradient[-1]:
        #         self.perToInvest = -math.tanh(
        #             params.a * (params.sellGradients.lowBoundSquareGradient - secondGradient[-1]) ** params.b)
        # elif type == "grad_crossZero":
        #     if macd.values[-2] < 0 < macd.values[-1]:
        #         self.perToInvest = math.tanh(params.a * firstGradient[-1] ** params.b)
        #     elif macd.values[-2] > 0 > macd.values[-1]:
        #         self.perToInvest = -math.tanh(params.a * (-firstGradient[-1]) ** params.b)
        # elif type == "grad_crossSignal":
        #     if (signal.values[-2] - macd.values[-2]) > 0 > (signal.values[-1] - macd.values[-1]):
        #         self.perToInvest = math.tanh(
        #             params.a * (firstGradient[-1] - firstGradientSignal[-1]) ** params.b)
        #     elif (signal.values[-2] - macd.values[-2]) < 0 < (signal.values[-1] - macd.values[-1]):
        #         self.perToInvest = -math.tanh(params.a * (firstGradientSignal[-1] - firstGradient[-1]) ** params.b)

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
            title="Evolution of Porfolio using MACD " + self.macdParams.type + " (" + self.record.index[0].strftime(
                "%d/%m/%Y") + "-" +
                  self.record.index[-1].strftime("%d/%m/%Y") + ")", xaxis_title="Date",
            yaxis_title="Value [$]", hovermode='x unified')
        fig.write_image("images/EvolutionPorfolioMACD" + self.macdParams.type + "(" + self.record.index[0].strftime(
                "%d_%m_%Y") + "-" +
                  self.record.index[-1].strftime("%d_%m_%Y") + ").png",scale=6, width=1080, height=1080)
        # fig.show()

        # Plot indicating the value of the indicator, the value of the stock market and the decisions made
        fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": True}], [{"secondary_y": False}]])
        if recordPredictedValue is not None:
            fig.add_trace(go.Scatter(name="Predicted Stock Market Value Close", x=recordPredictedValue.index,
                                     y=recordPredictedValue[0]), row=1, col=1,
                          secondary_y=False)
        fig.add_trace(go.Scatter(name="MACD " + self.macdParams.type, x=self.record.index,
                                 y=expData['macd' + self.macdParams.type + "macd"][-len(self.record.index):]), row=1, col=1,
                      secondary_y=True)
        if self.macdParams.type != "grad":
            fig.add_trace(go.Scatter(name="MACD " + self.macdParams.type + " Signal", x=self.record.index,
                                     y=expData['macd' + self.macdParams.type + "signal"][-len(self.record.index):]), row=1, col=1,
                              secondary_y=True)
        fig.add_trace(go.Scatter(name="Stock Market Value Open", x=self.record.index,
                                 y=stockMarketData.Open[-len(self.record.index):]), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(name="Stock Market Value Close", x=self.record.index,
                                 y=stockMarketData.Close[-len(self.record.index):]), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Bar(name="Money Invested Today", x=self.record.index, y=self.record["moneyInvestedToday"]), row=2, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_layout(
            title="Decision making under MACD " + self.macdParams.type + " (" + self.record.index[0].strftime("%d/%m/%Y") + "-" +
                  self.record.index[-1].strftime("%d/%m/%Y") + ")", hovermode='x unified')
        fig.write_image("images/DecisionmakingMACD" + self.macdParams.type + "(" + self.record.index[0].strftime("%d_%m_%Y") + "-" +
                  self.record.index[-1].strftime("%d_%m_%Y") + ").png",scale=6, width=1080, height=1080)
        # fig.show()


def movingAverageConvergenceDivergence(close, params: MACDInvestorParams):
    """
    Function that calculates the different returns of the MACD indicator
    :param close: Close market values
    :param params: Parameters to be used for the indicator calculation (fastWindow, slowWindow, signal)
    :return: dict with the following keys ["macd", "signal"]
    """
    macd = MACD(close, params.fastWindow, params.slowWindow, params.signal, True)
    diff = macd.macd() - macd.macd_signal()
    return {"macd" : macd.macd(), "signal" : macd.macd_signal(), 'diff': diff}

def exponentialMovingAverage(close, params: MAInvestorParams):
    ema = EMAIndicator(close, params.window, True)
    return {"ema" : ema.ema_indicator()}




