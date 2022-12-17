import math
from ta.volatility import BollingerBands
from classes.investorParamsClass import BBInvestorParams, NNInvestorParams
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from classes.investorClass import Investor
from classes.dataClass import DataManager
import pandas as pd
from DecisionFunction.decisionFunctionNN import NNDecisionFunction


class InvestorBB(Investor):
    def __init__(self, initialInvestment=10000, bbParams=None):
        super().__init__(initialInvestment)
        self.bbParams = bbParams

    def returnBrokerUpdate(self, moneyInvestedToday, moneySoldToday, data):
        return pd.DataFrame(
            {'bb': [data.bb], 'moneyToInvestBB': [moneyInvestedToday], 'moneyToSellBB': [moneySoldToday],
             'investedMoneyBB': [self.investedMoney], 'nonInvestedMoneyBB': [self.nonInvestedMoney]})

    def possiblyInvestTomorrow(self, data: DataManager):
        """
        Function that calls the buy function and updates the investment values
        :param data: Decision data based on the type of indicator
        """
        self.perToInvest = self.buyPredictionBB(data.bb[-1])

    def possiblySellTomorrow(self, data: DataManager):
        """
        Function that calls the sell function and updates the investment values
        :param data: Decision data based on the type of indicator
        """
        self.perToSell = self.sellPredictionBB(data.bb[-1])

    def buyPredictionBB(self, bb):
        """
        Function that returns the money to be invested
        :param bb: bollinger_pband() value
        :return:
        """
        params = self.bbParams

        if bb < params.lowerBound:
            return math.tanh(params.a * (params.lowerBound - bb) ** params.b)
        else:
            return 0

    def sellPredictionBB(self, bb):
        """
        Function that returns the money to be sold
        :param bb: bollinger_pband() value
        :return:
        """
        params = self.bbParams
        if bb > params.upperBound:
            return math.tanh(params.a * (bb - params.upperBound) ** params.b)
        else:
            return 0

    def plotDecisionRules(self):
        """
        Function that plots both how bollinger_pband() works and how the decisions are made
        :param params: BB investor parameters
        """
        testBB = np.arange(-2, 3, 0.01)
        buyPoints = []
        sellPoints = []
        for point in testBB:
            buyPoints = np.append(buyPoints, self.buyPredictionBB(point))
            sellPoints = np.append(sellPoints, self.sellPredictionBB(point))

        fig = go.Figure()
        fig.add_trace(go.Scatter(name="BuyPoints", x=testBB, y=buyPoints, fill='tozeroy'))
        fig.add_trace(go.Scatter(name="SellPoints", x=testBB, y=-sellPoints, fill='tozeroy'))
        fig.update_layout(title="Decision Rules for BB indicator", xaxis={"title": "BB Value"},
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
            title="Evolution of Porfolio using BB (" + self.record.index[0].strftime(
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
            title="Decision making under BB (" + self.record.index[0].strftime("%d/%m/%Y") + "-" +
                  self.record.index[-1].strftime("%d/%m/%Y") + ")", hovermode='x unified')
        fig.show()


class InvestorBBNN(Investor):
    def __init__(self, initialInvestment=10000, nnParams: NNInvestorParams = None):
        super().__init__(initialInvestment)
        self.nnParams = nnParams
        self.model = NNDecisionFunction()
        self.model.load(nnParams.file)

    def returnBrokerUpdate(self, moneyInvestedToday, moneySoldToday, data):
        return pd.DataFrame(
            {'bbNN': [data.bb], 'moneyToInvestBBNN': [moneyInvestedToday], 'moneyToSellBBNN': [moneySoldToday],
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
        y = self.model.predict(inputs.transpose())
        print(y)
        if y > 0:
            return y
        return 0

    def sellPredictionBB(self, bb):
        """
        Function that returns the money to be sold
        :param bb: bollinger_pband() value
        :return:
        """
        inputs = [bb[-1], bb[-2]]
        inputs = np.asarray(inputs)
        y = self.model.predict(inputs.transpose())
        print(y)
        if y < 0:
            return y
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
            title="Evolution of Porfolio using BB (" + self.record.index[0].strftime(
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
            title="Decision making under BB (" + self.record.index[0].strftime("%d/%m/%Y") + "-" +
                  self.record.index[-1].strftime("%d/%m/%Y") + ")", hovermode='x unified')
        fig.show()


def bollingerBands(values, params: BBInvestorParams):
    """
    Function that calcualtes the bollinger bands
    :param values: Open or Close value from the stock market series
    :param params: BB investor parameters
    :return: Dictionary with all the relevant features of the BB
    """
    bb = BollingerBands(values, params.window, params.stdDev, fillna=True)
    return {"pband": bb.bollinger_pband(), "mavg": bb.bollinger_mavg(), "hband": bb.bollinger_hband(), "lband": bb.bollinger_lband()}

