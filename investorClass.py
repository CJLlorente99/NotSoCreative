import pandas as pd
import ma
from dataClass import DataManager
import rsi
import bb
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Investor:
    def __init__(self, initialInvestment=10000, date=None, rsiParams=None, smaParams=None, emaParams=None, macdParams=None, bbParams=None):
        """

        :param initialInvestment: Initial money to be invested
        :param date: Day when the investment begins
        :param rsiParams:
        :param smaParams:
        :param emaParams:
        :param macdParams:
        :param bbParams:
        """
        self.initialInvestment = initialInvestment
        self.investedMoney = 0  # Value of money actually invested
        self.nonInvestedMoney = initialInvestment  # Value of money not currently invested
        self.record = pd.DataFrame()
        self.moneyToInvest = 0
        self.moneyToSell = 0
        if rsiParams:
            self.rsiParams = rsiParams
        if smaParams:
            self.smaParams = smaParams
        if emaParams:
            self.emaParams = emaParams
        if macdParams:
            self.macdParams = macdParams
        if bbParams:
            self.bbParams = bbParams

    def broker(self, data: DataManager, typeIndicator):
        """
        Function that takes decisions on buy/sell/hold based on today's value and predicted value for tomorrow
        :param data: Decision data based on the type of indicator
        :param typeIndicator: Type of indicator ('macd', 'sma', 'prediction'...)
        """
        # Update investedMoney value
        self.investedMoney *= data.actualStockValue / data.pastStockValue
        # Broker operation for today
        self.__investAndSellToday()
        out1, out2 = self.moneyToInvest, self.moneyToSell

        # print(f'Today-> invest {self.moneyToInvest}, sell {self.moneyToSell}')

        # Update porfolio record
        aux = pd.DataFrame({"moneyInvested": self.investedMoney, "moneyNotInvested": self.nonInvestedMoney,
                            "moneyInvestedToday": self.moneyToInvest, "moneySoldToday": self.moneyToSell,
                            "totalValue": (self.investedMoney + self.nonInvestedMoney)}, index=[data.date])
        self.record = pd.concat([self.record, aux])

        print(f'Date: {data.date}, moneyInvested {self.investedMoney}, moneyNonInvested {self.nonInvestedMoney}, actualInvestmentValue {self.record["totalValue"].iloc[-1]}')

        # Broker operations for next day
        self.__possiblySellTomorrow(data, typeIndicator)
        self.__possiblyInvestTomorrow(data, typeIndicator)

        # print(f'Tomorrow-> invest {self.moneyToInvest}, sell {self.moneyToSell}')

        return out1, out2, self.investedMoney, self.nonInvestedMoney

    def __investAndSellToday(self):
        """
        This function performs the operation given by signals established the day before
        """
        if self.moneyToInvest > self.nonInvestedMoney:
            self.investedMoney += self.nonInvestedMoney
            self.nonInvestedMoney = 0
        else:
            self.investedMoney += self.moneyToInvest
            self.nonInvestedMoney -= self.moneyToInvest

        if self.moneyToSell > self.investedMoney:
            self.nonInvestedMoney += self.investedMoney
            self.investedMoney = 0
        else:
            self.investedMoney -= self.moneyToSell
            self.nonInvestedMoney += self.moneyToSell

    def __possiblyInvestTomorrow(self, data: DataManager, typeIndicator):
        """
        Function that calls the buy function and updates the investment values
        :param data: Decision data based on the type of indicator
        :param typeIndicator: Type of indicator ('macd', 'sma', 'prediction'...)
        """
        moneyToInvest = 0
        if typeIndicator == 'rsi':
            moneyToInvest = rsi.buyFunctionPredictionRSI(data.rsi, self.rsiParams)
        elif typeIndicator == 'sma':
            moneyToInvest = ma.buyPredictionSMA(data.sma, self.smaParams)
        elif typeIndicator == 'ema':
            moneyToInvest = ma.buyPredictionEMA(data.ema, self.emaParams)
        elif typeIndicator == 'macd':
            moneyToInvest = ma.buyPredictionMACD(data.macd, self.macdParams)
        elif typeIndicator == 'bb':
            moneyToInvest = bb.buyPredictionBB(data.bb, self.bbParams)

        self.moneyToInvest = moneyToInvest

    def __possiblySellTomorrow(self, data: DataManager, typeIndicator):
        """
        Function that calls the sell function and updates the investment values
        :param data: Decision data based on the type of indicator
        :param typeIndicator: Type of indicator ('macd', 'sma', 'prediction'...)
        """
        moneyToSell = 0
        if typeIndicator == 'rsi':
            moneyToSell = rsi.sellFunctionPredictionRSI(data.rsi, self.rsiParams)
        elif typeIndicator == 'sma':
            moneyToSell = ma.sellPredictionSMA(data.sma, self.smaParams)
        elif typeIndicator == 'ema':
            moneyToSell = ma.sellPredictionEMA(data.ema, self.emaParams)
        elif typeIndicator == 'macd':
            moneyToSell = ma.sellPredictionMACD(data.macd, self.macdParams)
        elif typeIndicator == 'bb':
            moneyToSell = bb.sellPredictionBB(data.bb, self.bbParams)

        self.moneyToSell = moneyToSell

    def calculateMetrics(self):
        """
        Function that calculates the metric for a given investor
        :return: metrics calculated
        """
        porcentualGain = self.record["totalValue"].iloc[-1]/self.initialInvestment * 100 - 100
        meanPortfolioValue = self.record["totalValue"].mean()
        return porcentualGain, meanPortfolioValue

    def plotEvolution(self, indicatorData, stockMarketData, typeIndicator):
        # Plot indicating the evolution of the total value and contain (moneyInvested and moneyNotInvested)
        fig = go.Figure()
        fig.add_trace(go.Scatter(name="Money Invested", x=self.record.index, y=self.record["moneyInvested"], stackgroup="one"))
        fig.add_trace(go.Scatter(name="Money Not Invested", x=self.record.index, y=self.record["moneyNotInvested"], stackgroup="one"))
        fig.add_trace(go.Scatter(name="Total Value", x=self.record.index, y=self.record["totalValue"]))
        fig.update_layout(title="Evolution of Porfolio using " + typeIndicator, xaxis_title="Date",
                          yaxis_title="Value [$]")
        fig.show()

        # Plot indicating the value of the indicator, the value of the stock market and the decisions made
        fig = go.Figure()
        fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": True}], [{"secondary_y": False}]])
        fig.add_trace(go.Scatter(name=typeIndicator, x=self.record.index, y=indicatorData[:-len(self.record.index)+1]), row=1, col=1, secondary_y=True)
        fig.add_trace(go.Scatter(name="Stock Market Value", x=self.record.index, y=stockMarketData[:-len(self.record.index)+1]), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Bar(name="Money Invested", x=self.record.index, y=self.record["moneyInvestedToday"], marker_color="green"), row=2, col=1)
        fig.add_trace(go.Bar(name="Money Sold", x=self.record.index, y=-self.record["moneySoldToday"], marker_color="red"), row=2, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_layout(title="Decision making under " + typeIndicator)
        fig.show()
