import pandas as pd
import ma
from dataClass import DataManager
import rsi
import bb


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
        self.investmentArray = pd.DataFrame([initialInvestment], columns=['investmentArray'], index=pd.DatetimeIndex([date]))
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
        aux = pd.DataFrame([self.investedMoney+self.nonInvestedMoney], index=[data.date], columns=['investmentArray'])
        self.investmentArray = pd.concat([self.investmentArray, aux])
        print(f'Date: {data.date}, moneyInvested {self.investedMoney}, moneyNonInvested {self.nonInvestedMoney}, actualInvestmentValue {self.investmentArray["investmentArray"].iloc[-1]}')
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
        porcentualGain = self.investmentArray["investmentArray"].iloc[-1]/self.initialInvestment * 100 - 100
        meanPortfolioValue = self.investmentArray["investmentArray"].mean()
        return porcentualGain, meanPortfolioValue
