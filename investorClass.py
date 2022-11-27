import pandas as pd

import ma
from dataClass import DataManager
import rsi


class Investor:
    def __init__(self, initialInvestment=10000, date=None, maxBuy=10000, maxSell=2500, rsiParams=None\
                 , smaParams=None, emaParams=None, macdParams=None):
        """

        :param initialInvestment: Initial money to be invested
        :param date: Day when the investment begins
        :param maxBuy: Maximum quantity buy per day
        :param maxSell: Maximum quantity sell per day
        :param RSIupperBound: RSI investor function parameters (RSI value from where sell signal is given)
        :param RSIlowerBound: RSI investor function parameters (RSI value up to where buy signal is given)
        """
        self.initialInvestment = initialInvestment
        self.investedMoney = 0  # Value of money actually invested
        self.nonInvestedMoney = initialInvestment  # Value of money not currently invested
        self.investmentArray = pd.DataFrame([initialInvestment], columns=['investmentArray'], index=pd.DatetimeIndex([date]))
        self.maxBuy = maxBuy
        self.maxSell = maxSell
        self.moneyToInvest = 0
        self.moneyToSell = 0
        if rsiParams:
            self.RSIupperBound = rsiParams.RSIupperBound
            self.RSIlowerBound = rsiParams.RSIlowerBound
        if smaParams:
            self.smaSellParams = smaParams.sellGradients
            self.smaBuyParams = smaParams.buyGradients
        if emaParams:
            self.emaSellParams = emaParams.sellGradients
            self.emaBuyParams = emaParams.buyGradients
        if macdParams:
            self.macdParams = macdParams

    def broker(self, data: DataManager, typeIndicator):
        """
        Function that takes decisions on buy/sell/hold based on today's value and predicted value for tomorrow
        :param data: Decision data based on the type of indicator
        :param typeIndicator: Type of indicator ('macd', 'sma', 'prediction'...)
        IN FURTHER IMPLEMENTATION IT CAN BE AN ARRAY, SO DECISION IS TAKEN NOT ONLY BASED ON ONE DAY
        """
        # Update investedMoney value
        self.investedMoney *= data.actualStockValue / data.pastStockValue
        # Broker operation for today
        self.__investAndSellToday()
        out1, out2 = self.moneyToInvest, self.moneyToSell
        # print(f'Today-> invest {self.moneyToInvest}, sell {self.moneyToSell}')
        aux = pd.DataFrame([self.investedMoney+self.nonInvestedMoney], index=[data.date], columns=['investmentArray'])
        self.investmentArray = pd.concat([self.investmentArray, aux])
        # print(f'Date: {data.date}, moneyInvested {self.investedMoney}, moneyNonInvested {self.nonInvestedMoney}, actualInvestmentValue {self.investmentArray["investmentArray"].iloc[-1]}')
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
        if typeIndicator == 'prediction':
            moneyToInvest = self.__buyFunctionPrediction(data.actualStockValue, data.nextStockValue) * data.actualStockValue
        elif typeIndicator == 'rsi':
            moneyToInvest = rsi.buyFunctionPredictionRSI(data.rsi, lowerBound=self.RSIlowerBound, maxBuy=self.maxBuy)
        elif typeIndicator == 'sma':
            moneyToInvest = ma.buyPredictionSMA(data.sma, self.smaBuyParams, self.maxBuy)
        elif typeIndicator == 'ema':
            moneyToInvest = ma.buyPredictionEMA(data.ema, self.emaBuyParams, self.maxBuy)
        elif typeIndicator == 'macd':
            moneyToInvest = ma.buyPredictionMACD(data.macd, self.macdParams, self.maxBuy)

        self.moneyToInvest = moneyToInvest

    def __possiblySellTomorrow(self, data: DataManager, typeIndicator):
        """
        Function that calls the sell function and updates the investment values
        :param data: Decision data based on the type of indicator
        :param typeIndicator: Type of indicator ('macd', 'sma', 'prediction'...)
        """
        moneyToSell = 0
        if typeIndicator == 'prediction':
            moneyToSell = self.__sellFunctionPrediction(data.actualStockValue, data.nextStockValue) * data.actualStockValue
        elif typeIndicator == 'rsi':
            moneyToSell = rsi.sellFunctionPredictionRSI(data.rsi, upperBound=self.RSIupperBound, maxSell=self.maxSell)
        elif typeIndicator == 'sma':
            moneyToSell = ma.sellPredictionSMA(data.sma, self.smaSellParams, self.maxSell)
        elif typeIndicator == 'ema':
            moneyToSell = ma.sellPredictionEMA(data.ema, self.emaSellParams, self.maxSell)
        elif typeIndicator == 'macd':
            moneyToSell = ma.sellPredictionMACD(data.macd, self.macdParams, self.maxSell)

        self.moneyToSell = moneyToSell

    def __buyFunctionPrediction(self, actualStockValue, nextStockValue):
        """
        Function that represents the buying behavior
        :param actualStockValue: Opening value of today's S&P 500
        :param nextStockValue: Predicted value for tomorrow's S&P 500
        """
        # Dummy implementation that just holds
        return 0

    def __sellFunctionPrediction(self, actualStockValue, nextStockValue):
        """
        Function that represents the selling behavior
        :param actualStockValue: Opening value of today's S&P 500
        :param nextStockValue: Predicted value for tomorrow's S&P 500
        """
        # Dummy implementation that just holds
        return 0

    def calculateMetrics(self):
        """
        Function that calculates the metric for a given investor
        :return: metrics calculated
        """
        porcentualGain = self.investmentArray["investmentArray"].iloc[-1]/self.initialInvestment * 100 - 100
        meanPortfolioValue = self.investmentArray["investmentArray"].mean()
        return porcentualGain, meanPortfolioValue
