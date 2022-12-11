import pandas as pd
from dataClass import DataManager
from abc import ABC, abstractmethod


class Investor(ABC):
    """
    Investor is an abstract class that is inherited and expanded by all the different strategies' classes (benchmarks and
    our own strategies)
    """
    def __init__(self, initialInvestment=10000):
        """
        Initialization function for the Investor class
        :param initialInvestment: Initial money to be invested
        :param date: Day when the investment begins
        """
        self.initialInvestment = initialInvestment
        self.investedMoney = 0  # Value of money actually invested
        self.nonInvestedMoney = initialInvestment  # Value of money not currently invested
        self.record = pd.DataFrame()
        self.moneyToInvest = 0
        self.moneyToSell = 0

    """
    CONCRETE METHODS
    """

    def broker(self, data: DataManager):
        """
        Function that takes decisions on buy/sell/hold based on today's value and predicted value for tomorrow
        :param data: Decision data based on the type of indicator
        """
        # Update investedMoney value
        self.investedMoney *= data.actualStockValue / data.pastStockValue
        # Broker operation for today
        self.__investAndSellToday()
        out1, out2 = self.moneyToInvest, self.moneyToSell

        # Update porfolio record
        aux = pd.DataFrame({"moneyInvested": self.investedMoney, "moneyNotInvested": self.nonInvestedMoney,
                            "moneyInvestedToday": self.moneyToInvest, "moneySoldToday": self.moneyToSell,
                            "totalValue": (self.investedMoney + self.nonInvestedMoney), "openValue": data.actualStockValue}, index=[data.date])
        self.record = pd.concat([self.record, aux])

        # print(f'Date: {data.date}, moneyInvested {self.investedMoney}, moneyNonInvested {self.nonInvestedMoney}, actualInvestmentValue {self.record["totalValue"].iloc[-1]}')

        # Broker operations for next day
        self.possiblySellTomorrow(data)
        self.possiblyInvestTomorrow(data)

        # print(f'\nToday-> invest {out1}, sell {out2}')
        # print(f'Tomorrow-> invest {self.moneyToInvest}, sell {self.moneyToSell}')

        return self.returnBrokerUpdate(out1, out2, data)

    def __investAndSellToday(self):
        """
        This function performs the operation given by signals established the day before
        """
        # In one day, we should only be able to sell or buy (not both at the same time)
        if self.moneyToInvest > self.moneyToSell:
            self.moneyToInvest -= self.moneyToSell
            self.moneyToSell = 0
        elif self.moneyToInvest < self.moneyToSell:
            self.moneyToSell -= self.moneyToInvest
            self.moneyToInvest = 0

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

    """
    ABSTRACT METHODS
    """

    @abstractmethod
    def returnBrokerUpdate(self, moneyInvestedToday, moneySoldToday, data):
        """
        Function prototype that should return a df
        :param moneyInvestedToday:
        :param moneySoldToday:
        :param data:
        """
        pass

    @abstractmethod
    def possiblyInvestTomorrow(self, data: DataManager):
        """
        Function prototype that calls the buy function and updates the investment values
        :param data: Decision data based on the type of indicator
        """
        pass

    @abstractmethod
    def possiblySellTomorrow(self, data: DataManager):
        """
        Function prototype that calls the sell function and updates the investment values
        :param data: Decision data based on the type of indicator
        """
        pass

    @abstractmethod
    def plotEvolution(self, indicatorData, stockMarketData, recordPredictedValue=None):
        """
        Function prototype that plots the actual status of the investor investment as well as the decisions that
        have been made
        :param indicatorData: Data belonging to the indicator used to take decisions
        :param stockMarketData: df with the stock market data
        :param recordPredictedValue: Predicted data dataframe
        """
        pass
