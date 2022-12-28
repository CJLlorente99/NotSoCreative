import pandas as pd
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
        """
        self.initialInvestment = initialInvestment
        self.investedMoney = 0  # Value of money actually invested
        self.nonInvestedMoney = initialInvestment  # Value of money not currently invested
        self.record = pd.DataFrame() # Record that is further use for the calculation of testCriteria (contains basic metrics)
        self.perToInvest = 0  # [0,1], being 1 = 100% of the non invested money
        self.perToSell = 0  # [0,1], being 1 = 100% of the invested money

    """
    CONCRETE METHODS
    """

    def broker(self, data) -> pd.DataFrame:
        """
        Function that takes decisions on buy/sell/hold based on today's value and predicted value for tomorrow
        :param data: Decision data based on the type of indicator
        :return dataFrame with the data relevant to the actual strategy used and actions taken out that day
        """
        # Update investedMoney value
        self.investedMoney *= data["actualStockValue"] / data["pastStockValue"]

        # Broker operation for today
        moneyInvested, moneySold = self.__investAndSellToday()
        todayBuy, todaySell = self.perToInvest, self.perToSell

        # Update porfolio record
        aux = pd.DataFrame({"moneyInvested": self.investedMoney, "moneyNotInvested": self.nonInvestedMoney,
                            "moneyInvestedToday": moneyInvested, "moneySoldToday": moneySold,
                            "totalValue": (self.investedMoney + self.nonInvestedMoney), "openValue": data["actualStockValue"]}, index=[data["date"]])
        self.record = pd.concat([self.record, aux])

        # print(f'Date: {data.date}, moneyInvested {self.investedMoney}, moneyNonInvested {self.nonInvestedMoney}, actualInvestmentValue {self.record["totalValue"].iloc[-1]}')

        # Broker operations for next day
        self.possiblySellTomorrow(data)
        self.possiblyInvestTomorrow(data)

        # print(f'\nToday-> invest {todayBuy}, sell {todaySell}')
        # print(f'Tomorrow-> invest {self.perToInvest}, sell {self.perToSell}')

        return self.returnBrokerUpdate(todayBuy, todaySell, data)

    def __investAndSellToday(self) -> (float, float):
        """
        This function performs the operation given by signals established the day before
        :return tuple of float representing the money that has been finally bought and sold
        """
        # In one day, we should only be able to sell or buy (not both at the same time)
        if self.perToInvest > self.perToSell:
            self.perToInvest -= self.perToSell
            self.perToSell = 0
        elif self.perToInvest < self.perToSell:
            self.perToSell -= self.perToInvest
            self.perToInvest = 0

        # Calculate the money bought and sold depending on the actual nonInvested and Invested.
        # Even though money Invested and money sold are returned, only one of those actually contains data
        moneyInvested = self.perToInvest * self.nonInvestedMoney
        self.investedMoney += self.perToInvest * self.nonInvestedMoney
        self.nonInvestedMoney -= self.perToInvest * self.nonInvestedMoney

        moneySold = self.perToSell * self.investedMoney
        self.nonInvestedMoney += self.perToSell * self.investedMoney
        self.investedMoney -= self.perToSell * self.investedMoney

        return moneyInvested, moneySold

    """
    ABSTRACT METHODS
    """

    @abstractmethod
    def returnBrokerUpdate(self, moneyInvestedToday, moneySoldToday, data) -> pd.DataFrame:
        """
        Function prototype that should return a df
        :param moneyInvestedToday:
        :param moneySoldToday:
        :param data:
        :return dataFrame with all the data important for proper debugging or further plotting after the operation
        """
        pass

    @abstractmethod
    def possiblyInvestTomorrow(self, data) -> float:
        """
        Function prototype that calls the buy function and updates the investment values
        :param data: Decision data based on the type of indicator
        :return percentage of moneyNotInvested (between 0 and 1) that should be boght tomorrow
        """
        pass

    @abstractmethod
    def possiblySellTomorrow(self, data) -> float:
        """
        Function prototype that calls the sell function and updates the investment values
        :param data: Decision data based on the type of indicator
        :return percentage of moneyInvested (between 0 and 1) that should be sold tomorrow
        """
        pass

    @abstractmethod
    def plotEvolution(self, expData, stockMarketData, recordPredictedValue=None):
        """
        Function prototype that plots the actual status of the investor investment as well as the decisions that
        have been made
        :param indicatorData: Data belonging to the indicator used to take decisions
        :param stockMarketData: df with the stock market data
        :param recordPredictedValue: Predicted data dataframe
        """
        pass