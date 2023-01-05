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
        self.perToInvest = 0  # [-1,1], being 1 = 100% of the non invested money

    """
    CONCRETE METHODS
    """

    def brokerMorning(self, data) -> pd.DataFrame:
        """
        Function that takes decisions on buy/sell/hold based on today's value and predicted value for tomorrow
        :param data: Decision data based on the type of indicator
        :return dataFrame with the data relevant to the actual strategy used and actions taken out that day
        """
        # Update investedMoney value
        self.investedMoney *= data["actualStockValue"] / data["pastStockValue"]

        # Broker operations for next day
        self.possiblyInvestMorning(data)

        # Broker operation for today
        moneyInvested = self.__investAndSellToday()
        todayInvest = self.perToInvest

        # Update porfolio record
        aux = pd.DataFrame({"moneyInvested": self.investedMoney, "moneyNotInvested": self.nonInvestedMoney,
                            "moneyInvestedToday": moneyInvested,
                            "totalValue": (self.investedMoney + self.nonInvestedMoney), "actualStockValue": data["actualStockValue"]}, index=[data["date"]])
        self.record = pd.concat([self.record, aux])

        # print(f'Date: {data.date}, moneyInvested {self.investedMoney}, moneyNonInvested {self.nonInvestedMoney}, actualInvestmentValue {self.record["totalValue"].iloc[-1]}')

        # print(f'\nToday-> invest {todayBuy}, sell {todaySell}')

        return self.returnBrokerUpdate(todayInvest, data)

    def brokerAfternoon(self, data) -> pd.DataFrame:
        """
        Function that takes decisions on buy/sell/hold based on today's value and predicted value for tomorrow
        :param data: Decision data based on the type of indicator
        :return dataFrame with the data relevant to the actual strategy used and actions taken out that day
        """
        # Update investedMoney value
        self.investedMoney *= data["actualStockValue"] / data["pastStockValue"]

        # Broker operations for next day
        self.possiblyInvestAfternoon(data)

        # Broker operation for today
        moneyInvested = self.__investAndSellToday()
        todayInvest = self.perToInvest

        # Update porfolio record
        aux = pd.DataFrame({"moneyInvested": self.investedMoney, "moneyNotInvested": self.nonInvestedMoney,
                            "moneyInvestedToday": moneyInvested,
                            "totalValue": (self.investedMoney + self.nonInvestedMoney), "actualStockValue": data["actualStockValue"]}, index=[data["date"]])
        self.record = pd.concat([self.record, aux])

        # print(f'Date: {data.date}, moneyInvested {self.investedMoney}, moneyNonInvested {self.nonInvestedMoney}, actualInvestmentValue {self.record["totalValue"].iloc[-1]}')

        # print(f'\nToday-> invest {todayBuy}, sell {todaySell}')

        return self.returnBrokerUpdate(todayInvest, data)

    def __investAndSellToday(self) -> (float, float):
        """
        This function performs the operation given by signals established the day before
        :return tuple of float representing the money that has been finally bought and sold
        """

        # Calculate the money bought and sold depending on the actual nonInvested and Invested.
        # Even though money Invested and money sold are returned, only one of those actually contains data
        moneyInvested = 0
        if self.perToInvest > 0:
            moneyInvested = self.perToInvest * self.nonInvestedMoney
            self.investedMoney += self.perToInvest * self.nonInvestedMoney
            self.nonInvestedMoney -= self.perToInvest * self.nonInvestedMoney
        elif self.perToInvest < 0:
            moneyInvested = self.perToInvest * self.investedMoney
            self.nonInvestedMoney += -self.perToInvest * self.investedMoney
            self.investedMoney -= -self.perToInvest * self.investedMoney

        return moneyInvested

    """
    ABSTRACT METHODS
    """

    @abstractmethod
    def returnBrokerUpdate(self, moneyInvestedToday, data) -> pd.DataFrame:
        """
        Function prototype that should return a df
        :param moneyInvestedToday:
        :param moneySoldToday:
        :param data:
        :return dataFrame with all the data important for proper debugging or further plotting after the operation
        """
        pass

    @abstractmethod
    def possiblyInvestMorning(self, data):
        """
        Function prototype that calls the buy function and updates the investment values
        :param data: Decision data based on the type of indicator
        """
        pass

    @abstractmethod
    def possiblyInvestAfternoon(self, data):
        """
		Function prototype that calls the buy function and updates the investment values
		:param data: Decision data based on the type of indicator
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