import datetime as dt
import pandas_datareader as web
import pandas as pd
from pandas.tseries.offsets import CDay
from pandas.tseries.holiday import USFederalHolidayCalendar


class DataManager:
    def __init__(self):
        """
        Initialization function for DataManager
        """
        self.pastStockValue = 0
        self.actualStockValue = 0
        self.nextStockValue = 0
        self.date = dt.date.today()
        self.rsi = 0
        self.sma = None
        self.ema = None
        self.macd = None
        self.bb = None


class DataGetter:
    dataLen = 60

    def __init__(self):
        """
        Initialization function for the DataGetter. This class is supposed to be used as intermediary to get new data
        """
        self.ticker = '^GSPC'
        # define US business days
        us_bus = CDay(calendar=USFederalHolidayCalendar())
        self.today = pd.bdate_range('2014-01-01', '2014-01-31', freq=us_bus)[0]
        self.start = self.today - CDay(self.dataLen)

    def getPastData(self):
        """
        Function used to retrieve data from yesterday. Try/except clause to deal with days when the stock was closed.
        :return: Yesterday's data
        """
        yesterday = self.today
        while True:
            yesterday -= CDay(calendar=USFederalHolidayCalendar())
            try:
                data = web.DataReader(self.ticker, 'yahoo', yesterday, yesterday)
                return data
            except:
                continue

    def getUntilToday(self):
        """
        Function used to retrieve data until today. Try/except clause to deal with days when the stock was closed.
        :return: Data until today
        """
        while True:
            self.start = self.today - CDay(self.dataLen)
            try:
                data = web.DataReader(self.ticker, 'yahoo', self.start, self.today)
                return data
            except:
                self.today += CDay(calendar=USFederalHolidayCalendar())

    def getToday(self):
        """
        Function to get today's data. Try/except clause to deal with days when the stock was closed.
        :return: Today's data
        """
        while True:
            try:
                data = web.DataReader(self.ticker, 'yahoo', self.today, self.today)
                return data
            except:
                self.today += CDay(calendar=USFederalHolidayCalendar())

    def getNextDay(self):
        """
        Function to advance one day and get new data. Try/except clause to deal with days when the stock was closed.
        :return: Next day's data
        """
        nextDay = self.today
        while True:
            nextDay += CDay(calendar=USFederalHolidayCalendar())
            try:
                data = web.DataReader(self.ticker, 'yahoo', nextDay, nextDay)
                return data
            except:
                continue

    def goNextDay(self):
        """
        Function that just advances one day
        """
        self.today += CDay(calendar=USFederalHolidayCalendar())

