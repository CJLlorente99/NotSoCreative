from datetime import datetime
import pandas_datareader as web
import pandas as pd
import yfinance as yf
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
        self.nextnextStockValueOpen = 0
        self.nextStockValueOpen = 0
        self.date = datetime.today()
        self.rsi = 0
        self.stochrsi = 0
        self.adx = 0
        self.aroon = 0
        self.obv = 0
        self.atr = 0
        self.adi = 0
        self.macd = None
        self.bb = None
        self.nDay = 0
        self.dt = {}


class DataGetter:
    dataLen = 60

    def __init__(self):
        """
        Initialization function for the DataGetter. This class is supposed to be used as intermediary to get new data
        """
        yf.pdr_override()
        self.ticker = "^GSPC"
        # define US business days
        us_bus = CDay(calendar=USFederalHolidayCalendar())
        self.today = pd.bdate_range('2014-01-01', '2018-01-31', freq=us_bus)[0]
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
                data = yf.download(self.ticker, yesterday, self.today)
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
                data = yf.download(self.ticker, self.start, self.today)
                return data
            except:
                self.today += CDay(calendar=USFederalHolidayCalendar())

    def getToday(self):
        """
        Function to get today's data. Try/except clause to deal with days when the stock was closed.
        :return: Today's data
        """
        aux = self.today
        aux += CDay(calendar=USFederalHolidayCalendar())
        while True:
            try:
                data = yf.download(self.ticker, self.today, aux)
                return data
            except:
                self.today += CDay(calendar=USFederalHolidayCalendar())
                aux += CDay(calendar=USFederalHolidayCalendar())

    def getNextDay(self):
        """
        Function to advance one day and get new data. Try/except clause to deal with days when the stock was closed.
        :return: Next day's data
        """
        nextDay = self.today
        aux = nextDay + CDay(calendar=USFederalHolidayCalendar())
        while True:
            nextDay += CDay(calendar=USFederalHolidayCalendar())
            aux += CDay(calendar=USFederalHolidayCalendar())
            try:
                data = yf.download(self.ticker, nextDay, aux)
                return data
            except:
                continue

    def getNextNextDay(self):
        """
        Function to advance one day and get new data. Try/except clause to deal with days when the stock was closed.
        :return: Next day's data
        """
        nextDay = self.today
        nextDay += CDay(calendar=USFederalHolidayCalendar())
        aux = nextDay + CDay(calendar=USFederalHolidayCalendar())
        while True:
            nextDay += CDay(calendar=USFederalHolidayCalendar())
            aux += CDay(calendar=USFederalHolidayCalendar())
            try:
                data = yf.download(self.ticker, nextDay, aux)
                return data
            except:
                continue

    def goNextDay(self):
        """
        Function that just advances one day
        """
        self.today += CDay(calendar=USFederalHolidayCalendar())

