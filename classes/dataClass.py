from datetime import datetime
import pandas_datareader as web
import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import CDay
from pandas.tseries.holiday import USFederalHolidayCalendar


class DataManager:
    def __init__(self):
        """
        Initialization function for DataManager. This function is used as a common wrapper for info used during
        broker activity.
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
        self.lstm = {}


class DataGetter:
    dataLen = 400

    def __init__(self):
        """
        Initialization function for the DataGetter. This class is supposed to be used as intermediary to get new data
        """
        yf.pdr_override()
        self.ticker = "^GSPC"
        # define US business days
        us_bus = CDay(calendar=USFederalHolidayCalendar())
        self.today = pd.bdate_range('2018-01-01', '2018-01-31', freq=us_bus)[0]
        self.start = self.today - CDay(self.dataLen)

    def getPastData(self) -> pd.DataFrame:
        """
        Function used to retrieve data from yesterday. Try/except clause to deal with days when the stock was closed.
        :return: dataFrame: Yesterday's data
        """
        yesterday = self.today
        while True:
            yesterday -= CDay(calendar=USFederalHolidayCalendar())
            data = yf.download(self.ticker, yesterday, self.today)
            if len(data) == 0:
                yesterday -= CDay(calendar=USFederalHolidayCalendar())
            else:
                return data

    def getUntilToday(self) -> pd.DataFrame:
        """
        Function used to retrieve data until today.
        :return: Data until today
        """
        while True:
            self.start = self.today - CDay(self.dataLen)
            data = yf.download(self.ticker, self.start, self.today)
            if len(data) == 0:
                self.today += CDay(calendar=USFederalHolidayCalendar())
            else:
                return data

    def getToday(self) -> pd.DataFrame:
        """
        Function to get today's data.
        :return: Today's data
        """
        aux = self.today
        aux += CDay(calendar=USFederalHolidayCalendar())
        while True:
            data = yf.download(self.ticker, self.today, aux)
            if len(data) == 0:
                self.today += CDay(calendar=USFederalHolidayCalendar())
                aux += CDay(calendar=USFederalHolidayCalendar())
            else:
                return data

    def getNextDay(self) -> pd.DataFrame:
        """
        Function to advance one day and get new data.
        :return: Next day's data
        """
        nextDay = self.today
        aux = nextDay + CDay(calendar=USFederalHolidayCalendar())
        while True:
            nextDay += CDay(calendar=USFederalHolidayCalendar())
            aux += CDay(calendar=USFederalHolidayCalendar())
            data = yf.download(self.ticker, nextDay, aux)
            if len(data) != 0:
                return data


    def getNextNextDay(self) -> pd.DataFrame:
        """
        Function that retrieves data from the day after tomorrow.
        :return: Next day's data
        """
        nextDay = self.today
        nextDay += CDay(calendar=USFederalHolidayCalendar())
        aux = nextDay + CDay(calendar=USFederalHolidayCalendar())
        while True:
            nextDay += CDay(calendar=USFederalHolidayCalendar())
            aux += CDay(calendar=USFederalHolidayCalendar())
            data = yf.download(self.ticker, nextDay, aux)
            if len(data) != 0:
                return data

    def goNextDay(self):
        """
        Function that just advances one day
        """
        self.today += CDay(calendar=USFederalHolidayCalendar())

