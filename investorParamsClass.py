class InvestorParams:
    def __init__(self, maxBuy, maxSell):
        """
        Basic parent class inherited by all the other investor parameters classes
        :param maxBuy: Maximum money to be invested in a single operation
        :param maxSell: Maximum money to be sold in a single operation
        """
        self.maxBuy = maxBuy
        self.maxSell = maxSell


class BBInvestorParams(InvestorParams):
    def __init__(self, window, stdDev, lowerBound, upperBound, buyingSlope, sellingSlope, maxBuy=2500, maxSell=10000):
        super().__init__(maxBuy, maxSell)
        self.window = window
        self.stdDev = stdDev
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        self.buyingSlope = buyingSlope
        self.sellingSlope = sellingSlope

    def __str__(self):
        string = "BB, Window, StdDev, LowerBound, UpperBound, BuyingSlope, SellingSlope, maxBuy, maxSell\nSMA," \
                 + str(self.window) + "," + str(self.stdDev) + "," + str(self.lowerBound) + "," + str(self.upperBound) \
                 + "," + str(self.buyingSlope) + "," + str(self.sellingSlope) + "," + str(self.maxBuy) + "," + str(self.maxSell)
        return string


class MAInvestorParams(InvestorParams):
    def __init__(self, buyGradients, sellGradients, window, maxBuy=2500, maxSell=10000):
        super().__init__(maxBuy, maxSell)
        self.buyGradients = buyGradients
        self.sellGradients = sellGradients
        self.window = window

    def __str__(self):
        string = "SMA, LowerBoundBuy, UpperBoundBuy, LowBoundSquareBuy, UpperBoundSquareBuy, LowerBoundSell, UpperBoundSell, LowBoundSquareSell, UpperBoundSquareSell, Window, MaxBuy, MaxSell\nSMA," \
                 + str(self.buyGradients) + "," + str(self.sellGradients) + "," + str(self.window) + "," + str(
            self.maxBuy) + "," + str(self.maxSell)
        return string


class MACDInvestorParams(InvestorParams):
    def __init__(self, upperBound=50, lowerBound=50, fastWindow=12, slowWindow=26, signal=9, maxBuy=2500, maxSell=10000):
        super().__init__(maxBuy, maxSell)
        self.upperBound = upperBound
        self.lowerBound = lowerBound
        self.fastWindow = fastWindow
        self.slowWindow = slowWindow
        self.signal = signal

    def __str__(self):
        string = "MACD, UpperBound, LowerBound, FastWindow, SlowWindow, Signal, MaxBuy, MaxSell\n"\
                + "MACD," + str(self.upperBound) + "," + str(self.lowerBound) + "," + str(self.fastWindow) + ","\
                + str(self.slowWindow) + "," + str(self.signal) + "," + str(self.maxBuy) + "," + str(self.maxSell)
        return string


class RSIInvestorParams(InvestorParams):
    def __init__(self, upperBound, lowerBound, window, maxBuy=2500, maxSell=10000):
        super().__init__(maxBuy, maxSell)
        self.upperBound = upperBound
        self.lowerBound = lowerBound
        self.window = window

    def __str__(self):
        return f'RSI,UpperBound, LowerBound, Window, MaxBuy, MaxSell\nRSI,{self.upperBound},' \
               f'{self.lowerBound},{self.window},{self.maxBuy},{self.maxSell}'


# Useful classes
class GradientQuarter:
    def __init__(self, lowerBoundGradient, upperBoundGradient, lowBoundSquareGradient, upperBoundSquareGradient):
        self.lowerBoundGradient = lowerBoundGradient
        self.upperBoundGradient = upperBoundGradient
        self.lowBoundSquareGradient = lowBoundSquareGradient
        self.upperBoundSquareGradient = upperBoundSquareGradient

    def __str__(self):
        return f'{self.lowerBoundGradient},{self.upperBoundGradient},{self.lowBoundSquareGradient},{self.upperBoundSquareGradient}'