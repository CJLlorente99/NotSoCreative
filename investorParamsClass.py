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
    def __init__(self, window, stdDev, lowerBound, upperBound, maxBuy=2500, maxSell=10000, a=1, b=3):
        super().__init__(maxBuy, maxSell)
        self.window = window
        self.stdDev = stdDev
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        self.a = a
        self.b = b

    def __str__(self):
        string = "BB, Window, StdDev, LowerBound, UpperBound, MaxBuy, MaxSell, a, b\nBB," \
                 + str(self.window) + "," + str(self.stdDev) + "," + str(self.lowerBound) + "," + str(self.upperBound) \
                 + "," + str(self.maxBuy) + "," + str(self.maxSell) + "," + str(self.a) + "," + str(self.b)
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


class MACDInvestorParams(MAInvestorParams):
    def __init__(self, buyGradients, sellGradients, fastWindow=12, slowWindow=26, signal=9, maxBuy=2500, maxSell=10000, a=1, b=3, type="grad"):
        super().__init__(buyGradients, sellGradients, None, maxBuy, maxSell)
        self.fastWindow = fastWindow
        self.slowWindow = slowWindow
        self.signal = signal
        self.a = a
        self.b = b
        self.type = type

    def __str__(self):
        string = "MACD, LowerBoundBuy, UpperBoundBuy, LowBoundSquareBuy, UpperBoundSquareBuy, LowerBoundSell, UpperBoundSell, LowBoundSquareSell, UpperBoundSquareSell, FastWindow, SlowWindow, Signal, MaxBuy, MaxSell, a, b, type\nMACD," \
                 + str(self.buyGradients) + "," + str(self.sellGradients) + "," + str(self.fastWindow) + "," + str(
            self.slowWindow) + "," + str(self.signal) + "," + str(self.maxBuy) + "," + str(self.maxSell) + "," + str(
            self.a) + "," + str(self.b) + "," + str(self.type)
        return string


class RSIInvestorParams(InvestorParams):
    def __init__(self, upperBound, lowerBound, window, maxBuy=2500, maxSell=10000, a=1, b=3):
        super().__init__(maxBuy, maxSell)
        self.upperBound = upperBound
        self.lowerBound = lowerBound
        self.window = window
        self.a = a
        self.b = b

    def __str__(self):
        return f'RSI,UpperBound, LowerBound, Window, MaxBuy, MaxSell, a, b\nRSI,{self.upperBound},' \
               f'{self.lowerBound},{self.window},{self.maxBuy},{self.maxSell},{self.a},{self.b}'


# Useful classes
class GradientQuarter:
    def __init__(self, lowerBoundGradient, upperBoundGradient, lowBoundSquareGradient, upperBoundSquareGradient):
        self.lowerBoundGradient = lowerBoundGradient
        self.upperBoundGradient = upperBoundGradient
        self.lowBoundSquareGradient = lowBoundSquareGradient
        self.upperBoundSquareGradient = upperBoundSquareGradient

    def __str__(self):
        return f'{self.lowerBoundGradient},{self.upperBoundGradient},{self.lowBoundSquareGradient},{self.upperBoundSquareGradient}'