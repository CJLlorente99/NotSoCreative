class InvestorParams:
    def __init__(self):
        """
        Basic parent class inherited by all the other investor parameters classes.
        All children classes should have an initializer method and a overriding of the __str__ method
        """


class BBInvestorParams(InvestorParams):
    def __init__(self, window, stdDev, lowerBound=0, upperBound=0, a=1, b=3):
        super().__init__()
        self.window = window
        self.stdDev = stdDev
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        self.a = a
        self.b = b

    def __str__(self):
        string = "BB, Window, StdDev, LowerBound, UpperBound, a, b\nBB," \
                 + str(self.window) + "," + str(self.stdDev) + "," + str(self.lowerBound) + "," + str(self.upperBound) \
                 + "," + str(self.a) + "," + str(self.b)
        return string


class MAInvestorParams(InvestorParams):
    def __init__(self, buyGradients, sellGradients, window):
        super().__init__()
        self.buyGradients = buyGradients
        self.sellGradients = sellGradients
        self.window = window

    def __str__(self):
        string = "SMA, LowerBoundBuy, UpperBoundBuy, LowBoundSquareBuy, UpperBoundSquareBuy, LowerBoundSell, UpperBoundSell, LowBoundSquareSell, UpperBoundSquareSell, Window\nSMA," \
                 + str(self.buyGradients) + "," + str(self.sellGradients) + "," + str(self.window)
        return string


class MACDInvestorParams(MAInvestorParams):
    def __init__(self, buyGradients, sellGradients, fastWindow=12, slowWindow=26, signal=9, a=1, b=3, type="grad"):
        super().__init__(buyGradients, sellGradients, None)
        self.fastWindow = fastWindow
        self.slowWindow = slowWindow
        self.signal = signal
        self.a = a
        self.b = b
        self.type = type

    def __str__(self):
        string = "MACD, LowerBoundBuy, UpperBoundBuy, LowBoundSquareBuy, UpperBoundSquareBuy, LowerBoundSell, UpperBoundSell, LowBoundSquareSell, UpperBoundSquareSell, FastWindow, SlowWindow, Signal, a, b, type\nMACD," \
                 + str(self.buyGradients) + "," + str(self.sellGradients) + "," + str(self.fastWindow) + "," + str(
            self.slowWindow) + "," + str(self.signal) + "," + str(
            self.a) + "," + str(self.b) + "," + str(self.type)
        return string


class RSIInvestorParams(InvestorParams):
    def __init__(self, upperBound, lowerBound, window, a=1, b=3):
        super().__init__()
        self.upperBound = upperBound
        self.lowerBound = lowerBound
        self.window = window
        self.a = a
        self.b = b

    def __str__(self):
        return f'RSI,UpperBound, LowerBound, Window, a, b\nRSI,{self.upperBound},' \
               f'{self.lowerBound},{self.window},{self.a},{self.b}'


class StochasticRSIInvestorParams(InvestorParams):
    def __init__(self, window, smooth1, smooth2, a=1, b=3):
        super().__init__()
        self.window = window
        self.smooth1 = smooth1
        self.smooth2 = smooth2
        self.a = a
        self.b = b

    def __str__(self):
        return f'StochRSI, Window, Smooth1, Smooth2, a, b\nStochRSI,{self.window},' \
               f'{self.smooth1},{self.smooth2},{self.a},{self.b}'


class ADXInvestorParams(InvestorParams):
    def __init__(self, window, a=1, b=3):
        super().__init__()
        self.window = window
        self.a = a
        self.b = b

    def __str__(self):
        return f'ADX, Window, a, b\nADX,{self.window},' \
               f'{self.a},{self.b}'


class AroonInvestorParams(InvestorParams):
    def __init__(self, window, a=1, b=3):
        super().__init__()
        self.window = window
        self.a = a
        self.b = b

    def __str__(self):
        return f'Aroon, Window, a, b\nAroon,{self.window},' \
               f'{self.a},{self.b}'


class ADIInvestorParams(InvestorParams):
    def __init__(self, a=1, b=3):
        super().__init__()
        self.a = a
        self.b = b

    def __str__(self):
        return f'ADI, a, b\nADI,' \
               f'{self.a},{self.b}'


class OBVInvestorParams(InvestorParams):
    def __init__(self, a=1, b=3):
        super().__init__()
        self.a = a
        self.b = b

    def __str__(self):
        return f'OBV, a, b\nOBV,' \
               f'{self.a},{self.b}'


class ATRInvestorParams(InvestorParams):
    def __init__(self, window, a=1, b=3):
        super().__init__()
        self.window = window
        self.a = a
        self.b = b

    def __str__(self):
        return f'OBV, a, b\nOBV,' \
               f'{self.a},{self.b}'

class NNInvestorParams(InvestorParams):
    def __init__(self, file):
        super().__init__()
        self.file = file

    def __str__(self):
        return f'NN, File\nNN,{self.file}'

class DTInvestorParams(InvestorParams):
    def __init__(self, filename, orderedListArguments):
        super().__init__()
        self.filename = filename
        self.orderedListArguments = orderedListArguments

    def __str__(self):
        return f'DT, filename\nNN,{self.filename}'

class LSTMInvestorParams(InvestorParams):
    def __init__(self, filename, threshold):
        super().__init__()
        self.filename = filename
        self.threshold = threshold

    def __str__(self):
        return f'LSTM, filename, threshold\nNN,{self.filename},{self.threshold}'


# Useful classes
class GradientQuarter:
    def __init__(self, lowerBoundGradient, upperBoundGradient, lowBoundSquareGradient, upperBoundSquareGradient):
        self.lowerBoundGradient = lowerBoundGradient
        self.upperBoundGradient = upperBoundGradient
        self.lowBoundSquareGradient = lowBoundSquareGradient
        self.upperBoundSquareGradient = upperBoundSquareGradient

    def __str__(self):
        return f'{self.lowerBoundGradient},{self.upperBoundGradient},{self.lowBoundSquareGradient},{self.upperBoundSquareGradient}'