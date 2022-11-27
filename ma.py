from ta.trend import SMAIndicator, EMAIndicator, MACD
import numpy as np


class GradientQuarter:
    def __init__(self, lowerBoundGradient, upperBoundGradient, lowBoundSquareGradient, upperBoundSquareGradient):
        self.lowerBoundGradient = lowerBoundGradient
        self.upperBoundGradient = upperBoundGradient
        self.lowBoundSquareGradient = lowBoundSquareGradient
        self.upperBoundSquareGradient = upperBoundSquareGradient

    def __str__(self):
        return f'{self.lowerBoundGradient},{self.upperBoundGradient},{self.lowBoundSquareGradient},{self.upperBoundSquareGradient}'


class SMAInvestorParams:
    def __init__(self, buyGradients, sellGradients, window):
        self.buyGradients = buyGradients
        self.sellGradients = sellGradients
        self.window = window

    def __str__(self):
        string = "SMA, LowerBoundBuy, UpperBoundBuy, LowBoundSquareBuy, UpperBoundSquareBuy, LowerBoundSell, UpperBoundSell, LowBoundSquareSell, UpperBoundSquareSell, Window\nSMA,"\
                + str(self.buyGradients) + "," + str(self.sellGradients) + "," + str(self.window)
        return string


def simpleMovingAverage(values, window):
    sma = SMAIndicator(values, window, True)
    return sma.sma_indicator()


def buyPredictionSMA(sma, parameters: GradientQuarter, maxBuy=2500):
    """

    :param sma: Series with the values of the SMA
    :param parameters: Gradient parameters
    :param maxBuy: Maximum money to be invested in a single operation
    """
    firstGradient = np.gradient(sma.values)
    secondGradient = np.gradient(firstGradient)

    if (parameters.lowerBoundGradient < firstGradient[0] <= parameters.upperBoundGradient) and (
            parameters.lowBoundSquareGradient < secondGradient[0] < parameters.upperBoundSquareGradient):
        return (secondGradient[0] - parameters.lowBoundSquareGradient) * maxBuy / (
                    parameters.upperBoundSquareGradient - parameters.lowBoundSquareGradient)
    else:
        return 0


def sellPredictionSMA(sma, parameters: GradientQuarter, maxSell=10000):
    """

    :param sma: Series with the values of the SMA
    :param parameters: Gradient parameters
    :param maxSell: Maximum money to be invested in a single operation
    """
    firstGradient = np.gradient(sma.values)
    secondGradient = np.gradient(firstGradient)

    if (parameters.lowerBoundGradient < firstGradient[0] <= parameters.upperBoundGradient) and (
            parameters.lowBoundSquareGradient < secondGradient[0] < parameters.upperBoundSquareGradient):
        return (secondGradient[0] - parameters.upperBoundSquareGradient) * maxSell / (
                    parameters.lowBoundSquareGradient - parameters.upperBoundSquareGradient)
    else:
        return 0


class EMAInvestorParams:
    def __init__(self, buyGradients, sellGradients, window):
        self.buyGradients = buyGradients
        self.sellGradients = sellGradients
        self.window = window

    def __str__(self):
        string = "EMA, LowerBoundBuy, UpperBoundBuy, LowBoundSquareBuy, UpperBoundSquareBuy, LowerBoundSell, UpperBoundSell, LowBoundSquareSell, UpperBoundSquareSell, Window\nSMA,"\
                + str(self.buyGradients) + "," + str(self.sellGradients) + "," + str(self.window)
        return string


def exponentialMovingAverage(values, window):
    ema = EMAIndicator(values, window, True)
    return ema.ema_indicator()


def buyPredictionEMA(sma, parameters: GradientQuarter, maxBuy=2500):
    """

    :param sma: Series with the values of the SMA
    :param parameters: Gradient parameters
    :param maxBuy: Maximum money to be invested in a single operation
    """
    firstGradient = np.gradient(sma.values)
    secondGradient = np.gradient(firstGradient)

    if (parameters.lowerBoundGradient < firstGradient[0] <= parameters.upperBoundGradient) and (
            parameters.lowBoundSquareGradient < secondGradient[0] < parameters.upperBoundSquareGradient):
        return (secondGradient[0] - parameters.lowBoundSquareGradient) * maxBuy / (
                    parameters.upperBoundSquareGradient - parameters.lowBoundSquareGradient)
    else:
        return 0


def sellPredictionEMA(sma, parameters: GradientQuarter, maxSell=10000):
    """

    :param sma: Series with the values of the SMA
    :param parameters: Gradient parameters
    :param maxSell: Maximum money to be invested in a single operation
    """
    firstGradient = np.gradient(sma.values)
    secondGradient = np.gradient(firstGradient)

    if (parameters.lowerBoundGradient < firstGradient[0] <= parameters.upperBoundGradient) and (
            parameters.lowBoundSquareGradient < secondGradient[0] < parameters.upperBoundSquareGradient):
        return (secondGradient[0] - parameters.upperBoundSquareGradient) * maxSell / (
                    parameters.lowBoundSquareGradient - parameters.upperBoundSquareGradient)
    else:
        return 0


class MACDInvestorParams:
    def __init__(self, upperBound=50, lowerBound=50, fastWindow=12, slowWindow=26, signal=9):
        self.upperBound = upperBound
        self.lowerBound = lowerBound
        self.fastWindow = fastWindow
        self.slowWindow = slowWindow
        self.signal = signal

    def __str__(self):
        string = "MACD, UpperBound, LowerBound, FastWindow, SlowWindow, Signal\n"\
                + "MACD," + str(self.upperBound) + "," + str(self.lowerBound) + "," + str(self.fastWindow) + ","\
                + str(self.slowWindow) + "," + str(self.signal)
        return string


def movingAverageConvergenceDivergence(values, windowSlow, windowFast, signal=9):
    macd = MACD(values, windowFast, windowSlow, signal, True)
    return macd.macd()


def buyPredictionMACD(macd, parameters: MACDInvestorParams, maxBuy=2500):
    """

    :param macd: Series with the values of the MACD
    :param parameters: Upper and lower parameters
    :param maxBuy: Maximum money to be invested in a single operation
    """
    if macd > parameters.upperBound:  # Buy linearly then with factor f
        return (macd - parameters.upperBound) * maxBuy / 9*parameters.upperBound
    else:
        return 0


def sellPredictionMACD(macd, parameters: MACDInvestorParams, maxSell=10000):
    """

    :param macd: Series with the values of the MACD
    :param parameters: Upper and lower parameters
    :param maxSell: Maximum money to be invested in a single operation
    """
    if macd < parameters.lowerBound:  # Buy linearly then with factor f
        return (parameters.lowerBound - macd) * maxSell / 0.9*parameters.lowerBound
    else:
        return 0

