from ta.trend import SMAIndicator, EMAIndicator, MACD
from investorParamsClass import MAInvestorParams, MACDInvestorParams
import numpy as np


def simpleMovingAverage(values, params: MAInvestorParams):
    sma = SMAIndicator(values, params.window, True)
    return sma.sma_indicator()


def buyPredictionSMA(sma, params: MAInvestorParams):
    """

    :param sma: Series with the values of the SMA
    :param params: MA parameters
    """
    firstGradient = np.gradient(sma.values)
    secondGradient = np.gradient(firstGradient)

    parameters = params.buyGradients

    if (parameters.lowerBoundGradient < firstGradient[0] <= parameters.upperBoundGradient) and (
            parameters.lowBoundSquareGradient < secondGradient[0] < parameters.upperBoundSquareGradient):
        return (secondGradient[0] - parameters.lowBoundSquareGradient) * params.maxBuy / (
                    parameters.upperBoundSquareGradient - parameters.lowBoundSquareGradient)
    else:
        return 0


def sellPredictionSMA(sma, params: MAInvestorParams):
    """

    :param sma: Series with the values of the SMA
    :param params: MA parameters
    """
    firstGradient = np.gradient(sma.values)
    secondGradient = np.gradient(firstGradient)

    parameters = params.sellGradients

    if (parameters.lowerBoundGradient < firstGradient[0] <= parameters.upperBoundGradient) and (
            parameters.lowBoundSquareGradient < secondGradient[0] < parameters.upperBoundSquareGradient):
        return (secondGradient[0] - parameters.upperBoundSquareGradient) * params.maxSell / (
                    parameters.lowBoundSquareGradient - parameters.upperBoundSquareGradient)
    else:
        return 0


def exponentialMovingAverage(values, params: MAInvestorParams):
    ema = EMAIndicator(values, params.window, True)
    return ema.ema_indicator()


def buyPredictionEMA(sma, params: MAInvestorParams):
    """

    :param sma: Series with the values of the SMA
    :param params: MA parameters
    """
    firstGradient = np.gradient(sma.values)
    secondGradient = np.gradient(firstGradient)

    parameters = params.buyGradients

    if (parameters.lowerBoundGradient < firstGradient[0] <= parameters.upperBoundGradient) and (
            parameters.lowBoundSquareGradient < secondGradient[0] < parameters.upperBoundSquareGradient):
        return (secondGradient[0] - parameters.lowBoundSquareGradient) * params.maxBuy / (
                    parameters.upperBoundSquareGradient - parameters.lowBoundSquareGradient)
    else:
        return 0


def sellPredictionEMA(sma, params: MAInvestorParams):
    """

    :param sma: Series with the values of the SMA
    :param params: MA parameters
    """
    firstGradient = np.gradient(sma.values)
    secondGradient = np.gradient(firstGradient)

    parameters = params.sellGradients

    if (parameters.lowerBoundGradient < firstGradient[0] <= parameters.upperBoundGradient) and (
            parameters.lowBoundSquareGradient < secondGradient[0] < parameters.upperBoundSquareGradient):
        return (secondGradient[0] - parameters.upperBoundSquareGradient) * params.maxSell / (
                    parameters.lowBoundSquareGradient - parameters.upperBoundSquareGradient)
    else:
        return 0


def movingAverageConvergenceDivergence(values, params: MACDInvestorParams):
    macd = MACD(values, params.fastWindow, params.slowWindow, params.signal, True)
    return macd.macd()


def buyPredictionMACD(macd, params: MACDInvestorParams):
    """

    :param macd: Series with the values of the MACD
    :param params: MACD params
    """
    if macd > params.upperBound:  # Buy linearly then with factor f
        return (macd - params.upperBound) * params.maxBuy / 9*params.upperBound
    else:
        return 0


def sellPredictionMACD(macd, params: MACDInvestorParams):
    """

    :param macd: Series with the values of the MACD
    :param params: MACD params
    """
    if macd < params.lowerBound:  # Buy linearly then with factor f
        return (params.lowerBound - macd) * params.maxSell / 0.9*params.lowerBound
    else:
        return 0

