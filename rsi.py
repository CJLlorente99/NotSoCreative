import ta


class RSIInvestorParams:
    def __init__(self, RSIupperBound, RSIlowerBound, window):
        self.RSIupperBound = RSIupperBound
        self.RSIlowerBound = RSIlowerBound
        self.window = window

    def __str__(self):
        return f'RSI,UpperBound, LowerBound, Window\nRSI,{self.RSIupperBound},{self.RSIlowerBound},{self.window}'

def normalRSI(values, window):
    """
    Function that calculates the RSI values
    :param values: Data series containing data from the stock market
    :param window: Main parameter used for RSI calculation
    :return: RSI values over the given values
    """
    rsi = ta.momentum.RSIIndicator(values, window, True)
    return rsi.rsi()


def buyFunctionPredictionRSI(rsi, lowerBound=40, maxBuy=10000):
    """
    Function that represents the buying behavior
    :param rsi: RSI value for today
    :param lowerBound: Investor function parameter, that determines the boundary up to where the investor buys
    :param maxBuy: Maximum buy (buy at RSI = 0)
    :return: Money advised to be bought
    """
    if rsi < lowerBound:  # Buy linearly then with factor f
        return (lowerBound - rsi) * maxBuy / lowerBound
    else:
        return 0


def sellFunctionPredictionRSI(rsi, upperBound=70, maxSell=10000):
    """
    Function that represents the selling behavior
    :param rsi: RSI value for today
    :param upperBound: Investor function parameter, that determines the boundary from to where the investor sells
    :param maxSell: Maximum sell (sell at RSI = 100)
    :return:
    """
    if rsi > upperBound:  # Buy linearly then with factor f
        return (rsi - upperBound) * maxSell / (100 - upperBound)
    else:
        return 0