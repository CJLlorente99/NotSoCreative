import ta
from investorParamsClass import RSIInvestorParams


def relativeStrengthIndex(values, params: RSIInvestorParams):
    """
    Function that calculates the RSI values
    :param values:
    :param params: RSI parameters
    """
    rsi = ta.momentum.RSIIndicator(values, params.window, True)
    return rsi.rsi()


def buyFunctionPredictionRSI(rsi, params: RSIInvestorParams):
    """
    Function that represents the buying behavior
    :param rsi: RSI value for today
    :param params: RSI parameters
    """
    if rsi < params.lowerBound:  # Buy linearly then with factor f
        return (params.lowerBound - rsi) * params.maxBuy / params.lowerBound
    else:
        return 0


def sellFunctionPredictionRSI(rsi, params: RSIInvestorParams):
    """
    Function that represents the selling behavior
    :param rsi: RSI value for today
    :param params: RSI parameters
    """
    if rsi > params.upperBound:  # Buy linearly then with factor f
        return (rsi - params.upperBound) * params.maxSell / (100 - params.upperBound)
    else:
        return 0