import ta
from classes.investorParamsClass import RSIInvestorParams


def relativeStrengthIndex(close, params: RSIInvestorParams):
    """
    Function that calculates the RSI values
    :param close:
    :param params: RSI parameters
    :return dict with the following keys ["rsi"]
    """
    rsi = ta.momentum.RSIIndicator(close, params.window, True)
    return {"rsi": rsi.rsi()}
