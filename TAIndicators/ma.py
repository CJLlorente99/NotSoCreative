from ta.trend import MACD, EMAIndicator
from classes.investorParamsClass import MACDInvestorParams, MAInvestorParams


def movingAverageConvergenceDivergence(close, params: MACDInvestorParams):
    """
    Function that calculates the different returns of the MACD indicator
    :param close: Close market values
    :param params: Parameters to be used for the indicator calculation (fastWindow, slowWindow, signal)
    :return: dict with the following keys ["macd", "signal"]
    """
    macd = MACD(close, params.fastWindow, params.slowWindow, params.signal, True)
    diff = macd.macd() - macd.macd_signal()
    return {"macd" : macd.macd(), "signal" : macd.macd_signal(), 'diff': diff}

def exponentialMovingAverage(close, params: MAInvestorParams):
    ema = EMAIndicator(close, params.window, True)
    return {"ema" : ema.ema_indicator()}




