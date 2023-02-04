from ta.volatility import BollingerBands
from classes.investorParamsClass import BBInvestorParams


def bollingerBands(close, params: BBInvestorParams):
    """
    Function that calcualtes the bollinger bands
    :param values: Open or Close value from the stock market series
    :param params: BB investor parameters
    :return: dict with the following keys ["pband", "mavg", "hband", "lband"]
    """
    bb = BollingerBands(close, params.window, params.stdDev, fillna=True)
    return {"pband": bb.bollinger_pband(), "mavg": bb.bollinger_mavg(), "hband": bb.bollinger_hband(), "lband": bb.bollinger_lband()}

