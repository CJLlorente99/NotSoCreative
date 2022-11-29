import numpy as np
import ta
from investorParamsClass import RSIInvestorParams
import plotly.graph_objects as go


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


def plotRSIDecisionRules(params: RSIInvestorParams):
    testRSI = np.arange(0, 100, 0.25)
    buyPoints = []
    sellPoints = []
    for point in testRSI:
        buyPoints = np.append(buyPoints, buyFunctionPredictionRSI(point, params))
        sellPoints = np.append(sellPoints, sellFunctionPredictionRSI(point, params))

    fig = go.Figure()
    fig.add_trace(go.Scatter(name="BuyPoints", x=testRSI, y=buyPoints, fill='tozeroy'))
    fig.add_trace(go.Scatter(name="SellPoints", x=testRSI, y=-sellPoints, fill='tozeroy'))
    fig.update_layout(title="Decision Rules for RSI indicator", xaxis={"title": "RSI Value"}, yaxis={"title": "Sell/Buy/Hold [$]"})
    fig.show()
