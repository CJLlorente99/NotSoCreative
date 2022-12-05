import numpy as np
import ta
from investorParamsClass import RSIInvestorParams
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math


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
        return params.maxBuy * math.tanh(params.a * (params.lowerBound - rsi) ** params.b)
    else:
        return 0


def sellFunctionPredictionRSI(rsi, params: RSIInvestorParams):
    """
    Function that represents the selling behavior
    :param rsi: RSI value for today
    :param params: RSI parameters
    """
    if rsi > params.upperBound:  # Buy linearly then with factor f
        return params.maxSell * math.tanh(params.a * (rsi-params.upperBound) ** params.b)
    else:
        return 0


def plotRSIDecisionRules(params: RSIInvestorParams):
    """
    Function that plots the decision rule with the given parameters
    :param params: RSI parameters
    """
    testRSI = np.arange(0, 100, 0.01)
    buyPoints = []
    sellPoints = []
    for point in testRSI:
        buyPoints = np.append(buyPoints, buyFunctionPredictionRSI(point, params))
        sellPoints = np.append(sellPoints, sellFunctionPredictionRSI(point, params))

    fig = go.Figure()
    fig.add_trace(go.Scatter(name="BuyPoints", x=testRSI, y=buyPoints, fill='tozeroy'))
    fig.add_trace(go.Scatter(name="SellPoints", x=testRSI, y=-sellPoints, fill='tozeroy'))
    fig.update_layout(title="Decision Rules for RSI indicator", xaxis={"title": "RSI Value"},
                      yaxis={"title": "Sell/Buy/Hold [$]"}, hovermode='x unified')
    fig.show()


def tryRSIDecisionRules():
    """
    Function that tries a bunch of a and b values in order to feel how the decision rule behaves
    """
    a = np.arange(0.1, 1.7, 0.4)
    b = np.arange(0.1, 1.7, 0.4)
    titles = []
    for aVal in a:
        for bVal in b:
            titles = np.append(titles, "a=" + str(aVal) + " b=" + str(bVal))

    testRSI = np.arange(0, 100, 0.25)
    i = 1
    j = 1
    fig = make_subplots(rows=len(a), cols=len(b), subplot_titles=tuple(titles))
    for aVal in a:
        for bVal in b:
            params = RSIInvestorParams(70, 30, 10, 2500, 10000, aVal, bVal)
            buyPoints = []
            sellPoints = []
            for point in testRSI:
                buyPoints = np.append(buyPoints, buyFunctionPredictionRSI(point, params))
                sellPoints = np.append(sellPoints, sellFunctionPredictionRSI(point, params))
            fig.add_trace(go.Scatter(name="BuyPoints", x=testRSI, y=buyPoints, fill='tozeroy'), row=i, col=j)
            fig.add_trace(go.Scatter(name="SellPoints", x=testRSI, y=-sellPoints, fill='tozeroy'), row=i, col=j)
            fig.update_layout(title="Decision Rules for RSI indicator")
            j += 1
        j = 1
        i += 1
    fig.show()
