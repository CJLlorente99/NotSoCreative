import math
from ta.volatility import BollingerBands
from investorParamsClass import BBInvestorParams
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def bollingerBands(values, params: BBInvestorParams):
    """
    Function that calcualtes the bollinger bands
    :param values: Open or Close value from the stock market series
    :param params: BB investor parameters
    :return: < 0 => below lower band; > 1 over upper band; = 0.5 in the middle of the band
    """
    bb = BollingerBands(values, params.window, params.stdDev, fillna=True)
    return {"pband": bb.bollinger_pband(), "mavg": bb.bollinger_mavg(), "hband": bb.bollinger_hband(), "lband": bb.bollinger_lband()}


def buyPredictionBB(bb, params: BBInvestorParams):
    """
    Function that returns the money to be invested
    :param bb: bollinger_pband() value
    :param params: BB investor parameters
    :return:
    """
    if bb < params.lowerBound:
        return params.maxBuy*math.tanh(params.a * (params.lowerBound - bb) ** params.b)
    else:
        return 0


def sellPredictionBB(bb, params: BBInvestorParams):
    """
    Function that returns the money to be sold
    :param bb: bollinger_pband() value
    :param params: BB investor parameters
    :return:
    """
    if bb > params.upperBound:
        return params.maxSell*math.tanh(params.a * (bb-params.upperBound) ** params.b)
    else:
        return 0


def plotBBDecisionRules(params: BBInvestorParams):
    """
    Function that plots both how bollinger_pband() works and how the decisions are made
    :param params: B  investor parameters
    """
    testBB = np.arange(-2, 3, 0.01)
    buyPoints = []
    sellPoints = []
    for point in testBB:
        buyPoints = np.append(buyPoints, buyPredictionBB(point, params))
        sellPoints = np.append(sellPoints, sellPredictionBB(point, params))

    fig = go.Figure()
    fig.add_trace(go.Scatter(name="BuyPoints", x=testBB, y=buyPoints, fill='tozeroy'))
    fig.add_trace(go.Scatter(name="SellPoints", x=testBB, y=-sellPoints, fill='tozeroy'))
    fig.update_layout(title="Decision Rules for BB indicator", xaxis={"title": "BB Value"}, yaxis={"title": "Sell/Buy/Hold [$]"})
    fig.show()


def tryBBDecisionRules():
    a = np.arange(1, 10, 3)
    b = np.arange(1, 15, 4)
    titles = []
    for aVal in a:
        for bVal in b:
            titles = np.append(titles, "a=" + str(aVal) + " b=" + str(bVal))

    testBB = np.arange(-2, 3, 0.01)
    i = 1
    j = 1
    fig = make_subplots(rows=len(a), cols=len(b), subplot_titles=tuple(titles))
    for aVal in a:
        for bVal in b:
            params = BBInvestorParams(10, 3, 0.3, 0.7, 0, 0, 2500, 10000, aVal, bVal)
            buyPoints = []
            sellPoints = []
            for point in testBB:
                buyPoints = np.append(buyPoints, buyPredictionBB(point, params))
                sellPoints = np.append(sellPoints, sellPredictionBB(point, params))
            fig.add_trace(go.Scatter(name="BuyPoints", x=testBB, y=buyPoints, fill='tozeroy'), row=i, col=j)
            fig.add_trace(go.Scatter(name="SellPoints", x=testBB, y=-sellPoints, fill='tozeroy'), row=i, col=j)
            fig.update_layout(title="Decision Rules for BB indicator")
            j += 1
        j = 1
        i += 1
    fig.show()
