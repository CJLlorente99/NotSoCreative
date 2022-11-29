import math

from ta.trend import SMAIndicator, EMAIndicator, MACD
from investorParamsClass import MAInvestorParams, MACDInvestorParams
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


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
                    parameters.upperBoundSquareGradient - parameters.lowBoundSquareGradient), firstGradient, secondGradient
    else:
        return 0, firstGradient, secondGradient


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
                    parameters.lowBoundSquareGradient - parameters.upperBoundSquareGradient), firstGradient, secondGradient
    else:
        return 0, firstGradient, secondGradient


def plotSMADecisionRules(params: MAInvestorParams):
    x = np.arange(0, 4*math.pi, 0.05)
    testSMA = pd.Series(np.sin(x))
    buyPoints = []
    sellPoints = []
    for i in range(len(testSMA)-2):
        buyPoint, firstGradient, secondGradient = buyPredictionSMA(testSMA[0:i+2], params)
        buyPoints = np.append(buyPoints, buyPoint)
        sellPoints = np.append(sellPoints, sellPredictionSMA(testSMA[0:i+2], params))

    fig = go.Figure()
    fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": True}], [{"secondary_y": True}]])
    fig.add_trace(go.Scatter(name="SMAValues", x=x, y=testSMA.values), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(name="BuyPoints", x=x, y=buyPoints, fill='tozeroy'), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(name="SellPoints", x=x, y=-sellPoints, fill='tozeroy'), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(name="FirstGradient", x=x, y=firstGradient), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(name="SecondGradient", x=x, y=secondGradient), row=2, col=1, secondary_y=True)
    fig.update_layout(title="Decision Rules for SMA indicator", xaxis={"title": "x"}, yaxis={"title": "Sell/Buy/Hold [$]"})
    fig.show()


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
                    parameters.upperBoundSquareGradient - parameters.lowBoundSquareGradient), firstGradient, secondGradient
    else:
        return 0, firstGradient, secondGradient


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
                    parameters.lowBoundSquareGradient - parameters.upperBoundSquareGradient), firstGradient, secondGradient
    else:
        return 0, firstGradient, secondGradient


def plotEMADecisionRules(params: MAInvestorParams):
    x = np.arange(0, 4 * math.pi, 0.05)
    testEMA = pd.Series(np.sin(x))
    buyPoints = []
    sellPoints = []
    for i in range(len(testEMA)-2):
        buyPoint, firstGradient, secondGradient = buyPredictionSMA(testEMA[0:i+2], params)
        buyPoints = np.append(buyPoints, buyPoint)
        sellPoints = np.append(sellPoints, sellPredictionSMA(testEMA[0:i+2], params))

    fig = go.Figure()
    fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": True}], [{"secondary_y": True}]])
    fig.add_trace(go.Scatter(name="EMAValues", x=x, y=testEMA.values), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(name="BuyPoints", x=x, y=buyPoints, fill='tozeroy'), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(name="SellPoints", x=x, y=-sellPoints, fill='tozeroy'), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(name="FirstGradient", x=x, y=firstGradient), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(name="SecondGradient", x=x, y=secondGradient), row=2, col=1, secondary_y=True)
    fig.update_layout(title="Decision Rules for EMA indicator", xaxis={"title": "x"}, yaxis={"title": "Sell/Buy/Hold [$]"})
    fig.show()


def movingAverageConvergenceDivergence(values, params: MACDInvestorParams):
    macd = MACD(values, params.fastWindow, params.slowWindow, params.signal, True)
    return macd.macd()


def buyPredictionMACD(macd, params: MACDInvestorParams):
    """

    :param macd: Series with the values of the MACD
    :param params: MACD params
    """
    if macd < params.lowerBound:  # Buy linearly then with factor f
        return (macd - params.lowerBound) * params.maxBuy / 9*params.lowerBound
    else:
        return 0


def sellPredictionMACD(macd, params: MACDInvestorParams):
    """

    :param macd: Series with the values of the MACD
    :param params: MACD params
    """
    if macd > params.upperBound:  # Buy linearly then with factor f
        return (macd - params.upperBound) * params.maxSell / 0.9*params.upperBound
    else:
        return 0


def plotMACDDecisionRules(params: MACDInvestorParams):
    testMACD = np.arange(-4, 5, 0.1)
    buyPoints = []
    sellPoints = []
    for point in testMACD:
        buyPoints = np.append(buyPoints, buyPredictionMACD(point, params))
        sellPoints = np.append(sellPoints, sellPredictionMACD(point, params))

    fig = go.Figure()
    fig.add_trace(go.Scatter(name="BuyPoints", x=testMACD, y=buyPoints, fill='tozeroy'))
    fig.add_trace(go.Scatter(name="SellPoints", x=testMACD, y=-sellPoints, fill='tozeroy'))
    fig.update_layout(title="Decision Rules for MACD indicator", xaxis={"title": "MACD Value"}, yaxis={"title": "Sell/Buy/Hold [$]"})
    fig.show()

