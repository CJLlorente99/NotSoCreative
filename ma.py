import math
from ta.trend import SMAIndicator, EMAIndicator, MACD
from investorParamsClass import MAInvestorParams, MACDInvestorParams, GradientQuarter
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def simpleMovingAverage(values, params: MAInvestorParams):
    """

    :param values:
    :param params:
    :return:
    """
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
    """

    :param params:
    """
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
    """

    :param values:
    :param params:
    :return:
    """
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
    """

    :param params:
    """
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
    """

    :param values:
    :param params:
    :return:
    """
    macd = MACD(values, params.fastWindow, params.slowWindow, params.signal, True)
    return {"macd" : macd.macd(), "signal" : macd.macd_signal()}


def buyPredictionMACD(macdDict, params: MACDInvestorParams):
    """
    Function that is used to predict next day buying behavior
    :param macdDict: Dict with the values of the MACD (signal and macd)
    :param params: MACD params
    """
    # Unpackage macdDict
    macd = macdDict["macd"]
    signal = macdDict["signal"]
    type = params.type

    # Calculate gradients of the macd value
    firstGradient = np.gradient(macd.values)
    secondGradient = np.gradient(firstGradient)

    # Calculate gradient of the signal
    firstGradientSignal = np.gradient(signal.values)

    # Depending on the strategy, act accordingly
    if type == "grad":
        if params.buyGradients.lowerBoundGradient < firstGradient[
            -1] < params.buyGradients.upperBoundGradient and params.buyGradients.lowBoundSquareGradient < secondGradient[-1]:
            return firstGradient, secondGradient, params.maxBuy * math.tanh(
                params.a * (secondGradient[-1] - params.buyGradients.lowBoundSquareGradient) ** params.b)
        return firstGradient, secondGradient, 0
    elif type == "grad_crossZero":
        if firstGradient[-1] > 0 and -0.005 < macd.values[-1] < 0.005:
            return firstGradient, secondGradient, params.maxBuy * math.tanh(params.a * firstGradient[-1] ** params.b)
        return firstGradient, secondGradient, 0
    elif type == "grad_crossSignal":
        if abs(macd.values[-1] - signal.values[-1]) < 0.01 and firstGradient[-1] > firstGradientSignal[-1]:
            return firstGradient, secondGradient, params.maxBuy * math.tanh(params.a * firstGradient[-1] ** params.b)
        return firstGradient, secondGradient, 0


def sellPredictionMACD(macdDict, params: MACDInvestorParams):
    """
    Function that is used to predict next day selling behavior
    :param macdDict: Dict with the values of the MACD (signal and macd)
    :param params: MACD params
    """
    # Unpackage macdDict
    macd = macdDict["macd"]
    signal = macdDict["signal"]
    type = params.type

    # Calculate gradients of the macd value
    firstGradient = np.gradient(macd.values)
    secondGradient = np.gradient(firstGradient)

    # Calculate gradient of the signal
    firstGradientSignal = np.gradient(signal.values)

    # Depending on the strategy, act accordingly
    if type == "grad":
        if params.sellGradients.lowerBoundGradient < firstGradient[-1] < params.sellGradients.upperBoundGradient and params.sellGradients.lowBoundSquareGradient > secondGradient[-1]:
            return params.maxSell * math.tanh(params.a * (params.sellGradients.lowBoundSquareGradient - secondGradient[-1]) ** params.b)
        else:
            return 0
    elif type == "grad_crossZero":
        if firstGradient[-1] < 0 and -0.005 < macd.values[-1] < 0.005:
            return params.maxSell * math.tanh(params.a * (-firstGradient[-1]) ** params.b)
        return 0
    elif type == "grad_crossSignal":
        if abs(macd.values[-1] - signal.values[-1]) < 0.01 and firstGradient[-1] < firstGradientSignal[-1]:
            return params.maxSell * math.tanh(params.a * (-firstGradient[-1]) ** params.b)
        return 0


def plotMACDDecisionRules(params: MACDInvestorParams):
    """
    Function that plots the decision rule used
    :param params: MACD params
    """
    testMACDdata = pd.Series(np.random.normal(0, 1, 200))
    buyPoints = []
    sellPoints = []
    for i in range(len(testMACDdata) - params.fastWindow):
        testMACD = movingAverageConvergenceDivergence(testMACDdata[0:i+2], params)
        firstGradient, secondGradient, buyPoint = buyPredictionMACD(testMACD, params)
        buyPoints = np.append(buyPoints, buyPoint)
        sellPoints = np.append(sellPoints, sellPredictionMACD(testMACD, params))

    x = np.arange(len(testMACDdata))

    fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": True}], [{"secondary_y": True}]])
    fig.add_trace(go.Scatter(name="Stock data", x=x, y=testMACDdata.values), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(name="MACDValues", x=x[3:], y=testMACD["macd"].values), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(name="BuyPoints", x=x[4:], y=buyPoints, fill='tozeroy'), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(name="SellPoints", x=x[4:], y=-sellPoints, fill='tozeroy'), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(name="FirstGradient", x=x[3:], y=firstGradient), row=2, col=1, secondary_y=False)
    if params.type == "grad":
        fig.add_trace(go.Scatter(name="SecondGradient", x=x[3:], y=secondGradient), row=2, col=1, secondary_y=True)
        fig.update_layout(title="Decision Rules for MACD indicator (Grad)", xaxis={"title": "x"},
                          yaxis={"title": "Sell/Buy/Hold [$]"})
    elif params.type == "grad_crossZero":
        fig.update_layout(title="Decision Rules for MACD indicator (Grad+CrossZero)", xaxis={"title": "x"},
                          yaxis={"title": "Sell/Buy/Hold [$]"})
    elif params.type == "grad_crossSignal":
        fig.add_trace(go.Scatter(name="SignalValues", x=x[3:], y=testMACD["signal"].values), row=1, col=1, secondary_y=False)
        fig.update_layout(title="Decision Rules for MACD indicator (Grad+CrossSignal)", xaxis={"title": "x"},
                          yaxis={"title": "Sell/Buy/Hold [$]"})
    fig.show()

