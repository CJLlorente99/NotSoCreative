from ta.volatility import BollingerBands
from investorParamsClass import BBInvestorParams
import plotly.graph_objects as go
import numpy as np

def bollingerBands(values, params: BBInvestorParams):
    # Important to take into account that pband works as follows:
    # < 0 => below lower band; > 1 over upper band; = 0.5 in the middle of the band
    bb = BollingerBands(values, params.window, params.stdDev, fillna=True)
    return bb.bollinger_pband()


def buyPredictionBB(bb, params: BBInvestorParams):
    if bb < params.lowerBound:
        return min((params.lowerBound - bb) * params.buyingSlope, params.maxBuy)
    else:
        return 0


def sellPredictionBB(bb, params: BBInvestorParams):
    if bb > params.upperBound:
        return min((bb - params.upperBound) * params.sellingSlope, params.maxSell)
    else:
        return 0


def plotBBDecisionRules(params: BBInvestorParams):
    testBB = np.arange(-2, 3, 0.1)
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
