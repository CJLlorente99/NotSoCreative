from ta.volatility import BollingerBands
from investorParamsClass import BBInvestorParams


def bollingerBands(values, params: BBInvestorParams):
    # Important to take into account that pband works as follows:
    # < 0 => below lower band; > 1 over upper band; = 0.5 in the middle of the band
    bb = BollingerBands(values, params.window, params.stdDev, fillna=True)
    return bb.bollinger_pband()


def buyPredictionBB(bb, params: BBInvestorParams):
    if bb < params.lowerBound:
        return max((params.lowerBound - bb) * params.buyingSlope, params.maxBuy)
    else:
        return 0

def sellPredictionBB(bb, params: BBInvestorParams):
    if bb > params.upperBound:
        return max((bb - params.upperBound) * params.sellingSlope, params.maxSell)
    else:
        return 0



