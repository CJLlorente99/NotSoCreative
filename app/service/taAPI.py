from ta.volume import AccDistIndexIndicator, OnBalanceVolumeIndicator
from ta.trend import ADXIndicator, AroonIndicator, MACD, EMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.momentum import RSIIndicator, StochRSIIndicator

def accDistIndexIndicator(high, low, close, volume):
	"""
	Function that returns the ADI values
	:param high: Market high value
	:param low: Market low value
	:param close: Market close value
	:param volume: Market volume value
	:return: dict with only ["acc_dist_index"] as keys
	"""
	adi = AccDistIndexIndicator(high, low, close, volume, True)
	return {"acc_dist_index" : adi.acc_dist_index()}

def averageDirectionalMovementIndex(high, low, close, window):
	"""
	Function that returns the ADX values
	:param high: Market high value
	:param low: Market low value
	:param close: Market close value
	:param window: ADX parameter
	:return: dict with the following keys ["adx", "adx_neg", "adx_pos"]
	"""
	adx = ADXIndicator(high, low, close, window, True)
	return {"adx" : adx.adx(), "adx_neg" : adx.adx_neg(), "adx_pos": adx.adx_pos()}

def aroon(close, window):
	"""
	Function that returns the different values related to the Aroon Indicator
	:param close: market close value
	:param window: Aroon parameter
	:return: dict with the following keys ["aroon_indicator", "aroon_down", "aroon_up"]
	"""
	aroon = AroonIndicator(close, window, True)
	return {"aroon_indicator" : aroon.aroon_indicator(), "aroon_down" : aroon.aroon_down(), "aroon_up": aroon.aroon_up()}

def averageTrueRange(high, low, close, window):
	"""
	Function that returns the values for ATR indicator
	:param high: Market high value
	:param low: Market low value
	:param close: Market close value
	:param window: ATR parameter
	:return: dict with the following keys ["average_true_range"]
	"""
	obv = AverageTrueRange(high, low, close, window, True)
	return {"average_true_range" : obv.average_true_range()}

def bollingerBands(close, window, stdDev):
	"""
	Function that calcualtes the bollinger bands
	:param close: Close value from the stock market series
	:param window: BB parameter
	:param stdDev: BB parameter
	:return: dict with the following keys ["pband", "mavg", "hband", "lband"]
	"""
	bb = BollingerBands(close, window, stdDev, fillna=True)
	return {"pband": bb.bollinger_pband(), "mavg": bb.bollinger_mavg(), "hband": bb.bollinger_hband(), "lband": bb.bollinger_lband()}

def movingAverageConvergenceDivergence(close, fastWindow, slowWindow, signal):
	"""
	Function that calculates the different returns of the MACD indicator
	:param close: Close market values
	:param fastWindow: MACD parameter
	:param slowWindow: MACD parameter
	:param signal: MACD parameter
	:return: dict with the following keys ["macd", "signal"]
	"""
	macd = MACD(close, fastWindow, slowWindow, signal, True)
	diff = macd.macd() - macd.macd_signal()
	return {"macd" : macd.macd(), "signal" : macd.macd_signal(), 'diff': diff}

def exponentialMovingAverage(close, window):
	"""
	Function that calculates the different returns of the EMA indicator
	:param close: Close market values
	:param window: EMA parameter
	:return:
	"""
	ema = EMAIndicator(close, window, True)
	return {"ema" : ema.ema_indicator()}

def on_balance_volume(close, volume):
	"""
	Function that calculates the different returns for the OBV indicator
	:param close: Market close
	:param volume: Market volume
	:return: dict with the following keys ["on_balance_volume"]
	"""
	obv = OnBalanceVolumeIndicator(close, volume, True)
	return {"on_balance_volume" : obv.on_balance_volume()}

def relativeStrengthIndex(close, window):
	"""
	Function that calculates the RSI values
	:param close:
	:param window: RSI parameter
	:return dict with the following keys ["rsi"]
	"""
	rsi = RSIIndicator(close, window, True)
	return {"rsi": rsi.rsi()}

def stochasticRSI(close, window, smooth1, smooth2):
	"""
	Function that calculates the stochastic RSI values
	:param close: Market close value
	:param window: StochRSI parameter
	:param smooth1: StochRSI parameter
	:param smooth2: StochRSI parameter
	:return dict with the following keys ["stochrsi", "k", "d"]
	"""
	stochRsi = StochRSIIndicator(close, window, smooth1, smooth2, True)
	return {"stochrsi": stochRsi.stochrsi(), "k": stochRsi.stochrsi_k(), "d": stochRsi.stochrsi_d()}