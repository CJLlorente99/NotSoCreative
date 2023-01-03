from TAIndicators.adi import accDistIndexIndicator
from TAIndicators.adx import averageDirectionalMovementIndex
from classes.investorParamsClass import ADXInvestorParams
from TAIndicators.aroon import aroon
from classes.investorParamsClass import AroonInvestorParams
from TAIndicators.atr import averageTrueRange
from classes.investorParamsClass import ATRInvestorParams
from TAIndicators.bb import bollingerBands
from classes.investorParamsClass import BBInvestorParams
from TAIndicators.ma import movingAverageConvergenceDivergence
from classes.investorParamsClass import MACDInvestorParams
from TAIndicators.ma import exponentialMovingAverage
from classes.investorParamsClass import MAInvestorParams
from TAIndicators.obv import on_balance_volume
from TAIndicators.rsi import relativeStrengthIndex
from classes.investorParamsClass import RSIInvestorParams
from TAIndicators.stochasticRsi import stochasticRSI
from classes.investorParamsClass import StochasticRSIInvestorParams
import yfinance as yf
import numpy as np
import numpy.random as random
import pandas as pd
import json

# JSON for description
jsonString = []

# Retrive all data available from S&P500
df = yf.download("^GSPC", "2015-12-30", "2022-12-27")

# Calculate return
df["Return"] = df["Open"].shift(-1) - df["Open"]

# Calculate last return

df["ReturnBefore"] = df["Open"] - df["Open"].shift(1)

# Calculate log return
df["log(Open)"] = np.log(df["Open"])
df["LogReturn"] = df["log(Open)"].shift(-1) - df["log(Open)"]

# Calculate log return before
df["LogReturnBefore"] = df["log(Open)"] - df["log(Open)"].shift(1)

# Calculate log return before close
df["log(Close)"] = np.log(df["Close"])
df["LogReturnBeforeClose"] = df["log(Close)"] - df["log(Close)"].shift(1)

# Calculate class return
df["Class"] = [1 if df.Return[i] > 0 else 0 for i in range(len(df))]

# Accumulation Distribution Indicator (ADI)
# Kein veränderbar Parameter

print("Anfang ADI")

description = {"indicatorName": "adi", "dfName": "adi", "key": "acc_dist_index", "parameters": {}}
jsonString.append(json.dumps(description))

aux = pd.DataFrame(accDistIndexIndicator(df["High"], df["Low"], df["Close"], df["Volume"])["acc_dist_index"], columns=["adi"])
df = pd.concat([df, aux.shift()], axis=1)

print(df)

print("Schluss ADI")

# Average Directional Index (ADX)
# Zufällige Auswahl der Parameter

print("Anfang ADX")

windows = np.unique(random.randint(1, 51, 10))

i = 0
for window in windows:
	print(f"{i+1}/{str(len(windows))}")
	i += 1

	tag = "w" + str(window)

	description = {"indicatorName": "adx", "dfName": "adx_" + tag, "key": "adx", "parameters": {"window": int(window)}}
	jsonString.append(json.dumps(description))
	description = {"indicatorName": "adx", "dfName": "adx_neg_" + tag, "key": "adx_neg", "parameters": {"window": int(window)}}
	jsonString.append(json.dumps(description))
	description = {"indicatorName": "adx", "dfName": "adx_pos_" + tag, "key": "adx_pos", "parameters": {"window": int(window)}}
	jsonString.append(json.dumps(description))

	params = ADXInvestorParams(window)
	adx = averageDirectionalMovementIndex(df["High"], df["Low"], df["Close"], params)

	aux = pd.concat([pd.DataFrame(adx["adx"].values, columns=["adx_" + tag], index=adx["adx"].index),
					 pd.DataFrame(adx["adx_neg"].values, columns=["adx_neg_" + tag], index=adx["adx_neg"].index),
					 pd.DataFrame(adx["adx_pos"].values, columns=["adx_pos_" + tag], index=adx["adx_pos"].index)], axis=1)
	df = pd.concat([df, aux.shift()], axis=1)

print(df)

print("Schluss ADX")

# Aroon
# Zufällige Auswahl der Parameter

print("Anfang Aroon")

windows = np.unique(random.randint(1, 51, 10))

i = 0
for window in windows:
	print(f"{i + 1}/{str(len(windows))}")
	i += 1

	tag = "w" + str(window)

	description = {"indicatorName": "aroon", "dfName": "aroon_indicator_" + tag, "key": "aroon_indicator",
				   "parameters": {"window": int(window)}}
	jsonString.append(json.dumps(description))
	description = {"indicatorName": "aroon", "dfName": "aroon_down_" + tag, "key": "aroon_down",
				   "parameters": {"window": int(window)}}
	jsonString.append(json.dumps(description))
	description = {"indicatorName": "aroon", "dfName": "aroon_up_" + tag, "key": "aroon_up",
				   "parameters": {"window": int(window)}}

	params = AroonInvestorParams(window)
	ar = aroon(df["Close"], params)

	aux = pd.concat(
		[pd.DataFrame(ar["aroon_indicator"].values, columns=["aroon_indicator_" + tag], index=ar["aroon_indicator"].index),
		 pd.DataFrame(ar["aroon_down"].values, columns=["aroon_down_" + tag], index=ar["aroon_down"].index),
		 pd.DataFrame(ar["aroon_up"].values, columns=["aroon_up_" + tag], index=ar["aroon_up"].index)], axis=1)
	df = pd.concat([df, aux.shift()], axis=1)

	# description = {"indicatorName": "aroonReturnLogOpen", "dfName": "aroonReturnLogOpen_indicator_" + tag, "key": "aroon_indicator",
	# 			   "parameters": {"window": int(window)}}
	# jsonString.append(json.dumps(description))
	# description = {"indicatorName": "aroonReturnLogOpen", "dfName": "aroonReturnLogOpen_down_" + tag, "key": "aroon_down",
	# 			   "parameters": {"window": int(window)}}
	# jsonString.append(json.dumps(description))
	# description = {"indicatorName": "aroonReturnLogOpen", "dfName": "aroonReturnLogOpen_up_" + tag, "key": "aroon_up",
	# 			   "parameters": {"window": int(window)}}
	#
	# params = AroonInvestorParams(window)
	# ar = aroon(df["LogReturnBefore"], params)
	#
	# aux = pd.concat(
	# 	[pd.DataFrame(ar["aroon_indicator"].values, columns=["aroonReturnLogOpen_indicator_" + tag],
	# 				  index=ar["aroon_indicator"].index),
	# 	 pd.DataFrame(ar["aroon_down"].values, columns=["aroonReturnLogOpen_down_" + tag], index=ar["aroon_down"].index),
	# 	 pd.DataFrame(ar["aroon_up"].values, columns=["aroonReturnLogOpen_up_" + tag], index=ar["aroon_up"].index)], axis=1)
	# df = pd.concat([df, aux.shift()], axis=1)
	#
	# description = {"indicatorName": "aroonReturnLogClose", "dfName": "aroonReturnLogClose_indicator_" + tag, "key": "aroon_indicator",
	# 			   "parameters": {"window": int(window)}}
	# jsonString.append(json.dumps(description))
	# description = {"indicatorName": "aroonReturnLogClose", "dfName": "aroonReturnLogClose_down_" + tag, "key": "aroon_down",
	# 			   "parameters": {"window": int(window)}}
	# jsonString.append(json.dumps(description))
	# description = {"indicatorName": "aroonReturnLogClose", "dfName": "aroonReturnLogClose_up_" + tag, "key": "aroon_up",
	# 			   "parameters": {"window": int(window)}}
	#
	# params = AroonInvestorParams(window)
	# ar = aroon(df["LogReturnBeforeClose"], params)
	#
	# aux = pd.concat(
	# 	[pd.DataFrame(ar["aroon_indicator"].values, columns=["aroonReturnLogClose_indicator_" + tag],
	# 				  index=ar["aroon_indicator"].index),
	# 	 pd.DataFrame(ar["aroon_down"].values, columns=["aroonReturnLogClose_down_" + tag], index=ar["aroon_down"].index),
	# 	 pd.DataFrame(ar["aroon_up"].values, columns=["aroonReturnLogClose_up_" + tag], index=ar["aroon_up"].index)], axis=1)
	# df = pd.concat([df, aux.shift()], axis=1)

print(df)

print("Schluss Aroon")

# Average True Range (ATR)
# Zufällige Auswahl der Parameter

print("Anfang ATR")

windows = np.unique(random.randint(1, 51, 10))

i = 0
for window in windows:
	print(f"{i + 1}/{str(len(windows))}")
	i += 1

	tag = "w" + str(window)

	description = {"indicatorName": "atr", "dfName": "average_true_range_" + tag, "keys": "average_true_range",
				   "parameters": {"window": int(window)}}
	jsonString.append(json.dumps(description))

	params = ATRInvestorParams(window)
	atr = averageTrueRange(df["High"], df["Low"], df["Close"], params)

	aux = pd.DataFrame(atr["average_true_range"].values, columns=["average_true_range_" + tag], index=atr["average_true_range"].index)
	df = pd.concat([df, aux.shift()], axis=1)

print(df)

print("Schluss ATR")

# Bollinger Bands (BB)
# Zufällige Auswahl der Parameter

print("Anfang BB")

n = 20
windows = random.randint(1, 51, n)
stdDevs = random.uniform(0.5, 4, n)
X = [windows, stdDevs]

i = 0
for j in range(n):
	print(f"{i + 1}/{n}")
	i += 1

	window = X[0][j]
	stdDev = X[1][j]

	tag = "w" + str(window) + "_stdDev" + str(stdDev)

	description = {"indicatorName": "bb", "dfName": "bb_pband_" + tag, "key": "pband",
				   "parameters": {"window": int(window), "stdDev": float(stdDev)}}
	jsonString.append(json.dumps(description))
	description = {"indicatorName": "bb", "dfName": "bb_mavg_" + tag, "key": "mavg",
				   "parameters": {"window": int(window), "stdDev": float(stdDev)}}
	jsonString.append(json.dumps(description))
	description = {"indicatorName": "bb", "dfName": "bb_hband_" + tag, "key": "hband",
				   "parameters": {"window": int(window), "stdDev": float(stdDev)}}
	jsonString.append(json.dumps(description))
	description = {"indicatorName": "bb", "dfName": "bb_lband_" + tag, "key": "lband",
				   "parameters": {"window": int(window), "stdDev": float(stdDev)}}
	jsonString.append(json.dumps(description))

	params = BBInvestorParams(window, stdDev, 0, 0)
	bb = bollingerBands(df["Close"], params)

	aux = pd.concat(
		[pd.DataFrame(bb["pband"].values, columns=["bb_pband_" + tag], index=bb["pband"].index),
		 pd.DataFrame(bb["mavg"].values, columns=["bb_mavg_" + tag], index=bb["mavg"].index),
		 pd.DataFrame(bb["hband"].values, columns=["bb_hband_" + tag], index=bb["hband"].index),
		 pd.DataFrame(bb["lband"].values, columns=["bb_lband_" + tag], index=bb["lband"].index)], axis=1)
	df = pd.concat([df, aux.shift()], axis=1)

	# description = {"indicatorName": "bbReturnLogOpen", "dfName": "bbReturnLogOpen_pband_" + tag, "key": "pband",
	# 			   "parameters": {"window": int(window), "stdDev": float(stdDev)}}
	# jsonString.append(json.dumps(description))
	# description = {"indicatorName": "bbReturnLogOpen", "dfName": "bbReturnLogOpen_mavg_" + tag, "key": "mavg",
	# 			   "parameters": {"window": int(window), "stdDev": float(stdDev)}}
	# jsonString.append(json.dumps(description))
	# description = {"indicatorName": "bbReturnLogOpen", "dfName": "bbReturnLogOpen_hband_" + tag, "key": "hband",
	# 			   "parameters": {"window": int(window), "stdDev": float(stdDev)}}
	# jsonString.append(json.dumps(description))
	# description = {"indicatorName": "bbReturnLogOpen", "dfName": "bbReturnLogOpen_lband_" + tag, "key": "lband",
	# 			   "parameters": {"window": int(window), "stdDev": float(stdDev)}}
	# jsonString.append(json.dumps(description))
	#
	# params = BBInvestorParams(window, stdDev, 0, 0)
	# bb = bollingerBands(df["LogReturnBefore"], params)
	#
	# aux = pd.concat(
	# 	[pd.DataFrame(bb["pband"].values, columns=["bbReturnLogOpen_pband_" + tag], index=bb["pband"].index),
	# 	 pd.DataFrame(bb["mavg"].values, columns=["bbReturnLogOpen_mavg_" + tag], index=bb["mavg"].index),
	# 	 pd.DataFrame(bb["hband"].values, columns=["bbReturnLogOpen_hband_" + tag], index=bb["hband"].index),
	# 	 pd.DataFrame(bb["lband"].values, columns=["bbReturnLogOpen_lband_" + tag], index=bb["lband"].index)], axis=1)
	# df = pd.concat([df, aux.shift()], axis=1)
	#
	# description = {"indicatorName": "bbReturnLogClose", "dfName": "bbReturnLogClose_pband_" + tag, "key": "pband",
	# 			   "parameters": {"window": int(window), "stdDev": float(stdDev)}}
	# jsonString.append(json.dumps(description))
	# description = {"indicatorName": "bbReturnLogClose", "dfName": "bbReturnLogClose_mavg_" + tag, "key": "mavg",
	# 			   "parameters": {"window": int(window), "stdDev": float(stdDev)}}
	# jsonString.append(json.dumps(description))
	# description = {"indicatorName": "bbReturnLogClose", "dfName": "bbReturnLogClose_hband_" + tag, "key": "hband",
	# 			   "parameters": {"window": int(window), "stdDev": float(stdDev)}}
	# jsonString.append(json.dumps(description))
	# description = {"indicatorName": "bbReturnLogClose", "dfName": "bbReturnLogClose_lband_" + tag, "key": "lband",
	# 			   "parameters": {"window": int(window), "stdDev": float(stdDev)}}
	# jsonString.append(json.dumps(description))
	#
	# params = BBInvestorParams(window, stdDev, 0, 0)
	# bb = bollingerBands(df["LogReturnBeforeClose"], params)
	#
	# aux = pd.concat(
	# 	[pd.DataFrame(bb["pband"].values, columns=["bbReturnLogClose_pband_" + tag], index=bb["pband"].index),
	# 	 pd.DataFrame(bb["mavg"].values, columns=["bbReturnLogClose_mavg_" + tag], index=bb["mavg"].index),
	# 	 pd.DataFrame(bb["hband"].values, columns=["bbReturnLogClose_hband_" + tag], index=bb["hband"].index),
	# 	 pd.DataFrame(bb["lband"].values, columns=["bbReturnLogClose_lband_" + tag], index=bb["lband"].index)], axis=1)
	# df = pd.concat([df, aux.shift()], axis=1)

print(df)

print("Schluss BB")

# Moving Average Convergence Divergence (MACD)
# Zufällige Auswahl der Parameter

print("Anfang MACD")

n = 40
fastWindows = random.randint(1, 20, n)
slowWindows = random.randint(5, 51, n)
signals = random.randint(1, 51, n)
X = [fastWindows, slowWindows, signals]

i = 0
for j in range(n):
	print(f"{i + 1}/{n}")
	i += 1

	fastWindow = X[0][j]
	slowWindow = X[1][j]
	signal = X[2][j]

	tag = "fW" + str(fastWindow) + "_sW" + str(slowWindow) + "_signal" + str(signal)

	description = {"indicatorName": "macd", "dfName": "macd_" + tag, "key": "macd",
				   "parameters": {"fastWindow": int(fastWindow), "slowWindow": int(slowWindow), "signal": int(signal)}}
	jsonString.append(json.dumps(description))
	description = {"indicatorName": "macd", "dfName": "macd_signal_" + tag, "key": "signal",
				   "parameters": {"fastWindow": int(fastWindow), "slowWindow": int(slowWindow), "signal": int(signal)}}
	jsonString.append(json.dumps(description))
	description = {"indicatorName": "macd", "dfName": "macd_diff" + tag, "key": "macd-signal",
				   "parameters": {"fastWindow": int(fastWindow), "slowWindow": int(slowWindow), "signal": int(signal)}}
	jsonString.append(json.dumps(description))

	params = MACDInvestorParams(None, None, fastWindow, slowWindow, signal)
	macd = movingAverageConvergenceDivergence(df["Close"], params)

	aux = pd.concat(
		[pd.DataFrame(macd["macd"].values, columns=["macd_" + tag], index=macd["macd"].index),
		 pd.DataFrame(macd["signal"].values, columns=["macd_signal_" + tag], index=macd["signal"].index),
		 pd.DataFrame((macd["macd"] - macd["signal"]).values, columns=["macd_diff" + tag], index=macd["macd"].index)], axis=1)
	df = pd.concat([df, aux.shift()], axis=1)

	# description = {"indicatorName": "macdReturnLogOpen", "dfName": "macdReturnLogOpen_" + tag, "key": "macd",
	# 			   "parameters": {"fastWindow": int(fastWindow), "slowWindow": int(slowWindow), "signal": int(signal)}}
	# jsonString.append(json.dumps(description))
	# description = {"indicatorName": "macdReturnLogOpen", "dfName": "macdReturnLogOpen_signal_" + tag, "key": "signal",
	# 			   "parameters": {"fastWindow": int(fastWindow), "slowWindow": int(slowWindow), "signal": int(signal)}}
	# jsonString.append(json.dumps(description))
	# description = {"indicatorName": "macdReturnLogOpen", "dfName": "macdReturnLogOpen_diff" + tag, "key": "macd-signal",
	# 			   "parameters": {"fastWindow": int(fastWindow), "slowWindow": int(slowWindow), "signal": int(signal)}}
	# jsonString.append(json.dumps(description))
	#
	# params = MACDInvestorParams(None, None, fastWindow, slowWindow, signal)
	# macd = movingAverageConvergenceDivergence(df["LogReturnBefore"], params)
	#
	# aux = pd.concat(
	# 	[pd.DataFrame(macd["macd"].values, columns=["macdReturnLogOpen_" + tag], index=macd["macd"].index),
	# 	 pd.DataFrame(macd["signal"].values, columns=["macdReturnLogOpen_signal_" + tag], index=macd["signal"].index),
	# 	 pd.DataFrame((macd["macd"] - macd["signal"]).values, columns=["macdReturnLogOpen_diff" + tag], index=macd["macd"].index)],
	# 	axis=1)
	# df = pd.concat([df, aux.shift()], axis=1)
	#
	# description = {"indicatorName": "macdReturnLogClose", "dfName": "macdReturnLogClose_" + tag, "key": "macd",
	# 			   "parameters": {"fastWindow": int(fastWindow), "slowWindow": int(slowWindow), "signal": int(signal)}}
	# jsonString.append(json.dumps(description))
	# description = {"indicatorName": "macdReturnLogClose", "dfName": "macdReturnLogClose_signal_" + tag, "key": "signal",
	# 			   "parameters": {"fastWindow": int(fastWindow), "slowWindow": int(slowWindow), "signal": int(signal)}}
	# jsonString.append(json.dumps(description))
	# description = {"indicatorName": "macdReturnLogClose", "dfName": "macdReturnLogClose_diff" + tag, "key": "macd-signal",
	# 			   "parameters": {"fastWindow": int(fastWindow), "slowWindow": int(slowWindow), "signal": int(signal)}}
	# jsonString.append(json.dumps(description))
	#
	# params = MACDInvestorParams(None, None, fastWindow, slowWindow, signal)
	# macd = movingAverageConvergenceDivergence(df["LogReturnBeforeClose"], params)
	#
	# aux = pd.concat(
	# 	[pd.DataFrame(macd["macd"].values, columns=["macdReturnLogClose_" + tag], index=macd["macd"].index),
	# 	 pd.DataFrame(macd["signal"].values, columns=["macdReturnLogClose_signal_" + tag], index=macd["signal"].index),
	# 	 pd.DataFrame((macd["macd"] - macd["signal"]).values, columns=["macdReturnLogClose_diff" + tag],
	# 				  index=macd["macd"].index)],
	# 	axis=1)
	# df = pd.concat([df, aux.shift()], axis=1)

print(df)

print("Schluss MACD")

# Exponential Moving Average (EMA)
# Zufällige Auswahl der Parameter

print("Anfang EMA")

windows = np.unique(random.randint(1, 101, 20))

i = 0
for window in windows:
	print(f"{i + 1}/{str(len(windows))}")
	i += 1

	tag = "w" + str(window)

	description = {"indicatorName": "ema", "dfName": "ema_" + tag, "key": "ema",
				   "parameters": {"window": int(window)}}
	jsonString.append(json.dumps(description))

	params = MAInvestorParams(None, None, window)
	ema = exponentialMovingAverage(df["Close"], params)

	aux = pd.DataFrame(ema["ema"].values, columns=["ema_" + tag], index=ema["ema"].index)
	df = pd.concat([df, aux.shift()], axis=1)

	# description = {"indicatorName": "emaReturnLogOpen", "dfName": "emaReturnLogOpen_" + tag, "key": "ema",
	# 			   "parameters": {"window": int(window)}}
	# jsonString.append(json.dumps(description))
	#
	# params = MAInvestorParams(None, None, window)
	# ema = exponentialMovingAverage(df["LogReturnBefore"], params)
	#
	# aux = pd.DataFrame(ema["ema"].values, columns=["emaReturnLogOpen_" + tag], index=ema["ema"].index)
	# df = pd.concat([df, aux.shift()], axis=1)
	#
	# description = {"indicatorName": "emaReturnLogClose", "dfName": "emaReturnLogClose_" + tag, "key": "ema",
	# 			   "parameters": {"window": int(window)}}
	# jsonString.append(json.dumps(description))
	#
	# params = MAInvestorParams(None, None, window)
	# ema = exponentialMovingAverage(df["LogReturnBeforeClose"], params)
	#
	# aux = pd.DataFrame(ema["ema"].values, columns=["emaReturnLogClose_" + tag], index=ema["ema"].index)
	# df = pd.concat([df, aux.shift()], axis=1)

print(df)

print("Schluss EMA")

# On Balance Volume (OBV)
# Kein veränderbar Parameter

print("Anfang OBV")

description = {"indicatorName": "obv", "dfName": "obv", "key": "on_balance_volume", "parameters": {}}
jsonString.append(json.dumps(description))

obv = on_balance_volume(df["Close"], df["Volume"])
aux = pd.DataFrame(obv["on_balance_volume"].values, columns=["obv"], index=obv["on_balance_volume"].index)
df = pd.concat([df, aux.shift()], axis=1)

print(df)

print("Schluss OBV")

# Relative Strength Indicator (RSI)
# Zufällige Auswahl der Parameter

print("Anfang RSI")

windows = np.unique(random.randint(1, 51, 10))

i = 0
for window in windows:
	print(f"{i + 1}/{str(len(windows))}")
	i += 1

	tag = "w" + str(window)

	description = {"indicatorName": "rsi", "dfName": "rsi_" + tag, "key": "rsi",
				   "parameters": {"window": int(window)}}
	jsonString.append(json.dumps(description))

	params = RSIInvestorParams(0, 0, window)
	rsi = relativeStrengthIndex(df["Close"], params)

	aux = pd.DataFrame(rsi["rsi"].values, columns=["rsi_" + tag], index=rsi["rsi"].index)
	df = pd.concat([df, aux.shift()], axis=1)

	# description = {"indicatorName": "rsiReturnLogOpen", "dfName": "rsiReturnLogOpen_" + tag, "key": "rsi",
	# 			   "parameters": {"window": int(window)}}
	# jsonString.append(json.dumps(description))
	#
	# params = RSIInvestorParams(0, 0, window)
	# rsi = relativeStrengthIndex(df["LogReturnBefore"], params)
	#
	# aux = pd.DataFrame(rsi["rsi"].values, columns=["rsiReturnLogOpen_" + tag], index=rsi["rsi"].index)
	# df = pd.concat([df, aux.shift()], axis=1)
	#
	# description = {"indicatorName": "rsiReturnLogClose", "dfName": "rsiReturnLogClose_" + tag, "key": "rsi",
	# 			   "parameters": {"window": int(window)}}
	# jsonString.append(json.dumps(description))
	#
	# params = RSIInvestorParams(0, 0, window)
	# rsi = relativeStrengthIndex(df["LogReturnBeforeClose"], params)
	#
	# aux = pd.DataFrame(rsi["rsi"].values, columns=["rsiReturnLogClose_" + tag], index=rsi["rsi"].index)
	# df = pd.concat([df, aux.shift()], axis=1)

print(df)

print("Schluss RSI")

# Stochastic RSI (SRSI)
# Zufällige Auswahl der Parameter

print("Anfang SRSI")

n = 40
windows = random.randint(1, 51, n)
smooth1s = random.randint(1, 51, n)
smooth2s = random.randint(1, 51, n)
X = [windows, smooth1s, smooth2s]

i = 0
for j in range(n):
	print(f"{i + 1}/{n}")
	i += 1

	window = X[0][j]
	smooth1 = X[1][j]
	smooth2 = X[2][j]

	tag = "w" + str(window) + "_s1" + str(smooth1) + "_s2" + str(smooth2)

	description = {"indicatorName": "stochRsi", "dfName": "stochRsi_stochrsi_" + tag, "key": "stochrsi",
				   "parameters": {"window": int(window), "smooth1": int(smooth1), "smooth2": int(smooth2)}}
	jsonString.append(json.dumps(description))
	description = {"indicatorName": "stochRsi", "dfName": "stochRsi_k_" + tag, "key": "k",
				   "parameters": {"window": int(window), "smooth1": int(smooth1), "smooth2": int(smooth2)}}
	jsonString.append(json.dumps(description))
	description = {"indicatorName": "stochRsi", "dfName": "stochRsi_d_" + tag, "key": "d",
				   "parameters": {"window": int(window), "smooth1": int(smooth1), "smooth2": int(smooth2)}}
	jsonString.append(json.dumps(description))

	params = StochasticRSIInvestorParams(window, smooth1, smooth2)
	stochRsi = stochasticRSI(df["Close"], params)

	aux = pd.concat(
		[pd.DataFrame(stochRsi["stochrsi"].values, columns=["stochRsi_stochrsi_" + tag], index=stochRsi["stochrsi"].index),
		 pd.DataFrame(stochRsi["k"].values, columns=["stochRsi_k_" + tag], index=stochRsi["k"].index),
		 pd.DataFrame(stochRsi["d"].values, columns=["stochRsi_d_" + tag], index=stochRsi["d"].index)], axis=1)
	df = pd.concat([df, aux.shift()], axis=1)

	# description = {"indicatorName": "stochRsiReturnLogOpen", "dfName": "stochRsiReturnLogOpen_stochrsi_" + tag, "key": "stochrsi",
	# 			   "parameters": {"window": int(window), "smooth1": int(smooth1), "smooth2": int(smooth2)}}
	# jsonString.append(json.dumps(description))
	# description = {"indicatorName": "stochRsiReturnLogOpen", "dfName": "stochRsiReturnLogOpen_k_" + tag, "key": "k",
	# 			   "parameters": {"window": int(window), "smooth1": int(smooth1), "smooth2": int(smooth2)}}
	# jsonString.append(json.dumps(description))
	# description = {"indicatorName": "stochRsiReturnLogOpen", "dfName": "stochRsiReturnLogOpen_d_" + tag, "key": "d",
	# 			   "parameters": {"window": int(window), "smooth1": int(smooth1), "smooth2": int(smooth2)}}
	# jsonString.append(json.dumps(description))
	#
	# params = StochasticRSIInvestorParams(window, smooth1, smooth2)
	# stochRsi = stochasticRSI(df["LogReturnBefore"], params)
	#
	# aux = pd.concat(
	# 	[pd.DataFrame(stochRsi["stochrsi"].values, columns=["stochRsiReturnLogOpen_stochrsi_" + tag],
	# 				  index=stochRsi["stochrsi"].index),
	# 	 pd.DataFrame(stochRsi["k"].values, columns=["stochRsiReturnLogOpen_k_" + tag], index=stochRsi["k"].index),
	# 	 pd.DataFrame(stochRsi["d"].values, columns=["stochRsiReturnLogOpen_d_" + tag], index=stochRsi["d"].index)], axis=1)
	# df = pd.concat([df, aux.shift()], axis=1)
	#
	# description = {"indicatorName": "stochRsiReturnLogClose", "dfName": "stochRsiReturnLogClose_stochrsi_" + tag,
	# 			   "key": "stochrsi",
	# 			   "parameters": {"window": int(window), "smooth1": int(smooth1), "smooth2": int(smooth2)}}
	# jsonString.append(json.dumps(description))
	# description = {"indicatorName": "stochRsiReturnLogClose", "dfName": "stochRsiReturnLogClose_k_" + tag, "key": "k",
	# 			   "parameters": {"window": int(window), "smooth1": int(smooth1), "smooth2": int(smooth2)}}
	# jsonString.append(json.dumps(description))
	# description = {"indicatorName": "stochRsiReturnLogClose", "dfName": "stochRsiReturnLogClose_d_" + tag, "key": "d",
	# 			   "parameters": {"window": int(window), "smooth1": int(smooth1), "smooth2": int(smooth2)}}
	# jsonString.append(json.dumps(description))
	#
	# params = StochasticRSIInvestorParams(window, smooth1, smooth2)
	# stochRsi = stochasticRSI(df["LogReturnBeforeClose"], params)
	#
	# aux = pd.concat(
	# 	[pd.DataFrame(stochRsi["stochrsi"].values, columns=["stochRsiReturnLogClose_stochrsi_" + tag],
	# 				  index=stochRsi["stochrsi"].index),
	# 	 pd.DataFrame(stochRsi["k"].values, columns=["stochRsiReturnLogClose_k_" + tag], index=stochRsi["k"].index),
	# 	 pd.DataFrame(stochRsi["d"].values, columns=["stochRsiReturnLogClose_d_" + tag], index=stochRsi["d"].index)],
	# 	axis=1)
	# df = pd.concat([df, aux.shift()], axis=1)

print(df)

print("Schluss SRSI")

print(df.shape)

# Save to csv
df.to_csv("featureSelectionDataset.csv")

# Save json
with open('data.json', 'w') as f:
	json.dump(jsonString, f)
