import datetime
import os
import urllib.request
import yfinance as yf
import pandas as pd
from pandas.tseries.offsets import CDay
from pandas.tseries.holiday import USFederalHolidayCalendar
import time
from strategies.bah import BaH
from strategies.bia import BIA
from strategies.randomStrategy import Random
from strategies.wia import WIA
from strategies.ca import CA
from strategies.idle import Idle
from strategies.rsi import RSI
from strategies.bb import BB
from strategies.lstmConfidenceOpenClose import LSTMConfidenceOpenClose
from logManager.logManager import LogManager
from jsonManagement.inversionStrategyJSONAPI import *
from taAPI import *
import numpy as np
"""
The .bat (Windows) or .sh (Mac/Linux) should install all needed packages
The script should do the following.
1) safety procedures (operations can be done/necessary files are there)
2) get date
3) check new data can be retrieved from S&P 500 (check S&P 500 opens today)
4) open csv file and read inversor status
5) calculate needed data
6) run algorithm (generate part of the csv related to the strategies)
7) write csv file with dataframe (also append date and indicator data)
"""

# Constants
logFile = 'C:/Users/carlo/OneDrive/Documentos/Universidad/MUIT/Segundo/Data Science II/NotSoCreative/app/script/ACHTUNGScriptData/log.txt'
logManager = LogManager(logFile)

csvDataFileHidden = 'C:/Users/carlo/OneDrive/Documentos/Universidad/MUIT/Segundo/Data Science II/NotSoCreative/app/script/ACHTUNGScriptData/myData.csv'

csvDataFile = 'C:/Users/carlo/OneDrive/Documentos/Universidad/MUIT/Segundo/Data Science II/NotSoCreative/app/script/scriptData/myData.csv'

jsonFile = 'C:/Users/carlo/OneDrive/Documentos/Universidad/MUIT/Segundo/Data Science II/NotSoCreative/app/script/ACHTUNGScriptData\strategies.json'

openingHour = 15
openingMinute = 30
closingHour = 22
closingMinute = 0

def main():
	# 1) safety procedures (operations can be done/necessary files are there)
	if False in runSafetyProcedures().values():
		pass
		# Launch message to user depending on the error

	# 2) get date
	# dateToday = datetime.datetime.now()
	# now = datetime.datetime.now()
	dateToday = datetime.datetime(2022, 12, 28)
	now = datetime.datetime(2022, 12, 28, openingHour, openingMinute+20, 0)

	openingTimeSP500 = now.replace(hour=openingHour, minute=openingMinute, second=0)
	closingTimeSP500 = now.replace(hour=closingHour, minute=closingMinute, second=0)

	operation = 'None'
	if openingTimeSP500 < now < closingTimeSP500:
		operation = 'Morning'
	elif now > closingTimeSP500:
		operation = 'Afternoon'

	# 3) check new data can be retrieved from S&P 500 (check S&P 500 opens today)
	out = retrieveStockData(dateToday, operation)
	data = out['data']
	if not out['status']:
		return
		# Launch message to user

	# 4) open csv file and read inversor status
	out = retrieveCSVData()
	investorinfo = out['data']
	if not len(investorinfo):  # No inversor info because it didn't exist
		investorinfo = pd.DataFrame()

	if not out['status']:
		pass
		# Launch message to user

	# Check we have not already performed this operation (same day and moment)
	if len(investorinfo) != 0:
		out = checkNoRepeat(investorinfo, operation, dateToday)
		if not out['status']:
			# We've already done this operation
			return

	# 5) calculate needed data
	jsonManager = JsonStrategyManager(jsonFile)
	inputs = jsonManager.listInputs()

	out = calculateInputs(data, inputs, operation)
	inputsDf = out['data']
	if not out['status']:
		pass
		# Launch message to user

	# 6) run algorithm (generate part of the csv related to the strategies)
	# investorStrategy | MoneyInvested | MoneyNotInvested | MoneyBoughtToday | MoneySoldToday | PerBoughtTomorrow |
	# PerSoldTomorrow | TotalPortfolioValue
	out = runStrategies(dateToday, operation, investorinfo, inputsDf, jsonManager.listStrategies())
	aux = out['data']

	if not out['status']:
		pass
		# Launch message to user

	# 7) write csv file with dataframe (also append nDay and other special columns)
	if len(aux) != 0:
		try:
			nextNDay = investorinfo['nDay'].values[-1] + 1
			aux['nDay'] = nextNDay * np.ones(len(aux))
		except:
			aux['nDay'] = np.zeros(len(aux))

		investorinfo = pd.concat([investorinfo, aux])
		investorinfo.to_csv(csvDataFileHidden, index_label=['Date'])
		logManager.writeLog('INFO', 'csvDateFile writing OK')

		try:
			investorinfo.to_csv(csvDataFile, index_label=['Date'])
			logManager.writeLog('INFO', 'csvDateFile writing OK')
		except:
			logManager.writeLog('ERROR', 'csvDateFile writing ERROR')

def runSafetyProcedures():
	status = False
	errMsg = ''
	# Check needed paths exist
	if not os.path.exists("ACHTUNGScriptData"):
		os.mkdir("ACHTUNGScriptData")
		if not os.path.exists(csvDataFileHidden):
			f = open(csvDataFileHidden, 'w')
			f.close()

	if not os.path.exists("scriptData"):
		os.mkdir("scriptData")
		f = open(csvDataFile, 'w')
		f.close()

	status = True
	errMsg = 'PathCreation OK'
	logManager.writeLog('INFO', errMsg)

	# Check there is internet connection (try 10 times)
	tries = 0
	while tries < 10:
		tries += 1
		try:
			urllib.request.urlopen('https://finance.yahoo.com/')
			status = True
			errMsg = 'Internet Connection OK'
			logManager.writeLog('INFO', errMsg)
			break
		except:
			status = False
			errMsg = 'Internet Connection ERROR'
			logManager.writeLog('ERROR', errMsg)

	return {'status': status, 'errorMsg': errMsg}

def retrieveStockData(todayDate: datetime.date, operation) -> pd.DataFrame():
	status = False
	errMsg = ''

	window = 600
	startDate = todayDate - CDay(window, calendar=USFederalHolidayCalendar())

	# First try to retrieve data
	data = yf.download('^GSPC', start=startDate, end=todayDate + CDay(calendar=USFederalHolidayCalendar()))

	# If not, try various times along certain period of time
	retryTime = 3 * 3600	# Retry for 3 hours
	intervalRetry = 0.25 * 3600	# Every 15 min
	numRetries = 0
	while len(data) == 0 and numRetries < retryTime/intervalRetry:
		time.sleep(intervalRetry)
		numRetries += 1
		data = yf.download('^GSPC', start=startDate, end=todayDate + CDay(calendar=USFederalHolidayCalendar()))

	res = checkStockOpened(data, todayDate)
	if res['status']:
		if len(data) != 0:
			status = True
			errMsg = "retrieveStockData OK"
			logManager.writeLog('INFO', errMsg)

			# Depending on the moment of operation some values are definitive for the day or not
			if operation == 'Morning':
				data['Open'] = data['Open'].shift(-1)
				data = data[:-1]
			elif operation == 'Afternoon':
				pass
		else:
			status = False
			errMsg = 'retrieveStockData ERROR'
			logManager.writeLog('ERROR', errMsg)
	else:
		status = res['status']
		errMsg = res['errorMsg']

	return {'status': status, 'errorMsg': errMsg, 'data': data}

def retrieveCSVData():
	status = False
	errMsg = ''
	data = pd.DataFrame()

	try:
		data = pd.read_csv(csvDataFileHidden, index_col=['Date'])
		status = True
		errMsg = 'retrieveCSVData OK'
		logManager.writeLog('INFO', errMsg)
	except:
		status = False
		errMsg = 'retrieveCSVData ERROR'
		logManager.writeLog('ERROR', errMsg)

	return {'status': status, 'errorMsg': errMsg, 'data': data}

def calculateInputs(df: pd.DataFrame, inputs: [StrategyInput], operation):
	status = False
	errMsg = ''

	data = pd.DataFrame()

	for inp in inputs:
		name = inp.name
		dfName = inp.dfName
		key = inp.keyName
		description = inp.description
		parameters = inp.listParameters

		if name == 'High':
			data[dfName] = df['High']

		elif name == 'Low':
			data[dfName] = df['Low']

		elif name == 'Volume':
			data[dfName] = df['Volume']

		elif name == 'Close':
			if key == 'Natural':
				data[dfName] = df['Close']
			elif key == 'Log':
				data[dfName] = np.log(df['Close'])

		elif name == 'Open':
			if key == 'Natural':
				data[dfName] = df['Open']
			elif key == 'Log':
				data[dfName] = np.log(df['Open'])

		elif name == 'Return_open':
			if key == 'Natural':
				data[dfName] = df['Open'] / df['Open'].shift()
			elif key == 'Log':
				data[dfName] = np.log(df['Open']) - np.log(df['Open'].shift())

		elif name == 'Return_intraday':
			if key == 'Natural':
				if operation == 'Morning':
					data[dfName] = df['Close'] / df['Open'].shift()
				elif operation == 'Afternoon':
					data[dfName] = df['Close'] / df['Open']
			elif key == 'Log':
				if operation == 'Morning':
					data[dfName] = np.log(df['Close']) - np.log(df['Open'].shift())
				elif operation == 'Afternoon':
					data[dfName] = np.log(df['Close']) - np.log(df['Open'])

		elif name == 'Return_interday':
			if key == 'Natural':
				if operation == 'Morning':
					data[dfName] = df['Open'] / df['Close']
				elif operation == 'Afternoon':
					data[dfName] = df['Open'] / df['Close'].shift()
			elif key == 'Log':
				if operation == 'Morning':
					data[dfName] = np.log(df['Open']) - np.log(df['Close'])
				elif operation == 'Afternoon':
					data[dfName] = np.log(df['Open']) - np.log(df['Close'].shift())

		elif name == 'adi':
			data[dfName] = accDistIndexIndicator(df['High'], df['Low'], df['Close'], df['Volume'])[key]
		elif name == 'adx':
			for param in parameters:
				if param.name == 'Window':
					window = param.value

			data[dfName] = averageDirectionalMovementIndex(df['High'], df['Low'], df['Close'], window)[key]
		elif name == 'aroon':
			for param in parameters:
				if param.name == 'Window':
					window = param.value

			data[dfName] = aroon(df['Close'], window)[key]
		elif name == 'atr':
			for param in parameters:
				if param.name == 'Window':
					window = param.value

			data[dfName] = averageTrueRange(df['High'], df['Low'], df['Close'], window)[key]
		elif name == 'bb':
			for param in parameters:
				if param.name == 'Window':
					window = param.value
				elif param.name == 'StdDev':
					stdDev = param.value

			data[dfName] = bollingerBands(df['Close'], window, stdDev)[key]
		elif name == 'ema':
			for param in parameters:
				if param.name == 'Window':
					window = param.value

			data[dfName] = exponentialMovingAverage(df['Close'], window)[key]
		elif name == 'macd':
			for param in parameters:
				if param.name == 'FastWindow':
					fastWindow = param.value
				elif param.name == 'SlowWindow':
					slowWindow = param.value
				elif param.name == 'Signal':
					signal = param.value

			data[dfName] = movingAverageConvergenceDivergence(df['Close'], fastWindow, slowWindow, signal)[key]
		elif name == 'obv':
			data[dfName] = on_balance_volume(df['Close'], df['Volume'])[key]
		elif name == 'rsi':
			for param in parameters:
				if param.name == 'Window':
					window = param.value

			data[dfName] = relativeStrengthIndex(df['Close'], window)[key]
		elif name == 'stochasticRsi':
			for param in parameters:
				if param.name == 'Window':
					window = param.value
				elif param.name == 'Smooth1':
					smooth1 = param.value
				elif param.name == 'Smooth2':
					smooth2 = param.value

			data[dfName] = stochasticRSI(df['Close'], window, smooth1, smooth2)[key]

	# Close and Open should ALWAYS be in
	data['Open'] = df['Open']
	data['Close'] = df['Close']

	status = True
	errMsg = 'calculateInputs OK'
	logManager.writeLog('INFO', errMsg)

	return {'status': status, 'errorMsg': errMsg, 'data': data}

def runStrategies(dateToday, operation, investorInfo: pd.DataFrame, inputsDf: pd.DataFrame, listStrategies: [Strategy]):
	status = False
	errMsg = ''
	data = pd.DataFrame()

	# Set fixed date value depending on the operation
	if operation == 'Morning':
		tm = datetime.time(9, 30, 0)
	elif operation == 'Afternoon':
		tm = datetime.time(16, 0, 0)
	else:
		logManager.writeLog('INFO', 'runStrategies run without any action')
		return {'status': status, 'errorMsg': errMsg, 'data': data}
	dateTag = dateToday.combine(dateToday, tm)
	lastDateTag = None
	if len(investorInfo) != 0:
		lastDateTag = datetime.datetime.strptime(investorInfo.index[-1], '%Y-%m-%d %H:%M:%S')

	entry = pd.DataFrame()
	for strategy in listStrategies:
		name = strategy.name

		if len(investorInfo) != 0:
			strategyInfo = investorInfo[investorInfo['investorStrategy'] == name]
		else:
			strategyInfo = pd.DataFrame()
		inputsData = inputsDf[['Open', 'Close'] + strategy.getListDfNameInputs()]

		aux = pd.DataFrame()
		if name == 'bia':
			aux = BIA(strategyInfo, strategy).broker(operation, inputsData)
		elif name == 'wia':
			aux = WIA(strategyInfo, strategy).broker(operation, inputsData)
		elif name == 'bah':
			aux = BaH(strategyInfo, strategy).broker(operation, inputsData)
		elif name == 'ca':
			aux = CA(strategyInfo, strategy).broker(operation, inputsData)
		elif name == 'random':
			aux = Random(strategyInfo, strategy).broker(operation, inputsData)
		elif name == 'idle':
			aux = Idle(strategyInfo, strategy).broker(operation, inputsData)
		elif name == 'rsi':
			aux = RSI(strategyInfo, strategy).broker(operation, inputsData)
		elif name == 'bb':
			aux = BB(strategyInfo, strategy).broker(operation, inputsData)
		elif name == 'lstmConfidenceOpenClose':
			aux = LSTMConfidenceOpenClose(strategyInfo, strategy).broker(operation, inputsData)

		if name in ['wia', 'bia'] and lastDateTag:
			aux = pd.concat([aux.reset_index(drop=True), inputsDf[-1:].reset_index(drop=True)], axis=1)
			aux['Date'] = lastDateTag
			aux.set_index('Date', inplace=True)
		else:
			aux = pd.concat([aux.reset_index(drop=True), inputsDf[-1:].reset_index(drop=True)], axis=1)
			aux['Date'] = dateTag
			aux.set_index('Date', inplace=True)

		entry = pd.concat([entry, aux])

	status = True
	errMsg = 'runStrategies OK'
	logManager.writeLog('INFO', errMsg)
	data = entry
	return {'status': status, 'errorMsg': errMsg, 'data': data}

def checkNoRepeat(investorInfo, operation, dateToday):
	status = False
	errMsg = ''
	data = None
	
	# Set fixed date value depending on the operation
	if operation == 'Morning':
		tm = datetime.time(9, 30, 0)
	elif operation == 'Afternoon':
		tm = datetime.time(16, 0, 0)
	else:
		logManager.writeLog('INFO', 'runStrategies run without any action')
		return {'status': status, 'errorMsg': errMsg, 'data': data}
	dateTag = dateToday.combine(dateToday, tm)
	
	# Check last date
	if datetime.datetime.strptime(investorInfo.index[-1], '%Y-%m-%d %H:%M:%S') < dateTag:
		status = True
		errMsg = 'checkNoRepeat OK'
		logManager.writeLog('INFO', errMsg)
	else:
		status = False
		errMsg = 'checkNoRepeat already computed this day and moment'
		logManager.writeLog('INFO', errMsg)

	return {'status': status, 'errorMsg': errMsg, 'data': data}

def checkStockOpened(stockData, todayDate):
	status = False
	errMsg = ''
	data = None

	if stockData.index[-1].to_pydatetime().date() != todayDate.date():
		errMsg = 'checkStockOpened. Not open today'
		status = False
		logManager.writeLog('INFO', errMsg)
	else:
		errMsg = 'checkStockOpened OK'
		status = True
		logManager.writeLog('INFO', errMsg)

	return {'status': status, 'errorMsg': errMsg, 'data': data}

if __name__ == "__main__":
	main()