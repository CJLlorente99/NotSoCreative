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
from strategies.lstmWindowRobustMMT2 import LSTMWindowRobMMT2
from strategies.bilstmWindowRobustMMT1Legacy import BiLSTMWindowRobMMT1Legacy
from strategies.bilstmWindowRobustMMT1T2Legacy import BiLSTMWindowRobMMT1T2Legacy
from logManager.logManager import LogManager
from jsonManagement.inversionStrategyJSONAPI import *
from taAPI import *
import numpy as np
from googleStorageAPI import readBlobDf, updateBlobDf, readBlobJson
from datetime import timedelta

# Constants
logFile = './log.txt'
logManager = LogManager(logFile)

openingHour = 14
openingMinute = 30
closingHour = 21
closingMinute = 0

def main():
	# 1) safety procedures (operations can be done/necessary files are there)
	if False in runSafetyProcedures().values():
		pass
		# Launch message to user depending on the error

	# 2) get date
	dateToday = datetime.datetime.now()
	now = datetime.datetime.now()
	print(f'Today is {dateToday}')
	# dateToday = datetime.datetime(2023, 1, 11)
	# now = datetime.datetime(2023, 1, 11, closingHour, closingMinute+20, 0)
	# if operation == 0:
	# 	now = datetime.datetime(2023, 1, 11, openingHour, openingMinute+20, 0)
	# else:
	# 	now = datetime.datetime(2023, 1, 11, closingHour, closingMinute + 20, 0)

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
	jsonManager = JsonStrategyManager(readBlobJson())
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

		if len(investorinfo) != 0:
			investorinfo = fillNewStrategies(aux, investorinfo)

		logManager.writeLog('INFO', 'csvDateFile writing OK')

		investorinfo = pd.concat([investorinfo, aux])
		updateBlobDf(investorinfo)

def runSafetyProcedures():
	status = False
	errMsg = ''

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
	intervalRetry = 0.02 * 3600	# Every 72 seconds
	numRetries = 0
	while len(data) == 0 and numRetries < retryTime/intervalRetry:
		time.sleep(intervalRetry)
		numRetries += 1
		logManager.writeLog('INFO', 'Try number ' + str(numRetries) + ' to retrieve Stock Data')
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
		data = readBlobDf()
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

		elif name == 'Diff_open':
			data[dfName] = df['Open'] - df['Open'].shift()

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

		logManager.writeLog('INFO', name + ' input calculated')

	# Close and Open should ALWAYS be in
	data['Open'] = df['Open']
	data['Close'] = df['Close']
	data['High'] = df['High']
	data['Low'] = df['Low']
	data['Volume'] = df['Volume']

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
			try:
				strategyInfo = investorInfo[investorInfo['investorStrategy'] == name]
			except:
				strategyInfo = pd.DataFrame()  # In case the strategy is new
		else:
			strategyInfo = pd.DataFrame()

		aux = []
		for i in ['Open', 'Close', 'High', 'Low', 'Volume']:
			if i not in strategy.getListDfNameInputs():
				aux.append(i)

		inputsData = inputsDf[aux + strategy.getListDfNameInputs()]

		aux = pd.DataFrame()
		if name.startswith('bia'):
			aux = BIA(strategyInfo, strategy).broker(operation, inputsData)
		elif name.startswith('wia'):
			aux = WIA(strategyInfo, strategy).broker(operation, inputsData)
		elif name.startswith('bah'):
			aux = BaH(strategyInfo, strategy).broker(operation, inputsData)
		elif name.startswith('ca'):
			aux = CA(strategyInfo, strategy).broker(operation, inputsData)
		elif name.startswith('random'):
			aux = Random(strategyInfo, strategy).broker(operation, inputsData)
		elif name.startswith('idle'):
			aux = Idle(strategyInfo, strategy).broker(operation, inputsData)
		elif name.startswith('lstmWindowRobMMT2'):
			aux = LSTMWindowRobMMT2(strategyInfo, strategy).broker(operation, inputsData)
		elif name.startswith('bilstmWindowRobMMT1Legacy'):
			aux = BiLSTMWindowRobMMT1Legacy(strategyInfo, strategy).broker(operation, inputsData)
		elif name.startswith('bilstmWindowRobMMT1T2Legacy'):
			aux = BiLSTMWindowRobMMT1T2Legacy(strategyInfo, strategy).broker(operation, inputsData)

		if (name.startswith('wia') or name.startswith('bia')) and lastDateTag:
			aux = pd.concat([aux.reset_index(drop=True), inputsDf[-1:].reset_index(drop=True)], axis=1)
			aux['Date'] = lastDateTag
			aux.set_index('Date', inplace=True)
		elif name.startswith('wia') or name.startswith('bia'):
			continue
		else:
			aux = pd.concat([aux.reset_index(drop=True), inputsDf[-1:].reset_index(drop=True)], axis=1)
			aux['Date'] = dateTag
			aux.set_index('Date', inplace=True)

		entry = pd.concat([entry, aux])

		logManager.writeLog('INFO', name + ' calculated')

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

def fillNewStrategies(newEntry: pd.DataFrame, record: pd.DataFrame):
	# In case new Strategy has been added
	for column in newEntry.columns:
		if column not in record.columns:
			record[column] = np.zeros(len(record))
			record[column].values[-len(newEntry)+3:] = newEntry[column].values[-1]

	# In case Strategy has been taken out
	for column in record.columns:
		if column not in newEntry.columns:
			newEntry[column] = np.zeros(len(newEntry))

	return record


if __name__ == "__main__":
	# today = datetime.datetime(2023, 1, 10)
	# while today <= datetime.datetime(2023, 1, 22):
	# 	operation = 0
	# 	main(today, operation)
	#
	# 	operation = 1
	# 	main(today, operation)
	#
	# 	today += timedelta(days=1)
	main()
