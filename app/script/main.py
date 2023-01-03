import datetime
import os
from sys import platform
import urllib.request
from datetime import date
import yfinance as yf
import pandas as pd
from pandas.tseries.offsets import CDay
from pandas.tseries.holiday import USFederalHolidayCalendar
import time
from logManager import LogManager
import stat
from inversionStrategyJSONAPI import *
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
logFile = 'ACHTUNGScriptData/log.txt'
logManager = LogManager(logFile)

csvDataFileHidden = 'ACHTUNGScriptData/myData.csv'

csvDataFile = 'scriptData/myData.csv'

jsonFile = 'ACHTUNGScriptData/strategies.json'

def main():
	# 1) safety procedures (operations can be done/necessary files are there)
	if False in runSafetyProcedures().values():
		pass
		# Launch message to user depending on the error

	# 2) get date
	dateToday = date.today()

	# 3) check new data can be retrieved from S&P 500 (check S&P 500 opens today)
	out = retrieveStockData(dateToday)
	data = out['data']
	if not out['status']:
		pass
		# Launch message to user

	# 4) open csv file and read inversor status
	out = retrieveCSVData()
	inversorInfo = out['data']
	if not out['status']:
		pass
		# Launch message to user

	# 5) calculate needed data
	jsonManager = JsonStrategyManager(jsonFile)
	inputs = jsonManager.listInputs()

	out = calculateInputs(data, inputs)
	inputsDf = out['Data']
	if not out['status']:
		pass
		# Launch message to user

	# 6) run algorithm (generate part of the csv related to the strategies)
	# investorStrategy | MoneyInvested | MoneyNotInvested | MoneyBoughtToday | MoneySoldToday | PerBoughtTomorrow |
	# PerSoldTomorrow | TotalPortfolioValue
	out = runStrategies(inversorInfo, inputsDf, jsonManager.listStrategies())

	# 7) write csv file with dataframe (also append date and indicator data)



def runSafetyProcedures():
	status = False
	errMsg = ''
	# Check needed paths exist
	if not os.path.exists("ACHTUNGScriptData"):
		os.mkdir("ACHTUNGScriptData")
		if not os.path.exists(csvDataFileHidden):
			f = open(csvDataFileHidden, 'w')
			f.close()
			# Set file as hidden
			if platform.startswith('linux') or platform.startswith('darwin'):
				st = os.stat(csvDataFileHidden)
				os.chflags(csvDataFileHidden, st.st_flags ^ stat.UF_HIDDEN)
			elif platform.startswith('win'):
				os.system('attrib +h ' + csvDataFileHidden)

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
			urllib.request.urlopen(host='https://finance.yahoo.com/')
			status = True
			errMsg = 'Internet Connection OK'
			logManager.writeLog('INFO', errMsg)
		except:
			status = False
			errMsg = 'Internet Connection ERROR'
			logManager.writeLog('ERROR', errMsg)

	return {'status': status, 'errorMsg': errMsg}

def retrieveStockData(todayDate: datetime.date) -> pd.DataFrame():
	status = False
	errMsg = ''

	us_bus = CDay(calendar=USFederalHolidayCalendar())
	window = 300
	startDate = todayDate - CDay(window, calendar=us_bus)

	# First try to retrieve data
	data = yf.download('^GSPC', start=startDate, end=todayDate + CDay(calendar=us_bus))

	# If not, try various times along certain period of time
	retryTime = 3 * 3600	# Retry for 3 hours
	intervalRetry = 0.25 * 3600	# Every 15 min
	numRetries = 0
	while len(data) == 0 and numRetries < retryTime/intervalRetry:
		time.sleep(intervalRetry)
		numRetries += 1
		data = yf.download('^GSPC', start=startDate, end=todayDate + CDay(calendar=us_bus))

	if len(data) != 0:
		status = True
		errMsg = "retrieveStockData OK"
		logManager.writeLog('INFO', errMsg)

		# The only certain value we have from today is the open value. High, Low, Volume and Close are not teh definitive from today.
		data['Open'] = data['Open'].shift(-1)
		data = data[:-1]
	else:
		status = False
		errMsg = 'retrieveStockData ERROR'
		logManager.writeLog('ERROR', errMsg)
		errMsg = "NoDataRetrieved. Is S&P500 closed today?"
		logManager.writeLog('INFO', errMsg)

	return {'status': status, 'errorMsg': errMsg, 'data': data}

def retrieveCSVData():
	status = False
	errMsg = ''
	data = pd.DataFrame()

	try:
		data = pd.read_csv(csvDataFileHidden)
		status = True
		errMsg = 'retrieveCSVData OK'
		logManager.writeLog('INFO', errMsg)
	except:
		status = False
		errMsg = 'retrieveCSVData ERROR'
		logManager.writeLog('ERROR', errMsg)

	return {'status': status, 'errorMsg': errMsg, 'data': data}

def calculateInputs(df: pd.DataFrame, inputs: [StrategyInput]):
	status = False
	errMsg = ''

	data = pd.DataFrame()

	try:
		for inp in inputs:
			name = inp['Name']
			dfName = inp['DfName']
			key = inp['Key']
			description = inp['Description']
			parameters = inp['Parameters']

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
			elif name == 'adi':
				data[dfName] = accDistIndexIndicator(df['High'], df['Low'], df['Close'], df['Volume'])[key]
			elif name == 'adx':
				for param in parameters:
					if param['Name'] == 'Window':
						window = param['Value']

				data[dfName] = averageDirectionalMovementIndex(df['High'], df['Low'], df['Close'], window)[key]
			elif name == 'aroon':
				for param in parameters:
					if param['Name'] == 'Window':
						window = param['Value']

				data[dfName] = aroon(df['Close'], window)[key]
			elif name == 'atr':
				for param in parameters:
					if param['Name'] == 'Window':
						window = param['Value']

				data[dfName] = averageTrueRange(df['High'], df['Low'], df['Close'], window)[key]
			elif name == 'bb':
				for param in parameters:
					if param['Name'] == 'Window':
						window = param['Value']
					elif param['Name'] == 'StdDev':
						stdDev = param['Value']

				data[dfName] = bollingerBands(df['Close'], window, stdDev)[key]
			elif name == 'ema':
				for param in parameters:
					if param['Name'] == 'Window':
						window = param['Value']

				data[dfName] = exponentialMovingAverage(df['Close'], window)[key]
			elif name == 'macd':
				for param in parameters:
					if param['Name'] == 'FastWindow':
						fastWindow = param['Value']
					elif param['Name'] == 'slowWindow':
						slowWindow = param['Value']
					elif param['Name'] == 'signal':
						signal = param['Value']

				data[dfName] = movingAverageConvergenceDivergence(df['Close'], fastWindow, slowWindow, signal)[key]
			elif name == 'obv':
				data[dfName] = on_balance_volume(df['Close'], df['Volume'])[key]
			elif name == 'rsi':
				for param in parameters:
					if param['Name'] == 'Window':
						window = param['Value']

				data[dfName] = relativeStrengthIndex(df['Close'], window)[key]
			elif name == 'stochasticRsi':
				for param in parameters:
					if param['Name'] == 'Window':
						window = param['Value']
					elif param['Name'] == 'Smooth1':
						smooth1 = param['Value']
					elif param['Name'] == 'Smooth2':
						smooth2 = param['Value']

				data[dfName] = stochasticRSI(df['Close'], window, smooth1, smooth2)[key]

		status = True
		errMsg = 'calculateInputs OK'
		logManager.writeLog('INFO', errMsg)
	except:
		status = False
		errMsg = 'calculateInputs ERROR'
		logManager.writeLog('ERROR', errMsg)

	return {'status': status, 'errorMsg': errMsg, 'data': data}

def runStrategies(inversorInfo: pd.DataFrame, inputsDf: pd.DataFrame, listStrategies: [Strategy]):
	status = False
	errMsg = ''

	for strategy in listStrategies:
		name = strategy['Name']

		strategyInfo = inversorInfo[inversorInfo['investorStrategy'] == name]
		inputsData = inputsDf[strategy.getListDfNameInputs()]

		if name == 'bia':
			pass
		elif name == 'wia':
			pass
		elif name == 'bah':
			pass
		elif name == 'ca':
			pass
		elif name == 'random':
			pass
		elif name == 'idle':
			pass


	return {'status': status, 'errorMsg': errMsg, 'data': data}

if __name__ == "__main__":
	main()