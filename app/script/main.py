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
"""
The .bat (Windows) or .sh (Mac/Linux) should install all needed packages
The script should do the following.
1) safety procedures (operations can be done/necessary files are there)
2) get date
3) check new data can be retrieved from S&P 500 (check S&P 500 opens today)
4) open csv file and read inversor status
5) calculate needed data
6) run algorithm
7) write csv file with dataframe
"""

# Constants
logFile = 'ACHTUNGScriptData/log.txt'
logManager = LogManager(logFile)

csvDataFileHidden = 'ACHTUNGScriptData/myData.csv'

csvDataFile = 'scriptData/myData.csv'

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


if __name__ == "__main__":
	main()