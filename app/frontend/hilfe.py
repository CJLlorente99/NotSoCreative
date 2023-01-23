from googleStorageAPI import readBlobDf
import yfinance as yf
import numpy as np

# first we load the df from the cloud

df = readBlobDf()

# add an extra column indicating whether the row refers to a morning operation or afternoon

# we need this auxiliar function (This may take a long time, ca. 2 min at most)
def calculateOperation(value: str):
	if value.endswith('09:30:00'):
		return 0
	elif value.endswith('16:00:00'):
		return 1
	else:
		return 2  # never

df['operation'] = df['Date'].map(calculateOperation)

# let's get rid of all afternoon operations

df_morning = df.copy()
df_morning = df_morning[df_morning['operation'] == 0]

# for every investor in the df, let's calculate the mpv (only considering morning values)
mpvs = {}
for strategy in df_morning['investorStrategy'].unique():
	mpvs[strategy] = df_morning[df_morning['investorStrategy'] == strategy]['TotalPortfolioValue'].values.mean()

# for every indicator, let's calculate the gain
gains = {}
for strategy in df_morning['investorStrategy'].unique():
	aux = df_morning[df_morning['investorStrategy'] == strategy]['TotalPortfolioValue'].values
	gains[strategy] = (aux[-1] - aux[0])/aux[0]

# get rid of old strategies that are not updated anymore
# note latest date in the data frame, which strategies were alive then?
latestDate = df_morning['Date'].unique()[-1]
strategiesAlive = df_morning[df_morning['Date'] == latestDate]['investorStrategy'].values

df_morning = df_morning[df_morning['investorStrategy'].isin(strategiesAlive)]

# what if we have a new entry from s&p500

todayData = yf.download('^GSPC', '2023-01-23', '2023-01-24')
print(todayData)

# calculate updated MPV

updatedmpvs = {}
for strategy in df_morning['investorStrategy'].unique():
	aux = df_morning[df_morning['investorStrategy'] == strategy]['TotalPortfolioValue'].values
	lastMoneyInvested = df_morning[df_morning['investorStrategy'] == strategy]['MoneyInvested'].values[-1]
	updatedmpvs[strategy] = np.append(aux, todayData['Close'].values[-1]/todayData['Open'].values[-1] * lastMoneyInvested).mean()
	print(f'{strategy} updatedMPV {updatedmpvs[strategy]}')
print()

# calculate updated gain

updatedgains = {}
for strategy in df_morning['investorStrategy'].unique():
	aux = df_morning[df_morning['investorStrategy'] == strategy]['TotalPortfolioValue'].values
	updatedMoneyInvested = todayData['Close'].values[-1]/todayData['Open'].values[-1] * df_morning[df_morning['investorStrategy'] == strategy]['MoneyInvested'].values[-1]
	updatedPortfolioValue = df_morning[df_morning['investorStrategy'] == strategy]['MoneyNotInvested'].values[-1] + updatedMoneyInvested
	updatedgains[strategy] = (updatedPortfolioValue - aux[0])/aux[0]
	print(f'{strategy} updatedGain {updatedgains[strategy]}')