import pandas as pd


def biaDailyAction(strategy, record, openData):
	"""
	This function should calculate the bia for the LAST day (as it CANNOT be calculated for the same day)
	:param record: dataframe with the record from previous days
	:param inputs: Open data
	"""
	# investorStrategy | MoneyInvested | MoneyNotInvested | MoneyBoughtToday | MoneySoldToday | PerBoughtTomorrow |
	# PerSoldTomorrow | TotalPortfolioValue
	# Check it is not the first day (record is empty)
	if len(record) == 0:
		return pd.DataFrame({'investorStrategy': 'bia', 'MoneyInvested': -1, 'MoneyNotInvested': -1, 'MoneyBoughtToday': -1,
					  'MoneySoldToday': -1, 'PerBoughtTomorrow': -1, 'PerSoldTomorrow': -1, 'TotalPortfolioValue': -1})


