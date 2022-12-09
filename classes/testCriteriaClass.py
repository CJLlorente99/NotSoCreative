class testCriteriaClass:
	def __init__(self):
		pass
	
	@staticmethod
	def calculateCriteria(name, record):
		"""
		Function that calculates the test criteria for a given investor (strategy) in a single experiment (2 week run)
		- Mean Portfolio Value (MPV)
		- Mean of the Std of the PV (MStdPV)
		- Max PV (maxPV)
		- Min PV (minPV)
		- Percentual Gain over 2 weeks (PerGain)
		- Absolute Gain over 2 weeks (AbsGain)
		- Number of operation (nOperation)
		- Gain per Operation (GainPerOperation)
		- Mean of money not invested (meanNotInvested)
		- Mean of money invested (meanInvested)
		- Mean buying (MeanBuying)
		- Mean selling (MeanSelling)
		- Max gain in 1 day (maxGainOneDay)
		- Max loss in 1 day (maxLossOneDay)
		:return: metrics calculated
		"""
		results = {}
		initialInvestment = record["totalValue"][0]

		# Set name
		results["name"] = name

		# Calculation of the MPV
		MPV = record["totalValue"].mean()
		results["MPV"] = MPV

		# Calculation of the MStdPV
		MStdPV = record["totalValue"].std()
		results["MStdPV"] = MStdPV

		# Calculation of the max PV
		maxPV = record["totalValue"].max()
		results["maxPV"] = maxPV

		# Calculation of the min PV
		minPV = record["totalValue"].min()
		results["minPV"] = minPV

		# Calculation of the PerGain
		PerGain = (record["totalValue"].iloc[-1] - initialInvestment) / initialInvestment * 100
		results["PerGain"] = PerGain

		# Calculation of the AbsGain
		AbsGain = record["totalValue"].iloc[-1] - initialInvestment
		results["AbsGain"] = AbsGain

		# Calculation of the nOperation
		nOperation = record["moneyInvestedToday"][record["moneyInvestedToday"] != 0].count() + \
						record["moneySoldToday"][record["moneySoldToday"] != 0].count()
		results["nOperation"] = nOperation

		# Calculation of the GainPerOperation
		if nOperation > 0:
			GainPerOperation = AbsGain / nOperation
		else:
			GainPerOperation = 0
		results["GainPerOperation"] = GainPerOperation

		# Calculation of the meanNotInvested
		meanNotInvested = record["moneyNotInvested"].mean()
		results["meanNotInvested"] = meanNotInvested

		# Calculation of the meanInvested
		meanInvested = record["moneyInvested"].mean()
		results["meanInvested"] = meanInvested

		# Calculation of the meanBuying
		meanBuying = record["moneyInvestedToday"][record["moneyInvestedToday"] != 0].mean()
		results["meanBuying"] = meanBuying

		# Calculation of the meanSelling
		meanSelling = record["moneySoldToday"][record["moneySoldToday"] != 0].mean()
		results["meanSelling"] = meanSelling

		# Calculation of the maxGainOneDay
		maxGainOneDay = record["totalValue"].diff().max()
		results["maxGainOneDay"] = maxGainOneDay

		# Calculation of the maxLossOneDay
		maxLossOneDay = record["totalValue"].diff().min()
		results["maxLossOneDay"] = maxLossOneDay

		return results

	@staticmethod
	def calculateCriteriaVariousExperiments(arrayResults):
		"""
		Function that calculates the test criteria for various experiments
		- Mean of Mean Portfolio Value (MMPV)
		- Std of MPV (StdMPV)
		- Mean of the Mean of the Std of the PV (MMStdPV)
		- Max PV (maxPV)
		- Min PV (minPV)
		- Mean Percentual Gain over 2 weeks (MPerGain)
		- Mean Absolute Gain over 2 weeks (MAbsGain)
		- Mean Gain per Operation (MGainPerOperation)
		- Mean Number of Operations (MnOperation)
		- Mean of Mean of money not invested (MMNotInvested)
		- Mean of Mean of money invested (MMInvested)
		- Mean of Mean buying (MMeanBuying)
		- Mean of Mean selling (MMeanSelling)
		- Max gain in 1 day (maxGainOneDay)
		- Max loss in 1 day (maxLossOneDay)
		:return: metrics calculated
		"""
		result = {}

		# Calculation of the MMPV
		# TODO

		# Calculation of the StdMPV
		# TODO

		# Calculation of the MMStdPV
		# TODO

		# Calculation of the maxPV
		# TODO

		# Calculation of the minPV
		# TODO

		# Calculation of the MPerGain
		# TODO

		# Calculation of the MAbsGain
		# TODO

		# Calculation of the MGainPerOperation
		# TODO

		# Calculation of the MnOperation
		# TODO

		# Calculation of the MMNotInvested
		# TODO

		# Calculation of the MMInvested
		# TODO

		# Calculation of the MMeanBuying
		# TODO

		# Calculation of the MMeanSelling
		# TODO

		# Calculation of the maxGainOneDay
		# TODO

		# Calculation of the maxLossOneDay
		# TODO

		return result
