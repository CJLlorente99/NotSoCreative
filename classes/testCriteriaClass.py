import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import pandas_datareader as web


class testCriteriaClass:
	def __init__(self):
		pass
	
	@staticmethod
	def calculateCriteria(name, record):
		"""
		Function that calculates the test criteria for a given investor (strategy) in a single experiment (2 week run)
		- Mean Portfolio Value (MPV)
		- Std of the PV (MStdPV)
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
		- Treynor Measure (TreynorMeasure)
		- Sharpe Ratio (SharpeRatio)
		- Jensen Measure (JensenMeasure)
		:return: metrics calculated
		"""
		results = {}
		initialInvestment = record["totalValue"][0]

		# Set name
		results["name"] = name

		# Calculation of the MPV
		MPV = record["totalValue"].mean()
		results["MPV"] = MPV

		# Calculation of the StdPV
		StdPV = record["totalValue"].std()
		results["StdPV"] = StdPV

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
		nOperation = nOperation if nOperation != 0 else 0
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
		meanBuying = meanBuying if meanBuying != 0 else 0
		results["meanBuying"] = meanBuying

		# Calculation of the meanSelling
		meanSelling = record["moneySoldToday"][record["moneySoldToday"] != 0].mean()
		meanSelling = meanSelling if meanSelling != 0 else 0
		results["meanSelling"] = meanSelling

		# Calculation of the maxGainOneDay
		maxGainOneDay = record["totalValue"].diff().max()
		results["maxGainOneDay"] = maxGainOneDay

		# Calculation of the maxLossOneDay
		maxLossOneDay = record["totalValue"].diff().min()
		results["maxLossOneDay"] = maxLossOneDay

		# Calculation of TreynorMeasure
		beta = np.cov(record["totalValue"].diff()[1:].values, record["openValue"].diff()[1:].values)[0][1] / np.var(record["openValue"].diff()[1:].values)
		rfr = calculateRFR(record.index.values[0], record.index.values[-1])
		TreynorMeasure = (PerGain - rfr) / beta
		results["TreynorMeasure"] = TreynorMeasure

		# Calculation of SharpeRatio
		SP500std = record["openValue"].std()
		SharpeRatio = (PerGain - rfr) / SP500std
		results["SharpeRatio"] = SharpeRatio

		# Calculation of JensenMeasure
		capm = rfr + beta
		JensonMeasure = PerGain - capm
		results["JensenMeasure"] = JensonMeasure

		return results

	@staticmethod
	def plotCriteria(dfResult, title):
		fig = make_subplots(rows=3, cols=3, vertical_spacing=0.1, horizontal_spacing=0.04,
							subplot_titles=["MPV(StdPV)", "maxPV-minPV", "%Gain-AbsGain", "nOp",
											"GainPerOp-max1Day-max1Day", "Treynor-Sharpe-Jensen",
											"MNotInv-MInv-MBuy-MSell"],
							specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": True}],
								   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
								   [{"secondary_y": False, "colspan": 3}, {"secondary_y": False}, {"secondary_y": False}]])

		fig.add_trace(go.Scatter(name="MPV", x=dfResult["name"], y=dfResult["MPV"],
								 error_y=dict(type='data', array=dfResult["StdPV"].values, visible=True)), row=1,
					  			 col=1)

		fig.add_trace(go.Scatter(name="maxPV", x=dfResult["name"], y=dfResult["maxPV"]), row=1, col=2)
		fig.add_trace(go.Scatter(name="minPV", x=dfResult["name"], y=dfResult["minPV"]), row=1, col=2)

		fig.add_trace(go.Scatter(name="PerGain", x=dfResult["name"], y=dfResult["PerGain"]), row=1, col=3)
		fig.add_trace(go.Scatter(name="AbsGain", x=dfResult["name"], y=dfResult["AbsGain"]), row=1, col=3, secondary_y=True)

		fig.add_trace(go.Scatter(name="nOperation", x=dfResult["name"], y=dfResult["nOperation"]), row=2, col=1)

		fig.add_trace(go.Scatter(name="GainPerOperation", x=dfResult["name"],
							 	 y=dfResult["GainPerOperation"]), row=2, col=2)
		fig.add_trace(go.Scatter(name="maxGainOneDay", x=dfResult["name"],
								 y=dfResult["maxGainOneDay"]), row=2, col=2)
		fig.add_trace(go.Scatter(name="maxLossOneDay", x=dfResult["name"],
								 y=dfResult["maxLossOneDay"]), row=2, col=2)

		fig.add_trace(go.Scatter(name="TreynorMeasure", x=dfResult["name"],
								 y=dfResult["TreynorMeasure"]), row=2, col=3)
		fig.add_trace(go.Scatter(name="SharpeRatio", x=dfResult["name"],
								 y=dfResult["SharpeRatio"]), row=2, col=3)
		fig.add_trace(go.Scatter(name="JensenMeasure", x=dfResult["name"],
								 y=dfResult["JensenMeasure"]), row=2, col=3)

		fig.add_trace(go.Scatter(name="MNotInvested,", x=dfResult["name"],
								 y=dfResult["meanNotInvested"]), row=3, col=1)
		fig.add_trace(go.Scatter(name="MInvested", x=dfResult["name"],
								 y=dfResult["meanInvested"]), row=3, col=1)
		fig.add_trace(go.Scatter(name="MBuying", x=dfResult["name"],
								 y=dfResult["meanBuying"]), row=3, col=1)
		fig.add_trace(go.Scatter(name="MSelling", x=dfResult["name"],
								 y=dfResult["meanSelling"]), row=3, col=1)

		fig.update_layout(title_text=title, hovermode="x unified")

		fig.show()

	@staticmethod
	def calculateCriteriaVariousExperiments(dfResults):
		"""
		Function that calculates the test criteria for various experiments
		- Mean of Mean Portfolio Value (MMPV)
		- Std of Mean Portfolio Value (StdMPV)
		- Mean of the Std of the Portfolio Value (MStdPV)
		- Std of the Std of the Portfolio Value (StdStdPV)
		- Mean Percentual Gain over 2 weeks (MPerGain)
		- Std Percentual Gain over 2 weeks (StdPerGain)
		- Mean Absolute Gain over 2 weeks (MAbsGain)
		- Std Absolute Gain over 2 weeks (StdAbsGain)
		- Mean Gain per Operation (MGainPerOperation)
		- Std Gain per Operation (StdGainPerOperation)
		- Mean Number of Operations (MnOperation)
		- Std Number of Operations (StdnOperation)
		- Mean of Mean of money not invested (MMNotInvested)
		- Std of Mean of money not invested (StdMNotInvested)
		- Mean of Mean of money invested (MMInvested)
		- Std of Mean of money invested (StdMInvested)
		- Mean of Mean buying (MMeanBuying)
		- Std of Mean buying (StdMeanBuying)
		- Mean of Mean selling (MMeanSelling)
		- Std of Mean selling (StdMeanSelling)
		- Mean of max gain in 1 day (MmaxGainOneDay)
		- Std max gain in 1 day (StdmaxGainOneDay)
		- Mean Max loss in 1 day (MmaxLossOneDay)
		- Std Max loss in 1 day (StdmaxLossOneDay)
		- Mean Treynor Measure (MTreynorMeasure)
		- Mean Sharpe Ratio (MSharpeRatio)
		- Mean Jensen Measure (MJensenMeasure)
		:return: metrics calculated
		"""
		result = pd.DataFrame()
		i = 0

		for indicator in dfResults["name"].unique():
			df = dfResults[dfResults["name"] == indicator]
			aux = {}

			# Name of the indicator
			aux["name"] = indicator

			# Calculation of the MMPV
			MMPV = df["MPV"].mean()
			aux["MMPV"] = MMPV

			# Calculation of the StdMPV
			StdMPV = df["MPV"].std()
			aux["StdMPV"] = StdMPV

			# Calculation of the MStdPV
			MStdPV = df["StdPV"].mean()
			aux["MStdPV"] = MStdPV

			# Calculation fo the StdStdPV
			StdStdPV = df["StdPV"].std()
			aux["StdStdPV"] = StdStdPV

			# Calculation of the MPerGain
			MPerGain = df["PerGain"].mean()
			aux["MPerGain"] = MPerGain

			# Calculation of the StdMPerGain
			StdPerGain = df["PerGain"].std()
			aux["StdPerGain"] = StdPerGain

			# Calculation of the MAbsGain
			MAbsGain = df["AbsGain"].mean()
			aux["MAbsGain"] = MAbsGain

			# Calculation of the StdMAbsGain
			StdAbsGain = df["AbsGain"].std()
			aux["StdAbsGain"] = StdAbsGain

			# Calculation of the MGainPerOperation
			MGainPerOperation = df["GainPerOperation"].mean()
			aux["MGainPerOperation"] = MGainPerOperation

			# Calculation of the StdMGainPerOperation
			StdGainPerOperation = df["GainPerOperation"].std()
			aux["StdGainPerOperation"] = StdGainPerOperation

			# Calculation of the MnOperation
			MnOperation = df["nOperation"].mean()
			aux["MnOperation"] = MnOperation

			# Calculation of the StdMnOperation
			StdnOperation = df["nOperation"].std()
			aux["StdnOperation"] = StdnOperation

			# Calculation of the MMNotInvested
			MMNotInvested = df["meanNotInvested"].mean()
			aux["MMNotInvested"] = MMNotInvested

			# Calculation of the StdMMNotInvested
			StdMNotInvested = df["meanNotInvested"].std()
			aux["StdMNotInvested"] = StdMNotInvested

			# Calculation of the MMInvested
			MMInvested = df["meanInvested"].mean()
			aux["MMInvested"] = MMInvested

			# Calculation of the StdMMInvested
			StdMInvested = df["meanInvested"].std()
			aux["StdMInvested"] = StdMInvested

			# Calculation of the MMeanBuying
			MMBuying = df["meanBuying"].mean()
			aux["MMBuying"] = MMBuying

			# Calculation of the StdMMeanBuying
			StdMBuying = df["meanBuying"].std()
			aux["StdMBuying"] = StdMBuying

			# Calculation of the MMeanSelling
			MMSelling = df["meanSelling"].mean()
			aux["MMSelling"] = MMSelling

			# Calculation of the StdMMeanSelling
			StdMSelling = df["meanSelling"].std()
			aux["StdMSelling"] = StdMSelling

			# Calculation of the MmaxGainOneDay
			MmaxGainOneDay = df["maxGainOneDay"].mean()
			aux["MmaxGainOneDay"] = MmaxGainOneDay

			# Calculation of the StdMmaxGainOneDay
			StdmaxGainOneDay = df["maxGainOneDay"].std()
			aux["StdmaxGainOneDay"] = StdmaxGainOneDay

			# Calculation of the MmaxLossOneDay
			MmaxLossOneDay = df["maxLossOneDay"].mean()
			aux["MmaxLossOneDay"] = MmaxLossOneDay

			# Calculation of the StdMmaxLossOneDay
			StdmaxLossOneDay = df["maxLossOneDay"].std()
			aux["StdmaxLossOneDay"] = StdmaxLossOneDay

			# Calculation of the MTreynorMeasure
			MTreynorMeasure = df["TreynorMeasure"].mean()
			aux["MTreynorMeasure"] = MTreynorMeasure

			# Calculation of the MSharpeRatio
			MSharpeRatio = df["SharpeRatio"].mean()
			aux["MSharpeRatio"] = MSharpeRatio

			# Calculation of the MJensenMeasure
			MJensenMeasure = df["JensenMeasure"].mean()
			aux["MJensenMeasure"] = MJensenMeasure

			result = pd.concat([result, pd.DataFrame(aux, index=[i])])
			i += 1

		return result

	@staticmethod
	def plotCriteriaVariousExperiments(dfResult, title):
		fig = make_subplots(rows=3, cols=3, vertical_spacing=0.1, horizontal_spacing=0.04,
							subplot_titles=["MMPV", "MStdPV", "M%Gain-MAbsGain", "MnOp",
											"MGainPerOp-Mmax1Day-Mmax1Day", "Treynor-Sharpe-Jensen",
											"MMNotInv-MMInv-MMBuy-MMSell"],
							specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": True}],
								   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
								   [{"secondary_y": False, "colspan": 3}, {"secondary_y": False}, {"secondary_y": False}]])

		fig.add_trace(go.Scatter(name="MMPV", x=dfResult["name"], y=dfResult["MMPV"],
								 error_y=dict(type='data', array=dfResult["MStdPV"].values, visible=True)), row=1,
					  col=1)

		fig.add_trace(go.Scatter(name="MStdPV", x=dfResult["name"], y=dfResult["MStdPV"],
								 error_y=dict(type='data', array=dfResult["StdStdPV"].values, visible=True)), row=1,
					  col=2)

		fig.add_trace(go.Scatter(name="MPerGain", x=dfResult["name"], y=dfResult["MPerGain"],
								 error_y=dict(type='data', array=dfResult["StdPerGain"].values, visible=True)), row=1,
					  col=3)
		fig.add_trace(go.Scatter(name="MAbsGain", x=dfResult["name"], y=dfResult["MAbsGain"],
								 error_y=dict(type='data', array=dfResult["StdAbsGain"].values, visible=True)), row=1,
					  col=3, secondary_y=True)

		fig.add_trace(go.Scatter(name="MnOperation", x=dfResult["name"], y=dfResult["MnOperation"],
								 error_y=dict(type='data', array=dfResult["StdnOperation"].values, visible=True)),
					  row=2, col=1)

		fig.add_trace(go.Scatter(name="MGainPerOperation", x=dfResult["name"], y=dfResult["MGainPerOperation"],
								 error_y=dict(type='data', array=dfResult["StdGainPerOperation"].values, visible=True)),
					  row=2, col=2)
		fig.add_trace(go.Scatter(name="MmaxGainOneDay", x=dfResult["name"], y=dfResult["MmaxGainOneDay"],
								 error_y=dict(type='data', array=dfResult["StdmaxGainOneDay"].values, visible=True)),
					  row=2, col=2)
		fig.add_trace(go.Scatter(name="MmaxLossOneDay", x=dfResult["name"], y=dfResult["MmaxLossOneDay"],
								 error_y=dict(type='data', array=dfResult["StdmaxLossOneDay"].values, visible=True)),
					  row=2, col=2)

		fig.add_trace(go.Scatter(name="MTreynorMeasure", x=dfResult["name"], y=dfResult["MTreynorMeasure"]),
								 row=2, col=3)
		fig.add_trace(go.Scatter(name="MSharpeRatio", x=dfResult["name"], y=dfResult["MSharpeRatio"]),
								 row=2, col=3)
		fig.add_trace(go.Scatter(name="MJensenMeasure", x=dfResult["name"], y=dfResult["MJensenMeasure"]),
								 row=2, col=3)

		fig.add_trace(go.Scatter(name="MMNotInvested,", x=dfResult["name"], y=dfResult["MMNotInvested"],
								 error_y=dict(type='data', array=dfResult["StdMNotInvested"].values, visible=True)),
					  row=3, col=1)
		fig.add_trace(go.Scatter(name="MMInvested", x=dfResult["name"], y=dfResult["MMInvested"],
								 error_y=dict(type='data', array=dfResult["StdMInvested"].values, visible=True)),
					  row=3, col=1)
		fig.add_trace(go.Scatter(name="MMBuying", x=dfResult["name"], y=dfResult["MMBuying"],
								 error_y=dict(type='data', array=dfResult["StdMBuying"].values, visible=True)),
					  row=3, col=1)
		fig.add_trace(go.Scatter(name="MMSelling", x=dfResult["name"], y=dfResult["MMSelling"],
								 error_y=dict(type='data', array=dfResult["StdMSelling"].values, visible=True)),
					  row=3, col=1)
		fig.update_layout(title_text=title, hovermode="x unified")

		fig.show()


def calculateRFR(firstDate, lastDate):
	"""
	This function calculates the Risk-Free Return using the treasury bills as a proxy.
	:param firstDate:  First date from when data will be retrieved
	:param lastDate:  Last date until when data will be retrieved
	"""
	dfTreasuryBills = web.DataReader("^IRX", 'yahoo', firstDate, lastDate)
	return dfTreasuryBills.iloc[-1]["Open"] - dfTreasuryBills.iloc[0]["Open"]
