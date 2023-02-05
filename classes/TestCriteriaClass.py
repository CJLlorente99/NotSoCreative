import math
import yfinance as yf
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np


class TestCriteriaClass:
	def __init__(self, firstDate, lastDate):
		self.firstDate = firstDate
		self.lastDate = lastDate
		self.rfr = self.calculateRFR()

	def calculateRFR(self):
		"""
		This function calculates the Risk-Free Return using the treasury bills as a proxy.
		"""
		dfTreasuryBills = yf.download("^IRX", pd.Timestamp(self.firstDate), pd.Timestamp(self.lastDate))
		return dfTreasuryBills.iloc[-1]["Open"] - dfTreasuryBills.iloc[0]["Open"]

	def calculateCriteria(self, name, record):
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
		- Sortino Ratio (SortinoRatio)
		:return: metrics calculated
		"""
		results = {}
		initialInvestment = record["totalValue"][0]

		# Set name
		results["name"] = name

		# Calculation of the MPV
		MPV = record["totalValue"][::-2].mean()
		results["MPV"] = MPV

		# Calculation of the StdPV
		StdPV = record["totalValue"][::-2].std()
		results["StdPV"] = StdPV

		# Calculation of the max PV
		maxPV = record["totalValue"][::-2].max()
		results["maxPV"] = maxPV

		# Calculation of the min PV
		minPV = record["totalValue"][::-2].min()
		results["minPV"] = minPV

		# Calculation of the PerGain
		PerGain = (record["totalValue"].iloc[-1] - initialInvestment) / initialInvestment * 100
		results["PerGain"] = PerGain

		# Calculation of the AbsGain
		AbsGain = record["totalValue"].iloc[-1] - initialInvestment
		results["AbsGain"] = AbsGain

		# Calculation of the nOperation
		nOperation = record["moneyInvestedToday"][record["moneyInvestedToday"] != 0].count()
		nOperation = nOperation if nOperation != 0 else 0
		results["nOperation"] = nOperation

		# Calculation of the GainPerOperation
		if nOperation > 0:
			GainPerOperation = AbsGain / nOperation
		else:
			GainPerOperation = 0
		results["GainPerOperation"] = GainPerOperation

		# Calculation of the meanNotInvested
		meanNotInvested = record["moneyNotInvested"][::-2].mean()
		results["meanNotInvested"] = meanNotInvested

		# Calculation of the meanInvested
		meanInvested = record["moneyInvested"][::-2].mean()
		results["meanInvested"] = meanInvested

		# Calculation of the meanBuying
		meanBuying = record["moneyInvestedToday"][::-2][record["moneyInvestedToday"][::-2] > 0].mean()
		meanBuying = meanBuying if meanBuying != 0 else 0
		results["meanBuying"] = meanBuying

		# Calculation of the meanSelling
		meanSelling = record["moneyInvestedToday"][::-2][record["moneyInvestedToday"][::-2] < 0].mean()
		meanSelling = meanSelling if meanSelling != 0 else 0
		results["meanSelling"] = meanSelling

		# Calculation of the maxGainOneDay
		maxGainOneDay = record["totalValue"][::-2].diff().max()
		results["maxGainOneDay"] = maxGainOneDay

		# Calculation of the maxLossOneDay
		maxLossOneDay = record["totalValue"][::-2].diff().min()
		results["maxLossOneDay"] = maxLossOneDay

		# Calculation of TreynorMeasure
		beta = np.cov(record["totalValue"][::-2].diff()[1:].values, record["actualStockValue"][::-2].diff()[1:].values)[0][1] / np.var(
			record["actualStockValue"][::-2].diff()[1:].values)
		TreynorMeasure = (PerGain - self.rfr) / beta
		results["TreynorMeasure"] = TreynorMeasure

		# Calculation of SharpeRatio
		SP500std = record["actualStockValue"][::-2].std()
		SharpeRatio = (PerGain - self.rfr) / SP500std
		results["SharpeRatio"] = SharpeRatio

		# Calculation of JensenMeasure
		marketReturn = record["actualStockValue"][::-2].diff()[1:].sum() / 100
		betaJensen = beta * (marketReturn - self.rfr)
		capm = self.rfr + betaJensen
		JensonMeasure = PerGain - capm
		results["JensenMeasure"] = JensonMeasure

		# Calculation of SortinoRatio
		T = self.rfr
		returns = record["totalValue"][::-2].diff()[1:] / 100
		TDD = math.sqrt(1/returns.size * returns[returns < 0].sub(T).pow(2).sum())
		results["SortinoRatio"] = (PerGain - self.rfr) / TDD

		return results

	def plotCriteria(self, dfResult, title):
		"""
		This function plots the test criteria of all strategies belonging to one experiment.
		:param dfResult: dataFrame that contains the record of a specific strategy
		:param title: Title to be given to the whole figure (should identify uniquely the strategy)
		"""
		# Plot MPV
		fig = go.Figure()

		fig.add_trace(go.Bar(name="MPV", x=dfResult["name"], y=dfResult["MPV"] - 10000,
							 error_y=dict(type='data', array=dfResult["StdPV"].values, visible=True)))
		fig.update_layout(title_text=title + " MPV", hovermode="x unified", barmode="group",
						  paper_bgcolor="rgba(0,0,0,0)",
						  font=dict(size=40),
						  xaxis=dict(gridwidth=10, gridcolor='#2D3A44', linewidth=10, linecolor='#2D3A44',
									 zeroline=False, title_font_size=40, tickfont_size=50),
						  yaxis=dict(gridwidth=10, gridcolor='#2D3A44', linewidth=10, linecolor='#2D3A44',
									 zeroline=False, title_font_size=40, tickfont_size=50),
						  plot_bgcolor='black'
						  )
		fig.update_xaxes(ticks='inside', showgrid=True, griddash='dash', categoryorder='total descending')
		fig.write_image("images/" + title + "MPV.png", scale=6, width=2880, height=1800)

	def calculateCriteriaVariousExperiments(self, dfResults):
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
		- Mean Sortino Ratio (MSortinoRatio)
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

			# Calculation of the MSortinoRation
			MJSortinoRatio = df["SortinoRatio"].mean()
			aux["MSortinoRatio"] = MJSortinoRatio

			result = pd.concat([result, pd.DataFrame(aux, index=[i])])
			i += 1

		return result

	def plotCriteriaVariousExperiments(self, dfResult, title):
		"""
		Function used as a summary of all the experiments run in a given execution
		:param dfResult: dataFrame that contains the summaty test criteria of all strategies used in a given script execution
		:param title: Title to be given to the figure
		"""
		# Plot MMPV
		fig = go.Figure()

		fig.add_trace(go.Bar(name="MMPV", x=dfResult["name"], y=dfResult["MMPV"]-10000,
								 error_y=dict(type='data', array=dfResult["MStdPV"].values, visible=True)))

		fig.update_layout(title_text=title + " MMPV", hovermode="x unified", barmode="group")
		fig.update_xaxes(ticks='inside', showgrid=True, griddash='dash', categoryorder='total descending')
		fig.write_image("images/" + title + "MMPV.png", scale=6, width=2880, height=1800)


		# Plot M abs Gain
		fig = go.Figure()

		fig.add_trace(go.Bar(name="MAbsGain", x=dfResult["name"], y=dfResult["MAbsGain"],
								 error_y=dict(type='data', array=dfResult["StdAbsGain"].values, visible=True)))

		fig.update_layout(title_text=title + " MAbsGain", hovermode="x unified", barmode="group")
		fig.update_xaxes(ticks='inside', showgrid=True, griddash='dash', categoryorder='total descending')
		fig.write_image("images/" + title + "MAbsGain.png",scale=6, width=2880, height=1800)