# This file contains a class to be handled from main.py
import pandas as pd
from googleStorageAPI import readBlobDf
import jinja2
import plotly.graph_objects as go

shortDailyDigestTemplatePath = ''
longDailyDigestTemplatePath = ''

"""
TODO
How do we add a figure into the html?
Plotly has support for creating html strings. Does it work by its own?
"""

class Renderer:
	def __init__(self):
		self.stockData = self.getYfData()
		self.strategiesData = self.getStrategiesData()

	def getYfData(self) -> pd.DataFrame:
		"""
		This method should call yfinance and retrieve the data until today
		:return: pd.DataFrame containing the stock market data
		"""
		return pd.DataFrame()

	def getStrategiesData(self) -> pd.DataFrame:
		"""
		This method should retrieve the strategies data from Google storage via GoogleStorageAPI
		:return:
		"""
		return pd.DataFrame()

	def renderShortDailyDigest(self) -> str:
		"""
		This method should render the short daily digest template
		:return: html string
		"""
		return ''

	def renderLongDailyDigest(self) -> str:
		"""
		This method should render the short daily digest template
		:return: html string
		"""
		return ''
