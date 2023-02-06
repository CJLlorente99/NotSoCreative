from datetime import timedelta, datetime
import chart_studio.plotly as py
import chart_studio.tools
import pandas as pd
import pytz
import yfinance as yf
from googleStorageAPI import readBlobDf
from jinja2 import Environment, FileSystemLoader
import plotly.graph_objects as go
from plotly.subplots import make_subplots

shortDailyDigestTemplatePath = 'shortDailyDigest.html'
longDailyDigestTemplatePath = 'longDailyDigest.html'

usernameChartStudio = 'NotSoCreative'
apiChartStudio = 'c6a9xRk9hFTuEGqZ6xPT'

chart_studio.tools.set_credentials_file(username=usernameChartStudio, api_key=apiChartStudio)

"""
TODO
How do we add a figure into the html?
Plotly has support for creating html strings. Does it work by its own?
"""

class Renderer:
	def __init__(self, todayDate):
		self.todayDate = todayDate
		self.stockData = self.getYfData()
		self.strategiesData = self.getStrategiesData(['ca_25_1_2022', 'random_25_1_2022', 'idle_25_1_2022',
													  'bah_25_1_2022', 'bilstmWindowRobMMT1T2Legacy_25_1_2023'])
		self.ourStrategy = 'bilstmWindowRobMMT1T2Legacy_25_1_2023'
		self.env = Environment(loader=FileSystemLoader(searchpath='/home/carlosllocor/NotSoCreative/app/dailyDigestAutomatization/htmlTemplates/'))

	def getYfData(self) -> pd.DataFrame:
		"""
		This method should call yfinance and retrieve the data until today
		:return: pd.DataFrame containing the stock market data
		"""
		return yf.download('^GSPC', (self.todayDate-timedelta(10)).strftime('%Y-%m-%d'), (self.todayDate+timedelta(1)).strftime('%Y-%m-%d'))

	def getStrategiesData(self, listStrategies) -> dict:
		"""
		This method should retrieve the strategies data from Google storage via GoogleStorageAPI
		:return:
		"""
		res = {}
		df = readBlobDf()
		for strategy in listStrategies:
			res[strategy] = df.query('investorStrategy == @strategy')
		return res

	def renderShortDailyDigest(self, name) -> str:
		"""
		This method should render the short daily digest template
		:return: html string
		"""
		todayDecision = 'hold'
		if self.strategiesData['bilstmWindowRobMMT1T2Legacy_25_1_2023'].MoneyInvestedToday[-2] > 0:
			todayDecision = 'buy'
		elif self.strategiesData['bilstmWindowRobMMT1T2Legacy_25_1_2023'].MoneyInvestedToday[-2] < 0:
			todayDecision = 'sell'

		graph_url = self.generateHTMLImageMPV()

		template = self.env.get_template(shortDailyDigestTemplatePath)
		renderedHTML = template.render(name=name,
									   openPrice=self.stockData.Open[-1],
									   closePrice=self.stockData.Close[-1],
									   moneyInvestedToday=self.strategiesData['bilstmWindowRobMMT1T2Legacy_25_1_2023'].MoneyInvestedToday[-2],
									   totalPortfolioValueMorning=self.strategiesData['bilstmWindowRobMMT1T2Legacy_25_1_2023'].TotalPortfolioValue[-2],
									   totalPortfolioValueAfternoon=self.strategiesData['bilstmWindowRobMMT1T2Legacy_25_1_2023'].TotalPortfolioValue[-1],
									   meanPortfolioValue=self.strategiesData['bilstmWindowRobMMT1T2Legacy_25_1_2023'].MPV[-2],
									   caPortfolioValue=self.strategiesData['ca_25_1_2022'].TotalPortfolioValue[-2],
									   caMeanPortfolioValue=self.strategiesData['ca_25_1_2022'].MPV[-2],
									   bahPortfolioValue=self.strategiesData['bah_25_1_2022'].TotalPortfolioValue[-2],
									   bahMeanPortfolioValue=self.strategiesData['bah_25_1_2022'].MPV[-2],
									   randomPortfolioValue=self.strategiesData['random_25_1_2022'].TotalPortfolioValue[-2],
									   randomMeanPortfolioValue=self.strategiesData['random_25_1_2022'].MPV[-2],
									   idlePortfolioValue=self.strategiesData['idle_25_1_2022'].TotalPortfolioValue[-2],
									   idleMeanPortfolioValue=self.strategiesData['idle_25_1_2022'].MPV[-2],
									   todayDate=self.stockData.index[-1].to_pydatetime().strftime('%d-%m-%Y'),
									   todayDecision=todayDecision,
									   graph_url=graph_url)
		return renderedHTML

	def renderLongDailyDigest(self, name) -> str:
		"""
		This method should render the short daily digest template
		:return: html string
		"""
		todayDecision = 'hold'
		if self.strategiesData['bilstmWindowRobMMT1T2Legacy_25_1_2023'].MoneyInvestedToday[-2] > 0:
			todayDecision = 'buy'
		elif self.strategiesData['bilstmWindowRobMMT1T2Legacy_25_1_2023'].MoneyInvestedToday[-2] < 0:
			todayDecision = 'sell'

		graph_url_CompMPV = self.generateHTMLImageMPV()
		graph_url_CompTPV = self.generateHTMLImageTPV()
		graph_url_EvolTPV = self.generateHTMLImageTPVEvolution()
		graph_url_EvolMIT = self.generateHTMLImageMoneyInvestedEvolution()

		template = self.env.get_template(longDailyDigestTemplatePath)
		renderedHTML = template.render(name=name,
									   openPrice=self.stockData.Open[-1],
									   closePrice=self.stockData.Close[-1],
									   moneyInvestedToday=
									   self.strategiesData['bilstmWindowRobMMT1T2Legacy_25_1_2023'].MoneyInvestedToday[-2],
									   totalPortfolioValueMorning=self.strategiesData['bilstmWindowRobMMT1T2Legacy_25_1_2023'].TotalPortfolioValue[-2],
									   totalPortfolioValueAfternoon=self.strategiesData['bilstmWindowRobMMT1T2Legacy_25_1_2023'].TotalPortfolioValue[-1],
									   meanPortfolioValue=self.strategiesData['bilstmWindowRobMMT1T2Legacy_25_1_2023'].MPV[-2],
									   caPortfolioValue=self.strategiesData['ca_25_1_2022'].TotalPortfolioValue[-2],
									   caMeanPortfolioValue=self.strategiesData['ca_25_1_2022'].MPV[-2],
									   bahPortfolioValue=self.strategiesData['bah_25_1_2022'].TotalPortfolioValue[-2],
									   bahMeanPortfolioValue=self.strategiesData['bah_25_1_2022'].MPV[-2],
									   randomPortfolioValue=self.strategiesData['random_25_1_2022'].TotalPortfolioValue[-2],
									   randomMeanPortfolioValue=self.strategiesData['random_25_1_2022'].MPV[-2],
									   idlePortfolioValue=self.strategiesData['idle_25_1_2022'].TotalPortfolioValue[-2],
									   idleMeanPortfolioValue=self.strategiesData['idle_25_1_2022'].MPV[-2],
									   todayDate=self.stockData.index[-1].to_pydatetime().strftime('%d-%m-%Y'),
									   todayDecision=todayDecision,
									   graph_url_CompMPV=graph_url_CompMPV,
									   graph_url_CompTPV=graph_url_CompTPV,
									   graph_url_EvolTPV=graph_url_EvolTPV,
									   graph_url_EvolMIT=graph_url_EvolMIT)
		return renderedHTML

	def generateHTMLImageMPV(self) -> str:
		latestMorning = self.strategiesData[list(self.strategiesData.keys())[0]].index[-2]
		df = pd.DataFrame()

		for key in self.strategiesData:
			data = self.strategiesData[key]
			data = data.query('index == @latestMorning')
			df = pd.concat([df, data], ignore_index=True)

		df['investorStrategy'] = df['investorStrategy'].replace([self.ourStrategy], 'ourStrategy')

		fig = go.Figure()
		fig.add_trace(go.Bar(x=df['investorStrategy'], y=df['MPV'] - 10000))
		fig.update_layout(title='Mean Portfolio Value Comparison (10,000$ offset)')
		url = py.plot(fig, filename='compMPV', auto_open=False)
		return url

	def generateHTMLImageTPV(self) -> str:
		latestMorning = self.strategiesData[list(self.strategiesData.keys())[0]].index[-2]
		df = pd.DataFrame()

		for key in self.strategiesData:
			data = self.strategiesData[key]
			data = data.query('index == @latestMorning')
			df = pd.concat([df, data], ignore_index=True)

		df['investorStrategy'] = df['investorStrategy'].replace([self.ourStrategy], 'ourStrategy')

		fig = go.Figure()
		fig.add_trace(go.Bar(x=df['investorStrategy'], y=df['TotalPortfolioValue'] - 10000))
		fig.update_layout(title='Today Total Portfolio Value Comparison (10,000$ offset)')
		url = py.plot(fig, filename='compTPV', auto_open=False)
		return url

	def generateHTMLImageTPVEvolution(self) -> str:
		data = self.strategiesData[self.ourStrategy]

		fig = go.Figure()
		fig.add_trace(go.Bar(name=self.ourStrategy, x=data.index, y=data['TotalPortfolioValue'] - 10000))
		fig.update_layout(title='Our Strategy Total Portfolio Value Evolution (10,000$ offset)')
		url = py.plot(fig, filename='evolTPV', auto_open=False)
		return url

	def generateHTMLImageMoneyInvestedEvolution(self) -> str:
		data = self.strategiesData[self.ourStrategy]

		fig = make_subplots(specs=[[{"secondary_y": True}]])
		fig.add_trace(go.Bar(name='Money Invested', x=data.index, y=data['MoneyInvestedToday']), secondary_y=False)
		fig.add_trace(go.Scatter(name='Open Price', x=data.index, y=data['Open']), secondary_y=True)
		fig.update_layout(title='Our Strategy\'s Money Invested each Day and Stock Market Open Price',
						  legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.93))
		url = py.plot(fig, filename='evolMIT', auto_open=False)
		return url
