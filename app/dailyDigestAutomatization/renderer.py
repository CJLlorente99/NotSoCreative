from datetime import timedelta, datetime
import chart_studio.plotly as py
import chart_studio.tools
import pandas as pd
import pytz
import yfinance as yf
from googleStorageAPI import readBlobDf
from jinja2 import Environment, FileSystemLoader
import plotly.graph_objects as go

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
		self.env = Environment(loader=FileSystemLoader(searchpath='./htmlTemplates/'))

	def getYfData(self) -> pd.DataFrame:
		"""
		This method should call yfinance and retrieve the data until today
		:return: pd.DataFrame containing the stock market data
		"""
		return yf.download('^GSPC', (self.todayDate-timedelta(10)).strftime('%Y-%m-%d'), self.todayDate.strftime('%Y-%m-%d'))

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

	def renderShortDailyDigest(self) -> str:
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
		renderedHTML = template.render(name='Kim Erik',
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

	def renderLongDailyDigest(self) -> str:
		"""
		This method should render the short daily digest template
		:return: html string
		"""
		return ''

	def generateHTMLImageMPV(self) -> str:
		latestMorning = self.strategiesData[list(self.strategiesData.keys())[0]].index[-2]
		df = pd.DataFrame()

		for key in self.strategiesData:
			data = self.strategiesData[key]
			data = data.query('index == @latestMorning')
			df = pd.concat([df, data], ignore_index=True)

		fig = go.Figure()
		fig.add_trace(go.Bar(x=df['investorStrategy'], y=df['MPV'] - 10000))
		fig.update_layout(title='Mean Portfolio Value Comparison (10,000$ offset)')
		url = py.plot(fig, filename='example', auto_open=False)
		return url


if __name__ == '__main__':
	todayDate = datetime.now(pytz.timezone('America/New_York'))
	rend = Renderer(todayDate)
	renderedHTML = rend.renderShortDailyDigest()
