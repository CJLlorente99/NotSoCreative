import customtkinter as ctk
import pandas as pd
import matplotlib.pyplot as plt
import pytz
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
import mplfinance as mpf
from datetime import datetime, date, timedelta
from matplotlib.figure import Figure
from PIL import Image
import yfinance as yf
import numpy as np
from googleStorageAPI import readBlobDf
import webbrowser

# Basic structure of the window

# Constants
yfStartDate = '2021-01-01'
date_test = '2023-01-25'
strategyName = 'bilstmWindowRobMMT1T2Legacy_25_1_2023'

"""
AUX FUNCTIONS TO BE MAPPED TO DATAFRAMES IN ORDER TO GENERATE USEFUL COLUMNS
"""
# Aux function for calculation if the operation is done in the morning (0) or afternoon (1)
def calculateOperation(value: str):
	if value.endswith('09:30:00'):
		return 0
	elif value.endswith('16:00:00'):
		return 1
	else:
		return 2  # never

# Aux function to just get the year, month and day
def shortenedDate(value: str):
	return value[:-9]

# Aux function to get a string according to the decision taken
def strDecision(value):
	if value > 0:
		return 'Buy'
	elif value < 0:
		return 'Sell'
	else:
		return 'Hold'

# Aux function to get a int according to the decision taken
def decisionFunction(value):
	if value > 0:
		return 1
	elif value < 0:
		return -1
	else:
		return 0

# Aux function to return 0 only if hold
def holdDecision(value):
	if value == 0:
		return 0
	else:
		return np.nan

# Aux function to return -1 only if sell
def sellDecision(value):
	if value == -1:
		return -1
	else:
		return np.nan

# Aux function to return 1 only if buy
def buyDecision(value):
	if value == 1:
		return 1
	else:
		return np.nan

"""
DATA RETRIEVAL AND METRICS CALCULATION FUNCTIONS
"""
# Function that returns the updated stock market data
def refreshDataSP500():
	end = date.today() + timedelta(days=1)
	stock_data = yf.download('^GSPC', start=yfStartDate, end=end)
	stock_data = stock_data.reset_index()
	stock_data.set_index(pd.DatetimeIndex(stock_data['Date']), inplace=True)

	return stock_data

# Function that return the updated strategy data
def refreshDataStrategy():

	# strategy data
	df_morning = readBlobDf()

	# just get "active" strategies
	latestDate = df_morning['Date'].unique()[-1]
	strategiesAlive = df_morning[df_morning['Date'] == latestDate]['investorStrategy'].values

	df_morning = df_morning[df_morning['investorStrategy'].isin(strategiesAlive)]

	df_morning['operation'] = df_morning['Date'].map(calculateOperation)
	df_morning['shortenedDate'] = df_morning['Date'].map(shortenedDate)

	# let's get rid of all afternoon operations
	df_morning = df_morning[df_morning['operation'] == 0]

	# change format of index
	df_morning.set_index(pd.DatetimeIndex(df_morning["shortenedDate"]), inplace=True)
	df_morning.drop(['shortenedDate'], axis=1, inplace=True)
	df_morning.index.name = "Date"

	return df_morning[df_morning["investorStrategy"] == strategyName]

# Function that calculates useful metrics
def getCurrentValue_metric(stock_data, strategyData):

	res = {}
	# Date info

	nowStockData = stock_data.index[-1].to_pydatetime()
	now = datetime.now(pytz.timezone('America/New_York'))
	if nowStockData.day < now.day:  # Either we're in a holiday or the market has not opened yet
		now = nowStockData.replace(hour=16, minute=0, second=0)
	else: # nowStockData.day == now.day
		if now > now.replace(hour=16, minute=0, second=0): # the market has already closed (16.00 - 24.00)
			now = now.replace(hour=16, minute=0, second=0)
		elif now < now.replace(hour=9, minute=30, second=0): # the market has not opened yet (00.00 - 9.30)
			now = now.replace(day=now.day-1,hour=9, minute=30, second=0)
	now = now.strftime('%Y-%m-%d %H:%M:%S')
	res['Date'] = np.append(strategyData['Date'], now)

	# PortfolioValues
	actualPortfolioValue = strategyData['MoneyInvested'].values[-1] * stock_data['Close'].values[-1] / \
						   stock_data['Open'].values[-1] + strategyData['MoneyNotInvested'].values[-1]
	aux = np.append(strategyData['TotalPortfolioValue'].values, actualPortfolioValue)

	# Calculate Portfoliovalue (day t)
	res['PortfolioValue'] = aux

	# MPV
	x = np.array([])
	for i in range(len(aux)):
		x = np.append(x, aux[:i+1].mean())
	res['MPV'] = x

	# Gain (%)
	initialPV = strategyData['TotalPortfolioValue'][0]
	x = np.array([])
	for i in range(len(aux)):
		x = np.append(x, (aux[i] - initialPV)/initialPV*100)
	res['PerGain'] = x

	# Gain (absolute)
	initialPV = strategyData['TotalPortfolioValue'][0]
	x = np.array([])
	for i in range(len(aux)):
		x = np.append(x, aux[i] - initialPV)
	res['AbsGain'] = x

	# StdV
	x = np.array([])
	for i in range(len(aux)):
		x = np.append(x, aux[:i+1].std())
	res['StdPV'] = x

	# MoneyInvested
	actualMoneyInvested = strategyData['MoneyInvested'].values[-1] * stock_data['Close'].values[-1] / \
						   stock_data['Open'].values[-1]
	res['MoneyInvested'] = np.append(strategyData['MoneyInvested'].values, actualMoneyInvested)

	# Money Not Invested
	res['MoneyNotInvested'] = np.append(strategyData['MoneyNotInvested'].values, strategyData['MoneyNotInvested'].values[-1])

	# Money Invested Today
	res['MoneyInvestedToday'] = np.append(strategyData['MoneyInvestedToday'].values, strategyData['MoneyInvestedToday'].values[-1])

	# Max Gain
	x = np.array([0])
	for i in range(len(aux)-1):
		perGainsPerDay = ((aux - np.roll(aux, 1))[1:]/aux[:-1])*100
		x = np.append(x, perGainsPerDay[:i+1].max())
	res['MaxGain'] = x

	return res

# -------------------------------------------

"""
PLOT GENERATING FUNCTIONS
"""
# Function to generate the plot containing the metric var
def show_metrics(metrics, var):

	x = metrics['Date']
	y = metrics[var]

	f = Figure(dpi=100)
	a = f.add_subplot(111)

	if ctk.get_appearance_mode() == 'Dark':
		plt.style.use('dark_background')

	plt.grid(visible=True, axis='both')

	if var == 'PerGain':
		a.set_ylabel('[%]')
	else:
		a.set_ylabel('[$]')
	a.plot(x, y)
	return f

# Function that generates the candlesticks plot
def show_graph(stock_data, i):

	if ctk.get_appearance_mode() == 'Dark':
		mode ='nightclouds'
	else:
		mode='default'

	# plot candelsticks
	color_candels = mpf.make_marketcolors(up=get_Color_Buy_Up_Candlesticks(), down=get_Color_Sell_Down_Candlesticks(), wick="inherit",
										  edge="inherit", volume="in")

	mpf_style = mpf.make_mpf_style(base_mpf_style=mode, marketcolors=color_candels, rc={'font.size':5})
	fig, axl = mpf.plot(stock_data.iloc[-i:, :], type='candle', volume=False, style=mpf_style, returnfig=True, figsize=(5,5),xrotation=0)

	mpf_style = mpf.make_mpf_style(base_mpf_style=mode, marketcolors=color_candels)
	fig, axl = mpf.plot(stock_data.iloc[-i:, :], type='candle', volume=False, style=mpf_style, returnfig=True,
						datetime_format='%Y-%m-%d', xrotation=0)


	return fig


# Function that generates the candlesticks plot for our investing period
def show_graph_test(data_csv, stock_data):

	if ctk.get_appearance_mode() == 'Dark':
		mode ='nightclouds'
	else:
		mode='default'

	# Color of candlesticks
	color_candels = mpf.make_marketcolors(up=get_Color_Buy_Up_Candlesticks(), down=get_Color_Sell_Down_Candlesticks(), wick="inherit",
										  edge="inherit", volume="in")

	aux = pd.DataFrame()
	stock_data['Decision'] = data_csv['Decision']
	stock_data['Decision'] = data_csv['Decision'].values
	aux['Hold'] = stock_data['Decision'].map(holdDecision) + (stock_data['Close'] + stock_data['Open'])/2
	aux['Sell'] = stock_data['Decision'].map(sellDecision) + stock_data['Close'] + 5
	aux['Buy'] = stock_data['Decision'].map(buyDecision)  + stock_data['Open'] - 5

	if len(data_csv) > 5:
		markerSizeToday = 200
	else:
		markerSizeToday = 500

	apds = []
	if not aux['Hold'].isnull().all():

		apds.append(mpf.make_addplot(aux["Hold"], type='scatter', marker='s', markersize=10, color=get_Color_Hold_Decision(), secondary_y=False))
	if not aux['Buy'].isnull().all():
		apds.append(mpf.make_addplot(aux["Buy"], type='scatter', marker='^', markersize=30, color=get_Color_Buy_Up_Decision(), secondary_y=False))
	if not aux['Sell'].isnull().all():
		apds.append(mpf.make_addplot(aux["Sell"], type='scatter', marker='v', markersize=50, color=get_Color_Sell_Down_Decision(), secondary_y=False))

	# plot candlesticks
	mpf_style = mpf.make_mpf_style(base_mpf_style=mode, marketcolors=color_candels)
	fig, axl = mpf.plot(stock_data, type='candle', volume=False, style=mpf_style, returnfig=True, addplot=apds,
						datetime_format='%Y-%m-%d', xrotation=0)

	return fig

"""
SYSTEM FUNCTIONS
"""
# Function to be called once the window is closed (avoids hanging processes)
def _quit():
	root.quit()
	root.destroy()

"""
COLOR FUNCTIONS
"""
#color Mode for colorblind people
def color_mode():
	# returns 0 or 1 for differnet color modes
	color = 0
	if switch_var.get() == "off":
		color = 0
	elif switch_var.get() == "on":
		color = 1
	else:
		color = 0
	return color

def get_Color_Buy_Up_Candlesticks():
	if color_mode() == 0:
		color = "#A3FF87"
	elif color_mode() == 1:
		color = "#5D02FF"
	else:
		color = 0
	return color

def get_Color_Sell_Down_Candlesticks():
	if color_mode() == 0:
		color = "#FF8A8A"
	elif color_mode() == 1:
		color = "#E60400"
	else:
		color = 0
	return color

def get_Color_Buy_Up_Decision():
	if color_mode() == 0:
		color = "#00FF00"
	elif color_mode() == 1:
		color = "#5D02FF"
	else:
		color = 0
	return color


def get_Color_Hold_Decision():
	if color_mode() == 0:
		color = "#FFA500"
	elif color_mode() == 1:
		color = "#FFA500"
	else:
		color = 0
	return color


def get_Color_Sell_Down_Decision():
	if color_mode() == 0:
		color = "#ff0000"
	elif color_mode() == 1:
		color = "#E60400"
	else:
		color = 0
	return color


"""
FUNCTION CALLED FROM THE MAINLOOP (THIS IS, THAT ARE CALLED BY EVENTS IN THE MAIN WINDOW)
"""
# open new window for Information
def openNewWindow():
	# Toplevel object which will
	# be treated as a new window
	newWindow = ctk.CTkToplevel(root)
	textbox = ctk.CTkTextbox(newWindow, width= 400, height= 400)
	textbox.insert("0.0",
				   "Information\n\n "  + "Portfolio Value = Cash + Current Stock Value\n\n" + " Gain (%) " + "= Percentage difference between the purchase price and the current value of the shares\n\n" + " Gain (absolute) = Difference between the current value of the shares and the purchase price of the shares\n\n" + "Mean Portfolio Value= Mean of Portfolio Value for all test days Invested Money = Money that has been invested in the stock market\n\n" + " Invested Money Today = Money that is invested today\n\n" + " Money not Invested = Money that hasn`t been invested in the stock market" + " Standard Deviation = Describes the spread of the PVs (?) \n\n" + " Max. gain per Day = Maximum percentage gain on one day\n\n")

	# "\033[1m"
	#side= BOTTOM,
	textbox.pack(side = TOP, padx=0, pady=(15, 15))

	# sets the title of the
	# Toplevel widget
	newWindow.title("Information")
	newWindow.geometry("300x300")

"""
APPEARANCE AND SCALING FUNCTIONS
"""
# changes appearance mode
def change_appearance_mode_event(new_appearance_mode: str):
	ctk.set_appearance_mode(new_appearance_mode)
	ctk.get_appearance_mode()

# changes scaling
def change_scaling_event(new_scaling: str):
	new_scaling_float = int(new_scaling.replace("%", "")) / 100
	ctk.set_widget_scaling(new_scaling_float)

"""
UPDATE FUNCTION TO BE CALLED EVERY TIME DATA IS TO BE UPDATED
"""
def update():

	# Check appearance and scaling
	ctk.set_appearance_mode(appearance_mode_optionmenu.get())
	new_scaling_float = int(scaling_optionmenu.get().replace("%", "")) / 100
	ctk.set_widget_scaling(new_scaling_float)

	# Retrieve and calculate for new data
	strategyData = refreshDataStrategy()
	strategyDataTest = strategyData.iloc[strategyData.index.get_loc(date_test):]
	lastDataTest = strategyDataTest.index[-1]
	stockData = refreshDataSP500()
	stockDataTest = stockData.iloc[stockData.index.get_loc(date_test):stockData.index.get_loc(lastDataTest)+1]

	metrics = getCurrentValue_metric(stockDataTest, strategyDataTest)
	strategyDataTest['StrDecision'] = strategyDataTest['MoneyInvestedToday'].map(strDecision)
	strategyDataTest['Decision'] = strategyDataTest['MoneyInvestedToday'].map(decisionFunction)

	# nur letzter Wert
	sidebar_label_2.configure(text=str(round(metrics['PortfolioValue'][-1],2))+' $')
	sidebar_label_4.configure(text=str(round(metrics['PerGain'][-1], 2))+' %')
	sidebar_label_6.configure(text=str(round(metrics['MPV'][-1], 2))+' $')

	# Update last decisions
	updateLastDecisions(strategyDataTest)

	labelL2.configure(text=str(round(metrics['MoneyInvested'][-1], 2))+' $')
	labelL4.configure(text=str(round(metrics['MoneyNotInvested'][-1], 2))+' $')

	# nur letzter Wert
	labelL6.configure(text=str(round(metrics['StdPV'][-1], 2))+' $')
	labelL8.configure(text=str(round(metrics['MaxGain'][-1], 2))+' %')

	# Update plots
	updateCandlesticks(stockData, strategyDataTest, stockDataTest)
	updateMetrics(metrics)

"""
MAIN WINDOW AND WIDGET INITIALIZATION
"""

# Set default appearance
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

# Main Window
root = ctk.CTk()
root.geometry('1500x900')
root.minsize(1500, 900)
root.title("Stock Market Prediction Engine")
root.protocol("WM_DELETE_WINDOW", _quit)

# configure the grid
root.grid_columnconfigure(1, weight=1)  # Will contain logo, appearance, info, update button and apply changes on appearance button
root.grid_columnconfigure(2, weight=1)  # Will contain the main figures (MPV, AbsGain...)
root.grid_columnconfigure(3, weight=6)  # Will contain the plots
root.grid_rowconfigure(0, weight=1)  # Each column will contain a frame that will further divide the space or the widgets will be packed one on top of the other

switch_var = ctk.StringVar(value="off")

"""
FIRST COLUMN WIDGET CREATION
"""
# Create frame that will contain everything
sidebar_frame = ctk.CTkFrame(root, corner_radius=0)
sidebar_frame.grid(row=0, column=1, sticky="nsew")

# Title and logo
# Load image
img = ctk.CTkImage( light_image=Image.open('png.png'),dark_image=Image.open("png.png"),size=(200,200))
logo_label =ctk.CTkLabel(sidebar_frame,text='', image=img)

# Create label with the title
title_label = ctk.CTkLabel(sidebar_frame, text="Prediction Engine S&P 500",
						   font=ctk.CTkFont(size=26, weight="bold"))

# Pack the things one on top of the other
title_label.pack(side=TOP, padx=20, pady=(20, 10))
logo_label.pack(side=TOP, padx=20, pady=(10, 40))

# Option menu
option_label = ctk.CTkLabel(sidebar_frame, text='Options:', font=ctk.CTkFont(size=16))
option_label.pack(side=TOP, padx=20, pady=(0, 0))

# Update button
data_update_button= ctk.CTkButton(sidebar_frame, text="Update Data", command=update)
data_update_button.pack(side=TOP, padx=20, pady=(15, 15))

# Info button
sidebar_button = ctk.CTkButton(sidebar_frame, text="Info/Help", command=openNewWindow)
sidebar_button.pack(side=TOP, padx=20, pady=(15, 15))

# Color switch
switch_1 = ctk.CTkSwitch(sidebar_frame, text="Color", variable=switch_var, command=color_mode, onvalue="on",
						 offvalue="off")
switch_1.pack(side=TOP, padx=20, pady=(5, 0))

# Appearance label
appearance_mode_label = ctk.CTkLabel(sidebar_frame, text="Appearance Mode:", anchor="w")
appearance_mode_label.pack(side=TOP, padx=20, pady=(5, 0))

# Appearance menu
appearance_mode_optionmenu = ctk.CTkOptionMenu(sidebar_frame, values=["Light", "Dark", "System"])
appearance_mode_optionmenu.pack(side=TOP, padx=20, pady=(0, 0))

# Scaling label
scaling_label = ctk.CTkLabel(sidebar_frame, text="UI Scaling:", anchor="w")
scaling_label.pack(side=TOP, padx=20, pady=(10, 0))

# Scaling menu
scaling_optionmenu = ctk.CTkOptionMenu(sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"])
scaling_optionmenu.pack(side=TOP, padx=20, pady=(0, 10))

# Apply changes button
apply_update_button= ctk.CTkButton(sidebar_frame, text="Apply Changes", command=update)
apply_update_button.pack(side=TOP, padx=20, pady=(15, 15))

"""
SECOND COLUMN WIDGET CREATION
"""
# Creation of a frame with 3 different spaces
important_Values_frame = ctk.CTkFrame(root)
important_Values_frame.grid(row=0, column=2, padx=(20, 20), pady=(10, 0), sticky="nsew")
important_Values_frame.grid_columnconfigure(0, weight=1)
important_Values_frame.grid_rowconfigure(0, weight=1)
important_Values_frame.grid_rowconfigure(1, weight=1)
important_Values_frame.grid_rowconfigure(2, weight=1)

# FIRST ROW OF SECOND COLUMN
# Create a frame to contain all the labels in the first row
important_Values_frame1 = ctk.CTkFrame(important_Values_frame)
important_Values_frame1.grid(row=0, column=0, padx=(20, 20), pady=(10, 10), sticky="nsew")

# Portfolio label
sidebar_label_0=ctk.CTkLabel(important_Values_frame1, text='Portfolio', font=ctk.CTkFont(size=22, weight='bold'))
sidebar_label_0.pack(padx=20, pady=(20, 20))

# Portfolio value title label
sidebar_label_1 = ctk.CTkLabel(important_Values_frame1, text='Portfolio Value:', font=ctk.CTkFont(size=16, weight='bold'))
sidebar_label_1.pack( padx=20, pady=(0, 0))

# Portfolio value label
sidebar_label_2 = ctk.CTkLabel(important_Values_frame1,
							   text=str(0)+' $',
							   font=ctk.CTkFont(size=16))
sidebar_label_2.pack( padx=20, pady=(0,0))

# Percentage gain title label
sidebar_label_3 = ctk.CTkLabel(important_Values_frame1, text=' Gain(%):', font=ctk.CTkFont(size=16, weight='bold'))
sidebar_label_3.pack( padx=20, pady=(10, 0))

# Percentage gain value label
sidebar_label_4 = ctk.CTkLabel(important_Values_frame1, text=str(0)+' %',
							   font=ctk.CTkFont(size=16))
sidebar_label_4.pack(padx=20)

# Mean portfolio title label
sidebar_label_5 = ctk.CTkLabel(important_Values_frame1, text='Mean Portfolio Value:',
							   font=ctk.CTkFont(size=16, weight='bold'))
sidebar_label_5.pack( padx=20, pady=(10, 0))

# Mean portfolio value label
sidebar_label_6 = ctk.CTkLabel(important_Values_frame1, text=str(0)+' $',
							   font=ctk.CTkFont(size=16))
sidebar_label_6.pack( padx=20)

# SECOND ROW OF THE SECOND COLUMN
# Create a frame to contain all labels in the second row
recommendation_frame = ctk.CTkFrame(important_Values_frame)
recommendation_frame.grid(row=1, column=0, padx=(20, 20), pady=(10, 0), sticky="nsew")

# Recommend title label
labelrecommend = ctk.CTkLabel(recommendation_frame, text='Recommendation', font=ctk.CTkFont(size=22, weight="bold"))
labelrecommend.pack( padx=10, pady=(40, 5))

# Current recommendation label
labelcurrentRecommend = ctk.CTkLabel(recommendation_frame, text='Buy', font=ctk.CTkFont(size=20, weight='bold'),
									 text_color=get_Color_Buy_Up_Decision())
labelcurrentRecommend.pack( pady=(5,0), padx=20)

# Last recommendations title label
labelLastRecommend = ctk.CTkLabel(recommendation_frame, text='Last Recommendations', font=ctk.CTkFont(size=17, weight="bold"))
labelLastRecommend.pack( pady=(10, 5), padx=20)

# Second Last recommendation label
labelLRecommend1 = ctk.CTkLabel(recommendation_frame, text='No Last Recommendation', font=ctk.CTkFont(size=16))
labelLRecommend1.pack( pady=5, padx=20)

# Third last recommendation label
labelLRecommend2 = ctk.CTkLabel(recommendation_frame,text="",font=ctk.CTkFont(size=16))
labelLRecommend2.pack( pady=5, padx=20)

# Fourth last recommendation label
labelLRecommend3 = ctk.CTkLabel(recommendation_frame,text='', font=ctk.CTkFont(size=16))
labelLRecommend3.pack( pady=5, padx=20)

# Update function to be called on each update
def updateLastDecisions(strategyDataTest):
	# Check how many last recommendation we have and update appropriately
	if len(strategyDataTest) < 1:
		labelLRecommend1.configure(text='No Last Recommendation', font=ctk.CTkFont(size=16))
	elif len(strategyDataTest) == 2:
		labelLRecommend1.configure(
			text=pd.to_datetime(strategyDataTest.index.values[-2]).strftime('%Y-%m-%d') + ':    ' +
				 strategyDataTest['StrDecision'][
					 -2], font=ctk.CTkFont(size=16))
	elif len(strategyDataTest) == 3:
		labelLRecommend1.configure(
			text=pd.to_datetime(strategyDataTest.index.values[-2]).strftime('%Y-%m-%d') + ':    ' +
				 strategyDataTest['StrDecision'][
					 -2], font=ctk.CTkFont(size=16))
		labelLRecommend2.configure(
			text=pd.to_datetime(strategyDataTest.index.values[-3]).strftime('%Y-%m-%d') + ':    ' +
				 strategyDataTest['StrDecision'][
					 -3], font=ctk.CTkFont(size=16))
	elif len(strategyDataTest) > 3:
		labelLRecommend1.configure(
			text=pd.to_datetime(strategyDataTest.index.values[-2]).strftime('%Y-%m-%d') + ':    ' +
				 strategyDataTest['StrDecision'][
					 -2], font=ctk.CTkFont(size=16))
		labelLRecommend2.configure(
			text=pd.to_datetime(strategyDataTest.index.values[-3]).strftime('%Y-%m-%d') + ':    ' +
				 strategyDataTest['StrDecision'][
					 -3], font=ctk.CTkFont(size=16))
		labelLRecommend3.configure(
			text=pd.to_datetime(strategyDataTest.index.values[-4]).strftime('%Y-%m-%d') + ':    ' +
				 strategyDataTest['StrDecision'][
					 -4], font=ctk.CTkFont(size=16))

	# Current recommendation
	labelrecommend.configure(text='Recommendation\n' + pd.to_datetime(strategyDataTest.index.values[-1]).strftime('%Y-%m-%d'))
	if strategyDataTest['MoneyInvestedToday'][-1] > 0:
		labelcurrentRecommend.configure(text='Buy', font=ctk.CTkFont(size=20, weight='bold'),
										text_color=get_Color_Buy_Up_Decision())
	elif strategyDataTest['MoneyInvestedToday'][-1] < 0:
		labelcurrentRecommend.configure(text='Sell', font=ctk.CTkFont(size=20, weight='bold'),
										text_color=get_Color_Sell_Down_Decision())
	else:
		labelcurrentRecommend.configure(text='Hold', font=ctk.CTkFont(size=20, weight='bold'),
										text_color=get_Color_Hold_Decision())

# THIRD ROW OF THE SECOND COLUMN
# Create a frame to contain all labels in the third row
values_frame = ctk.CTkFrame(important_Values_frame)
values_frame.grid(row=2, column=0, padx=(20, 20), pady=(10, 10), sticky="nsew")

# Other metrics title label
labelL0=ctk.CTkLabel(values_frame, text='Other Metrics', font=ctk.CTkFont(size=22, weight='bold'))
labelL0.pack(pady=(20, 20), padx=20)

# Invested money title label
labelL1 = ctk.CTkLabel(values_frame, text='Invested Money:', font=ctk.CTkFont(size=16, weight="bold"))
labelL1.pack( pady=(10, 0), padx=20)

# Invested money value label
labelL2 = ctk.CTkLabel(values_frame, text=str(0)+' $',
					   font=ctk.CTkFont(size=16))
labelL2.pack( pady=(0, 0), padx=20)

# Not invested money title label
labelL3 = ctk.CTkLabel(values_frame, text='Money Not Invested:', font=ctk.CTkFont(size=16, weight="bold"))
labelL3.pack( pady=(0, 0), padx=20)

# Not invested money value label
labelL4 = ctk.CTkLabel(values_frame, text=str(0)+' $',
					   font=ctk.CTkFont(size=16))
labelL4.pack( pady=(0, 0), padx=20)

# Std title label
labelL5 = ctk.CTkLabel(values_frame, text='Standard Deviation:', font=ctk.CTkFont(size=16, weight="bold"))
labelL5.pack(side=TOP, pady=(0, 0), padx=20)

# Std value label
labelL6 = ctk.CTkLabel(values_frame, text=str(0)+' $',
					   font=ctk.CTkFont(size=16))
labelL6.pack( pady=(0, 0), padx=20)

# Max Gain title label
labelL7 = ctk.CTkLabel(values_frame, text='Max Gain per Day:', font=ctk.CTkFont(size=16, weight="bold"))
labelL7.pack(pady=(0, 0), padx=20)

# Max gain value label
labelL8 = ctk.CTkLabel(values_frame, text=str(0)+' %',
					   font=ctk.CTkFont(size=16))
labelL8.pack(pady=(0, 10), padx=20)

"""
THIRD COLUMN WIDGET CREATION
"""
# Create a frame that will contain the tabs of both "set" of plots
framePlots = ctk.CTkFrame(root)
framePlots.grid(row=0, column=3, padx=(20, 20), sticky="nsew")
framePlots.grid_rowconfigure(0, weight=1)
framePlots.grid_rowconfigure(1, weight=1)
framePlots.grid_columnconfigure(0, weight=1)

# Tab creation for candlestick plots
tabview = ctk.CTkTabview(framePlots)
tabview.grid(row=0, column=0, padx=(5, 5), pady=(5, 10), sticky="nsew")
tab2W = tabview.add("2 Weeks")
tab1M = tabview.add("1M")
tab6M = tabview.add("6M")
tab1Y = tabview.add("1Y")
tabview.tab("2 Weeks")
tabview.tab("1M")
tabview.tab("6M")
tabview.tab("1Y")

# Tab creation for metric plots
tabview = ctk.CTkTabview(framePlots)
tabview.grid(row=1, column=0, padx=(5, 5), pady=(5, 0), sticky="nsew")
tabMPV = tabview.add("Mean_PV")
tabTPV = tabview.add("TotalPortfolioValue")
tabAbsGain = tabview.add("Gain (absolute)")
tabPerGain = tabview.add("Gain (percentage)")
tabMIT = tabview.add("Money Invested Today")
tabview.tab("Mean_PV")
tabview.tab("TotalPortfolioValue")
tabview.tab("Gain (absolute)")
tabview.tab("Gain (percentage)")
tabview.tab("Money Invested Today")

# The plots will be embedded in the tabs through the appropriate update functions

"""
Plots update functions
"""
def updateCandlesticks(stockData, strategyDataTest, stockDataTest):
	# Before updating, destroy all past plots that could be hanging
	for obj in tab2W.winfo_children():
		obj.destroy()
	for obj in tab1M.winfo_children():
		obj.destroy()
	for obj in tab6M.winfo_children():
		obj.destroy()
	for obj in tab1Y.winfo_children():
		obj.destroy()

	# Embed 2W plot into tab
	line = FigureCanvasTkAgg(show_graph_test(strategyDataTest, stockDataTest), tab2W)
	line.draw()
	line.get_tk_widget().pack(side='top', fill='both', expand=True)
	# Navigation bar
	toolbarFrame = Frame(master=tab2W)
	toolbarFrame.place(relx=0, rely=0.95)
	NavigationToolbar2Tk(line, toolbarFrame)

	# Embed 1M plot into tab
	line = FigureCanvasTkAgg(show_graph(stockData, 18), tab1M)
	line.draw()
	line.get_tk_widget().pack(side='top', fill='both', expand=True)
	# Navigation bar
	toolbarFrame = Frame(master=tab1M)
	toolbarFrame.place(relx=0, rely=0.95)
	NavigationToolbar2Tk(line, toolbarFrame)

	# Embed 6M plot into tab
	line = FigureCanvasTkAgg(show_graph(stockData, 123), tab6M)
	line.draw()
	line.get_tk_widget().pack(side='top', fill='both', expand=True)
	# Navigation bar
	toolbarFrame = Frame(master=tab6M)
	toolbarFrame.place(relx=0, rely=0.95)
	NavigationToolbar2Tk(line, toolbarFrame)

	# Embed 1Y plot into tab
	line = FigureCanvasTkAgg(show_graph(stockData, 250), tab1Y)
	line.draw()
	line.get_tk_widget().pack(side='top', fill='both', expand=True)
	# Navigation bar
	toolbarFrame = Frame(master=tab1Y)
	toolbarFrame.place(relx=0, rely=0.95)
	NavigationToolbar2Tk(line, toolbarFrame)

def updateMetrics(metrics):
	# Before updating, destroy all past plots that could be hanging
	for obj in tabMPV.winfo_children():
		obj.destroy()
	for obj in tabMIT.winfo_children():
		obj.destroy()
	for obj in tabTPV.winfo_children():
		obj.destroy()
	for obj in tabPerGain.winfo_children():
		obj.destroy()
	for obj in tabAbsGain.winfo_children():
		obj.destroy()

	# Embed MPV plot into tab
	line = FigureCanvasTkAgg(show_metrics(metrics, 'MPV'), tabMPV)
	line.draw()
	line.get_tk_widget().pack(side='top', fill='both', expand=True)
	# Navigation bar
	toolbarFrame = Frame(master=tabMPV)
	toolbarFrame.place(relx=0, rely=0.945)
	NavigationToolbar2Tk(line, toolbarFrame)

	# Embed PV plot into tab
	line = FigureCanvasTkAgg(show_metrics(metrics, 'PortfolioValue'), tabTPV)
	line.draw()
	line.get_tk_widget().pack(side='top', fill='both', expand=True)
	# Navigation bar
	toolbarFrame = Frame(master=tabTPV)
	toolbarFrame.place(relx=0, rely=0.945)
	NavigationToolbar2Tk(line, toolbarFrame)

	# Embed AbsGain plot into tab
	line = FigureCanvasTkAgg(show_metrics(metrics, 'AbsGain'), tabAbsGain)
	line.draw()
	line.get_tk_widget().pack(side='top', fill='both', expand=True)
	# Navigation bar
	toolbarFrame = Frame(master=tabAbsGain)
	toolbarFrame.place(relx=0, rely=0.945)
	NavigationToolbar2Tk(line, toolbarFrame)

	# Embed PerGain plot into tab
	line = FigureCanvasTkAgg(show_metrics(metrics, 'PerGain'), tabPerGain)
	line.draw()
	line.get_tk_widget().pack(side='top', fill='both', expand=True)
	# Navigation bar
	toolbarFrame = Frame(master=tabPerGain)
	toolbarFrame.place(relx=0, rely=0.945)
	NavigationToolbar2Tk(line, toolbarFrame)

	# Embed MoneyInvestedToday plot into tab
	line = FigureCanvasTkAgg(show_metrics(metrics, 'MoneyInvestedToday'), tabMIT)
	line.draw()
	line.get_tk_widget().pack(side='top', fill='both', expand=True)
	# Navigation bar
	toolbarFrame = Frame(master=tabMIT)
	toolbarFrame.place(relx=0, rely=0.945)
	NavigationToolbar2Tk(line, toolbarFrame)

update()
root.mainloop()