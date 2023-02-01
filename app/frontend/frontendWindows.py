import customtkinter as ctk
import pandas as pd
import matplotlib.pyplot as plt
import pytz
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import *
import mplfinance as mpf
from datetime import datetime, date, timedelta
from matplotlib.figure import Figure
from PIL import Image
import yfinance as yf
import numpy as np
from googleStorageAPI import readBlobDf, subscribeBlobEmailsDf, unsubscribeBlobEmailsDf
import sys
import webbrowser

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

# Aux function to get an int according to the decision taken
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
def refreshDataStrategy(allData=False):

	# strategy data
	df_morning = readBlobDf()

	# just get "active" strategies
	latestDate = df_morning['Date'].unique()[-1]
	strategiesAlive = df_morning[df_morning['Date'] == latestDate]['investorStrategy'].values

	df_morning = df_morning[df_morning['investorStrategy'].isin(strategiesAlive)]

	df_morning['operation'] = df_morning['Date'].map(calculateOperation)
	df_morning['shortenedDate'] = df_morning['Date'].map(shortenedDate)

	# let's get rid of all afternoon operations
	if not allData:
		df_morning = df_morning[df_morning['operation'] == 0]

	# change format of index
	df_morning.set_index(pd.DatetimeIndex(df_morning["shortenedDate"]), inplace=True)
	df_morning.drop(['shortenedDate'], axis=1, inplace=True)
	df_morning.index.name = "Date"

	return df_morning[df_morning["investorStrategy"] == strategyName]

# Function that calculates useful metrics
def getCurrentValue_metric(stock_data, strategyData):

	res = {}
	offset = offset_var.get()
	offset_val = strategyData['TotalPortfolioValue'][0]
	# Date info

	nowStockData = stock_data.index[-1].to_pydatetime()
	now = datetime.now(pytz.timezone('America/New_York'))
	if nowStockData.replace(tzinfo=pytz.timezone('America/New_York')).date() < now.date():  # Either we're in a holiday or the market has not opened yet
		now = nowStockData.replace(hour=16, minute=0, second=0)
	else: # nowStockData.date() == now.date()
		if now > now.replace(hour=16, minute=0, second=0): # the market has already closed (16.00 - 24.00)
			now = now.replace(hour=16, minute=0, second=0)
	now = now.strftime('%Y-%m-%d %H:%M:%S')
	res['Date'] = np.append(strategyData['Date'], now)

	# PortfolioValues
	actualPortfolioValue = strategyData['MoneyInvested'].values[-1] * stock_data['Close'].values[-1] / \
						   stock_data['Open'].values[-1] + strategyData['MoneyNotInvested'].values[-1]
	aux = np.append(strategyData['TotalPortfolioValue'].values, actualPortfolioValue)

	# Calculate Portfoliovalue (day t)
	res['PortfolioValue'] = aux if not offset else aux - offset_val

	# MPV
	x = np.array([])
	for i in range(len(aux)):
		x = np.append(x, aux[:i+1].mean())
	res['MPV'] = x if not offset else x - offset_val

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
	res['MoneyInvestedToday'] = np.append(strategyData['MoneyInvestedToday'].values, 0)

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

	bar = bar_var.get()

	x = metrics['Date']
	y = metrics[var]

	fig = Figure(figsize=(0.7*tabviewMetrics.winfo_width()/ppi, 0.7*tabviewMetrics.winfo_height()/ppi))
	a = fig.add_subplot(111)

	if ctk.get_appearance_mode() == 'Dark':
		plt.style.use('dark_background')

	plt.grid(visible=True, axis='both')

	if var == 'PerGain':
		a.set_ylabel('[%]')
	else:
		a.set_ylabel('[$]')

	if bar:
		a.bar(x, y)
	elif not bar:
		a.plot(x,y)
	a.tick_params(axis='x', labelrotation=15, labelsize=ppi*0.13)
	a.tick_params(axis='y', labelsize=ppi * 0.13)
	return fig

# Function that generates the candlesticks plot
def show_graph(stock_data, i):

	if ctk.get_appearance_mode() == 'Dark':
		mode ='nightclouds'
	else:
		mode='default'

	# plot candelsticks
	color_candels = mpf.make_marketcolors(up=get_Color_Buy_Up_Candlesticks(), down=get_Color_Sell_Down_Candlesticks(), wick="inherit",
										  edge="inherit", volume="in")
	mpf_style = mpf.make_mpf_style(base_mpf_style=mode, marketcolors=color_candels, rc={'font.size':ppi*0.13})
	fig, axl = mpf.plot(stock_data.iloc[-i:, :], type='candle', volume=False, style=mpf_style, returnfig=True,
						datetime_format='%Y-%m-%d', xrotation=15, figsize=(0.7*tabviewCandlesticks.winfo_width()/ppi, 0.7*tabviewCandlesticks.winfo_height()/ppi))

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
	aux['Sell'] = stock_data['Decision'].map(sellDecision) + stock_data['Open'] + 5
	aux['Buy'] = stock_data['Decision'].map(buyDecision)  + stock_data['Open'] - 5

	if 8 >= len(data_csv) > 5:
		markerSizeToday = 0.05*ppi**2
	elif len(data_csv) > 8:
		markerSizeToday = 0.03 * ppi ** 2
	else:
		markerSizeToday = 0.13*ppi**2

	apds = []
	if not aux['Hold'].isnull().all():
		apds.append(mpf.make_addplot(aux["Hold"], type='scatter', marker='s', markersize=markerSizeToday, color=get_Color_Hold_Decision(), secondary_y=False))
	if not aux['Buy'].isnull().all():
		apds.append(mpf.make_addplot(aux["Buy"], type='scatter', marker='^', markersize=markerSizeToday, color=get_Color_Buy_Up_Decision(), secondary_y=False))
	if not aux['Sell'].isnull().all():
		apds.append(mpf.make_addplot(aux["Sell"], type='scatter', marker='v', markersize=markerSizeToday, color=get_Color_Sell_Down_Decision(), secondary_y=False))


	# plot candlesticks
	mpf_style = mpf.make_mpf_style(base_mpf_style=mode, marketcolors=color_candels, rc={'font.size':ppi*0.13})
	fig, axl = mpf.plot(stock_data, type='candle', volume=False, style=mpf_style, returnfig=True, addplot=apds,
						datetime_format='%Y-%m-%d', xrotation=15, figsize=(0.7*tabviewCandlesticks.winfo_width()/ppi, 0.7*tabviewCandlesticks.winfo_height()/ppi))

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
def get_Color_Buy_Up_Candlesticks():
	if color_var.get():
		color = "#A3FF87"
	elif not color_var.get():
		color = "#5D02FF"
	else:
		color = 0
	return color

def get_Color_Sell_Down_Candlesticks():
	if color_var.get():
		color = "#FF8A8A"
	elif not color_var.get():
		color = "#E60400"
	else:
		color = 0
	return color

def get_Color_Buy_Up_Decision():
	if color_var.get():
		color = "#00FF00"
	elif not color_var.get():
		color = "#5D02FF"
	else:
		color = 0
	return color


def get_Color_Hold_Decision():
	if color_var.get():
		color = "#FFA500"
	elif not color_var.get():
		color = "#FFA500"
	else:
		color = 0
	return color


def get_Color_Sell_Down_Decision():
	if color_var.get():
		color = "#ff0000"
	elif not color_var.get():
		color = "#E60400"
	else:
		color = 0
	return color

def get_Color_Market_Closed():
	if color_var.get():
		color = "#FF8A8A"
	elif not color_var.get():
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
	textbox = ctk.CTkTextbox(newWindow, width= 0.278*ws, height= 0.444*hs)
	textbox.insert("0.0",
				   "Information\n\n " + " Portfolio Value " + "= Cash (money not invested) + current value of shares that we hold\n\n" + " Gain (%) " + "= Percentage difference between the purchase price and the current value of the shares\n\n" + " Gain (absolute) " + "= Difference between the current value of the shares and the purchase price of the shares\n\n" + " Mean Portfolio Value " + "=  Mean of all portfolio values\n\n " + " Money Invested " + "=  Money that has been invested in the stock market \n\n" + " Money Invested Today " + "= Money that is invested today\n\n" + " Money not Invested " + "= Money that hasn`t been invested in the stock market\n\n" + " Standard Deviation " + "= Describes the spread of all portfolio values\n\n" + " Max. gain per day " + "= Maximum percentage gain on one day\n\n")

	# "\033[1m"
	#side= BOTTOM,
	textbox.pack(side = TOP, padx=0, pady=(0.010*ws, 0.017*hs))

	# sets the title of the
	# Toplevel widget
	newWindow.title("Information")
	newWindow.geometry("%dx%d+%d+%d" % (int(0.208*ws), int(0.333*hs), int(ws/2), int(hs/2)))
	newWindow.maxsize(0.208*ws, 0.333*hs)
	newWindow.minsize(0.208*ws, 0.333*hs)

def windowDailyDigest():
	def invalidNameCheck():
		res = True
		name_text_box.configure(fg_color='green')
		if "," in name_text_box.get(1.0, "end-1c"):
			pastText = email_text_box.get(1.0, "end-1c")
			name_text_box.delete('0.0', 'end')
			name_text_box.insert('0.0', pastText.replace(',',''))
			name_text_box.configure(fg_color='red')
			res = False

		email_text_box.configure(fg_color='green')
		if "," in email_text_box.get(1.0, "end-1c"):
			pastText = email_text_box.get(1.0, "end-1c")
			email_text_box.delete('0.0', 'end')
			email_text_box.insert("0.0", pastText.replace(',', ''))
			email_text_box.configure(fg_color='red')
			res = False
		return res

	def subscribeCallback():
		if not invalidNameCheck():
			return
		subscribeBlobEmailsDf(name_text_box.get(1.0, "end-1c"),
							  email_text_box.get(1.0, "end-1c"),
							  switch_digest.get())
		newWindow.destroy()

	def unsubscribeCallback():
		if not invalidNameCheck():
			return
		unsubscribeBlobEmailsDf(name_text_box.get(1.0, "end-1c"),
								email_text_box.get(1.0, "end-1c"),
								switch_digest.get())
		newWindow.destroy()

	# Toplevel object which will
	# be treated as a new window
	newWindow = ctk.CTkToplevel(root)

	# Name label
	name_label = ctk.CTkLabel(newWindow, text="Enter your name:", anchor="w")
	name_label.pack(side=TOP, padx=0.014*ws, pady=(0.006*hs, 0))

	# Name value
	name_text_box = ctk.CTkTextbox(newWindow, height=0.044*hs, width=0.208*ws)
	name_text_box.pack(side=TOP, padx=0.014*ws, pady=(0.006*hs, 0))

	# Email label
	email_label = ctk.CTkLabel(newWindow, text="Enter your email:", anchor="w")
	email_label.pack(side=TOP, padx=0.014*ws, pady=(0.006*hs, 0))

	# Email value
	email_text_box = ctk.CTkTextbox(newWindow, height=0.044*hs, width=0.208*ws)
	email_text_box.pack(side=TOP, padx=0.014*ws, pady=(0.006*hs, 0))

	# Switch short/long digest
	switch_digest = ctk.CTkSwitch(newWindow, text="Short/Long Digest", variable=digest_var, onvalue='Long',
								  offvalue='Short')
	switch_digest.pack(side=TOP, padx=0.014*ws, pady=(0.006*hs, 0))

	# Frame to hold buttons
	frame_buttons = ctk.CTkFrame(newWindow)
	frame_buttons.grid_rowconfigure(0, weight=1)
	frame_buttons.grid_columnconfigure(0, weight=1)
	frame_buttons.grid_columnconfigure(1, weight=1)
	frame_buttons.pack(side=TOP, padx=0.014*ws, pady=(0.006*hs, 0))

	# Button to subscribe
	button_subscribe = ctk.CTkButton(frame_buttons, text="Subscribe",
									 command=subscribeCallback)
	button_subscribe.grid(row=0, column=0, padx=(0.004*ws, 0.004*ws), pady=(0.006*hs, 0.011*hs), sticky="nsew")

	# Button to subscribe
	button_unsubscribe = ctk.CTkButton(frame_buttons, text="Unsubscribe",
									   command=unsubscribeCallback)
	button_unsubscribe.grid(row=0, column=1, padx=(0.004*ws, 0.004*ws), pady=(0.006*hs, 0.011*hs), sticky="nsew")

	# sets the title of the
	# Toplevel widget
	newWindow.title("Daily Digest Subscribe/Unsubscribe")
	newWindow.geometry("%dx%d+%d+%d" % (0.208*ws, 0.222*hs, ws/2, hs/2))
	newWindow.maxsize(0.208*ws, 0.222*hs)
	newWindow.minsize(0.208*ws, 0.222*hs)

# Function to save data into csv
def saveToCSV():
	df = refreshDataStrategy(allData=True)[['Date', 'MoneyInvested', 'MoneyNotInvested',
       'MoneyInvestedToday', 'PerInvestToday', 'TotalPortfolioValue', 'MPV',
       'StdPV', 'maxPV', 'minPV']]

	df.set_index(['Date'], inplace=True)

	now = datetime.now(pytz.timezone('America/New_York')).strftime('%Y_%m_%d_%H_%M_%S')
	df.to_csv('dataUntil' + now + '.csv', index=['Date'])

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
def update(menuChoice=None):

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

	if switch_update.get():
		dealWithTimer()

# Function called from the update switch that updates every 5 min
def dealWithTimer():
	if switch_update.get():
		global updateJob
		updateJob = root.after(5*60*1000, update)
	else:
		root.after_cancel(updateJob)

def openEmail():
	webbrowser.open('mailto:notsocreative2023@gmail.com', new=1)

"""
MAIN WINDOW AND WIDGET INITIALIZATION
"""

# Set default appearance
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

# Main Window
root = ctk.CTk()
hs = root.winfo_screenheight()
ws = root.winfo_screenwidth()
ppi = root.winfo_fpixels('1i')
root.geometry('%dx%d+%d+%d' % (ws*0.90,hs*0.90, 0, 0))
root.minsize(int(ws*0.85), int(hs*0.85))
root.title("Stock Market Prediction Engine")
root.protocol("WM_DELETE_WINDOW", _quit)

# configure the grid
root.grid_columnconfigure(1, weight=1)  # Will contain logo, appearance, info, update button and apply changes on appearance button
root.grid_columnconfigure(2, weight=1)  # Will contain the main figures (MPV, AbsGain...)
root.grid_columnconfigure(3, weight=6)  # Will contain the plots
root.grid_rowconfigure(0, weight=1)  # Each column will contain a frame that will further divide the space or the widgets will be packed one on top of the other

color_var = ctk.BooleanVar(value=True)
bar_var = ctk.BooleanVar(value=False)
offset_var = ctk.BooleanVar(value=False)
update_var = ctk.BooleanVar(value=False)
digest_var = ctk.StringVar(value='Short')

"""
FIRST COLUMN WIDGET CREATION
"""
# Create frame that will contain everything
sidebar_frame = ctk.CTkFrame(root)
sidebar_frame.grid(row=0, column=1, padx=(0.007*ws, 0.007*ws), pady=(0.011*hs, 0.011*hs), sticky="nsew")

# Title and logo
# Load image
img = ctk.CTkImage( light_image=Image.open('png.png'),dark_image=Image.open("png.png"),size=(0.14*ws, 0.22*hs))
logo_label =ctk.CTkLabel(sidebar_frame,text='', image=img)

# Create label with the title
title_label = ctk.CTkLabel(sidebar_frame, text="Prediction Engine S&P 500",
						   font=ctk.CTkFont(size=26, weight="bold"))

# Pack the things one on top of the other
title_label.pack(side=TOP, padx=0.014*ws, pady=(0.022*hs, 0.011*hs))
logo_label.pack(side=TOP, padx=0.014*ws, pady=(0.011*hs, 0.044*hs))

# Option menu
option_label = ctk.CTkLabel(sidebar_frame, text='Options:', font=ctk.CTkFont(size=16))
option_label.pack(side=TOP, padx=0.014*ws, pady=(0, 0))

# Update button
data_update_button= ctk.CTkButton(sidebar_frame, text="Update Data", command=update)
data_update_button.pack(side=TOP, padx=0.014*ws, pady=(0.017*hs, 0.017*hs))

# Automatic update button
switch_update = ctk.CTkSwitch(sidebar_frame, text="Update Every 5 Min", variable=update_var, command=dealWithTimer, onvalue=True,
						 offvalue=False)
switch_update.pack(side=TOP, padx=0.014*ws, pady=(0.006*hs, 0))

# Info button
sidebar_button = ctk.CTkButton(sidebar_frame, text="Info/Help", command=openNewWindow)
sidebar_button.pack(side=TOP, padx=0.014*ws, pady=(0.017*hs, 0.017*hs))

# Color switch
switch_color = ctk.CTkSwitch(sidebar_frame, text="Colorblind Mode", variable=color_var, command=update, onvalue=False,
						 offvalue=True)
switch_color.pack(side=TOP, padx=0.014*ws, pady=(0.006*hs, 0))

# Bar switch
switch_bar = ctk.CTkSwitch(sidebar_frame, text="Plot Line/Bar", variable=bar_var, command=update, onvalue=True,
						 offvalue=False)
switch_bar.pack(side=TOP, padx=0.014*ws, pady=(0.006*hs, 0))

# Offset switch
switch_offset = ctk.CTkSwitch(sidebar_frame, text="Plot Offset", variable=offset_var, command=update, onvalue=True,
						 offvalue=False)
switch_offset.pack(side=TOP, padx=0.014*ws, pady=(0.006*hs, 0))

# Appearance label
appearance_mode_label = ctk.CTkLabel(sidebar_frame, text="Appearance Mode:", anchor="w")
appearance_mode_label.pack(side=TOP, padx=0.014*ws, pady=(0.006*hs, 0))

# Appearance menu
appearance_mode_optionmenu = ctk.CTkOptionMenu(sidebar_frame, values=["Light", "Dark", "System"], command=update)
appearance_mode_optionmenu.pack(side=TOP, padx=0.014*ws, pady=(0, 0))

# Scaling label
scaling_label = ctk.CTkLabel(sidebar_frame, text="UI Scaling:", anchor="w")
scaling_label.pack(side=TOP, padx=0.014*ws, pady=(0.011*hs, 0))

# Scaling menu
scaling_optionmenu = ctk.CTkOptionMenu(sidebar_frame, values=["80%", "90%"], command=update)
scaling_optionmenu.pack(side=TOP, padx=0.014*ws, pady=(0, 0.011*hs))

# Save data button
data_save_button= ctk.CTkButton(sidebar_frame, text="Save Data", command=saveToCSV)
data_save_button.pack(side=TOP, padx=0.014*ws, pady=(0.017*hs, 0.017*hs))

# Send email button
send_email_button= ctk.CTkButton(sidebar_frame, text="Send Email to NotSoCreative", command=openEmail, fg_color='#FFFDC2', text_color='black')
send_email_button.pack(side=BOTTOM, padx=0.014*ws, pady=(0.017*hs, 0.017*hs))

# Daily digest button
daily_digest_button= ctk.CTkButton(sidebar_frame, text="Daily Digest\nSuscribe/Unsuscribe", command=windowDailyDigest, text_color='black')
daily_digest_button.pack(side=BOTTOM, padx=0.014*ws, pady=(0.017*hs, 0.017*hs))

"""
SECOND COLUMN WIDGET CREATION
"""
# Creation of a frame with 3 different spaces
important_Values_frame = ctk.CTkFrame(root)
important_Values_frame.grid(row=0, column=2, padx=(0.007*ws, 0.007*ws), pady=(0.011*hs, 0.011*hs), sticky="nsew")
important_Values_frame.grid_columnconfigure(0, weight=1)
important_Values_frame.grid_rowconfigure(0, weight=1)
important_Values_frame.grid_rowconfigure(1, weight=1)
important_Values_frame.grid_rowconfigure(2, weight=1)

# FIRST ROW OF SECOND COLUMN
# Create a frame to contain all the labels in the first row
important_Values_frame1 = ctk.CTkFrame(important_Values_frame)
important_Values_frame1.grid(row=0, column=0, padx=(0.014*ws, 0.014*ws), pady=(0.011*hs, 0.011*hs), sticky="nsew")

# Portfolio label
sidebar_label_0=ctk.CTkLabel(important_Values_frame1, text='Portfolio', font=ctk.CTkFont(size=22, weight='bold'))
sidebar_label_0.pack(padx=0.014*ws, pady=(0.022*hs, 0.022*hs))

# Portfolio value title label
sidebar_label_1 = ctk.CTkLabel(important_Values_frame1, text='Portfolio Value:', font=ctk.CTkFont(size=16, weight='bold'))
sidebar_label_1.pack( padx=0.014*ws, pady=(0, 0))

# Portfolio value label
sidebar_label_2 = ctk.CTkLabel(important_Values_frame1,
							   text=str(0)+' $',
							   font=ctk.CTkFont(size=16))
sidebar_label_2.pack( padx=0.014*ws, pady=(0,0))

# Percentage gain title label
sidebar_label_3 = ctk.CTkLabel(important_Values_frame1, text=' Gain(%):', font=ctk.CTkFont(size=16, weight='bold'))
sidebar_label_3.pack( padx=0.014*ws, pady=(0.011*hs, 0))

# Percentage gain value label
sidebar_label_4 = ctk.CTkLabel(important_Values_frame1, text=str(0)+' %',
							   font=ctk.CTkFont(size=16))
sidebar_label_4.pack(padx=0.014*ws)

# Mean portfolio title label
sidebar_label_5 = ctk.CTkLabel(important_Values_frame1, text='Mean Portfolio Value:',
							   font=ctk.CTkFont(size=16, weight='bold'))
sidebar_label_5.pack( padx=0.014*ws, pady=(0.011*hs, 0))

# Mean portfolio value label
sidebar_label_6 = ctk.CTkLabel(important_Values_frame1, text=str(0)+' $',
							   font=ctk.CTkFont(size=16))
sidebar_label_6.pack( padx=0.014*ws)

# SECOND ROW OF THE SECOND COLUMN
# Create a frame to contain all labels in the second row
recommendation_frame = ctk.CTkFrame(important_Values_frame)
recommendation_frame.grid(row=1, column=0, padx=(0.014*ws, 0.014*ws), pady=(0.011*hs, 0), sticky="nsew")

# Recommend title label
labelrecommend = ctk.CTkLabel(recommendation_frame, text='Recommendation', font=ctk.CTkFont(size=22, weight="bold"))
labelrecommend.pack( padx=0.014*ws, pady=(0.044*hs, 0.006*hs))

# Current recommendation label
labelcurrentRecommend = ctk.CTkLabel(recommendation_frame, text='Buy', font=ctk.CTkFont(size=20, weight='bold'),
									 text_color=get_Color_Buy_Up_Decision())
labelcurrentRecommend.pack( pady=(0.006*hs, 0), padx=0.014*ws)

# Last recommendations title label
labelLastRecommend = ctk.CTkLabel(recommendation_frame, text='Last Recommendations', font=ctk.CTkFont(size=17, weight="bold"))
labelLastRecommend.pack( pady=(0.011*hs, 0.006*hs), padx=0.014*ws)

# Second Last recommendation label
labelLRecommend1 = ctk.CTkLabel(recommendation_frame, text='No Last Recommendation', font=ctk.CTkFont(size=16))
labelLRecommend1.pack( pady=0.006*hs, padx=0.014*ws)

# Third last recommendation label
labelLRecommend2 = ctk.CTkLabel(recommendation_frame,text="",font=ctk.CTkFont(size=16))
labelLRecommend2.pack( pady=0.006*hs, padx=0.014*ws)

# Fourth last recommendation label
labelLRecommend3 = ctk.CTkLabel(recommendation_frame,text='', font=ctk.CTkFont(size=16))
labelLRecommend3.pack( pady=0.006*hs, padx=0.014*ws)

# Update function to be called on each update
def updateLastDecisions(strategyDataTest):
	# Check if there is a recommendation for today
	offset = 0
	now = datetime.now(pytz.timezone('America/New_York'))
	if now > datetime.strptime(strategyDataTest['Date'][-1], '%Y-%m-%d %H:%M:%S').replace(hour=16, minute=0, second=0, tzinfo=pytz.timezone('America/New_York')):  #  No last recommendation should be shown
		offset = 1
	# Check how many last recommendation we have and update appropriately
	if len(strategyDataTest)+offset < 1:
		labelLRecommend1.configure(text='No Last Recommendation', font=ctk.CTkFont(size=16))
	elif len(strategyDataTest)+offset == 2:
		labelLRecommend1.configure(
			text=pd.to_datetime(strategyDataTest.index.values[-2+offset]).strftime('%Y-%m-%d') + ':    ' +
				 strategyDataTest['StrDecision'][
					 -2+offset], font=ctk.CTkFont(size=16))
	elif len(strategyDataTest)+offset == 3:
		labelLRecommend1.configure(
			text=pd.to_datetime(strategyDataTest.index.values[-2+offset]).strftime('%Y-%m-%d') + ':    ' +
				 strategyDataTest['StrDecision'][
					 -2+offset], font=ctk.CTkFont(size=16))
		labelLRecommend2.configure(
			text=pd.to_datetime(strategyDataTest.index.values[-3+offset]).strftime('%Y-%m-%d') + ':    ' +
				 strategyDataTest['StrDecision'][
					 -3+offset], font=ctk.CTkFont(size=16))
	elif len(strategyDataTest)+offset > 3:
		labelLRecommend1.configure(
			text=pd.to_datetime(strategyDataTest.index.values[-2+offset]).strftime('%Y-%m-%d') + ':    ' +
				 strategyDataTest['StrDecision'][
					 -2+offset], font=ctk.CTkFont(size=16))
		labelLRecommend2.configure(
			text=pd.to_datetime(strategyDataTest.index.values[-3+offset]).strftime('%Y-%m-%d') + ':    ' +
				 strategyDataTest['StrDecision'][
					 -3+offset], font=ctk.CTkFont(size=16))
		labelLRecommend3.configure(
			text=pd.to_datetime(strategyDataTest.index.values[-4+offset]).strftime('%Y-%m-%d') + ':    ' +
				 strategyDataTest['StrDecision'][
					 -4+offset], font=ctk.CTkFont(size=16))

	# Current recommendation
	if offset == 0:
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
	else:
		labelrecommend.configure(
			text='Recommendation\n' + now.strftime('%Y-%m-%d'))
		labelcurrentRecommend.configure(text='Market closed', font=ctk.CTkFont(size=20, weight='bold'),
										text_color=get_Color_Market_Closed())

# THIRD ROW OF THE SECOND COLUMN
# Create a frame to contain all labels in the third row
values_frame = ctk.CTkFrame(important_Values_frame)
values_frame.grid(row=2, column=0, padx=(0.014*ws, 0.014*ws), pady=(0.011*hs, 0.011*hs), sticky="nsew")

# Other metrics title label
labelL0=ctk.CTkLabel(values_frame, text='Other Metrics', font=ctk.CTkFont(size=22, weight='bold'))
labelL0.pack(pady=(0.022*ws, 0.022*ws), padx=0.014*ws)

# Invested money title label
labelL1 = ctk.CTkLabel(values_frame, text='Invested Money:', font=ctk.CTkFont(size=16, weight="bold"))
labelL1.pack( pady=(0.011*hs, 0), padx=0.014*ws)

# Invested money value label
labelL2 = ctk.CTkLabel(values_frame, text=str(0)+' $',
					   font=ctk.CTkFont(size=16))
labelL2.pack( pady=(0, 0), padx=0.014*ws)

# Not invested money title label
labelL3 = ctk.CTkLabel(values_frame, text='Money Not Invested:', font=ctk.CTkFont(size=16, weight="bold"))
labelL3.pack( pady=(0, 0), padx=0.014*ws)

# Not invested money value label
labelL4 = ctk.CTkLabel(values_frame, text=str(0)+' $',
					   font=ctk.CTkFont(size=16))
labelL4.pack( pady=(0, 0), padx=0.014*ws)

# Std title label
labelL5 = ctk.CTkLabel(values_frame, text='Standard Deviation:', font=ctk.CTkFont(size=16, weight="bold"))
labelL5.pack(side=TOP, pady=(0, 0), padx=0.014*ws)

# Std value label
labelL6 = ctk.CTkLabel(values_frame, text=str(0)+' $',
					   font=ctk.CTkFont(size=16))
labelL6.pack( pady=(0, 0), padx=0.014*ws)

# Max Gain title label
labelL7 = ctk.CTkLabel(values_frame, text='Max Gain per Day:', font=ctk.CTkFont(size=16, weight="bold"))
labelL7.pack(pady=(0, 0), padx=0.014*ws)

# Max gain value label
labelL8 = ctk.CTkLabel(values_frame, text=str(0)+' %',
					   font=ctk.CTkFont(size=16))
labelL8.pack(pady=(0, 0.011*hs), padx=0.014*ws)

"""
THIRD COLUMN WIDGET CREATION
"""
# Create a frame that will contain the tabs of both "set" of plots
framePlots = ctk.CTkFrame(root)
framePlots.grid(row=0, column=3, padx=(0.007*ws, 0.007*ws), pady=(0.011*hs, 0.011*hs), sticky="nsew")
framePlots.grid_rowconfigure(0, weight=1)
framePlots.grid_rowconfigure(1, weight=1)
framePlots.grid_columnconfigure(0, weight=1)

# Tab creation for candlestick plots
tabviewCandlesticks = ctk.CTkTabview(framePlots)
tabviewCandlesticks.grid(row=0, column=0, padx=(0.004*ws, 0.004*ws), pady=(0.006*ws, 0.011*hs), sticky="nsew")
tab2W = tabviewCandlesticks.add("2 Weeks")
tab1M = tabviewCandlesticks.add("1M")
tab6M = tabviewCandlesticks.add("6M")
tab1Y = tabviewCandlesticks.add("1Y")
tabviewCandlesticks.tab("2 Weeks")
tabviewCandlesticks.tab("1M")
tabviewCandlesticks.tab("6M")
tabviewCandlesticks.tab("1Y")

# Tab creation for metric plots
tabviewMetrics = ctk.CTkTabview(framePlots)
tabviewMetrics.grid(row=1, column=0, padx=(0.004*ws, 0.004*ws), pady=(0.006*ws, 0.011*hs), sticky="nsew")
tabMPV = tabviewMetrics.add("Mean_PV")
tabTPV = tabviewMetrics.add("TotalPortfolioValue")
tabAbsGain = tabviewMetrics.add("Gain (absolute)")
tabPerGain = tabviewMetrics.add("Gain (percentage)")
tabMIT = tabviewMetrics.add("Money Invested Today")
tabviewMetrics.tab("Mean_PV")
tabviewMetrics.tab("TotalPortfolioValue")
tabviewMetrics.tab("Gain (absolute)")
tabviewMetrics.tab("Gain (percentage)")
tabviewMetrics.tab("Money Invested Today")

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
	toolbarFrame.place(x=0, y=0)
	NavigationToolbar2Tk(line, toolbarFrame)

	# Embed 1M plot into tab
	line = FigureCanvasTkAgg(show_graph(stockData, 18), tab1M)
	line.draw()
	line.get_tk_widget().pack(side='top', fill='both', expand=True)
	# Navigation bar
	toolbarFrame = Frame(master=tab1M)
	toolbarFrame.place(x=0, y=0)
	NavigationToolbar2Tk(line, toolbarFrame)

	# Embed 6M plot into tab
	line = FigureCanvasTkAgg(show_graph(stockData, 123), tab6M)
	line.draw()
	line.get_tk_widget().pack(side='top', fill='both', expand=True)
	# Navigation bar
	toolbarFrame = Frame(master=tab6M)
	toolbarFrame.place(x=0, y=0)
	NavigationToolbar2Tk(line, toolbarFrame)

	# Embed 1Y plot into tab
	line = FigureCanvasTkAgg(show_graph(stockData, 250), tab1Y)
	line.draw()
	line.get_tk_widget().pack(side='top', fill='both', expand=True)
	# Navigation bar
	toolbarFrame = Frame(master=tab1Y)
	toolbarFrame.place(x=0, y=0)
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
	toolbarFrame.place(x=0, y=0)
	NavigationToolbar2Tk(line, toolbarFrame)

	# Embed PV plot into tab
	line = FigureCanvasTkAgg(show_metrics(metrics, 'PortfolioValue'), tabTPV)
	line.draw()
	line.get_tk_widget().pack(side='top', fill='both', expand=True)
	# Navigation bar
	toolbarFrame = Frame(master=tabTPV)
	toolbarFrame.place(x=0, y=0)
	NavigationToolbar2Tk(line, toolbarFrame)

	# Embed AbsGain plot into tab
	line = FigureCanvasTkAgg(show_metrics(metrics, 'AbsGain'), tabAbsGain)
	line.draw()
	line.get_tk_widget().pack(side='top', fill='both', expand=True)
	# Navigation bar
	toolbarFrame = Frame(master=tabAbsGain)
	toolbarFrame.place(x=0, y=0)
	NavigationToolbar2Tk(line, toolbarFrame)

	# Embed PerGain plot into tab
	line = FigureCanvasTkAgg(show_metrics(metrics, 'PerGain'), tabPerGain)
	line.draw()
	line.get_tk_widget().pack(side='top', fill='both', expand=True)
	# Navigation bar
	toolbarFrame = Frame(master=tabPerGain)
	toolbarFrame.place(x=0, y=0)
	NavigationToolbar2Tk(line, toolbarFrame)

	# Embed MoneyInvestedToday plot into tab
	line = FigureCanvasTkAgg(show_metrics(metrics, 'MoneyInvestedToday'), tabMIT)
	line.draw()
	line.get_tk_widget().pack(side='top', fill='both', expand=True)
	# Navigation bar
	toolbarFrame = Frame(master=tabMIT)
	toolbarFrame.place(x=0, y=0)
	NavigationToolbar2Tk(line, toolbarFrame)

f = open('output.txt', 'w')
sys.stdout = f
update()
root.mainloop()
