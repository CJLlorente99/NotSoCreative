import customtkinter as ctk
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

import pytz
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
import mplfinance as mpf
from datetime import datetime, date
from matplotlib.figure import Figure
from PIL import Image
import yfinance as yf
import numpy as np
from googleStorageAPI import readBlobDf

# Basic structure of the window

# Constants
yfStartDate = '2021-01-01'
date_test = '2023-01-24'
strategyName = 'bilstmWindowRobMMT1T2Legacy_24_1_2023'

########## Functions #########

def calculateOperation(value: str):
	if value.endswith('09:30:00'):
		return 0
	elif value.endswith('16:00:00'):
		return 1
	else:
		return 2  # never

def shortenedDate(value: str):
	return value[:-9]

def strDecision(value):
	if value > 0:
		return 'Buy'
	elif value < 0:
		return 'Sell'
	else:
		return 'Hold'

def decisionFunction(value):
	if value > 0:
		return 1
	elif value < 0:
		return -1
	else:
		return 0

def holdDecision(value):
	if value == 0:
		return 0
	else:
		return np.nan

def sellDecision(value):
	if value == -1:
		return -1
	else:
		return np.nan

def buyDecision(value):
	if value == 1:
		return 1
	else:
		return np.nan

def refreshDataSP500():
	end = date.today() + timedelta(days=1)
	stock_data = yf.download('^GSPC', start=yfStartDate, end=end)
	stock_data = stock_data.reset_index()
	stock_data.set_index(pd.DatetimeIndex(stock_data['Date']), inplace=True)

	return stock_data

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

def show_metrics(metrics, var):

	x = metrics['Date']
	y = metrics[var]

	f = Figure(figsize=(10, 10), dpi=100)
	a = f.add_subplot(111)

	if ctk.get_appearance_mode() == 'Dark':
		plt.style.use('dark_background')
	else:
		# plt.style.use('default')
		pass
	plt.grid(visible=True, axis='both')
	a.plot(x, y)
	return f

# get the newest selected value of a selected strategy (kind=selected Value) .tail(number of investment strategies)

def show_graph(stock_data, i):

	if ctk.get_appearance_mode() == 'Dark':
		mode ='nightclouds'
	else:
		mode='default'

	# plot candelsticks
	color_candels = mpf.make_marketcolors(up=get_Color_Buy_Up_Candlesticks(), down=get_Color_Sell_Down_Candlesticks(), wick="inherit",
										  edge="inherit", volume="in")
	mpf_style = mpf.make_mpf_style(base_mpf_style=mode, marketcolors=color_candels)
	fig, axl = mpf.plot(stock_data.iloc[-i:, :], type='candle', volume=False, style=mpf_style, returnfig=True,
						figsize=(10, 10))
	return fig


# specifically for our test time window
def show_graph_test(data_csv, stock_data):

	if ctk.get_appearance_mode() == 'Dark':
		mode ='nightclouds'
	else:
		mode='default'

	# Color of candlesticks
	color_candels = mpf.make_marketcolors(up=get_Color_Buy_Up_Candlesticks(), down=get_Color_Sell_Down_Candlesticks(), wick="inherit",
										  edge="inherit", volume="in")

	aux = pd.DataFrame()
	aux['Hold'] = data_csv['Decision'].map(holdDecision) + (stock_data['Close'] + stock_data['Open'])/2
	aux['Sell'] = data_csv['Decision'].map(sellDecision) + stock_data['Close'] + 5
	aux['Buy'] = data_csv['Decision'].map(buyDecision)  + stock_data['Open'] - 5

	apds = []
	if not aux['Hold'].isnull().all():
		apds.append(mpf.make_addplot(aux["Hold"], type='scatter', marker='s', markersize=400, color=get_Color_Hold_Decision(), secondary_y=False))
	if not aux['Buy'].isnull().all():
		apds.append(mpf.make_addplot(aux["Buy"], type='scatter', marker='^', markersize=400, color=get_Color_Buy_Up_Decision(), secondary_y=False))
	if not aux['Sell'].isnull().all():
		apds.append(mpf.make_addplot(aux["Sell"], type='scatter', marker='v', markersize=400, color=get_Color_Sell_Down_Decision(), secondary_y=False))


	# plot candlesticks
	mpf_style = mpf.make_mpf_style(base_mpf_style=mode, marketcolors=color_candels)
	fig, axl = mpf.plot(stock_data, type='candle', volume=False, style=mpf_style, returnfig=True, addplot=apds,
						figsize=(10, 10))
	return fig

def _quit():
	root.quit()
	root.destroy()

def get_Color_Buy_Up_Candlesticks():
	if color_mode() == 0:
		color = "#A3FF87"
	elif color_mode() == 1:
		color = "#5D02FF"
	return color

def get_Color_Sell_Down_Candlesticks():
	if color_mode() == 0:
		color = "#FF8A8A"
	elif color_mode() == 1:
		color = "#E60400"
	return color

def get_Color_Buy_Up_Decision():
	if color_mode() == 0:
		color = "#00FF00"
	elif color_mode() == 1:
		color = "#5D02FF"
	return color


def get_Color_Hold_Decision():
	if color_mode() == 0:
		color = "#FFA500"
	elif color_mode() == 1:
		color = "#FFA500"
	return color


def get_Color_Sell_Down_Decision():
	if color_mode() == 0:
		color = "#ff0000"
	elif color_mode() == 1:
		color = "#E60400"
	return color


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
	# sets the geometry of toplevel
	newWindow.geometry("300x300")
	# A Label widget to show in toplevel
	#Label(newWindow,
	#).pack()


# changes appearance mode
def change_appearance_mode_event(new_appearance_mode: str):
	ctk.set_appearance_mode(new_appearance_mode)
	ctk.get_appearance_mode()

# changes scaling
def change_scaling_event(new_scaling: str):
	new_scaling_float = int(new_scaling.replace("%", "")) / 100
	ctk.set_widget_scaling(new_scaling_float)

########## Main Frontend#########

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

# Main Window
root = ctk.CTk()
root.geometry('1500x900')
root.minsize(1200, 720)
root.title("Stock Market Prediction Engine")
root.protocol("WM_DELETE_WINDOW", _quit)

# configure the grid
root.grid_columnconfigure(1, weight=1)
root.grid_columnconfigure(2, weight=1)
root.grid_columnconfigure(3, weight=3)
root.grid_rowconfigure(0, weight=1)

switch_var = ctk.StringVar(value="off")

####### UPDATE START##########
def update():

	ctk.set_appearance_mode(appearance_mode_optionmenu.get())
	new_scaling_float = int(scaling_optionmenu.get().replace("%", "")) / 100
	ctk.set_widget_scaling(new_scaling_float)

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

	# print last 3 decisions depending on the amounts of available taken decisions
	updateLastDecisions(strategyDataTest)


	labelL2.configure(text=str(round(metrics['MoneyInvested'][-1], 2))+' $')
	labelL4.configure(text=str(round(metrics['MoneyNotInvested'][-1], 2))+' $')

	# nur letzter Wert
	labelL6.configure(text=str(round(metrics['StdPV'][-1], 2))+' $')
	labelL8.configure(text=str(round(metrics['MaxGain'][-1], 2))+' %')

	#####Update Candlesticks######
	updateCandlesticks(stockData, strategyDataTest, stockDataTest)
	updateMetrics(metrics)


########## Update END#######

#color Mode for colorblind people
def color_mode():
	# returns 0 or 1 for differnet color modes
	color = 0
	if switch_var.get() == "off":
		color = 0
	elif switch_var.get() == "on":
		color = 1
	return color

fig = mpf.figure()

# Not actual open price but start value of new day

########### Start Widgets########

###############Left side##################

# create left sidebar frame with widgets
sidebar_frame = ctk.CTkFrame(root, width=140, corner_radius=0)
sidebar_frame.grid(row=0, column=1, sticky="nsew")

img = ctk.CTkImage( light_image=Image.open('png.png'),dark_image=Image.open("png.png"),size=(200,200))

title_label = ctk.CTkLabel(sidebar_frame, text="Prediction Engine S&P 500",
						   font=ctk.CTkFont(size=26, weight="bold"))

title_label.pack(side=TOP, padx=20, pady=(20, 10))
logo_label =ctk.CTkLabel(sidebar_frame,text='', image=img)
logo_label.pack(side=TOP, padx=20, pady=(10, 40))

######Ottion Menu########
option_label = ctk.CTkLabel(sidebar_frame, text='Options:', font=ctk.CTkFont(size=16))
option_label.pack(side=TOP, padx=20, pady=(0, 0))

data_update_button= ctk.CTkButton(sidebar_frame, text="Update Data", command=update)
data_update_button.pack(side=TOP, padx=20, pady=(15, 15))

sidebar_button = ctk.CTkButton(sidebar_frame, text="Info/Help", command=openNewWindow)
sidebar_button.pack(side=TOP, padx=20, pady=(15, 15))

switch_1 = ctk.CTkSwitch(sidebar_frame, text="Color", variable=switch_var, command=color_mode(), onvalue="on",
						 offvalue="off")
switch_1.pack(side=TOP, padx=20, pady=(5, 0))

appearance_mode_label = ctk.CTkLabel(sidebar_frame, text="Appearance Mode:", anchor="w")
appearance_mode_label.pack(side=TOP, padx=20, pady=(5, 0))

appearance_mode_optionmenu = ctk.CTkOptionMenu(sidebar_frame, values=["Light", "Dark", "System"])
appearance_mode_optionmenu.pack(side=TOP, padx=20, pady=(0, 0))

scaling_label = ctk.CTkLabel(sidebar_frame, text="UI Scaling:", anchor="w")
scaling_label.pack(side=TOP, padx=20, pady=(10, 0))

scaling_optionmenu = ctk.CTkOptionMenu(sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"])
scaling_optionmenu.pack(side=TOP, padx=20, pady=(0, 10))

apply_update_button= ctk.CTkButton(sidebar_frame, text="Apply Changes", command=update)
apply_update_button.pack(side=TOP, padx=20, pady=(15, 15))

########### Value Frames################
important_Values_frame = ctk.CTkFrame(root)
important_Values_frame.grid(row=0, column=2, padx=(20, 20), pady=(10, 0), sticky="nsew")
important_Values_frame.grid_columnconfigure(0, weight=1)
important_Values_frame.grid_rowconfigure(0, weight=1)
important_Values_frame.grid_rowconfigure(1, weight=1)
important_Values_frame.grid_rowconfigure(2, weight=1)

important_Values_frame1 = ctk.CTkFrame(important_Values_frame, height=300)
important_Values_frame1.grid(row=0, column=0, padx=(20, 20), pady=(10, 10), sticky="nsew")

sidebar_label_0=ctk.CTkLabel(important_Values_frame1, text='Portfolio', font=ctk.CTkFont(size=22, weight='bold'))
sidebar_label_0.pack(padx=20, pady=(20, 20))

sidebar_label_1 = ctk.CTkLabel(important_Values_frame1, text='Portfolio Value:', font=ctk.CTkFont(size=16, weight='bold'))
sidebar_label_1.pack( padx=20, pady=(0, 0))
# nur letzter Wert
sidebar_label_2 = ctk.CTkLabel(important_Values_frame1,
							   text=str(0)+' $',
							   font=ctk.CTkFont(size=16))
sidebar_label_2.pack( padx=20, pady=(0,0))

# nur letzter Wert
sidebar_label_3 = ctk.CTkLabel(important_Values_frame1, text=' Gain(%):', font=ctk.CTkFont(size=16, weight='bold'))
sidebar_label_3.pack( padx=20, pady=(10, 0))

sidebar_label_4 = ctk.CTkLabel(important_Values_frame1, text=str(0)+' %',
							   font=ctk.CTkFont(size=16))
sidebar_label_4.pack(padx=20)
# nur letzter Wert
sidebar_label_5 = ctk.CTkLabel(important_Values_frame1, text='Mean Portfolio Value:',
							   font=ctk.CTkFont(size=16, weight='bold'))
sidebar_label_5.pack( padx=20, pady=(10, 0))

sidebar_label_6 = ctk.CTkLabel(important_Values_frame1, text=str(0)+' $',
							   font=ctk.CTkFont(size=16))
sidebar_label_6.pack( padx=20)

# create value frame
values_frame = ctk.CTkFrame(important_Values_frame,height=300)
values_frame.grid(row=2, column=0, padx=(20, 20), pady=(10, 10), sticky="nsew")

labelL0=ctk.CTkLabel(values_frame, text='Other Metrics', font=ctk.CTkFont(size=22, weight='bold'),underline=0)
labelL0.pack(pady=(20, 20), padx=20)

labelL1 = ctk.CTkLabel(values_frame, text='Invested Money:', font=ctk.CTkFont(size=16, weight="bold"))
labelL1.pack( pady=(10, 0), padx=20)

labelL2 = ctk.CTkLabel(values_frame, text=str(0)+' $',
					   font=ctk.CTkFont(size=16))
labelL2.pack( pady=(0, 0), padx=20)

labelL3 = ctk.CTkLabel(values_frame, text='Money Not Invested:', font=ctk.CTkFont(size=16, weight="bold"))
labelL3.pack( pady=(0, 0), padx=20)

labelL4 = ctk.CTkLabel(values_frame, text=str(0)+' $',
					   font=ctk.CTkFont(size=16))
labelL4.pack( pady=(0, 0), padx=20)

labelL5 = ctk.CTkLabel(values_frame, text='Standard Deviation:', font=ctk.CTkFont(size=16, weight="bold"))
labelL5.pack(side=TOP, pady=(0, 0), padx=20)
# nur letzter Wert
labelL6 = ctk.CTkLabel(values_frame, text=str(0)+' $',
					   font=ctk.CTkFont(size=16))
labelL6.pack( pady=(0, 0), padx=20)

labelL7 = ctk.CTkLabel(values_frame, text='Max Gain per Day:', font=ctk.CTkFont(size=16, weight="bold"))
labelL7.pack(pady=(0, 0), padx=20)
# nur letzter Wert

labelL8 = ctk.CTkLabel(values_frame, text=str(0)+' %',
					   font=ctk.CTkFont(size=16))
labelL8.pack(pady=(0, 10), padx=20)

################Plot left side#######################

#########Graph########

framePlots = ctk.CTkFrame(root)
framePlots.grid(row=0, column=3, padx=(20, 20), sticky="nsew")
framePlots.grid_rowconfigure(0, weight=1)
framePlots.grid_rowconfigure(1, weight=1)
framePlots.grid_columnconfigure(0, weight=1)

# Plot candlesticks

tabview = ctk.CTkTabview(framePlots)
tabview.grid(row=0, column=0, padx=(5, 5), pady=(5, 0), sticky="nsew")
tab2W = tabview.add("2 Weeks")
tab1M = tabview.add("1M")
tab6M = tabview.add("6M")
tab1Y = tabview.add("1Y")
tabview.tab("2 Weeks")
tabview.tab("1M")
tabview.tab("6M")
tabview.tab("1Y")

def updateCandlesticks(stockData, strategyDataTest, stockDataTest):
	for obj in tab2W.winfo_children():
		obj.destroy()
	for obj in tab1M.winfo_children():
		obj.destroy()
	for obj in tab6M.winfo_children():
		obj.destroy()
	for obj in tab1Y.winfo_children():
		obj.destroy()

	# test
	line = FigureCanvasTkAgg(show_graph_test(strategyDataTest, stockDataTest), tab2W)
	line.draw()
	line.get_tk_widget().pack(side='top', fill='both', expand=True)
	# Navigation bar
	toolbarFrame = Frame(master=tab2W)
	toolbarFrame.place(relx=0, rely=0.94)
	NavigationToolbar2Tk(line, toolbarFrame)

	# 1M
	line = FigureCanvasTkAgg(show_graph(stockData, 18), tab1M)
	line.draw()
	line.get_tk_widget().pack(side='top', fill='both', expand=True)
	# Navigation bar
	toolbarFrame = Frame(master=tab1M)
	toolbarFrame.place(relx=0, rely=0.94)
	NavigationToolbar2Tk(line, toolbarFrame)

	# 6M
	line = FigureCanvasTkAgg(show_graph(stockData, 123), tab6M)
	line.draw()
	line.get_tk_widget().pack(side='top', fill='both', expand=True)
	# Navigation bar
	toolbarFrame = Frame(master=tab6M)
	toolbarFrame.place(relx=0, rely=0.94)
	NavigationToolbar2Tk(line, toolbarFrame)

	# 1Y
	line = FigureCanvasTkAgg(show_graph(stockData, 250), tab1Y)
	line.draw()
	line.get_tk_widget().pack(side='top', fill='both', expand=True)
	# Navigation bar
	toolbarFrame = Frame(master=tab1Y)
	toolbarFrame.place(relx=0, rely=0.94)
	NavigationToolbar2Tk(line, toolbarFrame)

# Visualization for important numbers
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

def updateMetrics(metrics):
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

	# Mean PV
	line = FigureCanvasTkAgg(show_metrics(metrics, 'MPV'), tabMPV)
	line.draw()
	line.get_tk_widget().pack(side='top', fill='both', expand=True)
	# Navigation bar
	toolbarFrame = Frame(master=tabMPV)
	toolbarFrame.place(relx=0, rely=0.95)
	NavigationToolbar2Tk(line, toolbarFrame)

	# TotalPortfolioValue
	line = FigureCanvasTkAgg(show_metrics(metrics, 'PortfolioValue'), tabTPV)
	line.draw()
	line.get_tk_widget().pack(side='top', fill='both', expand=True)
	# Navigation bar
	toolbarFrame = Frame(master=tabTPV)
	toolbarFrame.place(relx=0, rely=0.95)
	NavigationToolbar2Tk(line, toolbarFrame)

	# Gain (absolute)
	line = FigureCanvasTkAgg(show_metrics(metrics, 'AbsGain'), tabAbsGain)
	line.draw()
	line.get_tk_widget().pack(side='top', fill='both', expand=True)
	# Navigation bar
	toolbarFrame = Frame(master=tabAbsGain)
	toolbarFrame.place(relx=0, rely=0.95)
	NavigationToolbar2Tk(line, toolbarFrame)

	# Gain (percentage)
	line = FigureCanvasTkAgg(show_metrics(metrics, 'PerGain'), tabPerGain)
	line.draw()
	line.get_tk_widget().pack(side='top', fill='both', expand=True)
	# Navigation bar
	toolbarFrame = Frame(master=tabPerGain)
	toolbarFrame.place(relx=0, rely=0.95)
	NavigationToolbar2Tk(line, toolbarFrame)

	# Money Invested Today
	line = FigureCanvasTkAgg(show_metrics(metrics, 'MoneyInvestedToday'), tabMIT)
	line.draw()
	line.get_tk_widget().pack(side='top', fill='both', expand=True)
	# Navigation bar
	toolbarFrame = Frame(master=tabMIT)
	toolbarFrame.place(relx=0, rely=0.95)
	NavigationToolbar2Tk(line, toolbarFrame)

# Last Recommendations and current recommendation

# print last 3 decsions depending on the amounts of available taken decisions
recommendation_frame = ctk.CTkFrame(important_Values_frame, height=300)
recommendation_frame.grid(row=1, column=0, padx=(20, 20), pady=(10, 0), sticky="nsew")

labelrecommend = ctk.CTkLabel(recommendation_frame, text='Recommendation', font=ctk.CTkFont(size=22, weight="bold"))
labelrecommend.pack( padx=10, pady=(40, 5))

labelcurrentRecommend = ctk.CTkLabel(recommendation_frame, text='Buy', font=ctk.CTkFont(size=20, weight='bold'),
									 text_color=get_Color_Buy_Up_Decision())
labelcurrentRecommend.pack( pady=(5,0), padx=20)

labelLastRecommend = ctk.CTkLabel(recommendation_frame, text='Last Recommendations', font=ctk.CTkFont(size=17, weight="bold"))
labelLastRecommend.pack( pady=(10, 5), padx=20)

labelLRecommend1 = ctk.CTkLabel(recommendation_frame, text='No Last Recommendation', font=ctk.CTkFont(size=16))
labelLRecommend1.pack( pady=5, padx=20)

labelLRecommend2 = ctk.CTkLabel(recommendation_frame,text="",font=ctk.CTkFont(size=16))
labelLRecommend2.pack( pady=5, padx=20)

labelLRecommend3 = ctk.CTkLabel(recommendation_frame,text='', font=ctk.CTkFont(size=16))
labelLRecommend3.pack( pady=5, padx=20)

def updateLastDecisions(strategyDataTest):
	# Last recommendations
	if len(strategyDataTest) < 1:
		labelLRecommend1.configure(text='No Last Recommendation', font=ctk.CTkFont(size=16))
	elif len(strategyDataTest) == 2:
		labelLRecommend1.configure(
			text=str(pd.to_datetime(strategyDataTest.index.values[-2]).date()) + ':    ' +
				 strategyDataTest['StrDecision'][
					 -2], font=ctk.CTkFont(size=16))
	elif len(strategyDataTest) == 3:
		labelLRecommend1.configure(
			text=str(pd.to_datetime(strategyDataTest.index.values[-2]).date()) + ':    ' +
				 strategyDataTest['StrDecision'][
					 -2], font=ctk.CTkFont(size=16))
		labelLRecommend2.configure(
			text=str(pd.to_datetime(strategyDataTest.index.values[-3]).date()) + ':    ' +
				 strategyDataTest['StrDecision'][
					 -3], font=ctk.CTkFont(size=16))
	elif len(strategyDataTest) > 3:
		labelLRecommend1.configure(
			text=str(pd.to_datetime(strategyDataTest.index.values[-2]).date()) + ':    ' +
				 strategyDataTest['StrDecision'][
					 -2], font=ctk.CTkFont(size=16))
		labelLRecommend2.configure(
			text=str(pd.to_datetime(strategyDataTest.index.values[-3]).date()) + ':    ' +
				 strategyDataTest['StrDecision'][
					 -3], font=ctk.CTkFont(size=16))
		labelLRecommend3.configure(
			text=str(pd.to_datetime(strategyDataTest.index.values[-4]).date()) + ':    ' +
				 strategyDataTest['StrDecision'][
					 -4], font=ctk.CTkFont(size=16))

	# Current recommendation

	if strategyDataTest['MoneyInvestedToday'][-1] > 0:
		labelcurrentRecommend.configure(text='Buy', font=ctk.CTkFont(size=20, weight='bold'),
										text_color=get_Color_Buy_Up_Decision())
	elif strategyDataTest['MoneyInvestedToday'][-1] < 0:
		labelcurrentRecommend.configure(text='Sell', font=ctk.CTkFont(size=20, weight='bold'),
										text_color=get_Color_Sell_Down_Decision())
	else:
		labelcurrentRecommend.configure(text='Hold', font=ctk.CTkFont(size=20, weight='bold'),
										text_color=get_Color_Hold_Decision())

update()
root.mainloop()