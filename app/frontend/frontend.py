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

# Constants
yfStartDate = '2021-01-01'
date_test = '2023-01-24'
strategyName = 'bilstmWindowRobMMT1T2Legacy_24_1_2023'

# Globals
global metrics

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
	now = datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d %H:%M:%S')
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

def show_metrics(var):

	x = metrics['Date']
	y = metrics[var]

	if ctk.get_appearance_mode() == 'Dark':
		f = Figure(figsize=(10, 10), dpi=100)
		a = f.add_subplot(111)

		plt.style.use('dark_background')
		a.plot(x, y)
	else:
		f = Figure(figsize=(10, 10), dpi=100)
		plt.style.use('default')
		a = f.add_subplot(111)
		a.plot(x, y)
	return f

# get the newest selected value of a selected strategy (kind=selected Value) .tail(number of investment strategies)

def show_graph(stock_data, i):

	if ctk.get_appearance_mode() == 'Dark':
		mode ='nightclouds'
	else:
		mode='default'

	# plot candelsticks
	color_candels = mpf.make_marketcolors(up=get_Color_Buy_Up(), down=get_Color_Sell_Down(), wick="inherit",
										  edge="inherit", volume="in")
	mpf_style = mpf.make_mpf_style(base_mpf_style=mode, marketcolors=color_candels)
	fig, axl = mpf.plot(stock_data.iloc[-i:, :], type='candle', volume=False, style=mpf_style, returnfig=True,
						figsize=(3.5, 5))
	return fig


# specifically for our test time window
def show_graph_test(data_csv, stock_data):

	if ctk.get_appearance_mode() == 'Dark':
		mode ='nightclouds'
	else:
		mode='default'

	# Color of candlesticks
	color_candels = mpf.make_marketcolors(up=get_Color_Buy_Up(), down=get_Color_Sell_Down(), wick="inherit",
										  edge="inherit", volume="in")

	# Color of Investment Actions (Triangles)
	colors_apd = [get_Color_Buy_Up() if v == 1 else get_Color_Hold() if v == 0 else get_Color_Sell_Down() for v in
				  data_csv['Decision'].values]

	apd = mpf.make_addplot(data_csv["Decision"], type='scatter', marker='^', markersize=200, color=colors_apd)


	# plot candlesticks
	mpf_style = mpf.make_mpf_style(base_mpf_style=mode, marketcolors=color_candels)
	fig, axl = mpf.plot(stock_data, type='candle', volume=False, style=mpf_style, returnfig=True, addplot=apd,
						figsize=(8, 5))

	return fig

def _quit():
	root.quit()
	root.destroy()

def get_Color_Buy_Up():
	if color_mode() == 0:
		color = "#00ff00"
	elif color_mode() == 1:
		color = "#5D02FF"
	return color


def get_Color_Hold():
	if color_mode() == 0:
		color = "#FFA500"
	elif color_mode() == 1:
		color = "#FFA500"
	return color


def get_Color_Sell_Down():
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
root.title("Stock Market Prediction Engine")

# configure the grid
root.grid_columnconfigure(1, weight=1)
root.grid_columnconfigure(2, weight=1)
root.grid_columnconfigure(3, weight=3)
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_rowconfigure(2, weight=1)

switch_var = ctk.StringVar(value="off")

# initialize some things
strategyData = refreshDataStrategy()
strategyDataTest = strategyData.iloc[strategyData.index.get_loc(date_test):]
lastDataTest = strategyDataTest.index[-1]
stockData = refreshDataSP500()
stockDataTest = stockData.iloc[stockData.index.get_loc(date_test):stockData.index.get_loc(lastDataTest)+1]

metrics = getCurrentValue_metric(stockDataTest, strategyDataTest)
strategyDataTest['StrDecision'] = strategyDataTest['MoneyInvestedToday'].map(strDecision)
strategyDataTest['Decision'] = strategyDataTest['MoneyInvestedToday'].map(decisionFunction)


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

	global metrics
	metrics = getCurrentValue_metric(stockDataTest, strategyDataTest)
	strategyDataTest['StrDecision'] = strategyDataTest['MoneyInvestedToday'].map(strDecision)
	strategyDataTest['Decision'] = strategyDataTest['MoneyInvestedToday'].map(decisionFunction)

	# nur letzter Wert
	sidebar_label_2.configure(text=str(round(metrics['PortfolioValue'][-1],2))+' $')
	sidebar_label_4.configure(text=str(round(metrics['PerGain'][-1], 2))+' %')
	sidebar_label_6.configure(text=str(round(metrics['MPV'][-1], 2))+' $')

	if strategyData['MoneyInvestedToday'][-1] > 0:
		labelcurrentRecommend.configure(text='Buy', font=ctk.CTkFont(size=20, weight='bold'),
										text_color=get_Color_Buy_Up())
	elif strategyData['MoneyInvestedToday'][-1] < 0:
		labelcurrentRecommend.configure(text='Sell', font=ctk.CTkFont(size=20, weight='bold'),
										text_color=get_Color_Sell_Down())
	else:
		labelcurrentRecommend.configure(text='Hold', font=ctk.CTkFont(size=20, weight='bold'),
										text_color=get_Color_Hold())

	# print last 3 decisions depending on the amounts of available taken decisions
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

	labelL2.configure(text=str(round(metrics['MoneyInvested'][-1], 2))+' $')
	labelL4.configure(text=str(round(metrics['MoneyNotInvested'][-1], 2))+' $')

	# nur letzter Wert
	labelL6.configure(text=str(round(metrics['StdPV'][-1], 2))+' $')
	labelL8.configure(text=str(round(metrics['MaxGain'][-1], 2))+' %')

	#####Update Candlesticks######
	# test
	line = FigureCanvasTkAgg(show_graph_test(strategyDataTest, stockDataTest), label_test)
	line.get_tk_widget().place(width=800, height=500)
	# Navigation bar
	toolbarFrame = Frame(master=label_test)
	toolbarFrame.place(x=0, y=450)
	NavigationToolbar2Tk(line, toolbarFrame)
	# 1M
	line = FigureCanvasTkAgg(show_graph(stockData, 18), label_1m)
	line.get_tk_widget().place(width=800, height=500)
	# Navigation bar
	toolbarFrame = Frame(master=label_1m)
	toolbarFrame.place(x=0, y=450)
	NavigationToolbar2Tk(line, toolbarFrame)

	# 6M
	line = FigureCanvasTkAgg(show_graph(stockData, 123), label_6m)
	line.get_tk_widget().place(width=800, height=500)
	# Navigation bar
	toolbarFrame = Frame(master=label_6m)
	toolbarFrame.place(x=0, y=450)
	NavigationToolbar2Tk(line, toolbarFrame)

	# 1Y
	line = FigureCanvasTkAgg(show_graph(stockData, 250), label_1y)
	line.get_tk_widget().place(width=800, height=500)
	# Navigation bar
	toolbarFrame = Frame(master=label_1y)
	toolbarFrame.place(x=0, y=450)
	NavigationToolbar2Tk(line, toolbarFrame)

	###Update Value Graphs####
	# Mean PV
	line = FigureCanvasTkAgg(show_metrics("MPV"), label_v_1)
	line.get_tk_widget().place(width=800, height=500)
	# Navigation bar
	toolbarFrame = Frame(master=label_v_1)
	toolbarFrame.place(x=0, y=400)
	NavigationToolbar2Tk(line, toolbarFrame)

	# TotalPortfolioValue
	line = FigureCanvasTkAgg(show_metrics("PortfolioValue"), label_v_2)
	line.get_tk_widget().place(width=800, height=500)
	# Navigation bar
	toolbarFrame = Frame(master=label_v_2)
	toolbarFrame.place(x=0, y=400)
	NavigationToolbar2Tk(line, toolbarFrame)

	# Gain (absolute)
	line = FigureCanvasTkAgg(show_metrics("AbsGain"), label_v_3)
	line.get_tk_widget().place(width=800, height=500)
	# Navigation bar
	toolbarFrame = Frame(master=label_v_3)
	toolbarFrame.place(x=0, y=400)
	NavigationToolbar2Tk(line, toolbarFrame)

	# Gain (percentage)
	line = FigureCanvasTkAgg(show_metrics("PerGain"), label_v_4)
	line.get_tk_widget().place(width=800, height=500)
	# Navigation bar
	toolbarFrame = Frame(master=label_v_4)
	toolbarFrame.place(x=0, y=400)
	NavigationToolbar2Tk(line, toolbarFrame)

	# Money Invested Today
	line = FigureCanvasTkAgg(show_metrics("MoneyInvestedToday"), label_v_5)
	line.get_tk_widget().place(width=800, height=500)
	# Navigation bar
	toolbarFrame = Frame(master=label_v_5)
	toolbarFrame.place(x=0, y=400)
	NavigationToolbar2Tk(line, toolbarFrame)


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
sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
sidebar_frame.grid_rowconfigure(4, weight=1)

#img = ImageTk.PhotoImage(Image.open("png.png"))

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
important_Values_frame = ctk.CTkFrame(root, height=300)
important_Values_frame.grid(row=0, column=1, padx=(20, 20), pady=(10, 0), sticky="nsew")
important_Values_frame.grid_rowconfigure(3, weight=1)
important_Values_frame.grid_columnconfigure(3,weight=1)

sidebar_label_0=ctk.CTkLabel(important_Values_frame, text='Portfolio', font=ctk.CTkFont(size=22, weight='bold'))
sidebar_label_0.pack(padx=20, pady=(20, 20))
sidebar_label_1 = ctk.CTkLabel(important_Values_frame, text='Portfolio Value:', font=ctk.CTkFont(size=16, weight='bold'))
#sidebar_label_1.grid(row=1, column=0, padx=20,pady=(10,0))
sidebar_label_1.pack( padx=20, pady=(0, 0))
# nur letzter Wert
sidebar_label_2 = ctk.CTkLabel(important_Values_frame,
							   text=str(round(metrics["PortfolioValue"][-1], 2))+' $',
							   font=ctk.CTkFont(size=16))
#sidebar_label_2.grid(row=2, column=0, padx=20)
sidebar_label_2.pack( padx=20, pady=(0,0))

# nur letzter Wert
sidebar_label_3 = ctk.CTkLabel(important_Values_frame, text=' Gain(%):', font=ctk.CTkFont(size=16, weight='bold'))
#sidebar_label_3.grid(row=1, column=1, padx=20 ,pady=(10,0))
sidebar_label_3.pack( padx=20, pady=(10, 0))
sidebar_label_4 = ctk.CTkLabel(important_Values_frame, text=str(round(metrics['PerGain'][-1], 2))+' %',
							   font=ctk.CTkFont(size=16))
#sidebar_label_4.grid(row=2, column=1, padx=20)
sidebar_label_4.pack(padx=20)
# nur letzter Wert
sidebar_label_5 = ctk.CTkLabel(important_Values_frame, text='Mean Portfolio Value:',
							   font=ctk.CTkFont(size=16, weight='bold'))
#sidebar_label_5.grid(row=1, column=2, padx=20,pady=(10,0))
sidebar_label_5.pack( padx=20, pady=(10, 0))
sidebar_label_6 = ctk.CTkLabel(important_Values_frame, text=str(round(metrics['MPV'][-1], 2))+' $',
							   font=ctk.CTkFont(size=16))
#sidebar_label_6.grid(row=2, column=2, padx=20)
sidebar_label_6.pack( padx=20)

recommendation_frame = ctk.CTkFrame(root,height=300)
recommendation_frame.grid(row=1,rowspan=2, column=1, padx=(20, 20), pady=(10, 0), sticky="nsew")
recommendation_frame.grid_rowconfigure(4, weight=1)
recommendation_frame.grid_columnconfigure(2,weight=1)

labelrecommend = ctk.CTkLabel(recommendation_frame, text='Recommendation', font=ctk.CTkFont(size=22, weight="bold"))
#labelrecommend.grid(row=0,column=0,padx=(20,10), pady=(20, 5))
labelrecommend.pack( padx=10, pady=(40, 5))
labelcurrentRecommend = ctk.CTkLabel(recommendation_frame, text='Buy', font=ctk.CTkFont(size=20, weight='bold'),
									 text_color=get_Color_Buy_Up())
#labelcurrentRecommend.grid(row=1,column=0,pady=(5,5), padx=(20,10))
labelcurrentRecommend.pack( pady=(5,0), padx=20)

if strategyDataTest['MoneyInvestedToday'][-1] > 0:
	labelcurrentRecommend.configure(text='Buy', font=ctk.CTkFont(size=20, weight='bold'),
									text_color=get_Color_Buy_Up())
elif strategyDataTest['MoneyInvestedToday'][-1] < 0:
	labelcurrentRecommend.configure(text='Sell', font=ctk.CTkFont(size=20, weight='bold'),
									text_color=get_Color_Sell_Down())
else:
	labelcurrentRecommend.configure(text='Hold', font=ctk.CTkFont(size=20, weight='bold'),
									text_color=get_Color_Hold())

# print last 3 decsions depending on the amounts of available taken decisions
labelLastRecommend = ctk.CTkLabel(recommendation_frame, text='Last Recommendations', font=ctk.CTkFont(size=17, weight="bold"))
labelLastRecommend.pack( pady=(10, 5), padx=20)
#labelLastRecommend.grid(row=0,column=1,pady=(20, 5), padx=(100,20))
labelLRecommend1 = ctk.CTkLabel(recommendation_frame, text='No Last Recommendation', font=ctk.CTkFont(size=16))
labelLRecommend1.pack( pady=5, padx=20)
#labelLRecommend1.grid(row=1,column=1,pady=5, padx=(100,20))
labelLRecommend2 = ctk.CTkLabel(recommendation_frame,text="",font=ctk.CTkFont(size=16))
#labelLRecommend2.grid(row=2,column=1,pady=5, padx=(100,20))
labelLRecommend2.pack( pady=5, padx=20)
labelLRecommend3 = ctk.CTkLabel(recommendation_frame,text='', font=ctk.CTkFont(size=16))
#labelLRecommend3.grid(row=3,column=1,pady=5, padx=(100,20))
labelLRecommend3.pack( pady=5, padx=20)

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

# create value frame
values_frame = ctk.CTkFrame(root,height=300)
values_frame.grid(row=3, column=1, padx=(20, 20), pady=(10, 10), sticky="nsew")

labelL0=ctk.CTkLabel(values_frame, text='Other Metrics', font=ctk.CTkFont(size=22, weight='bold'),underline=0)
labelL0.pack(pady=(20, 20), padx=20)
labelL1 = ctk.CTkLabel(values_frame, text='Invested Money:', font=ctk.CTkFont(size=16, weight="bold"))
# labelL1.grid(row=1, column=1, pady=10, padx=20, sticky="n")
labelL1.pack( pady=(10, 0), padx=20)
labelL2 = ctk.CTkLabel(values_frame, text=str(round(strategyData["MoneyInvested"][-1], 2))+' $',
					   font=ctk.CTkFont(size=16))
# labelL2.grid(row=2, column=1, pady=10, padx=20, sticky="n")
labelL2.pack( pady=(0, 0), padx=20)
labelL3 = ctk.CTkLabel(values_frame, text='Money Not Invested:', font=ctk.CTkFont(size=16, weight="bold"))
# labelL3.grid(row=3, column=1, pady=10, padx=20, sticky="n")
labelL3.pack( pady=(0, 0), padx=20)
labelL4 = ctk.CTkLabel(values_frame, text=str(round(strategyData['MoneyNotInvested'][-1], 2))+' $',
					   font=ctk.CTkFont(size=16))
# labelL4.grid(row=4, column=1, pady=10, padx=20, sticky="n")
labelL4.pack( pady=(0, 0), padx=20)
labelL5 = ctk.CTkLabel(values_frame, text='Standard Deviation:', font=ctk.CTkFont(size=16, weight="bold"))
# labelL5.grid(row=5, column=1, pady=10, padx=20, sticky="n")
labelL5.pack(side=TOP, pady=(0, 0), padx=20)
# nur letzter Wert
labelL6 = ctk.CTkLabel(values_frame, text=str(round(metrics['StdPV'][-1], 2))+' $',
					   font=ctk.CTkFont(size=16))
# labelL6.grid(row=6, column=1, pady=10, padx=20, sticky="n")
labelL6.pack( pady=(0, 0), padx=20)
labelL7 = ctk.CTkLabel(values_frame, text='Max Gain per Day:', font=ctk.CTkFont(size=16, weight="bold"))
# labelL7.grid(row=7, column=1, pady=10, padx=20, sticky="n")
labelL7.pack(pady=(0, 0), padx=20)
# nur letzter Wert
labelL8 = ctk.CTkLabel(values_frame, text=str(round(metrics['MaxGain'][-1], 2))+' %',
					   font=ctk.CTkFont(size=16))
# labelL8.grid(row=8, column=1, pady=10, padx=20, sticky="n")
labelL8.pack(pady=(0, 10), padx=20)

################Plot left side#######################

#########Graph########

# Plot candlesticks

tabview = ctk.CTkTabview(root)
tabview.grid(row=0, column=2, columnspan=2,rowspan=2, padx=(20, 20), pady=(20, 0), sticky="nsew")
tabview.add("2 Weeks")
tabview.add("1M")
tabview.add("6M")
tabview.add("1Y")
tabview.tab("2 Weeks")
tabview.tab("1M")
tabview.tab("6M")
tabview.tab("1Y")

# frame inside tab "test"

label_test = ctk.CTkFrame(tabview.tab("2 Weeks"), width=800, height=500)
label_test.grid(row=0, column=1)

# frame inside tab "1M"
label_1m = ctk.CTkFrame(tabview.tab("1M"), width=800, height=500)
label_1m.grid(row=0, column=1)

# frame inside tab "6M"
label_6m = ctk.CTkFrame(tabview.tab("6M"), width=800, height=500)
label_6m.grid(row=0, column=1)

# frame inside tab "1Y"
label_1y = ctk.CTkFrame(tabview.tab("1Y"), width=800, height=500)
label_1y.grid(row=0, column=1)

# test
line = FigureCanvasTkAgg(show_graph_test(strategyDataTest, stockDataTest), label_test)
line.get_tk_widget().place(width=800, height=500)
# Navigation bar
toolbarFrame = Frame(master=label_test)
toolbarFrame.place(x=0, y=450)
toolbar = NavigationToolbar2Tk(line, toolbarFrame)

# 1M
line = FigureCanvasTkAgg(show_graph(stockData, 18), label_1m)
line.get_tk_widget().place(width=800, height=500)
# Navigation bar
toolbarFrame = Frame(master=label_1m)
toolbarFrame.place(x=0, y=450)
toolbar = NavigationToolbar2Tk(line, toolbarFrame)

# 6M
line = FigureCanvasTkAgg(show_graph(stockData, 123), label_6m)
line.get_tk_widget().place(width=800, height=500)
# Navigation bar
toolbarFrame = Frame(master=label_6m)
toolbarFrame.place(x=0, y=450)
toolbar = NavigationToolbar2Tk(line, toolbarFrame)

# 1Y
line = FigureCanvasTkAgg(show_graph(stockData, 250), label_1y)
line.get_tk_widget().place(width=800, height=500)
# Navigation bar
toolbarFrame = Frame(master=label_1y)
toolbarFrame.place(x=0, y=450)
toolbar = NavigationToolbar2Tk(line, toolbarFrame)

# Visualization for important numbers
tabview = ctk.CTkTabview(root)
tabview.grid(row=2, column=2, columnspan=2,rowspan=2, padx=(20, 20), pady=(20, 0), sticky="nsew")
tabview.add("Mean_PV")
tabview.add("TotalPortfolioValue")
tabview.add("Gain (absolute)")
tabview.add("Gain (percentage)")
tabview.add("Money Invested Today")
tabview.tab("Mean_PV")
tabview.tab("TotalPortfolioValue")
tabview.tab("Gain (absolute)")
tabview.tab("Gain (percentage)")
tabview.tab("Money Invested Today")

# frame inside tab mean PV
label_v_1 = ctk.CTkFrame(tabview.tab("Mean_PV"), width=800, height=450)
label_v_1.grid(row=0, column=1)

# frame inside tab TotalPortfolioValue
label_v_2 = ctk.CTkFrame(tabview.tab("TotalPortfolioValue"), width=800, height=450)
label_v_2.grid(row=0, column=1)

# frame inside tab Gain (absolute)
label_v_3 = ctk.CTkFrame(tabview.tab("Gain (absolute)"), width=800, height=450)
label_v_3.grid(row=0, column=1)

# frame inside tab Gain (percentage)
label_v_4 = ctk.CTkFrame(tabview.tab("Gain (percentage)"), width=800, height=450)
label_v_4.grid(row=0, column=1)

# frame inside tab Money Invested Today
label_v_5 = ctk.CTkFrame(tabview.tab("Money Invested Today"), width=800, height=450)
label_v_5.grid(row=0, column=1)

# Mean PV
line = FigureCanvasTkAgg(show_metrics('MPV'), label_v_1)
line.get_tk_widget().place(width=800, height=400)
# Navigation bar
toolbarFrame = Frame(master=label_v_1)
toolbarFrame.place(x=0, y=400)
NavigationToolbar2Tk(line, toolbarFrame)

# TotalPortfolioValue
line = FigureCanvasTkAgg(show_metrics('PortfolioValue'), label_v_2)
line.get_tk_widget().place(width=800, height=400)
# Navigation bar
toolbarFrame = Frame(master=label_v_2)
toolbarFrame.place(x=0, y=400)
NavigationToolbar2Tk(line, toolbarFrame)

# Gain (absolute)
line = FigureCanvasTkAgg(show_metrics('AbsGain'), label_v_3)
line.get_tk_widget().place(width=800, height=400)
# Navigation bar
toolbarFrame = Frame(master=label_v_3)
toolbarFrame.place(x=0, y=400)
NavigationToolbar2Tk(line, toolbarFrame)

# Gain (percentage)
line = FigureCanvasTkAgg(show_metrics('PerGain'), label_v_4)
line.get_tk_widget().place(width=800, height=400)
# Navigation bar
toolbarFrame = Frame(master=label_v_4)
toolbarFrame.place(x=0, y=400)
NavigationToolbar2Tk(line, toolbarFrame)

# Money Invested Today
line = FigureCanvasTkAgg(show_metrics('MoneyInvestedToday'), label_v_5)
line.get_tk_widget().place(width=800, height=400)
# Navigation bar
toolbarFrame = Frame(master=label_v_5)
toolbarFrame.place(x=0, y=400)
NavigationToolbar2Tk(line, toolbarFrame)

root.protocol("WM_DELETE_WINDOW", _quit)
# update()
root.mainloop()