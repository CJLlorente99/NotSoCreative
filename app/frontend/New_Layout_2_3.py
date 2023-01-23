import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
import plotly.graph_objects as go
import mplfinance as mpf
from datetime import datetime
from matplotlib.pyplot import figure
from matplotlib.figure import Figure
from PIL import Image
import yfinance as yf


########## Functions #########

# Define the dates that we want to consider for our candlestick graph (test) & our Finance metrics

# Option 1: define consider_date as 25th of January



# Yahoo finance data [Open, Close, High, Low]
# All data (incl. 1 Year in the past)
start = '2021-01-01'
end = '2023-01-21'
stock_data = yf.download('^GSPC', start=start, end= end)
# Convert yahoo Finance stock data in the right format (Remove times for each day)
stock_data = stock_data.reset_index()
stock_data['New Date'] = stock_data['Date'].dt.date
stock_data = stock_data.drop(["Date"], axis=1)
stock_data = stock_data.set_index(pd.DatetimeIndex(stock_data["New Date"]))
stock_data.index.name = "Date"
stock_data = stock_data.drop(["New Date"], axis=1)
stock_data["Decision"]=0
stock_data

# Add 1 Decision column


date_test = "2023-01-03"

def donut_plot(data,a,f):

    #widgets = important_Values_frame.grid_slaves(row=0,column=0)
    #widgets[0].destroy()

    colors=['#FFA500', '#0000FF']
    labels=['Cash'+'\n'+'{}$'.format(str(data[0])),'Stocks'+'\n'+'{}$'.format(str(data[0]))]
    slaves = important_Values_frame.grid_slaves(row=0,column=0)

    a.clear()

    if ctk.get_appearance_mode() == 'Dark':


        a.set_title( r"$\bf{Portfolio}$",color='white',fontsize=18, )
        plt.rcParams['text.color'] = 'white'
        f.patch.set_facecolor('#2B2B2B')
        a.pie(data, radius=1, wedgeprops=dict(width=0.5), colors=colors, labels=labels)


    else:

        f = Figure(figsize=(4,4))
        a = f.add_subplot(111)
        a.set_title(r"$\bf{Portfolio}$", color='black', fontsize=18, )
        a.set_title('Portfolio')
        f.patch.set_facecolor('#DBDBDB')
        plt.rcParams['text.color'] = 'black'
        a.pie(data, radius=1, wedgeprops=dict(width=0.5), colors=colors, labels=labels)

    return f

def show_metrics(data_csv, var, date_test):


    index_no = data_csv.index.get_loc(date_test)
    x = data_csv.index[index_no:]
    y = data_csv.loc[date_test:, var]

    for widgets in label_v_5.winfo_children():
        for widgets in label_v_1.winfo_children():
            widgets.destroy()
        for widgets in label_v_2.winfo_children():
            widgets.destroy()
        for widgets in label_v_3.winfo_children():
            widgets.destroy()
        for widgets in label_v_4.winfo_children():
            widgets.destroy()
        for widgets in label_v_5.winfo_children():
            widgets.destroy()


    if ctk.get_appearance_mode() == 'Dark':
        f = Figure(figsize=(2, 3))
        a = f.add_subplot(111)

        plt.style.use('dark_background')
        a.plot(x, y)
    else:
        f = Figure(figsize=(2, 3))
        plt.style.use('default')
        a = f.add_subplot(111)
        a.plot(x, y)
    return f

# get the newest selected value of a selected strategy (kind=selected Value) .tail(number of investment strategies)

def show_graph(stock_data, i):
    # Plot candlesticks
    # Create list filled with buy/hold/sell actions


    #! Be caureful with i here
    for widgets in label_1y.winfo_children():

        for widgets in label_1m.winfo_children():
            widgets.destroy()
        for widgets in label_6m.winfo_children():
            widgets.destroy()
        for widgets in label_1y.winfo_children():
            widgets.destroy()
        for widgets in label_test.winfo_children():
            widgets.destroy()

    if ctk.get_appearance_mode() == 'Dark':
        mode ='nightclouds'
    else:
        mode='default'

    # plot candelsticks
    color_candels = mpf.make_marketcolors(up=get_Color_Buy_Up(), down=get_Color_Sell_Down(), wick="inherit",
                                          edge="inherit", volume="in")
    mpf_style = mpf.make_mpf_style(base_mpf_style=mode, marketcolors=color_candels)
    fig, axl = mpf.plot(stock_data.iloc[-i:, :], type='candle', volume=False, style=mpf_style, returnfig=True,
                        figsize=(6, 3.5))
    return fig


# specificly for our test time window
def show_graph_test(data_csv, stock_data, date_test):
    # Plot candlesticks
    # Create list filled with buy/hold/sell actions

    i_csv = data_csv.index.get_loc(date_test)
    i_MoneyInvestedToday = data_csv.columns.get_loc("MoneyInvestedToday")
    i_date = stock_data.index.get_loc(date_test)



    for widgets in label_1y.winfo_children():
        for widgets in label_1m.winfo_children():
            widgets.destroy()
        for widgets in label_6m.winfo_children():
            widgets.destroy()
        for widgets in label_1y.winfo_children():
            widgets.destroy()
        for widgets in label_test.winfo_children():
            widgets.destroy()

    # Fill action list with investment decisions made
    action = []


    for (columnName, columnData) in data_csv.iloc[i_csv: ,i_MoneyInvestedToday].items():
        if columnData > 0:
            action.append(1)
        elif columnData < 0:
            action.append(-1)
        else:
            action.append(0)


    i_decision = stock_data.columns.get_loc("Decision")


    for i in range(len(action)):
        stock_data.iloc[i_date + i, i_decision] = action[i]

    if ctk.get_appearance_mode() == 'Dark':
        mode ='nightclouds'
    else:
        mode='default'


    # Color of candlesticks
    color_candels = mpf.make_marketcolors(up=get_Color_Buy_Up(), down=get_Color_Sell_Down(), wick="inherit",
                                          edge="inherit", volume="in")

    # Color of Investment Actions (Triangles)
    colors_apd = [get_Color_Buy_Up() if v == 1 else get_Color_Hold() if v == 0 else get_Color_Sell_Down() for v in
                  stock_data["Decision"].iloc[i_date:]]

    apd = mpf.make_addplot(stock_data["Decision"].iloc[i_date:], type='scatter', marker='^', markersize=200, color=colors_apd)


    # plot candelsticks
    mpf_style = mpf.make_mpf_style(base_mpf_style=mode, marketcolors=color_candels)
    fig, axl = mpf.plot(stock_data.iloc[i_date:, :], type='candle', volume=False, style=mpf_style, returnfig=True, addplot=apd,
                        figsize=(8, 5))

    return fig



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


def getCurrentValue(df, strategy, kind):
    # newest date
    newDate = df['Date'].max()
    newestData = df.loc[df['Date'] == newDate]
    strategyData = newestData.loc[newestData['investorStrategy'] == strategy]
    return strategyData.iloc[0][kind]


# get newest decison
def getDecision(MoneyInvestedToday):
    if MoneyInvestedToday > 0:
        decision = 'Buy'
    elif MoneyInvestedToday < 0:
        decision = 'Sell'
    else:
        decision = 'Hold'
    return decision


# open new window for Informations
def openNewWindow():
     # Toplevel object which will
    # be treated as a new window
    newWindow = ctk.CTkToplevel(root)
    textbox = ctk.CTkTextbox(newWindow, width= 400, height= 400)
    textbox.insert("0.0",
           "Information\n\n "  + "Portfolio Value = Cash + Current Stock Value\n\n" + " Gain (%) " + "= Percentage difference between the purchase price and the current value of the shares\n\n" + " Gain (absolute) = Difference between the current value of the shares and the purchase price of the shares\n\n" + "Mean Portfolio Value= Mean of Portfolio Value for all test days Invested Money = Money that has been invested in the stock market\n\n" + " Invested Money Today = Money that is invested today\n\n" + " Money not Invested = Money that hasn`t been invested in the stock market" + " Standard Deviation = Describes the spread of the PVs (?) \n\n" + " Max. gain per Day = Maximum percentage gain on one day\n\n")


    '''
    textbox.insert("0.0","Information\n\n ")

    textbox.insert("0.0","Portfolio Value")
    textbox.insert("=Cash + Current Stock Value\n\n")

    textbox.insert("0.0","Gain (%)")
    textbox.insert("=Percentage difference between the purchase price and the current value of the shares\n\n")

    textbox.insert("0.0","Gain (absolute)")
    textbox.insert("= Difference between the current value of the shares and the purchase price of the shares\n\n")

    textbox.insert("0.0","Mean Portfolio Value")
    textbox.insert("= Mean of Portfolio Value for all test days Invested Money = Money that has been invested in the stock market\n\n")

    textbox.insert("0.0","Invested Money Today")
    textbox.insert("= Money that is invested today\n\n")

    textbox.insert("0.0","Money not Invested ")
    textbox.insert("= Money that hasn`t been invested in the stock market")

    textbox.insert("0.0","Standard Deviation ")
    textbox.insert("= Describes the spread of the PVs (?) \n\n")

    textbox.insert("0.0","Standard Deviation ")
    textbox.insert("= Describes the spread of the PVs (?) \n\n")

    textbox.insert("0.0","Max. gain per Day")
    textbox.insert("= Maximum percentage gain on one day\n\n")
    '''


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


# changes dappearance mode
def change_appearance_mode_event(new_appearance_mode: str):
    ctk.set_appearance_mode(new_appearance_mode)
    ctk.get_appearance_mode()


# changes scaliing
def change_scaling_event(new_scaling: str):
    new_scaling_float = int(new_scaling.replace("%", "")) / 100
    ctk.set_widget_scaling(new_scaling_float)

########## Main Frontend#########

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

# Main Window
root = ctk.CTk()
root.geometry('800x600')
root.title("Stock Market Prediction Engine")

# configure the grid
root.grid_columnconfigure(1, weight=1)
root.grid_columnconfigure((2, 3), weight=0)
root.grid_rowconfigure((0, 1, 2), weight=1)

switch_var = ctk.StringVar(value="off")


####### UPDATE START##########
def update():

    ctk.set_appearance_mode(appearance_mode_optionmenu.get())
    new_scaling_float = int(scaling_optionmenu.get().replace("%", "")) / 100
    ctk.set_widget_scaling(new_scaling_float)

    stock_data = yf.download('^GSPC', start=start, end=end)
    # Convert yahoo Finance stock data in the right format (Remove times for each day)
    stock_data = stock_data.reset_index()
    stock_data['New Date'] = stock_data['Date'].dt.date
    stock_data = stock_data.drop(["Date"], axis=1)
    stock_data = stock_data.set_index(pd.DatetimeIndex(stock_data["New Date"]))
    stock_data.index.name = "Date"
    stock_data = stock_data.drop(["New Date"], axis=1)
    stock_data["Decision"] = 0
    stock_data

    df = pd.read_csv("NewData.csv")
    investorStrategy = 'ca'
    MoneyInvestedToday = getCurrentValue(df, investorStrategy, 'MoneyInvestedToday')

    # Not actual open price but start value of new day
    NewDayValue = df.iloc[11::22, :]

    cash = round(getCurrentValue(df, investorStrategy, 'MoneyNotInvested'), 2)
    stocks = round(getCurrentValue(df, investorStrategy, 'TotalPortfolioValue') - getCurrentValue(df, investorStrategy,
                                                                                                  'MoneyNotInvested'),
                   2)

    if stocks > 0:
        stocks = stocks
    else:
        stocks = 0

    pf = [cash, stocks]

    pfv = FigureCanvasTkAgg(donut_plot(pf, a, f), important_Values_frame)
    pfv.get_tk_widget().grid(row=0, column=0, padx=20, pady=(10, 0))

    sidebar_label_2.configure(text=str(round(getCurrentValue(df, investorStrategy, 'TotalPortfolioValue'),2)))
    sidebar_label_4.configure(text=str(round(getCurrentValue(df, investorStrategy, 'perGain'), 2)))
    sidebar_label_6.configure(text=str(round(getCurrentValue(df, investorStrategy, 'MPV'), 2)))


    if MoneyInvestedToday > 0:
        labelcurrentRecommend.configure(text='Buy', font=ctk.CTkFont(size=20, weight='bold'),text_color=get_Color_Buy_Up())

    elif MoneyInvestedToday < 0:

        labelcurrentRecommend.configure(text='Sell', font=ctk.CTkFont(size=20, weight='bold'),
                               text_color=get_Color_Sell_Down())

    else:

        labelcurrentRecommend.configure(text='Hold', font=ctk.CTkFont(size=20, weight='bold'),
                               text_color=get_Color_Hold())




    #dates for Last recomandations
    df_investorStrategy =df.loc[df['investorStrategy'] == investorStrategy]
    lastDates=[]

    for dates in df_investorStrategy.Date:
        dates = dates[:-9]
        lastDates.append(dates)



    lastDecisions = []
    for (columnName, columnData) in df_investorStrategy["MoneyInvestedToday"].items():
        if columnData > 0:
            lastDecisions.append('Buy')
        elif columnData < 0:
            lastDecisions.append('Sell')
        elif columnData==0:
            lastDecisions.append('Hold')
        else:
            lastDecisions.append('')



    # print last 3 decsions depending on the amounts of available taken decisions
    if len(lastDecisions) <= 1:
        labelLRecommend1.configure(text='No Last Recommendation', font=ctk.CTkFont(size=16))

    elif len(lastDecisions) == 2:
        labelLRecommend1.configure(text=str(lastDates[-2]) + ':    '+lastDecisions[-2], font=ctk.CTkFont(size=16))

    elif len(lastDecisions) == 3:
        labelLRecommend1.configure(text=str(lastDates[-2]) + ':    '+ lastDecisions[-2], font=ctk.CTkFont(size=16))
        labelLRecommend2.configure(text=str(lastDates[-4]) + ':    '+lastDecisions[-3], font=ctk.CTkFont(size=16))
    elif len(lastDecisions) > 3:
        labelLRecommend1.configure(text=str(lastDates[-2]) + ':    '+ lastDecisions[-2], font=ctk.CTkFont(size=16))
        labelLRecommend2.configure(text=str(lastDates[-3]) + ':    '+lastDecisions[-3], font=ctk.CTkFont(size=16))
        labelLRecommend3.configure(text=str(lastDates[-4]) + ':    '+ lastDecisions[-4], font=ctk.CTkFont(size=16))

    labelL2.configure(text=str(round(getCurrentValue(df, investorStrategy, 'MoneyInvested'), 2)))
    labelL4.configure(text=str(round(getCurrentValue(df, investorStrategy, 'MoneyNotInvested'), 2)))
    labelL6.configure(text=str(round(getCurrentValue(df, investorStrategy, 'StdPV'), 2)))
    labelL8.configure(text=str(round(getCurrentValue(df, investorStrategy, 'maxGain'), 2)))


    # ,"TotalPortfolioValue", "MPV", "absGain","perGain", "MoneyInvestedToday"
    # Preparing data for candlesticks: Necessary data for candlesticks: "Date","Open","High","Low","Close","Volume"
    data_csv = NewDayValue[
        ["Date", "Open", "High", "Low", "Close", "Volume", "TotalPortfolioValue", "MPV", "absGain", "perGain",
         "MoneyInvestedToday"]]
    # maybe we should add this to the beinning (Data Prperation)
    # Adding the date column to the dataframe
    lst = []
    for i in range(0, len(data_csv)):
        new_date = datetime.strptime(data_csv.iloc[i, 0], '%Y-%m-%d %H:%M:%S')
        lst.append(new_date.date())

    data_csv["New Date"] = lst
    data_csv = data_csv.drop(["Date"], axis=1)
    data_csv = data_csv.set_index(pd.DatetimeIndex(data_csv["New Date"]))
    data_csv.index.name = "Date"
    data_csv = data_csv.drop(["New Date"], axis=1)


    #####Update Candlesticks######
    # test
    line = FigureCanvasTkAgg(show_graph_test(data_csv, stock_data, date_test), label_test)
    line.get_tk_widget().place(width=600, height=350)
    # Navigation bar
    toolbarFrame = Frame(master=label_test)
    toolbarFrame.place(x=0, y=450)
    toolbar = NavigationToolbar2Tk(line, toolbarFrame)
    # 1M
    line = FigureCanvasTkAgg(show_graph(stock_data, 18), label_1m)
    line.get_tk_widget().place(width=800, height=450)
    # Navigation bar
    toolbarFrame = Frame(master=label_1m)
    toolbarFrame.place(x=0, y=450)
    toolbar = NavigationToolbar2Tk(line, toolbarFrame)

    # 6M
    line = FigureCanvasTkAgg(show_graph(stock_data, 123), label_6m)
    line.get_tk_widget().place(width=800, height=450)
    # Navigation bar
    toolbarFrame = Frame(master=label_6m)
    toolbarFrame.place(x=0, y=450)
    toolbar = NavigationToolbar2Tk(line, toolbarFrame)

    # 1Y
    line = FigureCanvasTkAgg(show_graph(stock_data, 250), label_1y)
    line.get_tk_widget().place(width=800, height=450)
    # Navigation bar
    toolbarFrame = Frame(master=label_1y)
    toolbarFrame.place(x=0, y=450)
    toolbar = NavigationToolbar2Tk(line, toolbarFrame)

    ###Update Value Graphs####
    # Mean PV
    line = FigureCanvasTkAgg(show_metrics(data_csv, "MPV", date_test), label_v_1)
    line.get_tk_widget().place(width=800, height=400)
    # Navigation bar
    toolbarFrame = Frame(master=label_v_1)
    toolbarFrame.place(x=0, y=400)
    toolbar = NavigationToolbar2Tk(line, toolbarFrame)

    # TotalPortfolioValue
    line = FigureCanvasTkAgg(show_metrics(data_csv, "TotalPortfolioValue", date_test), label_v_2)
    line.get_tk_widget().place(width=800, height=400)
    # Navigation bar
    toolbarFrame = Frame(master=label_v_2)
    toolbarFrame.place(x=0, y=400)
    toolbar = NavigationToolbar2Tk(line, toolbarFrame)

    # Gain (absolute)
    line = FigureCanvasTkAgg(show_metrics(data_csv, "absGain", date_test), label_v_3)
    line.get_tk_widget().place(width=800, height=400)
    # Navigation bar
    toolbarFrame = Frame(master=label_v_3)
    toolbarFrame.place(x=0, y=400)
    toolbar = NavigationToolbar2Tk(line, toolbarFrame)

    # Gain (percentage)
    line = FigureCanvasTkAgg(show_metrics(data_csv, "perGain", date_test), label_v_4)
    line.get_tk_widget().place(width=800, height=400)
    # Navigation bar
    toolbarFrame = Frame(master=label_v_4)
    toolbarFrame.place(x=0, y=400)
    toolbar = NavigationToolbar2Tk(line, toolbarFrame)

    # Money Invested Today
    line = FigureCanvasTkAgg(show_metrics(data_csv, "MoneyInvestedToday", date_test), label_v_5)
    line.get_tk_widget().place(width=800, height=400)
    # Navigation bar
    toolbarFrame = Frame(master=label_v_5)
    toolbarFrame.place(x=0, y=400)
    toolbar = NavigationToolbar2Tk(line, toolbarFrame)

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

df = pd.read_csv("NewData.csv")
investorStrategy = 'ca'
MoneyInvestedToday = getCurrentValue(df, investorStrategy, 'MoneyInvestedToday')

# Not actual open price but start value of new day
NewDayValue = df.iloc[11::22, :]


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
important_Values_frame = ctk.CTkFrame(root)
important_Values_frame.grid(row=0, column=1, padx=(20, 20), pady=(10, 5), sticky="nsew")
important_Values_frame.grid_rowconfigure(3, weight=1)
important_Values_frame.grid_columnconfigure(3,weight=1)

cash = round(getCurrentValue(df,investorStrategy,'MoneyNotInvested'),2)
stocks = round(getCurrentValue(df,investorStrategy,'TotalPortfolioValue')- getCurrentValue(df,investorStrategy,'MoneyNotInvested'),2)

if stocks>0:
    stocks =stocks
else:
    stocks=0

pf =[cash,stocks]

f = Figure(figsize=(4, 4), dpi=100)
a = f.add_subplot(111)

pfv = FigureCanvasTkAgg(donut_plot(pf,a,f),important_Values_frame)
pfv.get_tk_widget().grid(row=0, column=0, padx=20,pady=(10, 0))


sidebar_label_1 = ctk.CTkLabel(important_Values_frame, text='Portfolio Value ($):', font=ctk.CTkFont(size=16, weight='bold'))
sidebar_label_1.grid(row=1, column=0, padx=20,pady=(10,0))
#sidebar_label_1.pack(side=TOP, padx=20, pady=(50, 0))
sidebar_label_2 = ctk.CTkLabel(important_Values_frame,
                               text=str(round(getCurrentValue(df, investorStrategy, 'TotalPortfolioValue'), 2)),
                               font=ctk.CTkFont(size=16))
sidebar_label_2.grid(row=2, column=0, padx=20)
#sidebar_label_2.pack(side=TOP, padx=20)

sidebar_label_3 = ctk.CTkLabel(important_Values_frame, text=' Gain(%):', font=ctk.CTkFont(size=16, weight='bold'))
sidebar_label_3.grid(row=1, column=1, padx=20 ,pady=(10,0))
#sidebar_label_3.pack(side=TOP, padx=20, pady=(10, 0))
sidebar_label_4 = ctk.CTkLabel(important_Values_frame, text=str(round(getCurrentValue(df, investorStrategy, 'perGain'), 2)),
                               font=ctk.CTkFont(size=16))
sidebar_label_4.grid(row=2, column=1, padx=20)
#sidebar_label_4.pack(side=TOP, padx=20)
sidebar_label_5 = ctk.CTkLabel(important_Values_frame, text='Mean Portfolio Value ($):',
                               font=ctk.CTkFont(size=16, weight='bold'))
sidebar_label_5.grid(row=1, column=2, padx=20,pady=(10,0))
#sidebar_label_5.pack(side=TOP, padx=20, pady=(10, 0))
sidebar_label_6 = ctk.CTkLabel(important_Values_frame, text=str(round(getCurrentValue(df, investorStrategy, 'MPV'), 2)),
                               font=ctk.CTkFont(size=16))
sidebar_label_6.grid(row=2, column=2, padx=20)
#sidebar_label_6.pack(side=TOP, padx=20)




recommendation_frame = ctk.CTkFrame(root)
recommendation_frame.grid(row=1, column=1, padx=(20, 20), pady=(10, 5), sticky="nsew")
recommendation_frame.grid_rowconfigure(4, weight=1)
recommendation_frame.grid_columnconfigure(2,weight=1)

labelrecommend = ctk.CTkLabel(recommendation_frame, text='Recommendation', font=ctk.CTkFont(size=22, weight="bold"))
labelrecommend.grid(row=0,column=0,padx=(20,10), pady=(20, 5))
#labelrecommend.pack(side=TOP, padx=10, pady=(40, 5))
labelcurrentRecommend = ctk.CTkLabel(recommendation_frame, text='Buy', font=ctk.CTkFont(size=20, weight='bold'),
                       text_color=get_Color_Buy_Up())
labelcurrentRecommend.grid(row=1,column=0,pady=(5,5), padx=(20,10))
#labelcurrentRecommend.pack(side=TOP, pady=(5,0), padx=20)

if MoneyInvestedToday > 0:
    labelcurrentRecommend.configure(text='Buy', font=ctk.CTkFont(size=20, weight='bold'),
                       text_color=get_Color_Buy_Up())
    decision = 'Buy'
elif MoneyInvestedToday < 0:
    labelcurrentRecommend.configure(text='Sell', font=ctk.CTkFont(size=20, weight='bold'),
                   text_color=get_Color_Sell_Down())
    decision = 'sell'
else:
    labelcurrentRecommend.configure(text='Hold', font=ctk.CTkFont(size=20, weight='bold'),
                   text_color=get_Color_Hold())
    decision = 'hold'

    # list to save last descions
lastDecisions = []
lastDecisions.append(decision)

# print last 3 decsions depending on the amounts of available taken decisions
labelLastRecommend = ctk.CTkLabel(recommendation_frame, text='Last 3 Recommendations', font=ctk.CTkFont(size=18, weight="bold"))
# labelL3.grid(row=2, column=1, pady=10, padx=20, sticky="n")
#labelLastRecommend.pack(side=TOP, pady=(10, 5), padx=20)
labelLastRecommend.grid(row=0,column=1,pady=(20, 5), padx=(100,20))
labelLRecommend1 = ctk.CTkLabel(recommendation_frame, text='No Last Recommendation', font=ctk.CTkFont(size=16))
#labelLRecommend1.pack(side=TOP, pady=10, padx=20)
labelLRecommend1.grid(row=1,column=1,pady=5, padx=(100,20))
labelLRecommend2 = ctk.CTkLabel(recommendation_frame,text="",font=ctk.CTkFont(size=16))
labelLRecommend2.grid(row=2,column=1,pady=5, padx=(100,20))
#labelLRecommend2.pack(side=TOP, pady=10, padx=20)
labelLRecommend3 = ctk.CTkLabel(recommendation_frame,text='', font=ctk.CTkFont(size=16))
labelLRecommend3.grid(row=3,column=1,pady=5, padx=(100,20))
#labelLRecommend3.pack(side=TOP, pady=10, padx=20)

if len(lastDecisions) <= 1:
    labelLRecommend1.configure(text='No Last Recommendation', font=ctk.CTkFont(size=16))

elif len(lastDecisions) == 2:
    labelLRecommend1.configure(text=lastDecisions[-1], font=ctk.CTkFont(size=16))

elif len(lastDecisions) == 3:
    labelLRecommend1.configure(text=lastDecisions[-1], font=ctk.CTkFont(size=16))
    labelLRecommend2.configure(text=lastDecisions[-2], font=ctk.CTkFont(size=16))
elif len(lastDecisions) > 3:
    labelLRecommend1.configure(text=lastDecisions[-1], font=ctk.CTkFont(size=16))
    labelLRecommend2.configure(text=lastDecisions[-2], font=ctk.CTkFont(size=16))
    labelLRecommend3.configure(text=lastDecisions[-3], font=ctk.CTkFont(size=16))

# create value frame
values_frame = ctk.CTkFrame(root)
values_frame.grid(row=2, column=1, rowspan=2, padx=(20, 20), pady=(5, 10), sticky="nsew")

labelL1 = ctk.CTkLabel(values_frame, text='Invested Money ($):', font=ctk.CTkFont(size=16, weight="bold"))
# labelL1.grid(row=1, column=1, pady=10, padx=20, sticky="n")
labelL1.pack(side=TOP, pady=(60, 0), padx=20)
labelL2 = ctk.CTkLabel(values_frame, text=str(round(getCurrentValue(df, investorStrategy, 'MoneyInvested'), 2)),
                       font=ctk.CTkFont(size=16))
# labelL2.grid(row=2, column=1, pady=10, padx=20, sticky="n")
labelL2.pack(side=TOP, pady=(0, 10), padx=20)
labelL3 = ctk.CTkLabel(values_frame, text='Money Not Invested ($):', font=ctk.CTkFont(size=16, weight="bold"))
# labelL3.grid(row=3, column=1, pady=10, padx=20, sticky="n")
labelL3.pack(side=TOP, pady=(10, 0), padx=20)
labelL4 = ctk.CTkLabel(values_frame, text=str(round(getCurrentValue(df, investorStrategy, 'MoneyNotInvested'), 2)),
                       font=ctk.CTkFont(size=16))
# labelL4.grid(row=4, column=1, pady=10, padx=20, sticky="n")
labelL4.pack(side=TOP, pady=(0, 10), padx=20)
labelL5 = ctk.CTkLabel(values_frame, text='Standard Deviation ($):', font=ctk.CTkFont(size=16, weight="bold"))
# labelL5.grid(row=5, column=1, pady=10, padx=20, sticky="n")
labelL5.pack(side=TOP, pady=(10, 0), padx=20)
labelL6 = ctk.CTkLabel(values_frame, text=str(round(getCurrentValue(df, investorStrategy, 'StdPV'), 2)),
                       font=ctk.CTkFont(size=16))
# labelL6.grid(row=6, column=1, pady=10, padx=20, sticky="n")
labelL6.pack(side=TOP, pady=(0, 10), padx=20)
labelL7 = ctk.CTkLabel(values_frame, text='Max Gain per Day (%):', font=ctk.CTkFont(size=16, weight="bold"))
# labelL7.grid(row=7, column=1, pady=10, padx=20, sticky="n")
labelL7.pack(side=TOP, pady=(10, 0), padx=20)
labelL8 = ctk.CTkLabel(values_frame, text=str(round(getCurrentValue(df, investorStrategy, 'maxGain'), 2)),
                       font=ctk.CTkFont(size=16))
# labelL8.grid(row=8, column=1, pady=10, padx=20, sticky="n")
labelL8.pack(side=TOP, pady=(0, 10), padx=20)

################Plot left side#######################

#########Graph########


# ,"TotalPortfolioValue", "MPV", "absGain","perGain", "MoneyInvestedToday"
# Preparing data for candlesticks: Necessary data for candlesticks: "Date","Open","High","Low","Close","Volume"
data_csv = NewDayValue[
    ["Date", "Open", "High", "Low", "Close", "Volume", "TotalPortfolioValue", "MPV", "absGain", "perGain",
     "MoneyInvestedToday"]]
data_csv
# maybe we should add this to the beinning (Data Prperation)
# Adding the date column to the dataframe
lst = []
for i in range(0, len(data_csv)):
    new_date = datetime.strptime(data_csv.iloc[i, 0], '%Y-%m-%d %H:%M:%S')
    lst.append(new_date.date())

data_csv["New Date"] = lst
data_csv = data_csv.drop(["Date"], axis=1)
data_csv = data_csv.set_index(pd.DatetimeIndex(data_csv["New Date"]))
data_csv.index.name = "Date"
data_csv = data_csv.drop(["New Date"], axis=1)
data_csv.index
# -------------------------------------------

# Plot candlesticks

tabview = ctk.CTkTabview(root)
tabview.grid(row=0, column=2, columnspan=2, padx=(20, 20), pady=(20, 0), sticky="nsew")
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
line = FigureCanvasTkAgg(show_graph_test(data_csv, stock_data, date_test), label_test)
line.get_tk_widget().place(width=800, height=450)
# Navigation bar
toolbarFrame = Frame(master=label_test)
toolbarFrame.place(x=0, y=450)
toolbar = NavigationToolbar2Tk(line, toolbarFrame)

# 1M
line = FigureCanvasTkAgg(show_graph(stock_data, 18), label_1m)
line.get_tk_widget().place(width=800, height=450)
# Navigation bar
toolbarFrame = Frame(master=label_1m)
toolbarFrame.place(x=0, y=450)
toolbar = NavigationToolbar2Tk(line, toolbarFrame)

# 6M
line = FigureCanvasTkAgg(show_graph(stock_data, 123), label_6m)
line.get_tk_widget().place(width=800, height=450)
# Navigation bar
toolbarFrame = Frame(master=label_6m)
toolbarFrame.place(x=0, y=450)
toolbar = NavigationToolbar2Tk(line, toolbarFrame)

# 1Y
line = FigureCanvasTkAgg(show_graph(stock_data, 250), label_1y)
line.get_tk_widget().place(width=800, height=450)
# Navigation bar
toolbarFrame = Frame(master=label_1y)
toolbarFrame.place(x=0, y=450)
toolbar = NavigationToolbar2Tk(line, toolbarFrame)




# Visualization for important numbers
tabview = ctk.CTkTabview(root)
tabview.grid(row=2, column=2, columnspan=2, padx=(20, 20), pady=(20, 0), sticky="nsew")
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
label_v_1 = ctk.CTkFrame(tabview.tab("Mean_PV"), width=800, height=500)
label_v_1.grid(row=0, column=1)

# frame inside tab TotalPortfolioValue
label_v_2 = ctk.CTkFrame(tabview.tab("TotalPortfolioValue"), width=800, height=500)
label_v_2.grid(row=0, column=1)

# frame inside tab Gain (absolute)
label_v_3 = ctk.CTkFrame(tabview.tab("Gain (absolute)"), width=800, height=500)
label_v_3.grid(row=0, column=1)

# frame inside tab Gain (percentage)
label_v_4 = ctk.CTkFrame(tabview.tab("Gain (percentage)"), width=800, height=500)
label_v_4.grid(row=0, column=1)

# frame inside tab Money Invested Today
label_v_5 = ctk.CTkFrame(tabview.tab("Money Invested Today"), width=800, height=500)
label_v_5.grid(row=0, column=1)

# Mean PV
line = FigureCanvasTkAgg(show_metrics(data_csv, "MPV", date_test), label_v_1)
line.get_tk_widget().place(width=800, height=400)
# Navigation bar
toolbarFrame = Frame(master=label_v_1)
toolbarFrame.place(x=0, y=400)
toolbar = NavigationToolbar2Tk(line, toolbarFrame)

# TotalPortfolioValue
line = FigureCanvasTkAgg(show_metrics(data_csv, "TotalPortfolioValue", date_test), label_v_2)
line.get_tk_widget().place(width=800, height=400)
# Navigation bar
toolbarFrame = Frame(master=label_v_2)
toolbarFrame.place(x=0, y=400)
toolbar = NavigationToolbar2Tk(line, toolbarFrame)

# Gain (absolute)
line = FigureCanvasTkAgg(show_metrics(data_csv, "absGain", date_test), label_v_3)
line.get_tk_widget().place(width=800, height=400)
# Navigation bar
toolbarFrame = Frame(master=label_v_3)
toolbarFrame.place(x=0, y=400)
toolbar = NavigationToolbar2Tk(line, toolbarFrame)

# Gain (percentage)
line = FigureCanvasTkAgg(show_metrics(data_csv, "perGain", date_test), label_v_4)
line.get_tk_widget().place(width=800, height=400)
# Navigation bar
toolbarFrame = Frame(master=label_v_4)
toolbarFrame.place(x=0, y=400)
toolbar = NavigationToolbar2Tk(line, toolbarFrame)

# Money Invested Today
line = FigureCanvasTkAgg(show_metrics(data_csv, "MoneyInvestedToday", date_test), label_v_5)
line.get_tk_widget().place(width=800, height=400)
# Navigation bar
toolbarFrame = Frame(master=label_v_5)
toolbarFrame.place(x=0, y=400)
toolbar = NavigationToolbar2Tk(line, toolbarFrame)


root.mainloop()