import csv
import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
import plotly.graph_objects as go
import mplfinance as mpf
from datetime import datetime
from matplotlib.pyplot import figure



###############Data Preperation##############

#get the newest selected value of a selected strategy (kind=selected Value) .tail(number of investment strategies)

def show_graph(NewDayValue, data_n, i):
     # Plot candlesticks
     mpf_style = mpf.make_mpf_style(base_mpf_style="nightclouds", marketcolors=color_graph())

     #Create list filled with buy/hold/sell actions
     action = []
     for (columnName, columnData) in NewDayValue["MoneyInvestedToday"].items():
         if columnData > 0:
           action.append(1)
         elif columnData< 0:
             action.append(-1)
         else:
             action.append(0)
     data_n["Decision"] = action

     # Investment action visualization - Triangles
     colors_2 = ['g' if v == 1 else "#FFA500" if v==0 else 'r' for v in data_n["Decision"].iloc[-i:]]
     apd = mpf.make_addplot(data_n["Decision"].iloc[-i:], type='scatter', markersize=200, marker='^', color=colors_2)


     fig_1, axl = mpf.plot(data_n.iloc[-i:, :], type='candle', volume=False,
                           style=mpf_style, returnfig=True, addplot=apd, figsize=(6,3.5))
     return fig_1


def getCurrentValue(df, strategy, kind):
    #newest date
    newDate = df['Date'].max()
    newestData = df.loc[df['Date'] == newDate]
    strategyData=newestData.loc[newestData['investorStrategy'] == strategy]
    return strategyData.iloc[0][kind]

#get newest decison
def getDecision(MoneyInvestedToday):
   if MoneyInvestedToday>0:
       decision='Buy'
   elif MoneyInvestedToday<0:
       decision ='Sell'
   else:
       decision ='Hold'
   return decision

#open new window for Informations
def openNewWindow():
    # Toplevel object which will
    # be treated as a new window
    newWindow = ctk.CTkToplevel(root)
    # sets the title of the
    # Toplevel widget
    newWindow.title("Information")
    # sets the geometry of toplevel
    newWindow.geometry("200x200")
    # A Label widget to show in toplevel
    Label(newWindow,
          text="Information").pack()

#changes dappearance mode
def change_appearance_mode_event( new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

#changes scaliing
def change_scaling_event( new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        ctk.set_widget_scaling(new_scaling_float)


############################### Main Frontend #########################


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
def color_graph():
    #print("switch toggled, current value:", switch_var.get())

    if switch_var.get() == "off":
        colors = mpf.make_marketcolors(up="#00ff00", down="#ff0000", wick="inherit", edge="inherit",
                                       volume="in")

    elif switch_var.get() == "on":
        colors = mpf.make_marketcolors(up="#5D02FF", down="#E60400", wick="inherit", edge="inherit",
                                       volume="in")

    return colors


def update():
    df = pd.read_csv("NewCSV.csv")
    investorStrategy = 'ca'
    MoneyInvestedToday = getCurrentValue(df, investorStrategy, 'MoneyInvestedToday')

    # Not actual open price but start value of new day
    NewDayValue = df.iloc[::12, :]



############## START Widgets###########

###############Left side##################

    # create left sidebar frame with widgets
    sidebar_frame = ctk.CTkFrame(root, width=140, corner_radius=0)
    sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
    sidebar_frame.grid_rowconfigure(4, weight=1)
    logo_label = ctk.CTkLabel(sidebar_frame, text="Prediction Engine",
                                              font=ctk.CTkFont(size=26, weight="bold"))

    #logo_label.grid(row=0, column=0, padx=20, pady=(20, 40))
    logo_label.pack(side=TOP,padx=20, pady=(20, 40))

    sidebar_label_1 = ctk.CTkLabel(sidebar_frame,text='Portfolio Value ($):', font=ctk.CTkFont(size=16,weight='bold'))
    #sidebar_label_1.grid(row=1, column=0, padx=20,pady=(10,0))
    sidebar_label_1.pack(side=TOP,padx=20,pady=(10,0))
    sidebar_label_2 = ctk.CTkLabel(sidebar_frame,text=str(round(getCurrentValue(df, investorStrategy, 'TotalPortfolioValue'),2)), font=ctk.CTkFont(size=16))
    #sidebar_label_2.grid(row=2, column=0, padx=20)
    sidebar_label_2.pack(side=TOP,padx=20)
    sidebar_label_3 = ctk.CTkLabel(sidebar_frame, text=' Gain(%):', font=ctk.CTkFont(size=16,weight='bold'))
    #sidebar_label_3.grid(row=3, column=0, padx=20 ,pady=(10,0))
    sidebar_label_3.pack(side=TOP,padx=20 ,pady=(10,0))
    sidebar_label_4 = ctk.CTkLabel(sidebar_frame,text=str(round(getCurrentValue(df, investorStrategy,'perGain'),2)), font=ctk.CTkFont(size=16))
    #sidebar_label_4.grid(row=4, column=0, padx=20)
    sidebar_label_4.pack(side=TOP,padx=20)
    sidebar_label_5 = ctk.CTkLabel(sidebar_frame, text='Mean Portfolio Value ($):', font=ctk.CTkFont(size=16,weight='bold'))
    #sidebar_label_5.grid(row=5, column=0, padx=20,pady=(10,0))
    sidebar_label_5.pack(side=TOP,padx=20,pady=(10,0))
    sidebar_label_6 = ctk.CTkLabel(sidebar_frame,text=str(round(getCurrentValue(df, investorStrategy, 'MPV'),2)), font=ctk.CTkFont(size=16))
    #sidebar_label_6.grid(row=6, column=0, padx=20)
    sidebar_label_6.pack(side=TOP,padx=20)

    sidebar_button = ctk.CTkButton(sidebar_frame, text="Info", command=openNewWindow)
    # sidebar_button.grid(row=12, column=0, padx=20, pady=(10,10))
    sidebar_button.pack(side=BOTTOM, padx=20, pady=(15, 15))

    scaling_optionmenu = ctk.CTkOptionMenu(sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                           command=change_scaling_event)
    # scaling_optionmenu.grid(row=11, column=0, padx=20, pady=(0, 10))
    scaling_optionmenu.pack(side=BOTTOM, padx=20, pady=(0, 10))
    scaling_label = ctk.CTkLabel(sidebar_frame, text="UI Scaling:", anchor="w")
    # scaling_label.grid(row=10, column=0, padx=20, pady=(10, 0))
    scaling_label.pack(side=BOTTOM, padx=20, pady=(10, 0))

    appearance_mode_optionmenu = ctk.CTkOptionMenu(sidebar_frame, values=["Light", "Dark", "System"], command=change_appearance_mode_event)
    # appearance_mode_optionmenu.grid(row=9, column=0, padx=20, pady=(0, 10))
    appearance_mode_optionmenu.pack(side=BOTTOM, padx=20, pady=(0, 0))
    appearance_mode_label = ctk.CTkLabel(sidebar_frame, text="Appearance Mode:", anchor="w")
    #appearance_mode_label.grid(row=8, column=0, padx=20, pady=(10, 0))
    appearance_mode_label.pack(side=BOTTOM,padx=20, pady=(5, 0))

    #switch_var = ctk.StringVar(value="on")

    switch_1 = ctk.CTkSwitch(sidebar_frame,text="Color", variable=switch_var, command=color_graph, onvalue="on", offvalue="off")

    switch_1.pack(side=BOTTOM,padx=20, pady=(5, 0))

    option_label = ctk.CTkLabel(sidebar_frame, text='Options', font=ctk.CTkFont(size=16))
    option_label.pack(side=BOTTOM,padx=20, pady=(0,0))





######################Value on left side (Middle)################

    # create recommendation frame frame
    recommendation_frame = ctk.CTkFrame(root)
    recommendation_frame.grid(row=0, column=1,padx=(20, 20), pady=(10,5), sticky="nsew")


    if MoneyInvestedToday > 0:
        labelL1 = ctk.CTkLabel(recommendation_frame, text='Recommendation', font=ctk.CTkFont(size=22, weight="bold"))
        #labelL1.grid(row=0, column=1, padx=10, pady=10, sticky="")
        labelL1.pack(side=TOP, padx=10, pady=(10,5))
        labelL2 = ctk.CTkLabel(recommendation_frame, text='Buy', font=ctk.CTkFont(size=20, weight='bold'), text_color='green')
        #labelL2.grid(row=1, column=1, pady=10, padx=20, sticky="n")
        labelL2.pack(side=TOP,pady=(0,10), padx=20)
        decision = 'Buy'
    elif MoneyInvestedToday < 0:
        labelL1 = ctk.CTkLabel(recommendation_frame, text='Recommendation', font=ctk.CTkFont(size=20, weight="bold"))
        # labelL1.grid(row=0, column=1, padx=10, pady=10, sticky="")
        labelL1.pack(side=TOP, padx=10, pady=(10,5))
        labelL2 = ctk.CTkLabel(recommendation_frame, text='Sell', font=ctk.CTkFont(size=20, weight='bold'), text_color='red')
        # labelL2.grid(row=1, column=1, pady=10, padx=20, sticky="n")
        labelL2.pack(side=TOP, pady=(0, 10), padx=20)
        decision = 'sell'
    else:
        labelL1 = ctk.CTkLabel(recommendation_frame, text='Recommendation', font=ctk.CTkFont(size=20, weight="bold"))
        # labelL1.grid(row=0, column=1, padx=10, pady=10, sticky="")
        labelL1.pack(side=TOP, padx=10, pady=(10,5))
        labelL2 = ctk.CTkLabel(recommendation_frame, text='Hold', font=ctk.CTkFont(size=20, weight='bold'))
        # labelL2.grid(row=1, column=1, pady=10, padx=20, sticky="n")
        labelL2.pack(side=TOP, pady=(0, 10), padx=20)
        decision = 'hold'

        # list to save last descions
    lastDecisions = []
    lastDecisions.append(decision)

    # print last 3 decsions depending on the amounts of available taken decisions
    labelL3 = ctk.CTkLabel(recommendation_frame, text='Last Recommendations', font=ctk.CTkFont(size=18, weight="bold"))
    #labelL3.grid(row=2, column=1, pady=10, padx=20, sticky="n")
    labelL3.pack(side=TOP,pady=(15,5), padx=20)

    if len(lastDecisions) <= 1:
        labelL4 = ctk.CTkLabel(recommendation_frame, text='No Last Recommendation', font=ctk.CTkFont(size=16))
        #labelL4.grid(row=3, column=1, pady=10, padx=20, sticky="n")
        labelL4.pack(side=TOP, pady=10, padx=20)
    elif len(lastDecisions) == 2:
        labelL4 = ctk.CTkLabel(recommendation_frame, text=lastDecisions[0], font=ctk.CTkFont(size=16))
        #labelL4.grid(row=3, column=1, pady=10, padx=20, sticky="n")
        labelL4.pack(side=TOP, pady=10, padx=20)
    elif len(lastDecisions) == 3:
        labelL4 = ctk.CTkLabel(recommendation_frame, text=lastDecisions[0], font=ctk.CTkFont(size=16))
        #labelL4.grid(row=3, column=1, pady=10, padx=20, sticky="n")
        labelL4.pack(side=TOP, pady=10, padx=20)
        labelL5 = ctk.CTkLabel(recommendation_frame, text=lastDecisions[1], font=ctk.CTkFont(size=16))
        #labelL5.grid(row=4, column=1, pady=10, padx=20, sticky="n")
        labelL5.pack(side=TOP, pady=10, padx=20)
    elif len(lastDecisions) > 3:
        labelL4 = ctk.CTkLabel(recommendation_frame, text=lastDecisions[-4], font=ctk.CTkFont(size=16))
        #labelL4.grid(row=3, column=1, pady=10, padx=20, sticky="n")
        labelL4.pack(side=TOP, pady=10, padx=20)
        labelL5 = ctk.CTkLabel(recommendation_frame, text=lastDecisions[-3], font=ctk.CTkFont(size=16))
        #labelL5.grid(row=4, column=1, pady=10, padx=20, sticky="n")
        labelL5.pack(side=TOP, pady=10, padx=20)
        labelL6 = ctk.CTkLabel(recommendation_frame, text=lastDecisions[-2], font=ctk.CTkFont(size=16))
        #labelL6.grid(row=5, column=1, pady=10, padx=20, sticky="n")
        labelL4.pack(side=TOP, pady=10, padx=20)

    # create value frame
    values_frame = ctk.CTkFrame(root)
    values_frame.grid(row=1, column=1, rowspan=2, padx=(20, 20), pady=(5, 10), sticky="nsew")

    labelL1 = ctk.CTkLabel(values_frame, text='Invested Money ($):', font=ctk.CTkFont(size=16, weight="bold"))
    #labelL1.grid(row=1, column=1, pady=10, padx=20, sticky="n")
    labelL1.pack(side=TOP, pady=(10,0), padx=20)
    labelL2 = ctk.CTkLabel(values_frame,text= str(round(getCurrentValue(df, investorStrategy, 'MoneyInvested'),2)),font=ctk.CTkFont(size=16))
    #labelL2.grid(row=2, column=1, pady=10, padx=20, sticky="n")
    labelL2.pack(side=TOP, pady=(0,10), padx=20)
    labelL3 = ctk.CTkLabel(values_frame, text='Money Not Invested ($):', font=ctk.CTkFont(size=16,weight="bold"))
    #labelL3.grid(row=3, column=1, pady=10, padx=20, sticky="n")
    labelL3.pack(side=TOP, pady=(10,0), padx=20)
    labelL4 = ctk.CTkLabel(values_frame, text=str(round(getCurrentValue(df, investorStrategy, 'MoneyNotInvested'),2)), font=ctk.CTkFont(size=16))
    #labelL4.grid(row=4, column=1, pady=10, padx=20, sticky="n")
    labelL4.pack(side=TOP, pady=(0,10), padx=20)
    labelL5 = ctk.CTkLabel(values_frame, text='Standard Deviation ($):', font=ctk.CTkFont(size=16,weight="bold"))
    #labelL5.grid(row=5, column=1, pady=10, padx=20, sticky="n")
    labelL5.pack(side=TOP, pady=(10,0), padx=20)
    labelL6 = ctk.CTkLabel(values_frame, text=str(round(getCurrentValue(df, investorStrategy, 'StdPV'),2)), font=ctk.CTkFont(size=16))
    #labelL6.grid(row=6, column=1, pady=10, padx=20, sticky="n")
    labelL6.pack(side=TOP, pady=(0,10), padx=20)
    labelL7 = ctk.CTkLabel(values_frame, text='Max Gain per Day (%):', font=ctk.CTkFont(size=16,weight="bold"))
    #labelL7.grid(row=7, column=1, pady=10, padx=20, sticky="n")
    labelL7.pack(side=TOP, pady=(10,0), padx=20)
    labelL8 = ctk.CTkLabel(values_frame, text=str(round(getCurrentValue(df, investorStrategy, 'maxGain'),2)), font=ctk.CTkFont(size=16))
    #labelL8.grid(row=8, column=1, pady=10, padx=20, sticky="n")
    labelL8.pack(side=TOP, pady=(0,10), padx=20)






################Plot left side#######################




    #########Graph########




    # Preparing data for candlesticks: Necessary data for candlesticks: "Date","Open","High","Low","Close","Volume"
    data_n=NewDayValue[["Date","Open","High","Low","Close","Volume"]]

    #maybe we should add this to the beinning (Data Prperation)
    # Adding the date column to the dataframe
    lst=[]
    for i in range(0,len(data_n)):
        new_date = datetime.strptime(data_n.iloc[i,0], '%Y-%m-%d %H:%M:%S')
        lst.append(new_date.date())

    data_n["New Date"]=lst
    data_n=data_n.drop(["Date"], axis=1)
    data_n=data_n.set_index(pd.DatetimeIndex(data_n["New Date"]))
    data_n.index.name = "Date"
    data_n=data_n.drop(["New Date"], axis=1)
    #-------------------------------------------


    # Plot candlesticks



    tabview = ctk.CTkTabview(root)
    tabview.grid(row=0, column=2, columnspan=2, padx=(20, 20), pady=(20, 0), sticky="nsew")
    tabview.add("Test")
    tabview.add("1M")
    tabview.add("6M")
    tabview.add("1Y")
    tabview.tab("Test")
    tabview.tab("1M")
    tabview.tab("6M")
    tabview.tab("1Y")

    # frame inside tab "test"
    label_test = ctk.CTkFrame(tabview.tab("Test"), width=800, height=500)
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
    line = FigureCanvasTkAgg(show_graph(NewDayValue, data_n, len(data_n)), label_test)
    line.get_tk_widget().place(width=800, height=450)
    # Navigation bar
    toolbarFrame = Frame(master=label_test)
    toolbarFrame.place(x=0, y=450)
    toolbar = NavigationToolbar2Tk(line, toolbarFrame)

    # 1M
    line = FigureCanvasTkAgg(show_graph(NewDayValue, data_n, 19), label_1m)
    line.get_tk_widget().place(width=800, height=450)
    # Navigation bar
    toolbarFrame = Frame(master=label_1m)
    toolbarFrame.place(x=0, y=450)
    toolbar = NavigationToolbar2Tk(line, toolbarFrame)

    # 6M
    line = FigureCanvasTkAgg(show_graph(NewDayValue, data_n, 123), label_6m)
    line.get_tk_widget().place(width=800, height=450)
    # Navigation bar
    toolbarFrame = Frame(master=label_6m)
    toolbarFrame.place(x=0, y=450)
    toolbar = NavigationToolbar2Tk(line, toolbarFrame)

    # 1Y
    line = FigureCanvasTkAgg(show_graph(NewDayValue, data_n, 250), label_1y)
    line.get_tk_widget().place(width=800, height=450)
    # Navigation bar
    toolbarFrame = Frame(master=label_1y)
    toolbarFrame.place(x=0, y=450)
    toolbar = NavigationToolbar2Tk(line, toolbarFrame)

    root.after(10000, update)

update()
root.mainloop()