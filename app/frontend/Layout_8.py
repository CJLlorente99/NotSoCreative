import csv
import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import pandas_datareader as web
import datetime as dt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import *
import plotly.graph_objects as go
import mplfinance as mpf
from datetime import datetime



ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

# Main Window
root = ctk.CTk()
root.geometry('800x600')
root.title("Stock Market Prediction Engine")

# configure the grid
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=3)
root.columnconfigure(2, weight=1)

root.rowconfigure(0, weight=1)
root.rowconfigure(1, weight=1)

#root.columnconfigure(0, weight=1)

###############Data Preperation##############

def update ():
    ps = pd.read_csv("NewData.csv")
    root.after(30000,update)
    return ps


#df = pd.read_csv('myData.csv', delimiter=',')
df = update()

# Look at the data
#df.iloc[:20,:]

investorStrategy = 'ca'

#Not actual open price but start value of new day
NewDayValue = df.iloc[11::22, :]
NewDayValue


#get the newest selected value of a selected strategy (kind=selected Value) .tail(number of investment strategies)
def getCurrentValue(df, strategy, kind):
    #newest date
    newDate = df['Date'].max()
    newestData = df.loc[df['Date'] == newDate]
    strategyData=newestData.loc[newestData['investorStrategy'] == strategy]
    return strategyData.iloc[0][kind]

def getDecision(MoneyInvestedToday):
   if MoneyInvestedToday>0:
       decision='Buy'
   elif MoneyInvestedToday<0:
       decision ='Sell'
   else:
       decision ='Hold'
   return decision


def openNewWindow():
    # Toplevel object which will
    # be treated as a new window
    newWindow = ctk.CTkToplevel(root)

    # sets the title of the
    # Toplevel widget
    newWindow.title("Information Button")

    # sets the geometry of toplevel
    newWindow.geometry("200x200")

    # A Label widget to show in toplevel
    Label(newWindow,
          text="Information").pack()


#def Show_Graph():

#padx = 10, pady = 10

############## START FRONTEND ###########

###############left side##################
frame_l = ctk.CTkFrame(root)
frame_l.grid(row=0, column=0)

labelL1 = ctk.CTkLabel(frame_l, text='Portfolio Value ($)', font=ctk.CTkFont(size=20, weight="bold"))
labelL1.pack()
labelL2 = ctk.CTkLabel(frame_l, text=str(round(getCurrentValue(df, investorStrategy, 'TotalPortfolioValue'),2)), font=ctk.CTkFont(size=20))
labelL2.pack()
labelL3 = ctk.CTkLabel(frame_l, text='Money Invested ($)', font=ctk.CTkFont(size=20, weight="bold"))
labelL3.pack()
labelL4 = ctk.CTkLabel(frame_l, text=str(round(getCurrentValue(df, investorStrategy, 'MoneyInvested'),2)), font=ctk.CTkFont(size=20))
labelL4.pack()
labelL5 = ctk.CTkLabel(frame_l, text='Money Invested ($)', font=ctk.CTkFont(size=20, weight="bold"))
labelL5.pack()
labelL6 = ctk.CTkLabel(frame_l, text=str(round(getCurrentValue(df, investorStrategy, 'MoneyNotInvested'),2)), font=ctk.CTkFont(size=20))
labelL6.pack()

# options = ["bia", "wia", "ca", "idle", "random", "bah", "rsi", "bb"]

mb =  Menubutton (frame_l, text="Select Category", relief=RAISED )
mb.menu  =  Menu ( mb, tearoff = 0 )
mb["menu"]  =  mb.menu

bia = IntVar()
wia = IntVar()
ca = IntVar()
idle = IntVar()
random = IntVar()
bah = IntVar()
rsi = IntVar()
bb = IntVar()


mb.menu.add_checkbutton ( label="bia", variable=bia)
mb.menu.add_checkbutton ( label="wia", variable=wia)
mb.menu.add_checkbutton ( label="ca", variable=ca)
mb.menu.add_checkbutton ( label="idle", variable=idle)
mb.menu.add_checkbutton ( label="random", variable=random)
mb.menu.add_checkbutton ( label="bah", variable=bah)
mb.menu.add_checkbutton ( label="rsi", variable=rsi)
mb.menu.add_checkbutton ( label="bb", variable=bb)

mb.pack()

######################middle################

#line = FigureCanvasTkAgg(figure, frame_m)
#line.get_tk_widget().pack()

# MIDDLE PLOT

# Preparing data for candlesticks: Necessary data for candlesticks: "Date","Open","High","Low","Close","Volume"
data_n=NewDayValue[["Date","Open","High","Low","Close","Volume"]]

data_n
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

#frame_m = ctk.CTkFrame(root, width=800, height=500)
#frame_m.grid(row=0, column=1)
#tabview
tabview = ctk.CTkTabview(root)
tabview.grid(row=0, column=1)
tabview.add("Test")
tabview.add("1M")
tabview.add("6M")
tabview.add("1Y")
tabview.tab("Test")
tabview.tab("1M")
tabview.tab("6M")
tabview.tab("1Y")

#frame inside tab "test"
label_test = ctk.CTkFrame(tabview.tab("Test"), width=800, height=500)
label_test.grid(row=0, column=1)

#frame inside tab "1M"
label_1m = ctk.CTkFrame(tabview.tab("1M"), width=800, height=500)
label_1m.grid(row=0, column=1)

#frame inside tab "6M"
label_6m = ctk.CTkFrame(tabview.tab("6M"), width=800, height=500)
label_6m.grid(row=0, column=1)

#frame inside tab "1Y"
label_1y = ctk.CTkFrame(tabview.tab("1Y"), width=800, height=500)
label_1y.grid(row=0, column=1)

def show_graph(NewDayValue, data_n, i):
    # Plot candlesticks
    colors = mpf.make_marketcolors(up="#00ff00", down="#ff0000", wick="inherit", edge="inherit", volume="in")
    mpf_style = mpf.make_mpf_style(base_mpf_style="nightclouds", marketcolors=colors)

    # Create list filled with buy/hold/sell actions
    action = []
    for (columnName, columnData) in NewDayValue["MoneyInvested"].items():
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
                          style=mpf_style, returnfig=True, addplot=apd)
    return fig_1



#test
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


####################right side############
frame_r = ctk.CTkFrame(root)
frame_r.grid(row=0, column=2)

MoneyInvestedToday = getCurrentValue(df, investorStrategy, 'MoneyInvestedToday')

btnInfo = ctk.CTkButton(frame_r,
                        text="Info",
                        command=openNewWindow)
btnInfo.pack()

# updbtn = ctk.CTkButton(fra)


#Decisions based on invested Money
if MoneyInvestedToday > 0:
    labelL1 = ctk.CTkLabel(frame_r, text='Recommendation', font=ctk.CTkFont(size=20, weight="bold"))
    labelL1.pack()
    labelL2 = ctk.CTkLabel(frame_r, text='Buy', font=ctk.CTkFont(size=20, weight='bold' ), text_color='green')
    labelL2.pack()
    decision='Buy'
elif MoneyInvestedToday < 0:
    labelL1 = ctk.CTkLabel(frame_r, text='Recommendation', font=ctk.CTkFont(size=20, weight="bold"))
    labelL1.pack()
    labelL2 = ctk.CTkLabel(frame_r, text='Sell', font=ctk.CTkFont(size=20, weight='bold'), text_color='red')
    labelL2.pack()
    decision='sell'
else:
    labelL1 = ctk.CTkLabel(frame_r, text='Recommendation', font=ctk.CTkFont(size=20, weight="bold"))
    labelL1.pack()
    labelL2 = ctk.CTkLabel(frame_r, text='Hold', font=ctk.CTkFont(size=20, weight='bold'))
    labelL2.pack()
    decision='hold'

#list to save last descions
lastDecisions=[]
lastDecisions.append(decision)

#print last 3 decsions depending on the amounts of available taken decisions
labelL3 = ctk.CTkLabel(frame_r, text='Last Recommendations', font=ctk.CTkFont(size=20, weight="bold"))
labelL3.pack()

if len(lastDecisions) <=1:
    labelL4 = ctk.CTkLabel(frame_r, text='No Last Recommendation', font=ctk.CTkFont(size=20))
    labelL4.pack()
elif len(lastDecisions) ==2:
    labelL4 = ctk.CTkLabel(frame_r, text=lastDecisions[0], font=ctk.CTkFont(size=20))
    labelL4.pack()
elif len(lastDecisions) == 3:
    label4 = ctk.CTkLabel(frame_r, text=lastDecisions[0], font=ctk.CTkFont(size=20))
    labelL4.pack()
    labelL5 = ctk.CTkLabel(frame_r, text=lastDecisions[1], font=ctk.CTkFont(size=20))
    labelL5.pack()
elif len(lastDecisions) > 3:
    label4 = ctk.CTkLabel(frame_r, text=lastDecisions[-4], font=ctk.CTkFont(size=20))
    labelL4.pack()
    labelL5 = ctk.CTkLabel(frame_r, text=lastDecisions[-3], font=ctk.CTkFont(size=20))
    labelL5.pack()
    labelL6 = ctk.CTkLabel(frame_r, text=lastDecisions[-2], font=ctk.CTkFont(size=20))
    labelL6.pack()




#############Bottom#############

frame_b = ctk.CTkFrame(root)
frame_b.grid(row=1, columnspan=2)

# Bottom side

total_rows = 7
total_columns = 3

# Names of the columns
list = ["Metrics", "MPV", "STD", "Max/Min (Total Assets)", "% & absolute gain", "Max. gain per day", "Mix. gain per day"]

# take the data
lst = [(list[0], 'Our Strategy', 'WIA' ),
       (list[1], '', ''),
       (list[2], '', ''),
       (list[3], '', ''),
       (list[4], '', ''),
       (list[5], '', ''),
       (list[6], '', '')]


# code for creating table
for i in range(total_rows):
    for j in range(total_columns):
        e = Entry(frame_b, width=20, fg='black',
                       font=('Arial', 16, 'bold'))
        e.grid(row=i, column=j)
        e.insert(END, lst[i][j])

root.mainloop()