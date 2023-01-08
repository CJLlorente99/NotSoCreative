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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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

#df = pd.read_csv('myData.csv', delimiter=',')
df = pd.read_csv('myDataVolume.csv')
df


investorStrategy = 'ca'

#Not actual open price but start value of new day
NewDayValue = df.iloc[::12, :]



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


######################middle################


#line = FigureCanvasTkAgg(figure, frame_m)
#line.get_tk_widget().pack()



# Plot in the middle

# Preparing data for candlesticks: Necessary data for candlesticks: "Date","Open","High","Low","Close","Volume"
data_n=NewDayValue[["Date","Open","High","Low","Close","Volume"]]

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

frame_m = ctk.CTkFrame(root)
frame_m.grid(row=0, column=1)

colors = mpf.make_marketcolors(up="#00ff00", down="#ff0000", wick="inherit", edge="inherit", volume="in")
mpf_style= mpf.make_mpf_style(base_mpf_style="nightclouds", marketcolors=colors)

#figure=mpf.plot(data_n,type="candle", style = mpf_style, volume=False)


fig, axlist = mpf.plot(data_n,type='candle',mav=(20),volume=False,
                       style=mpf_style,returnfig=True)
line = FigureCanvasTkAgg(fig, frame_m)
line.get_tk_widget().pack()

 # Copied from the web
#s = mpf.make_mpf_style(base_mpf_style='charles', rc={'font.size': 6})
#fig = mpf.figure(figsize=(10, 7), style=s) # pass in the self defined style to the whole canvas
#line = FigureCanvasTkAgg(fig, frame_m)
#line.get_tk_widget().pack()
#ax = fig.add_subplot(2,1,1) # main candle stick chart subplot, you can also pass in the self defined style here only for this subplot
#av = fig.add_subplot(2,1,2, sharex=ax)  # volume chart subplot
#mpf.plot(data_n, type='candle', ax=ax, volume=av)




####################right side############
frame_r = ctk.CTkFrame(root)
frame_r.grid(row=0, column=2)

MoneyInvestedToday = getCurrentValue(df, investorStrategy, 'MoneyInvestedToday')

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