import csv
import tkinter as tk
import customtkinter as ctk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from tkinter import Entry
from tkinter import END

#Main Window
root = tk.Tk()
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
df = pd.read_csv('myData.csv')
investorStrategy = 'ca'

#Not actual open price but start value of new day
NewDayValue = df.iloc[::12, :]
print(NewDayValue)

#get the newest selected value of a selected strategy (kind=selected Value) .tail(number of investment strategies)
def getCurrentValue(df, strategy, kind):

    # Strategy =
    # kind = selected value
    #newest date
    newDate = df['Date'].max()
    newestData = df.loc[df['Date'] == newDate]
    strategyData=newestData.loc[newestData['investorStrategy'] == strategy]
    return strategyData.iloc[0][kind]

#padx = 10, pady = 10

############## START FRONTEND ###########

#left side
frame_l = tk.Frame(root)
frame_l.grid(row=0, column=0)

labelL1 = tk.Label(frame_l, text='Portfolio Value', font=18)
labelL1.pack()
labelL2 = tk.Label(frame_l, text=str(round(getCurrentValue(df, investorStrategy, 'TotalPortfolioValue'),2)), font=18)
labelL2.pack()

#middle
frame_m = tk.Frame(root)
frame_m.grid(row=0, column=1)


#Plot in the middle
figure = plt.Figure(figsize=(3,2), dpi=100)
plt.style.use("bmh")
ax = figure.add_subplot(111)
line = FigureCanvasTkAgg(figure, frame_m)
line.get_tk_widget().pack()
open = NewDayValue[['Date','Open']].groupby('Date').sum()
close = NewDayValue[['Date','Close']].groupby('Date').sum()
open.plot(kind='line', legend=True, ax=ax)
close.plot(kind='line', legend=True, ax=ax)
ax.set_title('Open S&P 500')



#right side
frame_r = tk.Frame(root)
frame_r.grid(row=0, column=2)


labelR1 = tk.Label(frame_r, text='Money Invested', font=18)
labelR1.pack()

labelR2 = tk.Label(frame_r, text=str(round(getCurrentValue(df, investorStrategy, 'MoneyInvested'),2)), font=18)
labelR2.pack()


frame_b = tk.Frame(root)
frame_b.grid(row=1, columnspan=2)

# Bottom side

total_rows = 7
total_columns = 3

# Names of the columns
list=["Metrics", "MPV", "STD", "Max/Min (Total Assets)", "% & absolute gain", "Max. gain per day", "Mix. gain per day"]

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






#list
#for i in range(height): #Rows
    #for j in range(width): #Columns
        #b = frame_b
        #b.grid(row=i, column=j)




#def update (data):








root.mainloop()
