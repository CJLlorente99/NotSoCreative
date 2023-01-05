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


#Main Window
root = tk.Tk()
root.geometry('800x600')
root.title("Stock Market Prediction Engine")


###############Data Preperation##############
df = pd.read_csv('myData.csv')
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



############## START FRONTEND ###########

#left side
frame_l = tk.Frame(root)
frame_l.grid(row=0, column=0)

labelL1 = tk.Label(frame_l, text='PortfolioValue')
labelL1.pack()
labelL2 = tk.Label(frame_l, text=str(round(getCurrentValue(df, investorStrategy, 'TotalPortfolioValue'),2)))
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
test = NewDayValue[['Date','Open']].groupby('Date').sum()
test.plot(kind='line', legend=True, ax=ax)
ax.set_title('Open S&P 500')




#right side
frame_r = tk.Frame(root)
frame_r.grid(row=0, column=2)


labelR1 = tk.Label(frame_r, text='Money Invested')
labelR1.pack()
labelR2 = tk.Label(frame_r, text=str(round(getCurrentValue(df, investorStrategy, 'MoneyInvested'),2)))
labelR2.pack()


frame_b = tk.Frame(root)
frame_b.grid(row=1)






#def update (data):








root.mainloop()
