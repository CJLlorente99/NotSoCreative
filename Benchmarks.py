import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt
import numpy as np
import random



#Buy and sells random, random is uniform distributed. Buys random amount from available Money or sells random amount from portfolio
def buySellRandom(df,startmoney):
   startmoney = startmoney
   avialablemoney = startmoney
   pfvList= []
   moneyList=[]
   assets = 0.0
   currentMoney=startmoney

    #iterate over Data Frame
   for i in range (0, df.shape[0]): 
      #create random float value between -1 and 1 negative=sell poitiv=buy 1 equals buy assets fromm all available Money -1 equals sell assets
      rand = random.uniform(-1,1)
      index = df.index[i]
      openToday=df.at[index,'Open'].item()
      
      #Buy  
      if rand > 0 and avialablemoney!=0: 
        
         assets = (rand*avialablemoney)/openToday + assets
         pfv = assets*openToday
         avialablemoney = avialablemoney - (avialablemoney*rand)
         currentMoney = pfv + avialablemoney
       
      #sell  
      elif rand < 0 and avialablemoney!=0:

            #check if there already bought assets
            if assets==0.0:
                 currentMoney = currentMoney
                 assets = 0.0
                 pfv = 0 
                 

            #sells assets 
            else: 
                 avialablemoney = -rand*pfv +avialablemoney     
                 pfv= pfv-((-rand) * pfv)
                 assets = pfv/openToday
                 currentMoney = pfv+ avialablemoney
               
      moneyList.append(currentMoney)
      pfvList.append(pfv)   
   
   #df['Random_Money'] = moneyList   
   #df['Ranomd_PFV']= pfvList
   return moneyList,pfvList

#Worts Ivestement action --> Buy Asstes if Asset Value decreases  sell if Asset Value increses
def worstInvestmentAction(df, startmoney):
    assets = 0
    currentMoney = startmoney
    invested = 0
    pfvList =[]
    moneyList =[]
    
    #iterate over dataframe
    for i in range(0,df.shape[0]):
      
      #get value of current day
      index = df.index[i]
      openToday = df.at[index,'Open']
      
      #check for last day
      if i == df.shape[0]-1:
         if assets ==0:
          currentMoney=currentMoney
          pfv=0
         else:
          currentMoney= assets*openToday 
          pfv = currentMoney
        
         pfvList.append(pfv)
         moneyList.append(currentMoney)
         break   

      #get calue of next day
      shift = df.index[i+1]
      openTomorrow =df.at[shift,'Open']
     
   
      #check to buy 
      if openToday > openTomorrow: 
            #investing if possible
            if invested == 0:      
             assets = currentMoney/openToday
             pfv = assets*openToday
             currentMoney = pfv
             invested = 1
            # hold the investmennt investing wehn money is available
            elif invested == 1:
             pfv = assets*openToday
             currentMoney = pfv
             invested = 1

   
      elif openToday < openTomorrow:

            #kind of hold for no asstets/cant sell wehn no assets available
            if assets==0:
                 currentMoney = currentMoney
                 assets = 0
                 pfv = 0 
                 invested = 0

            #sells all assets 
            else:      
                 currentMoney = openToday*assets
                 assets = 0
                 pfv = 0
                 invested = 0
           

      moneyList.append(currentMoney)
      pfvList.append(pfv)
 
    #df['WIA_PortfolioValue']= pfvList
    #df['WIA_Money']= moneyList        
    return moneyList,pfvList

#Benchmark for Cost Average Effect you can define the amount of many to spend each day
def CostAverage(df,startmoney, dailywindow):
    
    availableMoney= startmoney
    pfvList= []
    moneyList=[]
    assets = 0

    #iterate over Data Frame
    for i in range(0,df.shape[0]):
        index = df.index[i]
        openToday=df.at[index,'Open']
        
        if availableMoney==0:
            money=assets*openToday
            pfv = money
        else:    
            assets = dailywindow/openToday + assets
            availableMoney = availableMoney-dailywindow
            money = assets*openToday + availableMoney
            pfv = assets*openToday
        
        pfvList.append(pfv)
        moneyList.append(money)

    #df['CostAverage_Money']=moneyList
    #df['CostAverage_PFV']=pfvList   
    return moneyList,pfvList

#Buys alle possible asstes on the first day
def BuyAndHold(df, startmoney):
    assets = startmoney/df['Open'].iat[0]
    pfvList= []
    moneyList=[]

    for i in range(0,df.shape[0]):
        index = df.index[i]
        money=df.at[index,'Open']*assets
        pfv= money
        moneyList.append(money)
        pfvList.append(pfv)
    
    #df['HoldBuy_Money']= moneyList
    #df['HoldBuy_PFV']= pfvList
    return moneyList,pfvList       

#no investign at all
def doNothing(df, startmoney):
 moneyList = [startmoney]*df.shape[0]
 pfvList =[0]*df.shape[0]
 return moneyList, pfvList

#Best possible investement action buys only when prices goes up and sells all money if the prices go down  
def bestInvestmentAction(df, startmoney):
    assets = 0
    currentMoney = startmoney
    invested = 0
    pfvList =[]
    moneyList =[]
    
    #iterate over dataframe
    for i in range(0,df.shape[0]):
      
      #get value of current day
      index = df.index[i]
      openToday = df.at[index,'Open']
      
      #check for last day
      if i == df.shape[0]-1:
         if assets ==0:
          currentMoney=currentMoney
          pfv=0
         else:
          currentMoney= assets*openToday 
          pfv = currentMoney
        
         pfvList.append(pfv)
         moneyList.append(currentMoney)
         break   

      #get calue of next day
      shift = df.index[i+1]
      openTomorrow =df.at[shift,'Open']
     
   
      #check to buy 
      if openToday < openTomorrow: 
            #investing if possible
            if invested == 0:      
             assets = currentMoney/openToday
             pfv = assets*openToday
             currentMoney = pfv
             invested = 1
            # hold the investmennt investing wehn money is available
            elif invested == 1:
             pfv = assets*openToday
             currentMoney = pfv
             invested = 1

   
      elif openToday > openTomorrow:

            #kind of hold for no asstets/cant sell wehn no assets available
            if assets==0:
                 currentMoney = currentMoney
                 assets = 0
                 pfv = 0 
                 invested = 0

            #sells all assets 
            else:      
                 currentMoney = openToday*assets
                 assets = 0
                 pfv = 0
                 invested = 0
           

      moneyList.append(currentMoney)
      pfvList.append(pfv)
 
    #df['BIA_PortfolioValue']= pfvList
    #df['BIA_Money']= moneyList        
    return moneyList , pfvList


#main returns alle benchmarks as lists in following order: BIA; DoNothing; BuyAndHolf; CostAverage; Random; WIA
def main(open,money,costAverageWindow):
 open = open
 money = money
 costAverageWindow = costAverageWindow   

 BIA_money,BIA_pfv = bestInvestmentAction(open,money)
 doNothing_money,doNothing_pfv = doNothing(open,money)
 BAH_money,BAH_pfv = BuyAndHold(open,money) 
 CA_money,CA_pfv = CostAverage(open,money,costAverageWindow) 
 Random_money,Random_pfv = buySellRandom(open,money)
 WIA_money, WIA_pfv =worstInvestmentAction(open,money)
 
 return  BIA_money,BIA_pfv, doNothing_money,doNothing_pfv, BAH_money,BAH_pfv, CA_money,CA_pfv,Random_money,Random_pfv,WIA_money, WIA_pfv


if __name__ == '__main__':
    main() 