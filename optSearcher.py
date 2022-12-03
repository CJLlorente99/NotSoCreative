import pandas as pd
import numpy as np

# This script just searches for optimization configuration achieving better results

fileName = "data/optimizationRSI_2022_12_02_22_57_01.csv"
indicatorName = "RSI"

portfolioName = "meanPortfolioValue" + indicatorName
percentageName = "percentage" + indicatorName

df = pd.read_csv(fileName)

# print(df.columns)

df.drop(df[df.nOpt == "nOpt"].index, inplace=True)
df.drop(["initDate", "lastDate", "experiment"], axis=1, inplace=True)
df = df.astype("float64")

# print(df)

dfPercentage = df.drop([portfolioName], axis=1)

# print(dfPercentage)

dfPortfolio = df.drop([percentageName], axis=1)

# print(dfPortfolio)

dfPercentage = dfPercentage.groupby(["nOpt"]).mean()
dfPortfolio = dfPortfolio.groupby(["nOpt"]).mean()

dfPercentage = dfPercentage[percentageName].to_frame(name=percentageName).reset_index().drop(["nOpt"], axis=1)
dfPortfolio = dfPortfolio[portfolioName].to_frame(name=portfolioName).reset_index().drop(["nOpt"], axis=1)

print(dfPercentage.sort_values([percentageName], ascending=False)[:10])
print(dfPortfolio.sort_values([portfolioName], ascending=False)[:10])
