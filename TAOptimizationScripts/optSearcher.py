import pandas as pd
import numpy as np
import plotly.express as px


def searchTopResults(filename, indicatorName,  nResults):
    portfolioName = "meanPortfolioValue" + indicatorName
    percentageName = "percentage" + indicatorName

    df = pd.read_csv(filename)

    print(df.columns)

    df.drop(df[df.nOpt == "nOpt"].index, inplace=True)
    df.drop(["initDate", "lastDate", "experiment"], axis=1, inplace=True)
    df = df.astype("float64")

    # print(df)

    dfPercentage = df.drop([portfolioName], axis=1)

    # print(dfPercentage)

    dfPortfolio = df.drop([percentageName], axis=1)

    # print(dfPortfolio)

    dfPercentageMean = dfPercentage.groupby(["nOpt"]).mean()
    dfPercentageStd = dfPercentage.groupby(["nOpt"]).std()
    dfPortfolioMean = dfPortfolio.groupby(["nOpt"]).mean()
    dfPortfolioStd = dfPortfolio.groupby(["nOpt"]).std()

    dfPercentageMean = dfPercentageMean[percentageName].to_frame(name=percentageName).reset_index().drop(["nOpt"], axis=1)
    dfPercentageStd = dfPercentageStd[percentageName].to_frame(name=percentageName).reset_index().drop(["nOpt"], axis=1)
    dfPortfolioMean = dfPortfolioMean[portfolioName].to_frame(name=portfolioName).reset_index().drop(["nOpt"], axis=1)
    dfPortfolioStd = dfPortfolioStd[portfolioName].to_frame(name=portfolioName).reset_index().drop(["nOpt"], axis=1)

    topPercentageMean = dfPercentageMean.sort_values([percentageName], ascending=False)[:nResults]
    topPercentageMean.rename(columns={"percentage" + indicatorName: "percentage" + indicatorName + "Mean"},
                             inplace=True)

    topPercentageStd = dfPercentageStd.iloc[
        dfPercentageMean.sort_values([percentageName], ascending=False)[:nResults].index]
    topPercentageStd.rename(columns={"percentage" + indicatorName: "percentage" + indicatorName + "Std"}, inplace=True)

    topPercentage = pd.concat([topPercentageMean, topPercentageStd], axis=1)

    topPortfolioMean = dfPortfolioMean.sort_values([portfolioName], ascending=False)[:nResults]
    topPortfolioMean.rename(columns={"meanPortfolioValue" + indicatorName: "meanPortfolio" + indicatorName + "Mean"},
                             inplace=True)

    topPortfolioStd = dfPortfolioStd.iloc[
        dfPortfolioMean.sort_values([portfolioName], ascending=False)[:nResults].index]
    topPortfolioStd.rename(columns={"meanPortfolioValue" + indicatorName: "meanPortfolio" + indicatorName + "Std"},
                          inplace=True)

    topPortfolio = pd.concat([topPortfolioMean, topPortfolioStd], axis=1)

    return topPercentage, topPortfolio


def searchTopParams(filename, indicatorName, locArray):
    df = pd.read_csv(filename)

    if indicatorName[:4] == "MACD":
        macdIndicatorType = indicatorName[4:]
        indicatorName = "MACD"

        if macdIndicatorType == "Grad":
            lastColumnIndicator = "grad"
        elif macdIndicatorType == "Zero":
            lastColumnIndicator = "grad_crossZero"
        elif macdIndicatorType == "Signal":
            lastColumnIndicator = "grad_crossSignal"

    df.drop(df[df[" MaxBuy"] == " MaxBuy"].index, inplace=True)
    df.drop("0:" + indicatorName, axis=1, inplace=True)

    if indicatorName == "MACD":
        df.drop(df[df[" type"] != lastColumnIndicator].index, inplace=True)
        df.drop(" type", axis=1, inplace=True)

    df.reset_index(inplace=True, drop=True)
    df = df.astype("float64")

    return df.loc[locArray]


def plotComparatives(dfPercentage, dfPortfolio, indicatorName):
    allColumnNames = dfPercentage.columns.to_numpy()
    columnNames = dfPercentage.columns.to_numpy()
    columnNames = columnNames[columnNames != "percentage" + indicatorName + "Mean"]
    columnNames = columnNames[columnNames != "percentage" + indicatorName + "Std"]

    for column in columnNames:
        fig = px.scatter(dfPercentage, x=column, y="percentage" + indicatorName + "Mean",
                         title="percentage" + indicatorName + "Mean" + " vs" + column, hover_data=allColumnNames)
        fig.show()

    # allColumnNames = dfPortfolio.columns.to_numpy()
    # columnNames = dfPortfolio.columns.to_numpy()
    # columnNames = columnNames[columnNames != "meanPortfolio" + indicatorName + "Mean"]
    # columnNames = columnNames[columnNames != "meanPortfolio" + indicatorName + "Std"]
    #
    # for column in columnNames:
    #     fig = px.scatter(dfPortfolio, x=column, y="meanPortfolio" + indicatorName + "Mean",
    #                      title="meanPortfolio" + indicatorName + "Mean" + " vs" + column, hover_data=allColumnNames)
    #     fig.show()


if __name__ == "__main__":
    fileNameData = "../data/optimizationMACD_2022_12_05_22_50_35.csv"
    fileNameParams = "../data/optimizationMACD_2022_12_05_22_50_35.txt"

    # indicatorName = "RSI"
    # indicatorName = "BB"
    # indicatorName = "MACDGrad"
    # indicatorName = "MACDZero"
    indicatorName = "MACDSignal"

    topPercentage, topPortfolio = searchTopResults(fileNameData, indicatorName, 500)

    topPercentageParams = searchTopParams(fileNameParams, indicatorName, topPercentage.index)
    topPortfolioParams = searchTopParams(fileNameParams, indicatorName, topPortfolio.index)

    topPercentage = pd.concat([topPercentage, topPercentageParams], axis=1)
    topPortfolio = pd.concat([topPortfolio, topPortfolioParams], axis=1)

    # plotComparatives(topPercentage, topPortfolio, indicatorName)

    print(topPercentage[:10])
    print(topPortfolio[:10])
