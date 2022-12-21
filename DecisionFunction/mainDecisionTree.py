from datetime import datetime
import numpy as np
import pandas as pd
from decisionFunctionTree import DecisionFunctionTree
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

tag = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# Load data from .csv containing TA values + BIA signal
df = pd.read_csv("../data/optimizationTrainingSet.csv", index_col=["n"])
df["output"][df["output"] < 0] = 0

# Divide df
dfRSI = df.rsiResults
dfBB = df.bbResults
dfADI = df.adiResults
dfADX = df.adxResults
dfAroon = df.aroonResults
dfATR = df.atrResults
dfOBV = df.obvResults
dfStochRSI = df.stochasticRsi
dfOutput = df.output

"""
Decision tree based on 
"""
inputs = np.asarray([dfRSI.values[:-1], dfBB.values[:-1], dfADI.values[:-1], dfADX.values[:-1], dfAroon.values[:-1], dfATR.values[:-1], dfOBV.values[:-1], dfStochRSI.values[:-1]]).transpose()
outputs = dfOutput.values[1:]
X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.05, random_state=432, stratify=outputs)

decisionTree1 = DecisionFunctionTree(10, 6, "gini", ["RSI", "BB", "ADI", "ADX", "Aroon" ,"ATR", "OBV", "StochRSI"])
decisionTree1.train(X_train, y_train)
# decisionTree1.show()
decisionTree1.save("../data/dt")
y = decisionTree1.predict_test(X_test, y_test, True)

fig = go.Figure()
fig.add_trace(go.Scatter(name="Predicted", x=np.arange(len(y)), y=y))
fig.add_trace(go.Scatter(name="Real", x=np.arange(len(y)), y=y_test))
fig.update_layout(hovermode="x unified")
fig.show()
