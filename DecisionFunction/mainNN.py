import pandas as pd
from decisionFunctionNN import NNDecisionFunction, NNDecisionFunctionClassification
import numpy as np
import plotly.graph_objects as go
from keras.utils import to_categorical

# Load data from .csv containing TA values + BIA signal
df = pd.read_csv("../data/optimizationTrainingSet.csv", index_col=["n"])

# Change the problem for only positive range (0-> sell; 1-> buy)
dfClassification = df
dfClassification["output"][dfClassification["output"] < 0] = 0

percentageEval = 0.1
lenTest = int(len(df) * 0.1)

# Divide dataset for training and evaluation
dfTrain = df[:-lenTest]
dfTest = df[-lenTest:]

dfTrainClassification = dfClassification[:-lenTest]
dfTestClassification = dfClassification[-lenTest:]

"""
Try for NN with 2 inputs and only RSI values
"""
# Create NN with 2 inputs (yesterday and the day before yesterday RSI value)
nnRSI = NNDecisionFunction(2)
inputs = [dfTrain["rsiResults"].iloc[1:-1].to_numpy(), dfTrain["rsiResults"].iloc[0:-2].to_numpy()]
inputsTrain = np.asmatrix(inputs).transpose()
nnRSI.train_model(inputsTrain, dfTrain["output"][2:])

# Save model
# nnRSI.save("nnRSI")
nnRSI.summary()

# Plot comparison with test
inputs = [dfTest["rsiResults"].iloc[1:-1].to_numpy(), dfTest["rsiResults"].iloc[0:-2].to_numpy()]
inputsTest = np.asmatrix(inputs).transpose()
y = nnRSI.predict(inputsTest)[:,0]

fig = go.Figure()
fig.add_trace(go.Scatter(name="Prediction", x=np.arange(lenTest), y=y, marker=dict(color="red")))
fig.add_trace(go.Scatter(name="Actual output", x=np.arange(lenTest), y=dfTest["output"][2:], marker=dict(color="green")))
fig.add_hline(y=0.5)
fig.update_layout(hovermode="x unified", title="Prediction based on RSI (non-classification)")

fig.show()

# Plot the decision function in 3D
inputs = np.mgrid[0:100, 0:100]
inp = np.asarray([inputs[0].flatten(), inputs[1].flatten()]).transpose()
results = pd.DataFrame()

for split in np.split(inp, len(inp)/200):
    aux = {"RSIYesterday": split[:, 0], "RSIBeforeYesterday": split[:, 1],  "y": nnRSI.predict(split)[:, 0]}
    results = pd.concat([results, pd.DataFrame(aux)])

fig = go.Figure()
fig.add_trace(go.Mesh3d(x=results.RSIYesterday, y=results.RSIBeforeYesterday, z=results.y))
fig.update_layout(scene=dict(
                    xaxis_title='RSIYesterday',
                    yaxis_title='RSIBeforeYesterday',
                    zaxis_title='Predicition'), title="Decision function of RSI (non-classification)")
fig.show()

"""
Try for NN with 2 inputs and only RSI values. Classification game
"""
# Create NN with 2 inputs (yesterday and the day before yesterday RSI value)
nnRSIClassification = NNDecisionFunctionClassification(2)
inputs = [dfTrain["rsiResults"].iloc[1:-1].to_numpy(), dfTrain["rsiResults"].iloc[0:-2].to_numpy()]
inputsTrain = np.asmatrix(inputs).transpose()
nnRSIClassification.train_model(inputsTrain, to_categorical(dfTrain["output"][2:]))

# Save model
# nnRSIClassification.save("nnRSI")
nnRSIClassification.summary()

# Plot comparison with test
inputs = [dfTest["rsiResults"].iloc[1:-1].to_numpy(), dfTest["rsiResults"].iloc[0:-2].to_numpy()]
inputsTest = np.asmatrix(inputs).transpose()
y = nnRSIClassification.predict(inputsTest)

fig = go.Figure()
fig.add_trace(go.Scatter(name="Prediction Sell", x=np.arange(lenTest), y=y[:, 0], marker=dict(color="red")))
fig.add_trace(go.Scatter(name="Prediction Buy", x=np.arange(lenTest), y=y[:, 1], marker=dict(color="orange")))
fig.add_trace(go.Scatter(name="Actual output", x=np.arange(lenTest), y=dfTest["output"][2:], marker=dict(color="green")))
fig.add_hline(y=0.5)
fig.update_layout(hovermode="x unified", title="Prediction based on RSI (classification)")

fig.show()

"""
Try for NN with 2 inputs and only BB values
"""
# Create NN with 2 inputs (yesterday and the day before yesterday BB value)
nnBB = NNDecisionFunction(2)
inputs = [dfTrain["bbResults"].iloc[1:].to_numpy(), dfTrain["bbResults"].iloc[0:-1].to_numpy()]
inputs = np.asarray(inputs)
nnBB.train_model(inputs.transpose(), dfTrain["output"][1:])

# Save model
# nnBB.save("nnBB")
nnBB.summary()

# Plot comparison with test
inputs = [dfTest["bbResults"].iloc[1:-1].to_numpy(), dfTest["bbResults"].iloc[0:-2].to_numpy()]
inputsTest = np.asmatrix(inputs).transpose()
y = nnBB.predict(inputsTest)[:,0]

fig = go.Figure()
fig.add_trace(go.Scatter(name="Prediction", x=np.arange(lenTest), y=y, marker=dict(color="red")))
fig.add_trace(go.Scatter(name="Actual output", x=np.arange(lenTest), y=dfTest["output"][2:], marker=dict(color="green")))
fig.add_hline(y=0.5)
fig.update_layout(hovermode="x unified", title="Prediction based on BB (non-classification)")

fig.show()

# Plot the decision function in 3D
inputs = np.mgrid[-3:3:0.05, -3:3:0.05]
results = pd.DataFrame()
for i in range(inputs.shape[1]):
	inp = inputs[:, i]
	aux = {"BBToday": inp[0], "BBYesterday": inp[1], "y": nnBB.predict(inp.transpose())[:, 0]}
	results = pd.concat([results, pd.DataFrame(aux)])

fig = go.Figure()
fig.add_trace(go.Mesh3d(x=results.BBToday, y=results.BBYesterday, z=results.y))
fig.update_layout(scene = dict(
                    xaxis_title='BBToday',
                    yaxis_title='BBYesterday',
                    zaxis_title='% Buy/Sell'), title="Decision function of BB (non-classification)")
fig.show()

"""
Try for NN with 2 inputs and only BB values. Classification
"""
# Create NN with 2 inputs (yesterday and the day before yesterday BB value)
nnBBClassification = NNDecisionFunctionClassification(2)
inputs = [dfTrain["bbResults"].iloc[1:].to_numpy(), dfTrain["bbResults"].iloc[0:-1].to_numpy()]
inputs = np.asarray(inputs).transpose()
nnBBClassification.train_model(inputs, to_categorical(dfTrain["output"][1:]))

# Save model
# nnBBClassification.save("nnBB")
nnBBClassification.summary()

# Plot comparison with test
inputs = [dfTest["bbResults"].iloc[1:-1].to_numpy(), dfTest["bbResults"].iloc[0:-2].to_numpy()]
inputsTest = np.asmatrix(inputs).transpose()
y = nnBBClassification.predict(inputsTest)

fig = go.Figure()
fig.add_trace(go.Scatter(name="Prediction Sell", x=np.arange(lenTest), y=y[:, 0], marker=dict(color="red")))
fig.add_trace(go.Scatter(name="Prediction Buy", x=np.arange(lenTest), y=y[:, 1], marker=dict(color="orange")))
fig.add_trace(go.Scatter(name="Actual output", x=np.arange(lenTest), y=dfTest["output"][2:], marker=dict(color="green")))
fig.add_hline(y=0.5)
fig.update_layout(hovermode="x unified", title="Prediction based on BB (classification)")

fig.show()

"""
Try for NN with 4 inputs, both BB and RSI values
"""
# Create NN with 4 inputs (yesterday and the day before yesterday BB value, and yesterday and the day before yesterday BB value)
nnTwo = NNDecisionFunction(4)
nnTwo.summary()
inputs = [dfTrain["bbResults"].iloc[1:].to_numpy(), dfTrain["bbResults"].iloc[0:-1].to_numpy(), dfTrain["rsiResults"].iloc[1:].to_numpy(), dfTrain["bbResults"].iloc[0:-1].to_numpy()]
inputs = np.asarray(inputs)
nnTwo.train_model(inputs.transpose(), dfTrain["output"].iloc[1:].to_numpy())

# Plot comparison with test
inputs = [dfTest["bbResults"].iloc[1:-1].to_numpy(), dfTest["bbResults"].iloc[0:-2].to_numpy(), dfTest["rsiResults"].iloc[1:-1].to_numpy(), dfTest["bbResults"].iloc[0:-2].to_numpy()]
inputsTest = np.asmatrix(inputs).transpose()
y = nnTwo.predict(inputsTest)[:,0]

fig = go.Figure()
fig.add_trace(go.Scatter(name="Prediction", x=np.arange(lenTest), y=y, marker=dict(color="red")))
fig.add_trace(go.Scatter(name="Actual output", x=np.arange(lenTest), y=dfTest["output"][2:], marker=dict(color="green")))
fig.add_hline(y=0.5)
fig.update_layout(hovermode="x unified", title="Prediction based on RSI and BB (non-classification)")

fig.show()

"""
Try for NN with 4 inputs, both BB and RSI values. Classification game
"""
# Create NN with 4 input as classification game
nnClassification = NNDecisionFunctionClassification(4)

inputs = [dfTrainClassification["bbResults"].iloc[1:-1].to_numpy(), dfTrainClassification["bbResults"].iloc[0:-2].to_numpy(), dfTrainClassification["rsiResults"].iloc[1:-1].to_numpy(), dfTrainClassification["bbResults"].iloc[0:-2].to_numpy()]
inputs = np.asarray(inputs).transpose()
nnClassification.train_model(inputs, to_categorical(dfTrainClassification["output"][2:]))

# Plot comparison with test
inputs = [dfTestClassification["bbResults"].iloc[1:-1].to_numpy(), dfTestClassification["bbResults"].iloc[0:-2].to_numpy(), dfTestClassification["rsiResults"].iloc[1:-1].to_numpy(), dfTestClassification["bbResults"].iloc[0:-2].to_numpy()]
inputsTest = np.asmatrix(inputs).transpose()
y = nnClassification.predict(inputsTest)

fig = go.Figure()
fig.add_trace(go.Scatter(name="Prediction Sell", x=np.arange(lenTest), y=y[:, 0], marker=dict(color="red")))
fig.add_trace(go.Scatter(name="Prediction Buy", x=np.arange(lenTest), y=y[:, 1], marker=dict(color="orange")))
fig.add_trace(go.Scatter(name="Actual output", x=np.arange(lenTest), y=dfTestClassification["output"][2:], marker=dict(color="green")))
fig.add_hline(y=0.5)
fig.update_layout(hovermode="x unified", title="Prediction based on RSI and BB (classification)")

fig.show()



