import pandas as pd
from decisionFunctionNN import NNDecisionFunction
import numpy as np
import plotly.graph_objects as go

df = pd.read_csv("../data/optimizationTrainingSet.csv", index_col=["n"])

nnRSI = NNDecisionFunction(2)
inputs = [df["rsiResults"].iloc[1:-1].to_numpy(), df["rsiResults"].iloc[0:-2].to_numpy()]
inputs = np.asmatrix(inputs)
nnRSI.train_model(inputs.transpose(), df["output"][2:])

# nnRSI.save("nnRSI")
nnRSI.summary()

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
                    zaxis_title='Predicition'))
fig.show()

nnBB = NNDecisionFunction(2)
inputs = [df["bbResults"].iloc[1:].to_numpy(), df["bbResults"].iloc[0:-1].to_numpy()]
inputs = np.asarray(inputs)
nnBB.train_model(inputs.transpose(), df["output"][1:])

# nnBB.save("nnBB")
nnBB.summary()

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
                    zaxis_title='% Buy/Sell'))
fig.show()

# nnTwo = NNDecisionFunction(4)
# nnTwo.summary()
# inputs = [df["bbResults"].iloc[1:].to_numpy(), df["bbResults"].iloc[0:-1].to_numpy(), df["rsiResults"].iloc[1:].to_numpy(), df["bbResults"].iloc[0:-1].to_numpy()]
# inputs = np.asarray(inputs)
# nnTwo.train_model(inputs.transpose(), df["output"].iloc[1:].to_numpy())

# inputs = np.mgrid[-2:2:0.05, -2:2:0.05, 0:100, 0:100]
# results = pd.DataFrame()
# for i in range(inputs.shape[1]):
# 	inp = inputs[:, i]
# 	aux = {"BBToday": inp[0], "BBTomorrow": inp[1], "RSIToday": inp[2], "RSITomorrow": inp[3], "y": nnTwo.predict(inp.transpose())[:, 0]}
# 	results = pd.concat([results, pd.DataFrame(aux)])

