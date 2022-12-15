import pandas as pd
from decisionFunctionNN import NNDecisionFunction
import numpy as np
import plotly.graph_objects as go

df = pd.read_csv("../data/optimizationTrainingSet.csv", index_col=["n"])

# nnRSI = NNDecisionFunction(1)
# nnRSI.train_model(df["rsiResults"], df["output"])
#
# nnRSI.summary()
#
# x = np.arange(0, 100, 1)
# y = nnRSI.predict(x)
#
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=x, y=y[:, 0]))
# fig.show()
#
# nnBB = NNDecisionFunction(1)
# nnBB.train_model(df["bbResults"], df["output"])
#
# nnBB.summary()
#
# x = np.arange(-2, 2, 0.05)
# y = nnBB.predict(x)
#
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=x, y=y[:, 0]))
# fig.show()

nnTwo = NNDecisionFunction(2)
nnTwo.summary()
inputs = [df["rsiResults"].to_numpy(), df["bbResults"].to_numpy()]
inputs = np.asarray(inputs)
nnTwo.train_model(inputs.transpose(), df["output"].to_numpy())

inputs = np.mgrid[0:100, -2:2:0.5]
results = pd.DataFrame()
for i in range(inputs.shape[1]):
		inp = inputs[:, i]
		aux = {"RSI": inp[0], "BB": inp[1], "y": nnTwo.predict(inp.transpose())[:, 0]}
		results = pd.concat([results, pd.DataFrame(aux)])

fig = go.Figure()
fig.add_trace(go.Mesh3d(x=results.RSI, y=results.BB, z=results.y))
fig.update_layout(scene = dict(
                    xaxis_title='RSI',
                    yaxis_title='BB',
                    zaxis_title='% Buy/Sell'))
fig.show()

