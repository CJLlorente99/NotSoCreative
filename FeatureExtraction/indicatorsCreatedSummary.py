import json

import numpy as np
import plotly.graph_objects as go

file = "data.json"

with open(file) as f:
	jsonData = json.load(f)

entries = []
for entry in jsonData:
	entries.append(json.loads(entry))

indicatorNames = []
parameterDict = {}
dimDict = {}

for entry in entries:
	if entry["indicatorName"] not in indicatorNames:
		indicatorNames.append(entry["indicatorName"])
		parameterDict[entry["indicatorName"]] = {}
		dimDict[entry["indicatorName"]] = 0
		for param in entry["parameters"]:
			if param not in parameterDict[entry["indicatorName"]]:
				parameterDict[entry["indicatorName"]][param] = []

for entry in entries:
	dim = 0
	for param in entry["parameters"]:
		dim += 1
		parameterDict[entry["indicatorName"]][param].append(entry["parameters"][param])
	dimDict[entry["indicatorName"]] = dim

auxDict = {}
for indicator in parameterDict:
	aux = []
	for param in parameterDict[indicator]:
		aux.append(np.array(parameterDict[indicator][param]).transpose())
	auxDict[indicator] = aux

for indicator in auxDict:
	keys = list(parameterDict[indicator].keys())
	if dimDict[indicator] == 1:
		fig = go.Figure()
		fig.add_trace(go.Scatter(x=auxDict[indicator][0], y=np.zeros(len(auxDict[indicator][0])), mode="markers"))
		fig.update_layout(title=indicator + " distribution", xaxis_title=keys[0])
		fig.show()
	elif dimDict[indicator] == 2:
		fig = go.Figure()
		fig.add_trace(go.Scatter(x=auxDict[indicator][0], y=auxDict[indicator][1], mode="markers"))
		fig.update_layout(title=indicator + " distribution", xaxis_title=keys[0], yaxis_title=keys[1])
		fig.show()
	elif dimDict[indicator] == 3:
		fig = go.Figure()
		fig.add_trace(go.Scatter3d(x=auxDict[indicator][0], y=auxDict[indicator][1], z=auxDict[indicator][2], mode="markers"))
		fig.update_layout(title=indicator + " distribution", scene=dict(xaxis_title=keys[0], yaxis_title=keys[1], zaxis_title=keys[2]))
		fig.show()