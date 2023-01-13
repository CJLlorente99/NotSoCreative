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
		fig.add_trace(go.Scatter(x=auxDict[indicator][0], y=np.zeros(len(auxDict[indicator][0])), mode="markers",
					  marker=dict(size=20)))
		fig.update_layout(title=indicator.upper() + " Features Distribution", xaxis=dict(title=keys[0], gridcolor='black'), paper_bgcolor = "rgba(0,0,0,0)",
                  font=dict(size=40))
		fig.write_image(indicator + " distribution.png", scale=1, width=2880, height=1800)
		# fig.show()
	elif dimDict[indicator] == 2:
		fig = go.Figure()
		fig.add_trace(go.Scatter(x=auxDict[indicator][0], y=auxDict[indicator][1], mode="markers",
					  marker=dict(size=30)))
		fig.update_layout(title=indicator.upper() + " Features Distribution", paper_bgcolor = "rgba(0,0,0,0)",
                  font=dict(size=40), 	xaxis=dict(title=keys[0], gridwidth=10, gridcolor='#2D3A44', linewidth=10, linecolor='#2D3A44', zeroline=False),
										yaxis=dict(title=keys[1], gridwidth=10, gridcolor='#2D3A44', linewidth=10, linecolor='#2D3A44', zeroline=False),
						  plot_bgcolor='black')
		fig.write_image(indicator + " distribution.png", scale=1, width=2880, height=1800)
		# fig.show()
	elif dimDict[indicator] == 3:
		fig = go.Figure()
		fig.add_trace(go.Scatter3d(x=auxDict[indicator][0], y=auxDict[indicator][1], z=auxDict[indicator][2],
								   mode="markers", marker=dict(size=20)))
		camera = dict(
			up=dict(x=0, y=0, z=1),
			center=dict(x=0, y=0, z=0),
			eye=dict(x=1.5, y=1.5, z=1.5)
		)
		title = {'text': indicator.upper() + " Features Distribution",
				 'y': 0.9,
				 'x': 0.18,
				 'xanchor': 'center',
				 'yanchor': 'top',
				 'font': dict(size=70)}
		fig.update_layout(scene=dict(xaxis=dict(title=keys[0], title_font_size=40, tickfont_size=20, gridwidth=10, gridcolor='#2D3A44', linewidth=10, linecolor='#2D3A44', zerolinewidth=10, zerolinecolor='#2D2A44', backgroundcolor="black"),
									 yaxis=dict(title=keys[1], title_font_size=40, tickfont_size=20, gridwidth=10, gridcolor='#2D3A44', linewidth=10, linecolor='#2D3A44', zerolinewidth=10, zerolinecolor='#2D2A44', backgroundcolor="black"),
									 zaxis=dict(title=keys[2], title_font_size=40, tickfont_size=20, gridwidth=10, gridcolor='#2D3A44', linewidth=10, linecolor='#2D3A44', zerolinewidth=10, zerolinecolor='#2D2A44', backgroundcolor="black")),
						  paper_bgcolor = "rgba(0,0,0,0)", scene_camera=camera, title=title)
		fig.write_image(indicator + " distribution.png", scale=1, width=3000, height=3000)
		# fig.show()
