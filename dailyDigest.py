from app.frontend.googleStorageAPI import readBlobDf
import plotly.graph_objects as go

df = readBlobDf()
df.reset_index(inplace=True)
df.set_index('Date', inplace=True)
df = df[['MPV', 'investorStrategy', 'TotalPortfolioValue']]

latestDate = df.index.unique()[-1]
importantStrategies = ['random_25_1_2022', 'idle_25_1_2022', 'ca_25_1_2022', 'bah_25_1_2022', 'bilstmWindowRobMMT1T2Legacy_25_1_2023']
df = df[df['investorStrategy'].isin(importantStrategies)]
df = df[df.index == latestDate]

fig = go.Figure()
fig.add_trace(go.Bar(x=df['investorStrategy'], y=df['MPV']-10000))
fig.update_layout(title='Mean Portfolio Value Comparison (10,000$ offset)')
fig.show()

fig = go.Figure()
fig.add_trace(go.Bar(x=df['investorStrategy'], y=df['TotalPortfolioValue']-10000))
fig.update_layout(title='Total Portfolio Value Comparison (10,000$ offset)')
fig.show()