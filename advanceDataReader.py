import pandas as pd
import plotly.graph_objects as go
import numpy as np

fileName = './images/advancedData.csv'

df = pd.read_csv(fileName, index_col='date')

df.drop(['nExperiment', 'nextStockValue', 'actualStockValue', 'nextStockValue.1', 'actualStockValue.1',
		 'actualStockValue.2', 'actualStockValue.3', 'actualStockValue.4'], axis=1, inplace=True)
# df.rename({'moneyToInvestRFClass.1': 'moneyToInvestRFClass2', 'investedMoneyRFClass.1': 'investedMoneyRFClass2',
# 		   'nonInvestedMoneyRFClass.1': 'nonInvestedMoneyRFClass2'}, axis=1, inplace=True)
# df.rename({'moneyToInvestXGB.1': 'moneyToInvestXGBWindow', 'investedMoneyXGB.1': 'investedMoneyXGBWindow',
# 		   'nonInvestedMoneyXGB.1': 'nonInvestedMoneyXGBWindow'}, axis=1, inplace=True)
# df.rename({'moneyToInvestXGB.2': 'moneyToInvestXGBReduced', 'investedMoneyXGB.2': 'investedMoneyXGBReduced',
# 		   'nonInvestedMoneyXGB.2': 'nonInvestedMoneyXGBReduced'}, axis=1, inplace=True)

df.rename({'nExperiment.1': 'nExperiment'}, axis=1, inplace=True)
df = df[df['nExperiment'] != 'nExperiment']
df = df[df.columns].astype(float)
df['nExperiment'] = df['nExperiment'].astype(int)

infoPerExp = []

for nExp in range(df['nExperiment'].max()+1):
	aux = df[df['nExperiment'] == nExp].drop(['nExperiment'], axis=1)
	infoPerExp.append(aux)
print(f'Number of experiments run: {len(infoPerExp)}')

# List of strategy tags
strategiesList = []

for col in df.columns:
	if col.startswith('moneyToInvest'):
		if col[len('moneyToInvest'):] not in strategiesList:
			strategiesList.append(col[len('moneyToInvest'):])
print(f'Strategies: {strategiesList}')

# Calculate MPV per experiment

print()
print(f'Mean Portfolio Value Calculation')

mpvs = []

for nExp in range(len(infoPerExp)):
	aux = {}
	for strategy in strategiesList:
		investedTag = 'investedMoney' + strategy
		nonInvestedTag = 'nonInvestedMoney' + strategy
		aux[strategy] = (infoPerExp[nExp][investedTag] + infoPerExp[nExp][nonInvestedTag]).mean()
		mpvString = '{:.2f}'.format(aux[strategy])
		print(f'Experiment number: {nExp}. Strategy: {strategy} yields a MPV {mpvString}$')
	mpvs.append(aux)

# Calculate MMPV

print()
print(f'Mean of Mean Portfolio Value Calculation')

mmpvs = pd.DataFrame(index=[0])
stdMpvs = pd.DataFrame(index=[0])

for strategy in strategiesList:
	mmpv = np.array([])
	for mpv in range(len(mpvs)):
		mmpv = np.append(mmpv,mpvs[mpv][strategy])
	mmpvs[strategy] = mmpv.mean()
	stdMpvs[strategy] = mmpv.std()
	mmpvString = '{:.2f}'.format(mmpvs[strategy].values[0])
	stdString = '{:.2f}'.format(stdMpvs[strategy].values[0])
	print(f'Strategy: {strategy} yields a MMPV {mmpvString}$ with Std {stdString}$')

fig = go.Figure()
fig.add_trace(go.Bar(x=mmpvs.columns, y=mmpvs.values[0]-10000,
					 error_y=dict(type='data', array=stdMpvs.values[0], visible=True)))
fig.update_layout(title_text='Summary of MMPV', hovermode="x unified")
fig.update_xaxes(ticks='inside', showgrid=True, griddash='dash', categoryorder='total descending')
fig.show()

# Calculate absolute gain

print()
print(f'Percentage Gains Calculation')

gains = []

for nExp in range(len(infoPerExp)):
	aux = {}
	for strategy in strategiesList:
		investedTag = 'investedMoney' + strategy
		nonInvestedTag = 'nonInvestedMoney' + strategy
		finalPortfolioValue = infoPerExp[nExp][investedTag].values[-1] + infoPerExp[nExp][nonInvestedTag].values[-1]
		initialPortfolioValue = infoPerExp[nExp][investedTag].values[0] + infoPerExp[nExp][nonInvestedTag].values[0]
		aux[strategy] = (finalPortfolioValue/initialPortfolioValue - 1) * 100
		perGainString = '{:.2f}'.format(aux[strategy])
		print(f'Experiment number: {nExp}. Strategy: {strategy} yields a Percentage Gain of {perGainString}%')
	gains.append(aux)

# Calculate Mean Percentage Gain

print()
print(f'Mean of Percentage Gain')

mPerGain = pd.DataFrame(index=[0])
stdPerGain = pd.DataFrame(index=[0])

for strategy in strategiesList:
	mPerGains = np.array([])
	for gain in range(len(gains)):
		mPerGains = np.append(mPerGains,gains[gain][strategy])
	mPerGain[strategy] = mPerGains.mean()
	stdPerGain[strategy] = mPerGains.std()
	mPerGainString = '{:.2f}'.format(mPerGain[strategy].values[0])
	stdPerGainString = '{:.2f}'.format(stdPerGain[strategy].values[0])
	print(f'Strategy: {strategy} yields a Mean Percentage Gain of {mPerGainString}% with Std {stdPerGainString}%')

fig = go.Figure()
fig.add_trace(go.Bar(x=mPerGain.columns, y=mPerGain.values[0],
					 error_y=dict(type='data', array=stdPerGain.values[0], visible=True)))
fig.update_layout(title_text='Summary of Percentage Gain', hovermode="x unified")
fig.update_xaxes(ticks='inside', showgrid=True, griddash='dash', categoryorder='total descending')
fig.show()


