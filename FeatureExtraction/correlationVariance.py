import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_regression
import plotly.graph_objects as go
from plotly.subplots import make_subplots

df = pd.read_csv("featureSelectionDataset.csv", index_col="Date")
df.dropna(inplace=True)

print(f"Original shape {df.shape}")
df = df.iloc[-500:]
print(df)
print(f"Studied shape {df.shape}")

X_supervised = df[df.columns.drop(["Return", "ReturnBefore", "Adj Close", "log(Open)", "Class", "LogReturn"])][:-1]
y_supervised = df["Class"][1:]

X_unsupervised = df[df.columns.drop(["Return", "ReturnBefore", "Adj Close", "log(Open)", "Class", "LogReturn"])][:-1]
y_unsupervised = df["Return"][1:]

# Straight correlation study (Return)

dfCorr = pd.DataFrame()
i = 1
for key in X_unsupervised.columns:
	print(f"{i}/{len(df.columns)}")
	i += 1

	pearsonCorr = pearsonr(X_unsupervised[key][:-1].to_numpy(), y_unsupervised[1:].to_numpy())[0]
	spearmanR = spearmanr(X_unsupervised[key][:-1].to_numpy(), y_unsupervised[1:].to_numpy())[0]
	kendallT = kendalltau(X_unsupervised[key][:-1].to_numpy(), y_unsupervised[1:].to_numpy())[0]

	aux = pd.DataFrame({key: [pearsonCorr, spearmanR, kendallT]}, index=["pearsonCorr", "SpearmanRho", "KendallTau"])

	dfCorr = pd.concat([dfCorr, aux], axis=1)

dfCorr = dfCorr.transpose()

fig = make_subplots(rows=3, cols=1)

fig.add_trace(go.Bar(name="PearsonCorrelation", y=dfCorr["pearsonCorr"].sort_values(ascending=False).values[:10], x=dfCorr["pearsonCorr"].sort_values(ascending=False).index), row=1, col=1)
fig.add_trace(go.Bar(name="SpearmanRho", y=dfCorr["SpearmanRho"].sort_values(ascending=False).values[:10], x=dfCorr["SpearmanRho"].sort_values(ascending=False).index), row=2, col=1)
fig.add_trace(go.Bar(name="KendallTau", y=dfCorr["KendallTau"].sort_values(ascending=False).values[:10], x=dfCorr["KendallTau"].sort_values(ascending=False).index), row=3, col=1)
fig.update_layout(title="Correlations for Unsupervised Learning (Return Prediction)")
fig.show()

# Straight correlation study (Return)

dfCorr = pd.DataFrame()
i = 1
for key in X_supervised.columns:
	print(f"{i}/{len(df.columns)}")
	i += 1

	pearsonCorr = pearsonr(X_supervised[key][:-1].to_numpy(), y_supervised[1:].to_numpy())[0]
	spearmanR = spearmanr(X_supervised[key][:-1].to_numpy(), y_supervised[1:].to_numpy())[0]
	kendallT = kendalltau(X_supervised[key][:-1].to_numpy(), y_supervised[1:].to_numpy())[0]

	aux = pd.DataFrame({key: [pearsonCorr, spearmanR, kendallT]}, index=["pearsonCorr", "SpearmanRho", "KendallTau"])

	dfCorr = pd.concat([dfCorr, aux], axis=1)

dfCorr = dfCorr.transpose()

fig = make_subplots(rows=3, cols=1)

fig.add_trace(go.Bar(name="PearsonCorrelation", y=dfCorr["pearsonCorr"].sort_values(ascending=False).values[:10], x=dfCorr["pearsonCorr"].sort_values(ascending=False).index), row=1, col=1)
fig.add_trace(go.Bar(name="SpearmanRho", y=dfCorr["SpearmanRho"].sort_values(ascending=False).values[:10], x=dfCorr["SpearmanRho"].sort_values(ascending=False).index), row=2, col=1)
fig.add_trace(go.Bar(name="KendallTau", y=dfCorr["KendallTau"].sort_values(ascending=False).values[:10], x=dfCorr["KendallTau"].sort_values(ascending=False).index), row=3, col=1)
fig.update_layout(title="Correlations for Supervised Learning (Class Prediction)")
fig.show()

# Variance study

scaler = StandardScaler()
dfScaled = pd.DataFrame(scaler.fit_transform(X_unsupervised), columns=X_unsupervised.columns)

selector = VarianceThreshold(1)
selector.fit(dfScaled)
print("Variables with variance greater than 1 after normalization")
print(dfScaled.columns[selector.get_support()])

# SelectKBest (intended for Supervised learning)

nFeatures = 10
selector = SelectKBest(mutual_info_regression, k=nFeatures)
selector.fit(X_supervised, y_supervised)
print("SelectKBest Result")
print(X_supervised.columns[selector.get_support()])