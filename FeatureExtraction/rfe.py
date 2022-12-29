from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import RFE
import pandas as pd

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

# Feature based on random forest (Return)

rfe_selector = RFE(estimator=RandomForestRegressor(), n_features_to_select=20, step=1)
rfe_selector.fit(X_unsupervised, y_unsupervised)
print("Features with RandomForestRegressor + RFE (Return)")
print(X_unsupervised.columns[rfe_selector.get_support()])

# Feature based on random forest (Class)

rfe_selector = RFE(estimator=RandomForestClassifier(), n_features_to_select=20, step=1)
rfe_selector.fit(X_supervised, y_supervised)
print("Features with RandomForestClassifier + RFE (Class)")
print(X_supervised.columns[rfe_selector.get_support()])