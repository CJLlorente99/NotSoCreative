from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SequentialFeatureSelector
import pandas as pd

df = pd.read_csv("featureSelectionDataset.csv", index_col="Date")
df.dropna(inplace=True)

print(f"Original shape {df.shape}")
df = df.iloc[-500:]
print(df)
print(f"Studied shape {df.shape}")

X_supervised = df[df.columns.drop(["Return", "Adj Close", "log(Open)", "Class", "LogReturn"])][:-1]
y_supervised = df["Class"][1:]

X_unsupervised = df[df.columns.drop(["Return", "Adj Close", "log(Open)", "Class", "LogReturn"])][:-1]
y_unsupervised = df["Return"][1:]

# Feature based on random forest (unsupervised == Return)

sfs_selector = SequentialFeatureSelector(estimator=RandomForestRegressor(), n_features_to_select=20, cv=10, direction="backward")
sfs_selector.fit(X_unsupervised, y_unsupervised)
print("Features with RandomForestRegressor + SFS-Backward (for Return)")
print(X_unsupervised.columns[sfs_selector.get_support()])

sfs_selector = SequentialFeatureSelector(estimator=RandomForestRegressor(), n_features_to_select=20, cv=10, direction="forward")
sfs_selector.fit(X_unsupervised, y_unsupervised)
print("Features with RandomForestRegressor + SFS-Forward (for Return)")
print(X_unsupervised.columns[sfs_selector.get_support()])

# Feature based on random forest (supervised == Class)

sfs_selector = SequentialFeatureSelector(estimator=RandomForestRegressor(), n_features_to_select=20, cv=10, direction="backward")
sfs_selector.fit(X_supervised, y_supervised)
print("Features with RandomForestRegressor + SFS-Backward (for Class)")
print(X_supervised.columns[sfs_selector.get_support()])

sfs_selector = SequentialFeatureSelector(estimator=RandomForestRegressor(), n_features_to_select=20, cv=10, direction="forward")
sfs_selector.fit(X_supervised, y_supervised)
print("Features with RandomForestRegressor + SFS-Forward (for Class)")
print(X_supervised.columns[sfs_selector.get_support()])