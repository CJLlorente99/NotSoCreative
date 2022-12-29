from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
import shap
import pandas as pd
import plotly.graph_objects as go

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

X_train, X_test, y_train, y_test = train_test_split(X_unsupervised, y_unsupervised, test_size=0.25, random_state=0)
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)

sorted_idx = rf.feature_importances_.argsort()

fig = go.Figure()
fig.add_trace(go.Bar(name="Feature Importance", x=X_unsupervised.columns[sorted_idx][-20:], y=rf.feature_importances_[sorted_idx][-20:], marker_color="red"))
fig.update_layout(title="Random Forest feature importance (Return)")
fig.show()

# Feature based on permutation (Return)

perm_importance = permutation_importance(rf, X_test, y_test)

sorted_idx = perm_importance.importances_mean.argsort()

fig = go.Figure()
fig.add_trace(go.Bar(name="Feature Importance", x=X_unsupervised.columns[sorted_idx][-20:], y=perm_importance.importances_mean[sorted_idx][-20:], marker_color="red"))
fig.update_layout(title="Permutation importance (Return)")
fig.show()

# Feature based with SHAP values (Return)

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)
# shap.summary_plot(shap_values, X_test, plot_type="bar")
shap.summary_plot(shap_values, X_test)

# Feature based on random forest (Class)

X_train, X_test, y_train, y_test = train_test_split(X_supervised, y_supervised, test_size=0.25, random_state=0)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

sorted_idx = rf.feature_importances_.argsort()

fig = go.Figure()
fig.add_trace(go.Bar(name="Feature Importance", x=X_supervised.columns[sorted_idx][-20:], y=rf.feature_importances_[sorted_idx][-20:], marker_color="red"))
fig.update_layout(title="Random Forest feature importance (Class)")
fig.show()

# Feature based on permutation (Class)

perm_importance = permutation_importance(rf, X_test, y_test)

sorted_idx = perm_importance.importances_mean.argsort()

fig = go.Figure()
fig.add_trace(go.Bar(name="Feature Importance", x=X_supervised.columns[sorted_idx][-20:], y=perm_importance.importances_mean[sorted_idx][-20:], marker_color="red"))
fig.update_layout(title="Permutation importance (Class)")
fig.show()

# Feature based with SHAP values (Class)

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)
# shap.summary_plot(shap_values, X_test, plot_type="bar")
shap.summary_plot(shap_values, X_test)

