# Random forest regression, SHAP interpretation, and plotting
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import textwrap
from sklearn.model_selection import LeaveOneOut, cross_validate, KFold
from sklearn.metrics import r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV
import shap

# STEP 1: Load and prepare normalized input data
path = '.../normalized data.csv'
df = pd.read_csv(path)

# Specify the column positions of the response and explanatory variables
response_col_idx = ...
predictor_cols_idx = ...

# Prepare X and y
df.iloc[:, response_col_idx] = df.iloc[:, response_col_idx].replace([9999, np.nan], np.nan)
df.iloc[:, predictor_cols_idx] = df.iloc[:, predictor_cols_idx].replace([9999, np.nan], np.nan)
df_clean = df.dropna()

X = df_clean.iloc[:, predictor_cols_idx].astype(np.float128)
y = df_clean.iloc[:, response_col_idx].astype(np.float128)

# STEP 2: Optimize random forest hyperparameters
model = RandomForestRegressor(random_state=42)
param_spaces = {
    'n_estimators': (100, 1000),
    'max_depth': (10, 110),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 4),
    'max_features': ['sqrt', 'log2']
}

bayes_search = BayesSearchCV(estimator=model, search_spaces=param_spaces, n_iter=32, cv=5, n_jobs=-1, verbose=0, random_state=42, refit=False)
bayes_search.fit(X, y)
best_params = bayes_search.best_params_

y_name = df_clean.columns[response_col_idx]
print("Response variable:", y_name)
print("Valid data points:", len(y)) 

# STEP 3: Evaluate the model with five-fold cross-validation
model = RandomForestRegressor(n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    max_features=best_params['max_features'],
    random_state=42)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

scoring = {'R2': 'r2'}

cv_results = cross_validate(model, X, y, cv=kfold, scoring=scoring, return_train_score=False)

print('\n1) K-Fold Test:')
print(f"R^2: {np.mean(cv_results['test_R2']):.4f}")

# STEP 4: Evaluate the model with leave-one-out cross-validation
X = df_clean.iloc[:, predictor_cols_idx].to_numpy()
y = df_clean.iloc[:, response_col_idx].to_numpy()

loo = LeaveOneOut()
predictions = []
test_values = []

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions.extend(y_pred) 
    test_values.extend(y_test)

overall_r2 = r2_score(test_values, predictions)
print('\n2) Leave-One-Out Test:')
print(f'R^2: {overall_r2:.4f}')

# STEP 5: Calculate feature importance using default approach
X = df_clean.iloc[:, predictor_cols_idx]
model.fit(X, y)
model_importances = pd.Series(model.feature_importances_, index=X.columns)
model_importances /= model_importances.sum()

# STEP 6: Calculate feature importance using SHAP approach, and average feature importance based on default and SHAP approach
explainer = shap.TreeExplainer(model)
shap_values = explainer(X)

shap_importances = pd.Series(np.abs(shap_values.values).mean(axis=0), index=X.columns)
shap_importances /= shap_importances.sum()
average_importances = (model_importances + shap_importances) / 2

# STEP 7: Plot average feature importance
average_importances_sorted = average_importances.sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(11.5, 7))
bars = ax.barh(average_importances_sorted.index, average_importances_sorted.values,
               color="#B5C99A", edgecolor="none", height=0.5)
ax.set_xlabel("Average feature importance [0–1]", fontsize=13, labelpad=8)
ax.set_ylabel("")
x_max = average_importances_sorted.max()
ax.set_xlim(0, x_max * 1.38)
ax.spines["top"].set_visible(True)
ax.spines["right"].set_visible(True)
ax.grid(False)
ax.tick_params(axis="y", length=0, labelsize=13, pad=6)
ax.tick_params(axis="x", labelsize=13)
for rect, value in zip(bars, average_importances_sorted.values):
    ax.text(value + x_max * 0.01, rect.get_y() + rect.get_height() / 2,
            f"{value:.3f}", va="center", ha="left", fontsize=14)
plt.tight_layout(pad=1.1)
plt.show()
plt.close()

# STEP 8: Plot SHAP scatter plots
sns.set_style("white")
num_features = X.shape[1]
cols_per_row = min(4, num_features)
rows = int(np.ceil(num_features / cols_per_row))
fig, axes = plt.subplots(rows, cols_per_row, figsize=(6 * cols_per_row, 5 * rows),
                         sharey=True, squeeze=False)
axes = axes.flatten()

# Scale each feature's positive and negative SHAP values separately while preserving zero
normalized_shap_values = shap_values.values.copy()
for i in range(normalized_shap_values.shape[1]):
    positive_mask = normalized_shap_values[:, i] > 0
    negative_mask = normalized_shap_values[:, i] < 0
    if positive_mask.any():
        normalized_shap_values[positive_mask, i] /= normalized_shap_values[positive_mask, i].max()
    if negative_mask.any():
        normalized_shap_values[negative_mask, i] /= -normalized_shap_values[negative_mask, i].min()
# Color all other panels by Poverty; show the Poverty panel in gray
color_values = pd.to_numeric(X.iloc[:, 0]).to_numpy(dtype=np.float64)
colored_scatter = None

for i in range(num_features):
    ax = axes[i]
    x_values = pd.to_numeric(X.iloc[:, i]).to_numpy(dtype=np.float64)

    if i == 0:
        ax.scatter(x_values, normalized_shap_values[:, i], color='#4d4d4d', s=16)
    else:
        colored_scatter = ax.scatter(x_values, normalized_shap_values[:, i],
                                     c=color_values, cmap=shap.plots.colors.red_blue, s=16)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, zorder=0)
    ax.set_ylim(-1.05, 1.05)
    ax.set_facecolor('white')
    ax.grid(False)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.set_xlabel(textwrap.fill(X.columns[i], width=35), fontsize=22, labelpad=9)
    ax.set_ylabel('SHAP value' if i % cols_per_row == 0 else '', fontsize=22)
    ax.tick_params(axis='x', labelsize=18, width=1, direction='out')
    ax.tick_params(axis='y', labelsize=20, width=1, direction='out',
                   left=i % cols_per_row == 0, labelleft=i % cols_per_row == 0)

for i in range(num_features, len(axes)):
    axes[i].remove()

plt.tight_layout(rect=[0, 0, 0.9, 1], pad=1.1)
if colored_scatter is not None:
    last_ax_position = axes[num_features - 1].get_position()
    cbar_ax = fig.add_axes([last_ax_position.x1 + 0.01, last_ax_position.y0,
                            0.008, last_ax_position.height])
    cbar = fig.colorbar(colored_scatter, cax=cbar_ax)
    cbar.set_label(f"{X.columns[0]}\n(point color scale)", fontsize=22, labelpad=12)
    cbar.ax.tick_params(labelsize=15)

plt.show()
plt.close()