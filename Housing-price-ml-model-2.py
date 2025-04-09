import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import plotly.express as px
from sklearn.datasets import fetch_california_housing

# Load California housing data
housing = fetch_california_housing(as_frame=True)
df_california = housing.frame
X = df_california.drop(columns='MedHouseVal')
y = df_california['MedHouseVal']

# Train and save the model
model = RandomForestRegressor(n_jobs=-1, random_state=42)
model.fit(X, y)
pickle.dump(model, open("housing_price_model.pkl", "wb"))

# Save a sample X for future EDA use
X_sample = X.sample(1000, random_state=42)
pickle.dump(X_sample, open("X_features.pkl", "wb"))

# 1. Univariate Histogram Plot
df_california.hist(bins=40, figsize=(10, 6), color='skyblue', edgecolor='black')
plt.tight_layout()
plt.savefig("eda_univariate_hist.png")
plt.close()

# 2. Correlation Matrix Heatmap
corr_mat = df_california.corr().round(2)
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
plt.figure(figsize=(8, 6))
sns.heatmap(corr_mat, mask=mask, cmap='plasma', vmin=-1, vmax=1, annot=True, square=True, linewidths=0.5)
plt.tight_layout()
plt.savefig("eda_correlation_matrix.png")
plt.close()

# 3. Bivariate Pairplot with Income Group Hue

tlist = ['MedInc', 'AveRooms', 'HouseAge', 'Latitude', 'MedHouseVal', 'Population']
df_pairplot = df_california[tlist].copy()
df_pairplot["IncomeGroup"] = pd.qcut(df_pairplot["MedInc"], q=4, labels=["Low", "Medium", "High", "Very High"])

grid = sns.PairGrid(df_pairplot, vars=tlist, hue="IncomeGroup", corner=True, diag_sharey=False)
grid.map_lower(sns.scatterplot, alpha=0.5)
grid.map_diag(sns.kdeplot, fill=True)
grid.add_legend()
plt.tight_layout()
plt.savefig("eda_pairplot_matrix.png", dpi=150)

print("✅ Model and EDA files saved:")
print("- housing_price_model.pkl")
print("- X_features.pkl")
print("- eda_univariate_hist.png")
print("- eda_correlation_matrix.png")
print("- eda_pairplot_matrix.png")
