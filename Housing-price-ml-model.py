import shap
import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

# Load and prepare the California housing data
housing = fetch_california_housing(as_frame=True)
df_california = housing.frame
X = df_california.drop(columns='MedHouseVal')
Y = df_california[['MedHouseVal']]

# Train the model
model = RandomForestRegressor(n_jobs=-1)
model.fit(X, Y.values.ravel())

# Save the model
with open('housing_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)

X_sample = X.sample(1000, random_state=42)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# Save these instead
pickle.dump(shap_values, open('shap_values.pkl', 'wb'))
pickle.dump(X_sample, open('X_features.pkl', 'wb'))

