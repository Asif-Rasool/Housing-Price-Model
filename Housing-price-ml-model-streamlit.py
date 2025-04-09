import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import pickle
import gdown  # <-- NEW
from PIL import Image
import os
from sklearn.datasets import fetch_california_housing
import plotly.express as px

# --- Page config ---
st.set_page_config(layout="wide")

# --- Helper to download from Google Drive using gdown ---
def download_from_drive(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)

# --- Google Drive file IDs ---
MODEL_FILE_ID = "1ZtvYIBsA5orD6i0FtXRDO9RIffSt4S4t"
SHAP_FILE_ID = "15Z44AWYmSu8gmOUN6RpkZMxx81r32OPy"
X_FILE_ID = "1gdIsz-HuVba7xEqG9UHWroaePtXJ8ULT"

# --- File paths ---
MODEL_PATH = "housing_price_model.pkl"
SHAP_PATH = "shap_values.pkl"
X_PATH = "X_features.pkl"

# --- Download necessary files ---
download_from_drive(MODEL_FILE_ID, MODEL_PATH)
download_from_drive(SHAP_FILE_ID, SHAP_PATH)
download_from_drive(X_FILE_ID, X_PATH)

# --- Load model and SHAP data ---
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(SHAP_PATH, 'rb') as f:
    shap_values = pickle.load(f)
with open(X_PATH, 'rb') as f:
    shap_X = pickle.load(f)

# --- Load and display header image ---
image_path = os.path.join(os.path.dirname(__file__), "Housing.jpg")
image = Image.open(image_path)
col_img1, col_img2, col_img3 = st.columns([1, 1, 1])
with col_img2:
    st.image(image, use_container_width=True)

# --- App title and intro ---
st.markdown("""
# 🏠 House Price Prediction Model

This app predicts the **California Median House Value** using a machine learning model trained on housing data.
""")
st.write('---')

# --- About section ---
with st.expander("ℹ️ About"):
    st.markdown("""
**Python libraries used:**  
- `streamlit` for building the app  
- `pandas` for data manipulation  
- `matplotlib` for plotting SHAP charts  
- `shap` for model explainability  
- `scikit-learn` for machine learning  
- `pickle` for model serialization  
- `Pillow` for image display  
- `gdown` for downloading large files from Google Drive  

**Data source:** [California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices?resource=download)

---
**Asif Rasool**, Ph.D.  
Research Economist, Southeastern Louisiana University  
📍 1514 Martens Drive, Hammond, LA 70401  
📞 985-549-3831  
📧 [asif.rasool@southeastern.edu](mailto:asif.rasool@southeastern.edu)  
_Last updated: April 7, 2025_
""")

# --- Load full dataset for input slider ranges ---
housing = fetch_california_housing(as_frame=True)
df_california = housing.frame
X_full = df_california.drop(columns='MedHouseVal')

# --- Sidebar input sliders ---
st.sidebar.header("Specify Input Parameters")

def user_input_features():
    return pd.DataFrame({
        'MedInc': [st.sidebar.slider('Median Income (10k USD)', float(X_full.MedInc.min()), float(X_full.MedInc.max()), float(X_full.MedInc.mean()))],
        'HouseAge': [st.sidebar.slider('House Age (years)', float(X_full.HouseAge.min()), float(X_full.HouseAge.max()), float(X_full.HouseAge.mean()))],
        'AveRooms': [st.sidebar.slider('Average Rooms', float(X_full.AveRooms.min()), float(X_full.AveRooms.max()), float(X_full.AveRooms.mean()))],
        'AveBedrms': [st.sidebar.slider('Average Bedrooms', float(X_full.AveBedrms.min()), float(X_full.AveBedrms.max()), float(X_full.AveBedrms.mean()))],
        'Population': [st.sidebar.slider('Block Population', float(X_full.Population.min()), float(X_full.Population.max()), float(X_full.Population.mean()))],
        'AveOccup': [st.sidebar.slider('Average Occupancy', float(X_full.AveOccup.min()), float(X_full.AveOccup.max()), float(X_full.AveOccup.mean()))],
        'Latitude': [st.sidebar.slider('Latitude', float(X_full.Latitude.min()), float(X_full.Latitude.max()), float(X_full.Latitude.mean()))],
        'Longitude': [st.sidebar.slider('Longitude', float(X_full.Longitude.min()), float(X_full.Longitude.max()), float(X_full.Longitude.mean()))],
    })

df_input = user_input_features()

# --- Make prediction ---
prediction = model.predict(df_input)

# --- Display maps ---
st.write("---")

# Sample 1000 points for map performance
df_sample = df_california.sample(1000, random_state=42).copy()
X_map = df_sample.drop(columns="MedHouseVal")
df_sample["PredictedPrice"] = model.predict(X_map) * 100000

# --- Layout with three columns ---
col_left, col_mid, col_right = st.columns([0.1, 1.5, 1])

# --- Middle Column: Show input and prediction ---
with col_mid:
    st.header("📋 Specified Input Parameters")
    st.dataframe(df_input, use_container_width=True)
    st.write("---")

    st.header("📈 Predicted Median House Value")
    st.success(f"🏡 **Estimated Value:** ${prediction[0] * 100000:,.2f}")
    st.write("---")

    with st.expander("🗺️ Geospatial View of Predicted Housing Prices", expanded=False):
        df_sample = df_california.sample(1000, random_state=42).copy()
        X_map = df_sample.drop(columns="MedHouseVal")
        df_sample["PredictedPrice"] = model.predict(X_map) * 100000
        df_sample.rename(columns={"MedInc": "MedianIncome"}, inplace=True)

        fig = px.scatter_mapbox(
            df_sample,
            lat="Latitude",
            lon="Longitude",
            color="PredictedPrice",
            size="PredictedPrice",
            size_max=15,
            zoom=4.5,
            mapbox_style="carto-positron",
            color_continuous_scale="YlOrRd",
            hover_data={
                "PredictedPrice": True,
                "Latitude": True,
                "Longitude": True,
                "MedianIncome": True,
                "HouseAge": True
            },
            title="Predicted Median House Values Across California"
        )

        st.plotly_chart(fig, use_container_width=True)

with col_right:
    with st.expander("🧠 Model Explainability with SHAP", expanded=True):
        st.subheader("🔍 Feature Importance & Model Overview")

        tab1, tab2, tab3 = st.tabs(["📊 Beeswarm Plot", "📉 Bar Plot", "🧠 Model Overview"])

        with tab1:
            st.markdown("""
            **SHAP Beeswarm Plot**  
            This plot shows the **impact** of each feature on individual predictions.  
            - **Color**: Red = high feature value, Blue = low  
            - **X-axis**: How much a feature pushes the prediction up or down  
            - Visualizes the range of SHAP impacts across all observations  
            """)
            shap.summary_plot(shap_values, shap_X, show=False)
            st.pyplot(plt.gcf())

        with tab2:
            st.markdown("""
            **SHAP Bar Plot**  
            This bar chart shows the **average importance** of each feature.  
            - Larger bars = more influence on model predictions  
            - Great for understanding overall feature significance  
            """)
            shap.summary_plot(shap_values, shap_X, plot_type="bar", show=False)
            st.pyplot(plt.gcf())

        with tab3:
            st.markdown("""
            ### 🤖 Model Overview: Random Forest Regressor  

            This model uses a **Random Forest Regressor**, an ensemble machine learning technique that builds multiple decision trees and averages their predictions for better accuracy and robustness.

            **Why Random Forest?**  
            - 🌲 Combines multiple trees to reduce overfitting  
            - ✅ Handles complex, nonlinear relationships  
            - 🔍 Provides feature importance for interpretation  
            - 📉 Robust to outliers and multicollinearity  

            ---
            ### 🏘️ Dataset Context: California Housing Data

            The model was trained on the **California Housing dataset** originally published by the U.S. Census Bureau.  
            It includes information about housing blocks across California, including:

            - **Median income**, **house age**, **average number of rooms and bedrooms**
            - **Block-level population and occupancy**
            - **Geographic data** like latitude and longitude

            This dataset is commonly used to predict **median house values** based on neighborhood characteristics.

            ---
            ### 📊 Descriptive Statistics of Input Features
            """)
            st.dataframe(X_full.describe().T.style.format("{:.2f}"))
