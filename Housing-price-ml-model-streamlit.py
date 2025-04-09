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
import numpy as np
import seaborn as sns
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
image_path = os.path.join(os.path.dirname(__file__), "Housing2.jpg")
image = Image.open(image_path)
col_img1, col_img2, col_img3 = st.columns([1, 1, 1])
# with col_img2:
#     st.image(image, use_container_width=True)
st.image(image, use_container_width=True)

# --- App title and intro ---
st.markdown("""
# AI-Powered House Price Prediction 
### Using Random Forest Ensembles for Real Estate Insights
##### Developed by Asif Rasool, Southeastern Louisiana University

""")
st.write('---')

# --- About section ---
with st.expander("ℹ️ About"):
    st.markdown("""
This interactive app demonstrates the use of **Artificial Intelligence** and **Machine Learning** to predict housing prices in California using real-world census data.  
It leverages exploratory data analysis (EDA), model interpretability techniques, and interactive visualizations to provide insights into the housing market.

---

### 🧰 **Tools & Libraries Used**

- **Streamlit** – App framework for interactive web dashboards  
- **scikit-learn** – Machine learning modeling (Random Forest Regressor)  
- **SHAP** – Explainable AI for feature importance  
- **pandas**, **numpy** – Data manipulation and numerical computing  
- **matplotlib**, **seaborn** – Statistical and SHAP visualizations  
- **plotly** – Interactive maps and geospatial visualizations  
- **gdown** – Securely download large model assets from Google Drive  
- **Pillow** – Display and manage image headers

---

### 📦 **Dataset Source**
- [California Housing Prices Dataset (Kaggle)](https://www.kaggle.com/datasets/camnugent/california-housing-prices?resource=download)

Originally published by the U.S. Census Bureau, this dataset includes:
- Median income  
- House age  
- Rooms & bedrooms per household  
- Population and occupancy  
- Latitude and longitude

---

**Asif Rasool, Ph.D.**  
Research Economist, Southeastern Louisiana University  
📍 1514 Martens Drive, Hammond, LA 70401  
📞 985-549-3831  
📧 [asif.rasool@southeastern.edu](mailto:asif.rasool@southeastern.edu)  
🌐 [Work Website](https://www.southeastern.edu/employee/asif-rasool/)  
🔗 [GitHub Repository](https://github.com/Asif-Rasool/Housing-Price-Model)

_Last updated: April 9, 2025_
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

    st.header("🗺️ Geospatial View of Predicted Housing Prices")

    # Sample and prepare data
    df_sample = df_california.sample(1000, random_state=42).copy()
    X_map = df_sample.drop(columns="MedHouseVal")
    df_sample["PredictedPrice"] = model.predict(X_map) * 100000
    df_sample.rename(columns={"MedInc": "MedianIncome"}, inplace=True)

    # Plotly map
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

# --- EDA Section ---
    st.header("🧪 Exploratory Data Analysis (EDA)")
    st.markdown("Explore the dataset's characteristics and distribution of features.")

# --- Univariate Histogram Section ---
with col_mid:
    with st.expander("📊 Univariate Analysis: Histograms of Housing Features", expanded=False):
        st.markdown("""
    Let's look at our data distribution using **univariate analysis** (analysis of one variable at a time).  
    Here's what we might look for when examining histograms:

    - **Data distribution**: Some models prefer less skewed distributions.
    - **Outliers**: Extreme values can harm model performance under low-noise assumptions.
    - **Odd patterns**: Abnormalities in data can negatively impact predictions.
    - **Axis scale**: Large differences in feature magnitudes can mislead models
        
    """)

    # Plot histograms for all numeric columns
        axes = df_california.hist(bins=40, figsize=(7.5, 4.5), color='skyblue', edgecolor='black')
        plt.tight_layout()
        st.pyplot(plt.gcf())

        # Add detailed histogram interpretation below plot
        st.markdown("""
### 🧾 Odd Patterns & Outliers  
**Data distributions which slightly stick out:**

- On first impression, a few outlier groups are present in our data — possibly due to the way in which the data was sampled (e.g., `housing_median_age` & `median_house_value`).
- `housing_median_age` shows some discontinuity and a sharp peak at its maximum value. This becomes more apparent when adjusting histogram bins.
- `median_house_value` has an unusual spike at the upper limit (around $500K), likely due to data clipping or capping.

**Less Noticeable Outliers:**

- Several features show **skewed** distributions — around 6 of them — which is concerning since we're using a relatively simple model.
- Some features, like `population`, have a broad range on the x-axis, suggesting many outliers.
- Features such as `population`, `total_bedrooms`, and `total_rooms`, which are related, also share a similar skew toward smaller values.

These characteristics should be kept in mind for potential transformation or scaling before modeling.
        """)

# --- Correlation Matrix Plotting Function ---
def corrMat(df):
    corr_mat = df.corr().round(2)

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(6, 6))

    # Create a mask for the upper triangle
    mask = np.zeros_like(corr_mat, dtype=np.bool_)  # Updated type
    mask[np.triu_indices_from(mask)] = True

    # Draw the heatmap
    sns.heatmap(
        corr_mat,
        mask=mask,
        vmin=-1,
        vmax=1,
        center=0,
        cmap='plasma',
        square=True,
        linewidths=0.5,
        annot=True,
        cbar=False,
        ax=ax
    )
    st.pyplot(f)


        # --- Bivariate correlation matrix ---
with col_mid:
    with st.expander("📊 Bivariate correlation matrix", expanded=False):
        st.markdown("""
    ### 🧮 Bivariate Correlation Matrix

Bivariate analysis involves examining the relationship between **two variables** at a time.  
A **correlation matrix** is a fast and effective way to gain insight into potential feature relationships in the dataset.

💡 **Key Points to Consider:**

- The matrix only captures **linear relationships** between features.
- It's useful for identifying **redundant features** — those that are too highly or too lowly correlated.
- Strong correlations may indicate that some features are essentially measuring the same thing.
- Weak correlations across the board might suggest **nonlinear** relationships or more complex interactions.

We use this matrix to understand which features are worth keeping, transforming, or dropping before modeling.
        
    """)
        # Call correlation matrix plot
        corrMat(df_california.drop(columns='MedHouseVal'))
        st.markdown("""
### 📌 Interpretation: Target Variable Relationships

- The target variable `median_house_value` is **very mildly correlated** with all features except one — **`median_income`**, which stands out as an important predictor.
- Features like `population` and `longitude` show very weak negative correlations (-0.02 and -0.05 respectively). While these may appear uninformative, **low correlation alone isn't a reason to drop a feature**.
- Such low values may indicate a **nonlinear relationship**, which can still carry predictive power — just not in a linear sense.
- That said, for **simpler models** (e.g., linear regressors), it's often advised to drop these weakly correlated features, as they’re less likely to contribute meaningfully and might introduce noise.
""")

with col_mid:
# --- Bivariate Scatter Plot Section ---
    with st.expander("🔍 Bivariate Analysis: Scatter Plot Matrix", expanded=False):
        st.markdown("""
Bivariate scatter plots (or pairplots) are **very insightful** for exploring relationships between two variables.  
They help reveal patterns, clusters, correlations, and potential outliers.

💡 **What to look for in scatter plots:**

- **Irregular two-feature patterns** or outliers — combining features reveals more than univariate views.
- **Two-dimensional clusters** in the data (can be emphasized using KDE density contours).
- **Visual signs of correlation** — is there a clear linear trend, or is the data spread randomly?

While we can use color labeling in scatter matrices, it often becomes hard to distinguish in crowded plots.
    """)

    # Use actual column names from the California dataset
        tlist = ['MedInc', 'AveRooms', 'HouseAge', 'Latitude', 'MedHouseVal', 'Population']
        selected_df = df_california[tlist].copy()

    # Add income group for color
        selected_df["IncomeGroup"] = pd.qcut(selected_df["MedInc"], q=4, labels=["Low", "Medium", "High", "Very High"])

        st.markdown("✅ **Color-coded by Median Income Group (quartiles)**")

    # Plot PairGrid
        grid = sns.PairGrid(
            selected_df,
            vars=tlist,
            hue="IncomeGroup",
            corner=True,
            diag_sharey=False
        )
        grid.map_lower(sns.scatterplot, alpha=0.5)
        grid.map_diag(sns.kdeplot, fill=True)
        grid.add_legend()

        st.pyplot(grid.fig)
        st.markdown("""
### 🧾 Relating to `median_house_value`

- The relationship between `median_house_value` and `median_income` appears **quite linear**, with some spread perpendicular to the line.  
  Notably, there's a **visible upper limit** for all values of `median_income` — a capped value that visually stands out.

- The relationship between `median_house_age` and `median_house_value` looks **highly scattered** with no obvious pattern.  
  The KDE plot reveals **two peaks roughly 20 years apart**, possibly indicating changes in affordability or housing policy.

- There's also a visible cluster near the **maximum values** for both features — again, highly nonlinear and broadly spread across the plot.

- The features `total_rooms` and `population` in relation to `median_house_value` show **complex, nonlinear structures**.  
  KDE shows heavy concentration at **low values**, with sparse but notable spread toward higher values — possibly **outliers**.

- Finally, many features differ significantly in **axis scale**.  
  Larger values might be mistakenly interpreted as more important by some models, so **feature scaling should definitely be considered**.
    """)

# Geospatial Plot Function (Plotly-Based)
def plot_geo_feature(df, feature, map_height=450):
    fig = px.scatter_mapbox(
        df,
        lat="Latitude",
        lon="Longitude",
        color=feature,
        size=feature,
        color_continuous_scale="plasma",
        size_max=12,
        zoom=4.5,
        height=map_height,
        mapbox_style="carto-positron",
        hover_data={"Latitude": True, "Longitude": True, feature: True}
    )
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, title=f"Geospatial Distribution of {feature}")
    return fig

with col_mid:
    # --- Geospatial Multivariate Section ---
    with st.expander("🗺️ Multivariate Analysis: Geospatial Feature Mapping", expanded=False):
        st.markdown("""
Multivariate visualization can often reveal relationships that simpler plots miss.  
By adding **color** and **location** to a scatter plot, we can explore how multiple features relate geographically.

📌 **Why this matters:**
- Some features may show strong **spatial clustering** or **hot spots**
- Patterns can emerge more clearly in **two dimensions**
- Visualizing geographic influence helps with **feature selection**

Common tools: `geopandas`, `folium`, `plotly`, and for 3D: `k3d`, `pyvista`
        """)

        # Show selected geospatial plots
        st.plotly_chart(plot_geo_feature(df_california, "Population"), use_container_width=True)
        st.plotly_chart(plot_geo_feature(df_california, "MedInc"), use_container_width=True)
        st.plotly_chart(plot_geo_feature(df_california, "HouseAge"), use_container_width=True)
        st.plotly_chart(plot_geo_feature(df_california, "MedHouseVal"), use_container_width=True)

        st.markdown("""
### 🧾 Interpretation

- For our target variable `median_house_value`, the influence of **geography** is immediately visible — values generally increase as we get closer to two main urban clusters.
- `median_income` shows a **strong spatial correlation** with `median_house_value`, especially in coastal areas.
- `housing_median_age` correlates well in some coastal regions but not as clearly inland, suggesting a **nonlinear relationship**.
- `population` is trickier — the data is skewed toward low values, and a few **extreme outliers** distort the pattern. This also aligns with what we observed in histograms and pairplots.
- These plots confirm that **location matters**, and combining features with spatial awareness can significantly improve model performance.
        """)


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
