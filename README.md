

### ✅ README.md — California Housing Price Prediction App

```markdown
# 🏡 California Housing Price Prediction App

This is an interactive [Streamlit](https://streamlit.io/) web application that predicts **median house prices in California** using a trained **Random Forest Regressor** model. It also provides model explainability through **SHAP** plots and a **map-based visualization** of predictions.

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://share.streamlit.io/)  
*Author: Asif Rasool, Ph.D. — Research Economist, Southeastern Louisiana University*

---

## 📦 Features

- 🧠 Predicts **median house value** from 8 housing and geographic inputs
- 📈 Interactive **SHAP beeswarm and bar plots** for feature importance
- 🗺️ **Geospatial Plotly map** showing predicted prices across California
- 📥 Loads model from **Google Drive** at runtime (no `.pkl` in repo)
- 🎨 Custom UI with a **top banner**, **expander sections**, and **tabs**

---

## 🧪 Technologies & Libraries Used

| Tool/Library     | Purpose                           |
|------------------|------------------------------------|
| `streamlit`      | Web app interface                 |
| `scikit-learn`   | Machine learning (Random Forest)  |
| `pandas`         | Data manipulation                 |
| `shap`           | Model explainability              |
| `matplotlib`     | Backend plotting for SHAP         |
| `plotly`         | Interactive map                   |
| `requests`       | Download model from Google Drive  |
| `Pillow`         | Load and render images            |

---

## 🔧 Project Structure

```
📁 Data-Science-Projects/
├── Housing-price-ml-model-streamlit.py   # Main Streamlit app
├── requirements.txt                      # Python dependencies
├── Procfile                              # For Heroku deployment
├── Housing.jpg                           # App header image
├── .gitignore                            # Ignore .pkl files
└── (No .pkl files — loaded from Google Drive)
```

---

## 🚀 How to Run the App Locally

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/california-housing-price-app.git
cd california-housing-price-app
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run Housing-price-ml-model-streamlit.py
```

✅ Model `.pkl` files will automatically download from Google Drive when the app runs.

---

## 🌐 Deployment Options

### 🚀 Streamlit Cloud

- Just connect this repo and deploy — no extra config needed!

### 🔁 Heroku (with Procfile)

```bash
heroku create housing-price-app
git push heroku main
```

---

## 📊 Input Features

- Median Income (`MedInc`)
- House Age (`HouseAge`)
- Average Rooms (`AveRooms`)
- Average Bedrooms (`AveBedrms`)
- Population (`Population`)
- Average Occupancy (`AveOccup`)
- Latitude / Longitude (`Latitude`, `Longitude`)

---

## 📂 Model & Data Files (Hosted Externally)

| File                  | Source Location |
|-----------------------|-----------------|
| `housing_price_model.pkl` | Google Drive |
| `shap_values.pkl`         | Google Drive |
| `X_features.pkl`          | Google Drive |

These are **downloaded at runtime** via direct links using `requests`.

---

## 👨‍💻 Author

**Asif Rasool, Ph.D.**  
Research Economist, Southeastern Louisiana University  
📍 1514 Martens Drive, Hammond, LA 70401  
📞 985-549-3831  
✉️ [asif.rasool@southeastern.edu](mailto:asif.rasool@southeastern.edu)

---

## 📅 Last Updated

**April 8, 2025**

---

## ⭐️ Give it a Star

If you found this project helpful or interesting, consider giving it a ⭐ on GitHub!
```

---
