# 📈 Demand Forecasting AI

A machine learning-powered web application built with **Python, Streamlit, Pandas, NumPy, and Scikit-learn** for predicting expected product demand based on historical sales patterns, pricing, promotions, and time-based features.

The application provides an interactive interface where users can enter sales-related information and receive an estimated demand prediction.

---

## 🚀 Project Overview

Accurate demand forecasting helps businesses make better decisions about:

* 📦 Inventory management
* 💰 Pricing strategies
* 📢 Promotional campaigns
* 🏪 Store-level planning
* 📊 Demand estimation
* 📈 Business growth

This project uses historical sales data and engineered time-series features to train a machine learning model capable of predicting future demand.

The trained model is integrated into a **Streamlit web application** for easy interaction and prediction.

---

## ✨ Features

### 📊 Demand Prediction

The application predicts expected demand in units based on user-provided information.

### 🏪 Store and Product Information

Users can specify:

* Store ID
* Item ID
* Price
* Promotion status

### 📅 Time-Based Features

The application accepts:

* Weekday
* Day
* Week
* Month
* Quarter
* Year
* Weekend indicator

### 📈 Historical Sales Features

The model uses historical sales patterns including:

* Sales Lag 1
* Sales Lag 7
* Sales Lag 14
* Sales Lag 28
* 7-Day Rolling Mean
* 28-Day Rolling Mean
* 7-Day Rolling Standard Deviation

### 💰 Price Features

The model considers:

* Current price
* Price change

### 📢 Promotion Features

The model also considers:

* Current promotion
* Previous-day promotion
* Promotion 7 days ago

### ⚡ Interactive Interface

The Streamlit interface allows users to enter the store, item, price, promotion status, and forecast date, then generate predictions instantly.

Historical sales features and calendar features are generated automatically from the application's historical sales data.

---

## 🧠 Machine Learning Approach

The project follows a time-series forecasting workflow.

### Workflow

```text
Historical Sales Data
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Time-Based Features
        ↓
Lag Features
        ↓
Rolling Statistics
        ↓
Data Preprocessing
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Best Model Selection
        ↓
Model Saving
        ↓
Streamlit Deployment
```

---

## 🔧 Feature Engineering

Several features were engineered from the historical sales data.

### Lag Features

Lag features capture previous sales values:

```text
sales_lag_1
sales_lag_7
sales_lag_14
sales_lag_28
```

For example, `sales_lag_7` represents sales from seven days earlier.

### Rolling Features

Rolling statistics help capture recent demand trends:

```text
sales_rolling_mean_7
sales_rolling_mean_28
sales_rolling_std_7
```

These features help the model understand both average demand and sales volatility.

### Promotion Features

Historical promotion information was also incorporated:

```text
promo_lag_1
promo_lag_7
```

This allows the model to learn how previous promotional activities may influence current sales.

---

## 🛠️ Technologies Used

| Technology       | Purpose                               |
| ---------------- | ------------------------------------- |
| Python           | Programming language                  |
| Pandas           | Data manipulation                     |
| NumPy            | Numerical operations                  |
| Scikit-learn     | Machine learning                      |
| Joblib           | Model serialization                   |
| Streamlit        | Web application                       |
| Jupyter Notebook | Model development and experimentation |

---

## 📁 Project Structure

```text
demand-forecasting-ai/
│
├── app.py
│
├── sales_forecasting_model.pkl
│
├── sales_model_features.pkl
│
├── requirements.txt
│
├── README.md
│
├── .gitignore
│
└── notebooks/
    └── demand_forecasting.ipynb
```

### File Descriptions

**`app.py`**

The Streamlit application responsible for collecting user inputs and generating predictions.

**`sales_forecasting_model.pkl`**

The trained machine learning model saved using Joblib.

**`sales_model_features.pkl`**

Contains the exact feature names and order used during model training.

**`sales_history.csv.gz`**

Compressed historical sales data used by the Streamlit application to automatically generate lag, rolling, promotion, and time-based features for predictions.

**`requirements.txt`**

Contains the Python dependencies required to run the application.

**`README.md`**

Project documentation.

**`.gitignore`**

Specifies files that should not be uploaded to GitHub.

---

## 💻 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Akahi08/demand-forecasting-ai.git
```

Move into the project directory:

```bash
cd demand-forecasting-ai
```

---

### 2. Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` does not yet exist, install the required packages:

```bash
pip install streamlit pandas numpy scikit-learn joblib
```

---

## ▶️ Running the Application

Start the Streamlit application with:

```bash
streamlit run app.py
```

Streamlit will provide a local URL similar to:

```text
http://localhost:8501
```

Open the URL in your browser.

---

## 📊 Using the Application

1. Launch the Streamlit application.
2. Enter the **Store ID**.
3. Enter the **Item ID**.
4. Enter the current **Price**.
5. Select whether the product is currently on promotion.
6. Select the **Forecast Date**.
7. Click **Predict Demand**.

The application automatically generates the required calendar, lag, rolling, price-change, and promotion-history features from the historical sales data.

For dates beyond the available historical dataset, the application generates forecasts recursively so that future lag and rolling features can still be constructed.

The application will display:

```text
Predicted Demand: XX.XX units
```

---

## 📅 Future-Date Forecasting

The application is not restricted to the historical dataset's end date.

When a forecast date is after the latest available historical observation:

1. The application calculates the forecasting horizon.
2. It generates predictions sequentially from the first future day up to the requested date.
3. Each generated prediction is added to the working history.
4. The generated predictions are then used to construct future lag and rolling features.
5. The requested date receives the user-selected price and promotion values.
6. Intermediate future dates use the latest known price and promotion values when their actual values are unavailable.

To keep long-range requests practical and avoid excessive recursive model calls, future forecasting is limited to **365 days beyond the latest historical date**.

If a requested date is beyond this horizon, the application displays an error asking the user to select a date within the supported forecasting range.

## 🔐 Model Compatibility

The application loads two Joblib files:

```python
sales_forecasting_model.pkl
sales_model_features.pkl
```

Both model files must be present in the same directory as `app.py`.

The application also requires `sales_history.csv.gz` in the same directory because it uses the historical sales data to generate prediction features automatically.

The application also preserves the exact feature order used during training:

```python
input_data = input_data[features]
```

This is important because machine learning models expect features in the same structure/order used during training.

---

## 🧪 Model Evaluation

Before deployment, the trained model should be evaluated using appropriate forecasting metrics such as:

* MAE — Mean Absolute Error
* RMSE — Root Mean Squared Error
* R² — R-Squared
* MAPE — Mean Absolute Percentage Error
* Prediction Bias

For time-series forecasting, the train/test split should respect chronological order to avoid data leakage.

---

## 🚀 Deployment

The application can be deployed using cloud platforms that support Streamlit applications.

Typical deployment requirements include:

```text
app.py
sales_forecasting_model.pkl
sales_model_features.pkl
sales_history.csv.gz
requirements.txt
```

After deployment, users can access the forecasting application through a web browser without installing Python locally.

---

## 🔮 Future Improvements

Possible improvements include:

### 1. CSV Upload

Allow users to upload sales data:

```text
Upload CSV
      ↓
Validate Data
      ↓
Generate Features
      ↓
Predict Demand
      ↓
Download Results
```

### 2. Forecast Multiple Days

Allow users to generate forecasts for:

* 7 days
* 14 days
* 30 days
* 90 days

### 3. Visualization Dashboard

Add charts showing:

* Historical sales/demand
* Forecasted demand
* Sales trends
* Promotion impact
* Price impact
* Store performance

### 4. Store-Level Analysis

Provide comparisons between different stores.

### 5. Product-Level Analysis

Show which products have the highest expected demand.

### 6. Automated Retraining

Create a pipeline that periodically retrains the model when new sales data becomes available.

---

## 📌 Limitations

The current application has several limitations:

* Future forecasts beyond the historical dataset are generated recursively and may accumulate prediction error.
* Future-date forecasting is limited to 365 days beyond the latest historical date.
* For future dates, intermediate unknown price and promotion values use the latest known values.
* The application does not automatically retrieve real-time sales information.
* Forecast accuracy depends on the performance of the trained model.
* Predictions should be treated as estimates rather than guaranteed future demands.

---

## 🎯 Project Goal

The primary goal of this project is to demonstrate how machine learning can be used to transform historical sales data into actionable demand forecasts through an accessible web application.

The project combines:

**Data Science + Feature Engineering + Machine Learning + Time-Series Analysis + Deployment**

into a practical end-to-end machine learning solution.

---

## 👨‍💻 Author

**Aka'aba Abdulshakur**

AI/ML Developer | Data Science & Machine Learning Enthusiast

GitHub: `https://github.com/Akahi08`

---

## ⭐ Acknowledgement

This project was developed as part of a practical machine learning workflow focused on applying predictive analytics to real-world demand forecasting problems.

If you find this project useful, consider giving the repository a ⭐ on GitHub.
