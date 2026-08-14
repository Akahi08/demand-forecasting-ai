# 📈 Sales Forecasting AI

A machine learning-powered web application built with **Python, Streamlit, Pandas, NumPy, and Scikit-learn** for predicting expected product sales based on historical sales patterns, pricing, promotions, and time-based features.

The application provides an interactive interface where users can enter sales-related information and receive an estimated sales prediction.

---

## 🚀 Project Overview

Accurate sales forecasting helps businesses make better decisions about:

* 📦 Inventory management
* 💰 Pricing strategies
* 📢 Promotional campaigns
* 🏪 Store-level planning
* 📊 Demand estimation
* 📈 Business growth

This project uses historical sales data and engineered time-series features to train a machine learning model capable of predicting future sales.

The trained model is integrated into a **Streamlit web application** for easy interaction and prediction.

---

## ✨ Features

### 📊 Sales Prediction

The application predicts expected sales in units based on user-provided information.

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

The Streamlit interface allows users to change input values and generate predictions instantly.

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
sales-forecasting-ai/
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
    └── sales_forecasting.ipynb
```

### File Descriptions

**`app.py`**

The Streamlit application responsible for collecting user inputs and generating predictions.

**`sales_forecasting_model.pkl`**

The trained machine learning model saved using Joblib.

**`sales_model_features.pkl`**

Contains the exact feature names and order used during model training.

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
git clone https://github.com/Akahi08/sales-forecasting-ai
```

Move into the project directory:

```bash
cd sales-forecasting-ai
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
6. Enter the relevant date and calendar information.
7. Enter historical sales lag values.
8. Enter rolling sales statistics.
9. Enter price-change information.
10. Enter historical promotion information.
11. Click **Predict Sales**.

The application will display:

```text
Predicted Sales: XX.XX units
```

---

## ⚠️ Important Input Consideration

The application requires historical features such as:

```text
Sales Lag 1
Sales Lag 7
Sales Lag 14
Sales Lag 28
7-Day Rolling Mean
28-Day Rolling Mean
7-Day Rolling Std
```

These values should ideally be calculated automatically from historical sales data rather than manually entered.

The current application exposes them as inputs to make the prediction interface simple and demonstrate how the trained model works.

---

## 🔐 Model Compatibility

The application loads two Joblib files:

```python
sales_forecasting_model.pkl
sales_model_features.pkl
```

Both files must be present in the same directory as `app.py`.

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
requirements.txt
```

After deployment, users can access the forecasting application through a web browser without installing Python locally.

---

## 🔮 Future Improvements

Possible improvements include:

### 1. Automatic Historical Feature Generation

Instead of manually entering lag and rolling features, the application could retrieve historical sales data and calculate them automatically.

### 2. CSV Upload

Allow users to upload sales data:

```text
Upload CSV
      ↓
Validate Data
      ↓
Generate Features
      ↓
Predict Sales
      ↓
Download Results
```

### 3. Forecast Multiple Days

Allow users to generate forecasts for:

* 7 days
* 14 days
* 30 days
* 90 days

### 4. Visualization Dashboard

Add charts showing:

* Historical sales
* Forecasted sales
* Sales trends
* Promotion impact
* Price impact
* Store performance

### 5. Store-Level Analysis

Provide comparisons between different stores.

### 6. Product-Level Analysis

Show which products have the highest expected demand.

### 7. Automated Retraining

Create a pipeline that periodically retrains the model when new sales data becomes available.

---

## 📌 Limitations

The current application has several limitations:

* Historical lag features must be supplied manually.
* Predictions depend heavily on the quality of the input data.
* The application does not automatically retrieve real-time sales information.
* Forecast accuracy depends on the performance of the trained model.
* Predictions should be treated as estimates rather than guaranteed future sales.

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

This project was developed as part of a practical machine learning workflow focused on applying predictive analytics to real-world sales forecasting problems.

If you find this project useful, consider giving the repository a ⭐ on GitHub.
