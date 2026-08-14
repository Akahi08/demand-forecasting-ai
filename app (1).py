import streamlit as st
import pandas as pd
import numpy as np
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Sales Forecasting AI",
    page_icon="📈",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "sales_forecasting_model.pkl"
    )

    features = joblib.load(
        "sales_model_features.pkl"
    )

    return model, features


model, features = load_model()


# =========================================================
# HEADER
# =========================================================

st.title("📈 Sales Forecasting AI")

st.write(
    """
    Predict expected sales using historical sales patterns,
    promotions, pricing and time-based features.
    """
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Prediction Inputs")


store_id = st.sidebar.number_input(
    "Store ID",
    min_value=1,
    max_value=50,
    value=1
)


item_id = st.sidebar.number_input(
    "Item ID",
    min_value=1,
    max_value=50,
    value=1
)


price = st.sidebar.number_input(
    "Price",
    min_value=0.0,
    value=10.0
)


promo = st.sidebar.selectbox(
    "Promotion",
    [0, 1]
)


weekday = st.sidebar.selectbox(
    "Weekday",
    list(range(7))
)


month = st.sidebar.selectbox(
    "Month",
    list(range(1, 13))
)


year = st.sidebar.number_input(
    "Year",
    min_value=2019,
    max_value=2030,
    value=2024
)


day = st.sidebar.number_input(
    "Day",
    min_value=1,
    max_value=31,
    value=1
)


week = st.sidebar.number_input(
    "Week",
    min_value=1,
    max_value=53,
    value=1
)


quarter = st.sidebar.selectbox(
    "Quarter",
    [1, 2, 3, 4]
)


is_weekend = st.sidebar.selectbox(
    "Weekend",
    [0, 1]
)


# =========================================================
# HISTORICAL FEATURES
# =========================================================

sales_lag_1 = st.sidebar.number_input(
    "Sales Lag 1",
    min_value=0.0,
    value=25.0
)

sales_lag_7 = st.sidebar.number_input(
    "Sales Lag 7",
    min_value=0.0,
    value=25.0
)

sales_lag_14 = st.sidebar.number_input(
    "Sales Lag 14",
    min_value=0.0,
    value=25.0
)

sales_lag_28 = st.sidebar.number_input(
    "Sales Lag 28",
    min_value=0.0,
    value=25.0
)


sales_rolling_mean_7 = st.sidebar.number_input(
    "7-Day Rolling Mean",
    min_value=0.0,
    value=25.0
)


sales_rolling_mean_28 = st.sidebar.number_input(
    "28-Day Rolling Mean",
    min_value=0.0,
    value=25.0
)


sales_rolling_std_7 = st.sidebar.number_input(
    "7-Day Rolling Std",
    min_value=0.0,
    value=5.0
)


price_change = st.sidebar.number_input(
    "Price Change",
    value=0.0
)


promo_lag_1 = st.sidebar.selectbox(
    "Previous Day Promotion",
    [0, 1]
)


promo_lag_7 = st.sidebar.selectbox(
    "Promotion 7 Days Ago",
    [0, 1]
)


# =========================================================
# CREATE INPUT
# =========================================================

input_data = pd.DataFrame({
    "store_id": [store_id],
    "item_id": [item_id],
    "price": [price],
    "promo": [promo],
    "weekday": [weekday],
    "month": [month],
    "year": [year],
    "day": [day],
    "week": [week],
    "quarter": [quarter],
    "is_weekend": [is_weekend],
    "sales_lag_1": [sales_lag_1],
    "sales_lag_7": [sales_lag_7],
    "sales_lag_14": [sales_lag_14],
    "sales_lag_28": [sales_lag_28],
    "sales_rolling_mean_7": [sales_rolling_mean_7],
    "sales_rolling_mean_28": [sales_rolling_mean_28],
    "sales_rolling_std_7": [sales_rolling_std_7],
    "price_change": [price_change],
    "promo_lag_1": [promo_lag_1],
    "promo_lag_7": [promo_lag_7]
})


# Make sure the exact training feature order is preserved
input_data = input_data[features]


# =========================================================
# PREDICTION
# =========================================================

if st.button("Predict Sales", type="primary"):

    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted Sales: **{prediction:.2f} units**"
    )