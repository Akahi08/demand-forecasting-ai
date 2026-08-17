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
# LOAD HISTORICAL DATA
# =========================================================

@st.cache_data
def load_history():
    history = pd.read_csv(
        "sales_history.csv.gz",
        parse_dates=["date"],
        converters={
            "store_id": lambda x: int(str(x).split("_")[-1]),
            "item_id": lambda x: int(str(x).split("_")[-1])
        },
        dtype={
            "sales": "int32",
            "price": "float32",
            "promo": "int8"
        }
    )

    history = history.sort_values(
        ["store_id", "item_id", "date"]
    ).reset_index(drop=True)

    return history


history = load_history()

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


forecast_date = st.sidebar.date_input(
    "Forecast Date",
    value=pd.Timestamp("2023-08-15").date(),
    min_value=(history["date"].min() + pd.Timedelta(days=28)).date(),
    max_value=history["date"].max().date()
)

# =========================================================
# AUTOMATIC FEATURE GENERATION
# =========================================================

def create_prediction_features(
    history,
    store_id,
    item_id,
    forecast_date,
    price,
    promo
):
    forecast_date = pd.Timestamp(forecast_date)

    # Get historical records for this store-item pair.
    pair_history = history[
        (history["store_id"] == store_id) &
        (history["item_id"] == item_id)
    ].sort_values("date").reset_index(drop=True)

    # The training features require at least 28 previous observations.
    prior_history = pair_history[
        pair_history["date"] < forecast_date
    ].reset_index(drop=True)

    if len(prior_history) < 28:
        raise ValueError(
            "There are not enough historical records for this "
            "store-item combination to calculate the 28-day features."
        )

    # Use the latest historical observation before the forecast date
    # for price-change calculation.
    previous_price = float(prior_history.iloc[-1]["price"])

    # Match the notebook's pct_change definition:
    # current price relative to the previous price.
    if previous_price == 0:
        price_change = 0.0
    else:
        price_change = (float(price) / previous_price) - 1.0

    sales = prior_history["sales"]

    # Match the notebook's shift/rolling logic:
    # previous 7/28 observations only.
    sales_lag_1 = float(sales.iloc[-1])
    sales_lag_7 = float(sales.iloc[-7])
    sales_lag_14 = float(sales.iloc[-14])
    sales_lag_28 = float(sales.iloc[-28])

    sales_rolling_mean_7 = float(sales.iloc[-7:].mean())
    sales_rolling_mean_28 = float(sales.iloc[-28:].mean())
    sales_rolling_std_7 = float(sales.iloc[-7:].std())

    promo_lag_1 = int(prior_history.iloc[-1]["promo"])
    promo_lag_7 = int(prior_history.iloc[-7]["promo"])

    # Calendar features are generated from the forecast date,
    # matching the notebook's date-based feature engineering.
    weekday = forecast_date.dayofweek
    month = forecast_date.month
    year = forecast_date.year
    day = forecast_date.day
    week = int(forecast_date.isocalendar().week)
    quarter = forecast_date.quarter
    is_weekend = int(forecast_date.dayofweek >= 5)

    input_data = pd.DataFrame({
        "store_id": [int(store_id)],
        "item_id": [int(item_id)],
        "price": [float(price)],
        "promo": [int(promo)],
        "weekday": [int(weekday)],
        "month": [int(month)],
        "year": [int(year)],
        "day": [int(day)],
        "week": [int(week)],
        "quarter": [int(quarter)],
        "is_weekend": [int(is_weekend)],
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

    return input_data


# =========================================================
# CREATE INPUT
# =========================================================

try:
    input_data = create_prediction_features(
        history=history,
        store_id=store_id,
        item_id=item_id,
        forecast_date=forecast_date,
        price=price,
        promo=promo
    )

    # Make sure the exact training feature order is preserved
    input_data = input_data[features]

    feature_error = None

except ValueError as e:
    input_data = None
    feature_error = str(e)


# =========================================================
# PREDICTION
# =========================================================

if st.button("Predict Sales", type="primary"):

    if feature_error:
        st.error(feature_error)

    else:
        prediction = model.predict(input_data)[0]

        st.success(
            f"Predicted Sales: **{prediction:.2f} units**"
        )
