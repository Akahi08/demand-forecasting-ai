import streamlit as st
import pandas as pd
import numpy as np
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Demand Forecasting AI",
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

st.title("📈 Demand Forecasting AI")

st.write(
    """
    Predict expected demand using historical sales patterns,
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
    value=(history["date"].max() + pd.Timedelta(days=1)).date(),
    min_value=(history["date"].min() + pd.Timedelta(days=28)).date()
)

st.sidebar.caption(
    "Future dates are forecast recursively beyond the historical dataset. "
    "For dates after the dataset, the selected price/promotion are applied "
    "to the target date; intermediate unknown values use the latest known values."
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

    pair_history = history[
        (history["store_id"] == store_id) &
        (history["item_id"] == item_id)
    ].sort_values("date").reset_index(drop=True)

    if len(pair_history) < 28:
        raise ValueError(
            "There are not enough historical records for this "
            "store-item combination to calculate the 28-day features."
        )

    # For dates beyond the dataset, recursively forecast only as far as
    # needed. A one-year maximum keeps very long requests practical while
    # still supporting useful future forecasts.
    if forecast_date > pair_history["date"].max():
        horizon_days = (forecast_date - pair_history["date"].max()).days
        max_horizon_days = 365

        if horizon_days > max_horizon_days:
            raise ValueError(
                f"Forecast date is {horizon_days} days beyond the available "
                f"data. Please choose a date within {max_horizon_days} days "
                "of the latest historical date."
            )

        working_history = pair_history.copy()
        current_date = working_history["date"].max() + pd.Timedelta(days=1)

        latest_known_price = float(working_history.iloc[-1]["price"])
        latest_known_promo = int(working_history.iloc[-1]["promo"])

        final_input = None

        while current_date <= forecast_date:
            prior_history = working_history[
                working_history["date"] < current_date
            ].reset_index(drop=True)

            sales = prior_history["sales"]

            current_price = (
                float(price) if current_date == forecast_date
                else latest_known_price
            )
            current_promo = (
                int(promo) if current_date == forecast_date
                else latest_known_promo
            )

            previous_price = float(prior_history.iloc[-1]["price"])

            if previous_price == 0:
                price_change = 0.0
            else:
                price_change = (current_price / previous_price) - 1.0

            input_row = pd.DataFrame({
                "store_id": [int(store_id)],
                "item_id": [int(item_id)],
                "price": [current_price],
                "promo": [current_promo],
                "weekday": [int(current_date.dayofweek)],
                "month": [int(current_date.month)],
                "year": [int(current_date.year)],
                "day": [int(current_date.day)],
                "week": [int(current_date.isocalendar().week)],
                "quarter": [int(current_date.quarter)],
                "is_weekend": [int(current_date.dayofweek >= 5)],
                "sales_lag_1": [float(sales.iloc[-1])],
                "sales_lag_7": [float(sales.iloc[-7])],
                "sales_lag_14": [float(sales.iloc[-14])],
                "sales_lag_28": [float(sales.iloc[-28])],
                "sales_rolling_mean_7": [float(sales.iloc[-7:].mean())],
                "sales_rolling_mean_28": [float(sales.iloc[-28:].mean())],
                "sales_rolling_std_7": [float(sales.iloc[-7:].std())],
                "price_change": [price_change],
                "promo_lag_1": [int(prior_history.iloc[-1]["promo"])],
                "promo_lag_7": [int(prior_history.iloc[-7]["promo"])]
            })

            # Preserve the exact feature order used during training.
            input_row = input_row[features]
            prediction = max(0.0, float(model.predict(input_row)[0]))

            final_input = input_row

            working_history = pd.concat([
                working_history,
                pd.DataFrame({
                    "date": [current_date],
                    "store_id": [int(store_id)],
                    "item_id": [int(item_id)],
                    "sales": [prediction],
                    "price": [current_price],
                    "promo": [current_promo]
                })
            ], ignore_index=True)

            current_date += pd.Timedelta(days=1)

        return final_input

    # Original behaviour for dates covered by the historical dataset.
    prior_history = pair_history[
        pair_history["date"] < forecast_date
    ].reset_index(drop=True)

    if len(prior_history) < 28:
        raise ValueError(
            "There are not enough historical records before this date "
            "for this store-item combination to calculate the 28-day features."
        )

    previous_price = float(prior_history.iloc[-1]["price"])

    if previous_price == 0:
        price_change = 0.0
    else:
        price_change = (float(price) / previous_price) - 1.0

    sales = prior_history["sales"]

    input_data = pd.DataFrame({
        "store_id": [int(store_id)],
        "item_id": [int(item_id)],
        "price": [float(price)],
        "promo": [int(promo)],
        "weekday": [int(forecast_date.dayofweek)],
        "month": [int(forecast_date.month)],
        "year": [int(forecast_date.year)],
        "day": [int(forecast_date.day)],
        "week": [int(forecast_date.isocalendar().week)],
        "quarter": [int(forecast_date.quarter)],
        "is_weekend": [int(forecast_date.dayofweek >= 5)],
        "sales_lag_1": [float(sales.iloc[-1])],
        "sales_lag_7": [float(sales.iloc[-7])],
        "sales_lag_14": [float(sales.iloc[-14])],
        "sales_lag_28": [float(sales.iloc[-28])],
        "sales_rolling_mean_7": [float(sales.iloc[-7:].mean())],
        "sales_rolling_mean_28": [float(sales.iloc[-28:].mean())],
        "sales_rolling_std_7": [float(sales.iloc[-7:].std())],
        "price_change": [price_change],
        "promo_lag_1": [int(prior_history.iloc[-1]["promo"])],
        "promo_lag_7": [int(prior_history.iloc[-7]["promo"])]
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

if st.button("Predict demand", type="primary"):

    if feature_error:
        st.error(feature_error)

    else:
        prediction = model.predict(input_data)[0]

        st.success(
            f"Predicted Demand: **{prediction:.2f} units**"
        )
