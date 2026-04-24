import pandas as pd
import streamlit as st
from prophet import Prophet


@st.cache_data(show_spinner="Fitting forecast model...")
def run(df: pd.DataFrame, periods: int = 30) -> pd.DataFrame:
    m = Prophet(
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        n_changepoints=10,
        changepoint_range=0.9,
        weekly_seasonality=True,
        yearly_seasonality=False,
        daily_seasonality=False,
        uncertainty_samples=300,
    )

    m.fit(df)

    future = m.make_future_dataframe(periods=periods, freq="D")
    forecast = m.predict(future)

    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]].copy()
