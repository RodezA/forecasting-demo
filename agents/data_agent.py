import io

import pyarrow.parquet as pq
import pandas as pd
import requests
import streamlit as st

DATA_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
)


@st.cache_data(show_spinner="Downloading trip data...")
def run(url: str) -> pd.DataFrame:
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    buffer = io.BytesIO(response.content)
    table = pq.read_table(buffer, columns=["tpep_pickup_datetime"])
    df = table.to_pandas()

    df["tpep_pickup_datetime"] = pd.to_datetime(
        df["tpep_pickup_datetime"], utc=True
    ).dt.tz_localize(None)

    # The parquet contains a small number of stray records outside January 2024
    df = df[
        (df["tpep_pickup_datetime"] >= "2024-01-01")
        & (df["tpep_pickup_datetime"] < "2024-02-01")
    ]

    daily = (
        df.assign(date=df["tpep_pickup_datetime"].dt.normalize())
        .groupby("date", as_index=False)
        .size()
        .rename(columns={"date": "ds", "size": "y"})
    )

    daily["ds"] = pd.to_datetime(daily["ds"])
    return daily.sort_values("ds").reset_index(drop=True)
