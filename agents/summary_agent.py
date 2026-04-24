import os

import anthropic
import pandas as pd
import streamlit as st

MODEL = "claude-sonnet-4-6"

_SYSTEM_PROMPT = (
    "You are a transportation analytics specialist providing executive-level briefings "
    "for city planners and transit authorities. Your summaries are concise, data-driven, "
    "and free of jargon. Write in flowing prose paragraphs — no bullet points, no numbered "
    "lists, no emojis, no informal language. Refer to the data in the third person. "
    "Structure every summary in exactly three paragraphs: "
    "(1) the current trend and its magnitude, "
    "(2) peak and trough days with percentage deviations from the daily mean, "
    "(3) the forecast outlook and the uncertainty range for the forecast period."
)


def _get_api_key() -> str:
    key = st.secrets.get("ANTHROPIC_API_KEY")
    if key:
        return key
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    raise ValueError(
        "ANTHROPIC_API_KEY not found. "
        "Set it in .streamlit/secrets.toml locally or in the Streamlit Cloud dashboard."
    )


def _build_user_message(forecast_df: pd.DataFrame, raw_df: pd.DataFrame) -> str:
    mean_daily = raw_df["y"].mean()
    peak_row = raw_df.loc[raw_df["y"].idxmax()]
    trough_row = raw_df.loc[raw_df["y"].idxmin()]
    peak_pct = (peak_row["y"] - mean_daily) / mean_daily * 100
    trough_pct = (trough_row["y"] - mean_daily) / mean_daily * 100

    first_week_avg = raw_df.head(7)["y"].mean()
    last_week_avg = raw_df.tail(7)["y"].mean()
    trend_pct = (last_week_avg - first_week_avg) / first_week_avg * 100
    trend_direction = "upward" if trend_pct > 0 else "downward"

    forecast_only = forecast_df[forecast_df["ds"] > raw_df["ds"].max()]
    forecast_mean = forecast_only["yhat"].mean()
    forecast_low = forecast_only["yhat_lower"].min()
    forecast_high = forecast_only["yhat_upper"].max()
    forecast_start = forecast_only["ds"].min().strftime("%B %d, %Y")
    forecast_end = forecast_only["ds"].max().strftime("%B %d, %Y")
    forecast_periods = len(forecast_only)

    return (
        f"Analyze the following NYC yellow taxi trip data for January 2024 "
        f"and produce a three-paragraph executive summary.\n\n"
        f"Historical data (January 2024):\n"
        f"- Total days: {len(raw_df)}\n"
        f"- Mean daily trips: {mean_daily:,.0f}\n"
        f"- Peak day: {peak_row['ds'].strftime('%A, %B %d')} "
        f"with {peak_row['y']:,.0f} trips ({peak_pct:+.1f}% above mean)\n"
        f"- Trough day: {trough_row['ds'].strftime('%A, %B %d')} "
        f"with {trough_row['y']:,.0f} trips ({trough_pct:+.1f}% below mean)\n"
        f"- Trend: {trend_direction} over the month "
        f"({trend_pct:+.1f}% from first week to last week)\n\n"
        f"Forecast ({forecast_start} to {forecast_end}, {forecast_periods} days):\n"
        f"- Projected mean daily trips: {forecast_mean:,.0f}\n"
        f"- Uncertainty range (80% interval): {forecast_low:,.0f} "
        f"to {forecast_high:,.0f} trips per day"
    )


def run(forecast_df: pd.DataFrame, raw_df: pd.DataFrame) -> str:
    client = anthropic.Anthropic(api_key=_get_api_key())
    user_message = _build_user_message(forecast_df, raw_df)

    response = client.messages.create(
        model=MODEL,
        max_tokens=600,
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_message}],
    )

    return response.content[0].text
