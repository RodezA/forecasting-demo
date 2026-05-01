import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agents import data_agent, forecasting_agent, summary_agent

DATA_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
)

st.set_page_config(
    page_title="NYC Taxi Demand Forecast",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)



def _build_chart(
    raw_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
) -> go.Figure:
    cutoff = raw_df["ds"].max()
    hist_fit = forecast_df[forecast_df["ds"] <= cutoff]
    future = forecast_df[forecast_df["ds"] > cutoff]

    fig = go.Figure()

    # Confidence band — upper bound (invisible anchor for fill)
    fig.add_trace(
        go.Scatter(
            x=future["ds"],
            y=future["yhat_upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            name="_upper",
        )
    )

    # Confidence band — lower bound fills to upper
    fig.add_trace(
        go.Scatter(
            x=future["ds"],
            y=future["yhat_lower"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(99, 110, 250, 0.15)",
            name="80% Confidence Interval",
            hoverinfo="skip",
        )
    )

    # In-sample model fit (dotted, muted)
    fig.add_trace(
        go.Scatter(
            x=hist_fit["ds"],
            y=hist_fit["yhat"],
            mode="lines",
            line=dict(color="rgba(99, 110, 250, 0.45)", width=1, dash="dot"),
            name="Model fit",
            hovertemplate="<b>%{x|%b %d}</b><br>Fitted: %{y:,.0f}<extra></extra>",
        )
    )

    # Historical actuals
    fig.add_trace(
        go.Scatter(
            x=raw_df["ds"],
            y=raw_df["y"],
            mode="lines+markers",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=5, color="#1f77b4"),
            name="Historical trips",
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Actual: %{y:,.0f} trips<extra></extra>",
        )
    )

    # Forecast line
    fig.add_trace(
        go.Scatter(
            x=future["ds"],
            y=future["yhat"],
            mode="lines",
            line=dict(color="#ff7f0e", width=2),
            name="Forecast",
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Forecast: %{y:,.0f} trips<extra></extra>",
        )
    )

    cutoff_str = cutoff.isoformat()
    fig.add_shape(
        type="line",
        x0=cutoff_str,
        x1=cutoff_str,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="gray", width=1, dash="dash"),
    )
    fig.add_annotation(
        x=cutoff_str,
        y=1,
        yref="paper",
        text="Forecast start",
        showarrow=False,
        xanchor="left",
        font=dict(size=12, color="gray"),
    )

    axis_style = dict(
        title_font=dict(size=13, color="#222222"),
        tickfont=dict(size=12, color="#222222"),
        showgrid=True,
        gridcolor="#cccccc",
        linecolor="#888888",
        linewidth=1,
        showline=True,
        ticks="outside",
        tickcolor="#888888",
    )

    fig.update_layout(
        title=dict(
            text="Daily Trip Volume: January 2024 Historical + Forecast",
            font=dict(size=16, color="#111111"),
        ),
        xaxis=dict(title="Date", **axis_style),
        yaxis=dict(title="Daily Trips", tickformat=",", **axis_style),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12, color="#222222"),
        ),
        hovermode="x unified",
        plot_bgcolor="#f7f7f7",
        paper_bgcolor="white",
        margin=dict(l=60, r=20, t=80, b=60),
        height=480,
    )

    return fig


# --- Sidebar ---
with st.sidebar:
    st.title("Forecast Controls")
    st.markdown("---")

    forecast_horizon = st.slider(
        label="Forecast horizon (days)",
        min_value=7,
        max_value=90,
        value=30,
        step=1,
        help="Number of days to forecast beyond the historical data.",
    )

    st.markdown("---")
    st.caption("Data: NYC TLC Yellow Taxi, January 2024")
    st.caption("Model: Prophet 1.3.0")
    st.caption("Summary: Gemini 2.0 Flash")

# --- Main ---
st.title("NYC Yellow Taxi Demand Forecast")
st.markdown(
    f"Daily trip volumes for January 2024 with a probabilistic "
    f"**{forecast_horizon}-day** forecast using Prophet."
)

try:
    raw_df = data_agent.run(DATA_URL)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

try:
    forecast_df = forecasting_agent.run(raw_df, periods=forecast_horizon)
except Exception as e:
    st.error(f"Forecast failed: {e}")
    st.stop()

st.plotly_chart(_build_chart(raw_df, forecast_df), use_container_width=True)

st.markdown("---")
st.subheader("Executive Summary")

try:
    with st.spinner("Generating executive summary..."):
        summary = summary_agent.run(forecast_df, raw_df)
    st.markdown(summary)
except ValueError as e:
    st.warning(str(e))
except Exception as e:
    st.error(f"Summary generation failed: {e}")
