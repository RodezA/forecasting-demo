# NYC Taxi Demand Forecast

A multi-agent forecasting demo built with Streamlit. Fetches NYC yellow taxi trip data, generates a probabilistic forecast using Prophet, and produces an executive summary via Claude.

## Live Demo

[forecasting-demo-edmrrbnggmqxk3pnsgsfw5.streamlit.app](https://forecasting-demo-edmrrbnggmqxk3pnsgsfw5.streamlit.app)

## Architecture

Three independent agents orchestrated by `app.py`:

| Agent | File | Role |
|-------|------|------|
| Data | `agents/data_agent.py` | Downloads NYC TLC parquet, aggregates to daily trip counts |
| Forecasting | `agents/forecasting_agent.py` | Fits Prophet model, returns probabilistic forecast |
| Summary | `agents/summary_agent.py` | Calls Claude Sonnet 4.6 to generate an executive summary |

## Stack

- [Streamlit](https://streamlit.io) — UI and deployment
- [Prophet](https://facebook.github.io/prophet/) — time series forecasting
- [Plotly](https://plotly.com) — interactive chart
- [Claude Sonnet 4.6](https://www.anthropic.com) — natural-language executive summary
- [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) — public dataset

## Local Setup

**Requirements:** Python 3.12

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml` (see `secrets.toml.example`):

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
PASSWORD = "your-password"
```

```bash
streamlit run app.py
```
