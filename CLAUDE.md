# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Scaffold-phase — directory structure and dataset exist; no Python source files yet. `agents/`, `scripts/`, and `data/` directories are in place. Implement `app.py` and the three agents when work begins.

## Intent

A multi-agent forecasting demo deployed to Streamlit Community Cloud using NYC yellow taxi trip data. Three agents:

1. **Data agent** — loads and preprocesses `data/yellow_tripdata_2024-01.parquet`.
2. **Forecasting agent** — probabilistic forecasting via Prophet.
3. **Executive-summary agent** — natural-language summaries via Claude (`claude-sonnet-4-6`).

No emojis anywhere in the UI or code. Professional appearance throughout.

## Environment

Python 3.14 venv at `.venv/`. Activate with `source .venv/bin/activate` before running anything. Packages already installed: `pandas`, `pyarrow`, `numpy`.

```bash
# Install remaining dependencies (once requirements.txt exists)
.venv/bin/pip install -r requirements.txt

# Run the app locally
streamlit run app.py

# Lint
ruff check .
```

No test runner wired up yet. Add `pytest` when the first testable unit exists.

## Architecture

```
app.py                  # Streamlit entry point — orchestrates agents, renders UI
agents/
  data_agent.py         # Loads + preprocesses the parquet file; run(path) -> DataFrame
  forecasting_agent.py  # Prophet model; run(df) -> forecast DataFrame
  summary_agent.py      # Claude API call; run(forecast_df) -> markdown string
data/                   # gitignored — fetched at runtime by data_agent
```

Each agent exposes a single `run(...)` entry point. `app.py` owns all orchestration — agents must not import each other.

## Constraints

- **Streamlit Community Cloud is CPU-only and memory-limited.** Use Prophet; avoid NeuralProphet or transformer fine-tuning at runtime.
- **Public repo** — API keys via `st.secrets` (production) and `.env` (local).
- **Forecasting ↔ summarization are decoupled** — the forecaster can be swapped without touching `summary_agent.py`.
- Anthropic SDK work must follow the `claude-api` skill: use prompt caching, model ID `claude-sonnet-4-6`.
