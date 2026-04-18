## Macrolens

Macrolens is an economic indicators pipeline and analytics app for tracking key U.S. macro signals from 2015 to 2024. It pulls data from FRED and BLS, applies basic quality checks, stores a curated dataset locally and in Snowflake, and exposes the data through both a CLI chat interface and a Streamlit dashboard backed by a Groq-powered LangChain SQL agent.

## What the project does

- Extracts eight economic indicators from public APIs.
- Merges and cleans the raw series into a single time-series dataset.
- Applies range-based quality checks and adds lineage metadata.
- Saves the curated dataset locally as CSV plus a data dictionary.
- Loads the transformed table into Snowflake as `ECONOMIC_DATA`.
- Provides:
  - a terminal chat assistant in `chat.py`
  - a Streamlit dashboard in `app.py`

## Indicators covered

- GDP
- Inflation (CPI)
- Federal funds rate
- Unemployment
- Housing starts
- Nonfarm payroll
- Average hourly wage
- Labor force participation

## Repository structure

```text
.
|-- app.py                 # Streamlit dashboard + AI query workflow
|-- agent.py               # LangChain / Groq router + SQL agent
|-- chat.py                # Terminal chat interface
|-- main.py                # End-to-end pipeline entrypoint
|-- extract.py             # FRED and BLS extraction logic
|-- transform.py           # Merge, cleaning, quality checks
|-- load_local.py          # CSV + data dictionary output
|-- load_snowflake.py      # Snowflake load step
|-- config.py              # Environment-driven configuration
`-- data/
    |-- economic_indicators.csv
    `-- data_dictionary.md
```

## Data flow

1. `extract.py` pulls time series from FRED and BLS.
2. `transform.py` outer-joins all series on `date`, forward/back-fills gaps, and flags rows outside expected bounds.
3. `load_local.py` writes the curated dataset to `data/economic_indicators.csv` and regenerates the data dictionary.
4. `load_snowflake.py` overwrites the target Snowflake table with the transformed dataset.
5. `agent.py` and `app.py` let users ask natural-language questions against the Snowflake table.

## Setup

Create a virtual environment and install the required packages used by the project:

```bash
pip install pandas requests python-dotenv snowflake-connector-python snowflake-sqlalchemy sqlalchemy streamlit plotly langchain langchain-community langchain-groq
```

Create a `.env` file with the credentials the project expects:

```env
FRED_API_KEY=...
BLS_API_KEY=...
GROQ_API_KEY=...

SNOWFLAKE_ACCOUNT=...
SNOWFLAKE_USER=...
SNOWFLAKE_PASSWORD=...
SNOWFLAKE_WAREHOUSE=...
SNOWFLAKE_DATABASE=...
SNOWFLAKE_SCHEMA=...
```

## Running the pipeline

Run the full extract-transform-load flow:

```bash
python main.py
```

This will refresh the local CSV and push the transformed table into Snowflake.

## Running the applications

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

Start the terminal chat assistant:

```bash
python chat.py
```

## Snowflake table expectation

The app and agent expect the transformed dataset to be available in Snowflake as:

```text
ECONOMIC_DATA
```

The current load step writes to the database and schema supplied in `.env` and uses `overwrite=True`, so each pipeline run refreshes the target table completely.

## Notes

- This repo intentionally excludes `.env`, `venv/`, and Python cache artifacts from version control.
- The included dataset and dictionary in `data/` are generated outputs intended to make the project easier to review.
- The source code currently targets U.S. macro data for 2015-2024.
