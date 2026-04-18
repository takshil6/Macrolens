# extract.py
import requests
import pandas as pd
import json
from config import FRED_API_KEY, BLS_API_KEY, FRED_SERIES, BLS_SERIES, START_DATE, END_DATE

def fetch_fred(series_name, series_id):
    """Pull a single series from the FRED API."""
    print(f"  Fetching FRED: {series_name}...")
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id"       : series_id,
        "api_key"         : FRED_API_KEY,
        "file_type"       : "json",
        "observation_start": START_DATE,
        "observation_end" : END_DATE,
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()["observations"]

    df = pd.DataFrame(data)[["date", "value"]]
    df.columns = ["date", series_name]
    df["date"] = pd.to_datetime(df["date"])
    df[series_name] = pd.to_numeric(df[series_name], errors="coerce")
    return df

def fetch_bls(series_name, series_id):
    """Pull a single series from the BLS API."""
    print(f"  Fetching BLS:  {series_name}...")
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    headers = {"Content-type": "application/json"}
    payload = json.dumps({
        "seriesid"  : [series_id],
        "startyear" : START_DATE[:4],
        "endyear"   : END_DATE[:4],
        "registrationkey": BLS_API_KEY,
    })
    response = requests.post(url, data=payload, headers=headers)
    response.raise_for_status()
    series_data = response.json()["Results"]["series"][0]["data"]

    rows = []
    for item in series_data:
        rows.append({
            "date"      : f"{item['year']}-{item['period'].replace('M','')}-01",
            series_name : float(item["value"]),
        })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

def extract_all():
    """Run all extractions and return raw DataFrames."""
    print("\n--- EXTRACT ---")
    fred_frames = [fetch_fred(name, sid) for name, sid in FRED_SERIES.items()]
    bls_frames  = [fetch_bls(name, sid)  for name, sid in BLS_SERIES.items()]
    return fred_frames + bls_frames