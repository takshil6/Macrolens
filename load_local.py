# load_local.py
import os
import pandas as pd

def save(df, path="data/economic_indicators.csv"):
    """Save transformed data locally with a data dictionary."""
    os.makedirs("data", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n--- LOAD ---")
    print(f"  Saved {len(df)} rows to {path}")

    # Write data dictionary
    dict_path = "data/data_dictionary.md"
    with open(dict_path, "w") as f:
        f.write("# Data Dictionary — Economic Indicators\n\n")
        f.write("| Column | Description | Source | Unit |\n")
        f.write("|--------|-------------|--------|------|\n")
        cols = {
            "date"              : ("Observation date", "FRED/BLS", "YYYY-MM-DD"),
            "gdp"               : ("Gross Domestic Product", "FRED", "Billions USD"),
            "inflation_cpi"     : ("Consumer Price Index", "FRED", "Index (1982=100)"),
            "federal_funds_rate": ("Fed Funds Rate", "FRED", "Percent"),
            "unemployment"      : ("Unemployment Rate", "FRED", "Percent"),
            "housing_starts"    : ("New Housing Starts", "FRED", "Thousands of units"),
            "nonfarm_payroll"   : ("Total Nonfarm Payroll", "BLS", "Thousands of jobs"),
            "avg_hourly_wage"   : ("Average Hourly Earnings", "BLS", "USD"),
            "labor_force_part"  : ("Labor Force Participation Rate", "BLS", "Percent"),
            "quality_flag"      : ("Row-level quality check result", "Pipeline", "PASS/FAIL"),
            "source"            : ("Data source tag", "Pipeline", "String"),
            "loaded_at"         : ("UTC timestamp of load", "Pipeline", "Timestamp"),
            "year"              : ("Extracted year", "Pipeline", "Integer"),
            "month"             : ("Extracted month", "Pipeline", "Integer"),
        }
        for col, (desc, src, unit) in cols.items():
            f.write(f"| `{col}` | {desc} | {src} | {unit} |\n")
    print(f"  Data dictionary written to {dict_path}")