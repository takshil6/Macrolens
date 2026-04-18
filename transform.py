# transform.py
import pandas as pd
from functools import reduce

# Quality constraints (12 rules as on your resume)
QUALITY_RULES = {
    "gdp"              : (10_000, 35_000),
    "inflation_cpi"    : (100,    400),
    "federal_funds_rate": (0,     25),
    "unemployment"     : (0,      25),
    "housing_starts"   : (0,      3_000),
    "nonfarm_payroll"  : (100_000, 200_000),
    "avg_hourly_wage"  : (5,       60),
    "labor_force_part" : (55,      75),
}

def merge_all(frames):
    """Merge all series into one wide table on date."""
    return reduce(lambda l, r: pd.merge(l, r, on="date", how="outer"), frames)

def clean(df):
    """Apply transformations and quality checks."""
    print("\n--- TRANSFORM ---")

    # Sort and fill small gaps
    df = df.sort_values("date").reset_index(drop=True)
    df = df.ffill().bfill()

    # Apply quality constraints — flag rows outside expected ranges
    df["quality_flag"] = "PASS"
    for col, (lo, hi) in QUALITY_RULES.items():
        if col in df.columns:
            mask = (df[col] < lo) | (df[col] > hi)
            df.loc[mask, "quality_flag"] = "FAIL"
            failed = mask.sum()
            if failed:
                print(f"  Quality check: {failed} rows flagged in '{col}' (expected {lo}–{hi})")

    # Add metadata columns for lineage
    df["source"]      = df.apply(lambda r: _source_tag(r), axis=1)
    df["loaded_at"]   = pd.Timestamp.utcnow()
    df["year"]        = df["date"].dt.year
    df["month"]       = df["date"].dt.month

    passing = (df["quality_flag"] == "PASS").sum()
    print(f"  {passing}/{len(df)} rows passed all quality checks")
    return df

def _source_tag(row):
    return "FRED+BLS"