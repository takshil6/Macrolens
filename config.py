# config.py
import os
from dotenv import load_dotenv

load_dotenv()

FRED_API_KEY = os.getenv("FRED_API_KEY")
BLS_API_KEY  = os.getenv("BLS_API_KEY")

# Economic indicators we'll track (8 categories)
FRED_SERIES = {
    "gdp"              : "GDP",
    "inflation_cpi"    : "CPIAUCSL",
    "federal_funds_rate": "FEDFUNDS",
    "unemployment"     : "UNRATE",
    "housing_starts"   : "HOUST",
}

BLS_SERIES = {
    "nonfarm_payroll"  : "CES0000000001",
    "avg_hourly_wage"  : "CES0500000003",
    "labor_force_part" : "LNS11300000",
}

START_DATE = "2015-01-01"
END_DATE   = "2024-12-31"