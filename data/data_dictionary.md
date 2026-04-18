# Data Dictionary — Economic Indicators

| Column | Description | Source | Unit |
|--------|-------------|--------|------|
| `date` | Observation date | FRED/BLS | YYYY-MM-DD |
| `gdp` | Gross Domestic Product | FRED | Billions USD |
| `inflation_cpi` | Consumer Price Index | FRED | Index (1982=100) |
| `federal_funds_rate` | Fed Funds Rate | FRED | Percent |
| `unemployment` | Unemployment Rate | FRED | Percent |
| `housing_starts` | New Housing Starts | FRED | Thousands of units |
| `nonfarm_payroll` | Total Nonfarm Payroll | BLS | Thousands of jobs |
| `avg_hourly_wage` | Average Hourly Earnings | BLS | USD |
| `labor_force_part` | Labor Force Participation Rate | BLS | Percent |
| `quality_flag` | Row-level quality check result | Pipeline | PASS/FAIL |
| `source` | Data source tag | Pipeline | String |
| `loaded_at` | UTC timestamp of load | Pipeline | Timestamp |
| `year` | Extracted year | Pipeline | Integer |
| `month` | Extracted month | Pipeline | Integer |
