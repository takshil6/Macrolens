"""
app.py  —  US Economic Pulse  (Streamlit)
Requires: streamlit, plotly, pandas, sqlalchemy, snowflake-sqlalchemy,
          langchain-groq, langchain-community, python-dotenv
"""

import os, re, json, math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv
from agent import build_agent, ask_structured   # see agent.py

load_dotenv()

# ── Page config ───────────────────────────────────────────────
st.set_page_config(page_title="US Economic Pulse", layout="wide", initial_sidebar_state="collapsed")

# ── CSS  (mirrors dashboard.html design system) ───────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.4rem 2rem 3rem !important; max-width: 1560px; }

/* ── Metric cards ── */
.mc {
    background: white; border-radius: 10px; padding: 18px 20px;
    border: 1px solid #e2e6ea; position: relative; overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,.06), 0 2px 8px rgba(0,0,0,.04);
    transition: box-shadow .2s;
}
.mc:hover { box-shadow: 0 4px 16px rgba(0,0,0,.08); }
.mc::before {
    content:''; position:absolute; left:0; top:0; bottom:0; width:3px;
    background: #1a6fd4; border-radius:3px 0 0 3px;
}
.mc.g::before { background:#0a7c59; }
.mc.r::before { background:#c0392b; }
.mc.a::before { background:#c07a00; }
.ml { font-size:10px; font-weight:600; text-transform:uppercase;
      letter-spacing:.07em; color:#8a96a3; margin-bottom:6px; }
.mv { font-size:28px; font-weight:700; letter-spacing:-.5px;
      line-height:1; margin-bottom:6px; color:#0d1f35; }
.md { font-size:11px; font-weight:500; }
.du { color:#0a7c59; }
.dd { color:#c0392b; }
.df { color:#8a96a3; }

/* ── Section headers ── */
.sec-hdr {
    font-size:13px; font-weight:600; color:#0d1f35;
    border-bottom:1px solid #e2e6ea; padding-bottom:10px; margin-bottom:12px;
}

/* ── Chat bubbles ── */
.msg-u {
    background:#0d1f35; color:white;
    border-radius:13px 13px 3px 13px;
    padding:9px 13px; margin:6px 0; margin-left:10%;
    font-size:13px; line-height:1.55;
}
.msg-a {
    background:#f0f2f5; color:#0d1f35;
    border-radius:13px 13px 13px 3px;
    padding:9px 13px; margin:6px 0; margin-right:10%;
    font-size:13px; line-height:1.55;
    border:1px solid #e2e6ea;
}
.pill {
    display:inline-flex; align-items:center;
    border-radius:3px; padding:2px 7px;
    font-size:9px; font-weight:700;
    text-transform:uppercase; letter-spacing:.05em; margin-top:6px;
}
.pill-b { background:#e8f1fb; color:#1a6fd4; }
.pill-g { background:#e6f4f0; color:#0a7c59; }
.pill-a { background:#fef6e4; color:#c07a00; }

/* ── Map mode / control buttons ── */
div[data-testid="stHorizontalBlock"] .stSelectbox label { font-size:11px !important; }
</style>
""", unsafe_allow_html=True)

# ── Top nav bar ───────────────────────────────────────────────
st.markdown("""
<div style="background:#0d1f35;padding:10px 24px;margin:-1.4rem -2rem 1.4rem;
            display:flex;align-items:center;gap:0;">
  <span style="display:flex;align-items:center;gap:8px;font-size:14px;
               font-weight:600;color:#fff;">
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
      <rect x="1" y="7" width="2" height="6" rx=".6" fill="white" opacity=".45"/>
      <rect x="4.2" y="4.5" width="2" height="8.5" rx=".6" fill="white" opacity=".65"/>
      <rect x="7.4" y="2" width="2" height="11" rx=".6" fill="white"/>
      <rect x="10.6" y="5.5" width="2" height="7.5" rx=".6" fill="white" opacity=".55"/>
    </svg>
    Economic Pulse
  </span>
  <span style="width:1px;height:18px;background:rgba(255,255,255,.15);margin:0 14px;"></span>
  <span style="font-size:11px;color:rgba(255,255,255,.5);">US Macro Indicators · 2015–2024</span>
  <span style="flex:1;"></span>
  <span style="display:flex;align-items:center;gap:5px;background:rgba(255,255,255,.08);
               border:1px solid rgba(255,255,255,.12);border-radius:100px;
               padding:4px 11px;font-size:10px;font-weight:600;
               color:rgba(255,255,255,.65);letter-spacing:.05em;text-transform:uppercase;">
    <span style="width:6px;height:6px;border-radius:50%;background:#22c55e;"></span>
    Live · Snowflake
  </span>
</div>
""", unsafe_allow_html=True)

# ── State geographic data ─────────────────────────────────────
STATES = ["AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN",
          "IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV",
          "NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN",
          "TX","UT","VT","VA","WA","WV","WI","WY"]
STATE_NAMES = {
    "AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas","CA":"California",
    "CO":"Colorado","CT":"Connecticut","DE":"Delaware","FL":"Florida","GA":"Georgia",
    "HI":"Hawaii","ID":"Idaho","IL":"Illinois","IN":"Indiana","IA":"Iowa",
    "KS":"Kansas","KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland",
    "MA":"Massachusetts","MI":"Michigan","MN":"Minnesota","MS":"Mississippi","MO":"Missouri",
    "MT":"Montana","NE":"Nebraska","NV":"Nevada","NH":"New Hampshire","NJ":"New Jersey",
    "NM":"New Mexico","NY":"New York","NC":"North Carolina","ND":"North Dakota","OH":"Ohio",
    "OK":"Oklahoma","OR":"Oregon","PA":"Pennsylvania","RI":"Rhode Island","SC":"South Carolina",
    "SD":"South Dakota","TN":"Tennessee","TX":"Texas","UT":"Utah","VT":"Vermont",
    "VA":"Virginia","WA":"Washington","WV":"West Virginia","WI":"Wisconsin","WY":"Wyoming"
}
YEARS = list(range(2015, 2025))

# State structural multipliers ─────────────────────────────────
S_UNEMP = {"AL":1.15,"AK":1.45,"AZ":.95,"AR":1.1,"CA":1.1,"CO":.82,"CT":.95,
           "DE":.88,"FL":.95,"GA":.93,"HI":1.0,"ID":.82,"IL":1.1,"IN":.9,
           "IA":.68,"KS":.73,"KY":1.07,"LA":1.18,"ME":.88,"MD":.83,"MA":.84,
           "MI":1.07,"MN":.73,"MS":1.35,"MO":.9,"MT":.88,"NE":.62,"NV":1.25,
           "NH":.68,"NJ":1.08,"NM":1.28,"NY":1.12,"NC":.9,"ND":.62,"OH":.95,
           "OK":.9,"OR":1.02,"PA":.95,"RI":1.08,"SC":.93,"SD":.62,"TN":.9,
           "TX":.9,"UT":.72,"VT":.78,"VA":.83,"WA":1.0,"WV":1.22,"WI":.78,"WY":.85}
COVID_S  = {"HI":2.9,"NV":2.5,"NY":1.85,"CA":1.65,"MA":1.6,"IL":1.55,"NJ":1.65,
            "WA":1.45,"MI":1.5,"RI":1.4,"CT":1.4,"FL":1.3,"OR":1.35,"CO":1.3,
            "PA":1.3,"MD":1.25,"MN":1.2,"VA":1.2,"GA":1.2,"TX":1.15,"AZ":1.15}
S_GDP   = {"AL":.78,"AK":1.1,"AZ":.9,"AR":.73,"CA":1.35,"CO":1.15,"CT":1.35,
           "DE":1.3,"FL":.95,"GA":.95,"HI":1.0,"ID":.8,"IL":1.1,"IN":.88,
           "IA":.92,"KS":.9,"KY":.8,"LA":.88,"ME":.82,"MD":1.3,"MA":1.45,
           "MI":.9,"MN":1.08,"MS":.68,"MO":.88,"MT":.82,"NE":.95,"NV":.93,
           "NH":1.15,"NJ":1.4,"NM":.75,"NY":1.55,"NC":.93,"ND":1.15,"OH":.9,
           "OK":.88,"OR":1.02,"PA":1.05,"RI":1.0,"SC":.82,"SD":.95,"TN":.88,
           "TX":1.1,"UT":1.0,"VT":.88,"VA":1.15,"WA":1.3,"WV":.73,"WI":.93,"WY":1.05}
S_WAGE  = {"AL":.84,"AK":1.05,"AZ":.9,"AR":.8,"CA":1.28,"CO":1.1,"CT":1.2,
           "DE":1.05,"FL":.88,"GA":.9,"HI":1.08,"ID":.83,"IL":1.05,"IN":.88,
           "IA":.88,"KS":.86,"KY":.83,"LA":.84,"ME":.87,"MD":1.15,"MA":1.3,
           "MI":.95,"MN":1.05,"MS":.78,"MO":.88,"MT":.84,"NE":.88,"NV":.92,
           "NH":1.05,"NJ":1.2,"NM":.82,"NY":1.25,"NC":.88,"ND":.95,"OH":.92,
           "OK":.84,"OR":1.05,"PA":1.0,"RI":1.03,"SC":.83,"SD":.83,"TN":.86,
           "TX":1.0,"UT":.92,"VT":.9,"VA":1.08,"WA":1.2,"WV":.78,"WI":.93,"WY":.92}
S_HOUS  = {"AL":.85,"AK":.45,"AZ":1.6,"AR":.8,"CA":.55,"CO":1.2,"CT":.45,
           "DE":.8,"FL":1.6,"GA":1.35,"HI":.5,"ID":1.45,"IL":.5,"IN":1.0,
           "IA":.75,"KS":.75,"KY":.8,"LA":.7,"ME":.65,"MD":.6,"MA":.45,
           "MI":.65,"MN":.8,"MS":.72,"MO":.85,"MT":.9,"NE":1.05,"NV":1.2,
           "NH":.72,"NJ":.45,"NM":.65,"NY":.3,"NC":1.2,"ND":1.2,"OH":.7,
           "OK":.85,"OR":.85,"PA":.5,"RI":.4,"SC":1.2,"SD":1.3,"TN":1.25,
           "TX":1.5,"UT":1.7,"VT":.55,"VA":.85,"WA":.95,"WV":.42,"WI":.75,"WY":.85}
S_LAB   = {"AL":.96,"AK":1.04,"AZ":.96,"AR":.95,"CA":.97,"CO":1.04,"CT":.99,
           "DE":.98,"FL":.95,"GA":.98,"HI":1.01,"ID":1.0,"IL":.99,"IN":1.0,
           "IA":1.06,"KS":1.03,"KY":.93,"LA":.95,"ME":.98,"MD":1.0,"MA":1.0,
           "MI":.97,"MN":1.07,"MS":.91,"MO":.98,"MT":1.02,"NE":1.07,"NV":.98,
           "NH":1.06,"NJ":.98,"NM":.93,"NY":.96,"NC":.98,"ND":1.1,"OH":.97,
           "OK":.98,"OR":1.0,"PA":.96,"RI":.98,"SC":.96,"SD":1.1,"TN":.96,
           "TX":1.01,"UT":1.06,"VT":1.02,"VA":1.01,"WA":1.01,"WV":.83,"WI":1.04,"WY":1.04}

NAT_YEARLY = {
    "unemployment": [5.3,4.7,4.1,3.7,3.5,8.1,5.4,3.6,3.5,4.0],
    "gdp_pc":       [52,54,56,58.5,61,57.5,63,68,71,74],
    "wages":        [25.0,25.9,26.8,27.6,28.4,30.0,31.8,33.1,34.0,35.0],
    "housing":      [1.1,1.17,1.22,1.26,1.30,1.38,1.60,1.55,1.38,1.40],
    "labor_part":   [62.7,62.8,62.9,63.1,63.3,61.7,61.9,62.2,62.6,62.7],
}

@st.cache_data
def build_state_df() -> pd.DataFrame:
    """Generate synthetic state-level time-series (replace with Snowflake query if you have a state table)."""
    rows = []
    rng = np.random.default_rng(42)
    for yi, yr in enumerate(YEARS):
        for s in STATES:
            cs = COVID_S.get(s, 1.0)
            base_u = NAT_YEARLY["unemployment"][yi] * S_UNEMP.get(s,1)
            u = base_u + (NAT_YEARLY["unemployment"][5]-NAT_YEARLY["unemployment"][4])*(cs-1) if yr==2020 else base_u
            rows.append({
                "state": s,
                "year": yr,
                "unemployment": round(max(1.5, u + rng.normal(0,.15)), 1),
                "gdp_pc":       round(NAT_YEARLY["gdp_pc"][yi]   * S_GDP.get(s,1)  + rng.normal(0,.5), 1),
                "wages":        round(NAT_YEARLY["wages"][yi]     * S_WAGE.get(s,1) + rng.normal(0,.2), 2),
                "housing":      round(NAT_YEARLY["housing"][yi]   * S_HOUS.get(s,1) + abs(rng.normal(0,.03)), 2),
                "labor_part":   round(NAT_YEARLY["labor_part"][yi]* S_LAB.get(s,1)  + rng.normal(0,.2), 1),
            })
    return pd.DataFrame(rows)

# ── Load national data from Snowflake (or fallback to synthetic) ──
@st.cache_data(ttl=300)
def load_national_data():
    try:
        from sqlalchemy import create_engine
        engine = create_engine(
            f"snowflake://{os.getenv('SNOWFLAKE_USER')}:{os.getenv('SNOWFLAKE_PASSWORD')}"
            f"@{os.getenv('SNOWFLAKE_ACCOUNT')}/{os.getenv('SNOWFLAKE_DATABASE')}"
            f"/{os.getenv('SNOWFLAKE_SCHEMA')}?warehouse={os.getenv('SNOWFLAKE_WAREHOUSE')}"
        )
        df = pd.read_sql("SELECT * FROM ECONOMIC_DATA ORDER BY DATE", engine)
        df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        return df, True
    except Exception:
        # Synthetic fallback
        months = pd.date_range("2015-01", "2024-12", freq="MS")
        n = len(months)
        rng = np.random.default_rng(0)
        i = np.arange(n)
        def spike(base,trend,noise,s_idx,s_mag):
            v = base + trend*i + np.sin(i*.4)*noise
            v[s_idx:s_idx+5] += s_mag * np.exp(-.6*np.arange(5))
            return np.round(v + rng.normal(0,.08,n), 2)
        df = pd.DataFrame({
            "date": months,
            "unemployment":       spike(5.6,-.025,.3,62,10),
            "gdp":                np.round(18200+i*100 - np.where((i>60)&(i<65),900,0) + rng.normal(0,150,n)),
            "inflation_cpi":      spike(1.6,.008,.2,75,3.5),
            "federal_funds_rate": [.25 if k<12 else .25+k*.04 if k<36 else max(.08,2.4-k*.01) if k<86 else .08+(k-86)*.18 for k in i],
            "housing_starts":     spike(1100,4,80,62,-320),
            "avg_hourly_wage":    spike(25.2,.12,.3,62,2),
            "nonfarm_payroll":    np.round(143000+i*600 - np.where((i>60)&(i<65),23000,0) + rng.normal(0,400,n)),
            "labor_force_part":   spike(63.2,-.02,.15,62,-3.5),
        })
        return df, False

# ── Chart helpers ─────────────────────────────────────────────
COLORSCALES = {
    "current_unemployment": [[0,'#e8f4ff'],[.5,'#1a6fd4'],[1,'#0a2a5e']],
    "current_gdp_pc":       [[0,'#e6f4f0'],[.5,'#0a7c59'],[1,'#053d2c']],
    "current_wages":        [[0,'#fef6e4'],[.5,'#c07a00'],[1,'#6b4200']],
    "current_housing":      [[0,'#f0eaff'],[.5,'#6d28d9'],[1,'#2d1063']],
    "current_labor_part":   [[0,'#fff0e8'],[.5,'#c0392b'],[1,'#7f1d1d']],
    "change":               [[0,'#c0392b'],[.5,'#f5f5f5'],[1,'#0a7c59']],
    "rank":                 [[0,'#0d1f35'],[.5,'#1a6fd4'],[1,'#e8f4ff']],
}
MAP_LABELS = {
    "unemployment": "Unemployment %",
    "gdp_pc":       "GDP/Capita ($K)",
    "wages":        "Avg Hourly Wage ($/hr)",
    "housing":      "Housing Starts / 1K pop",
    "labor_part":   "Labor Force Participation %",
}
IND_MAP_METRIC = {
    "unemployment":"unemployment","gdp":"gdp_pc","inflation_cpi":"unemployment",
    "federal_funds_rate":"unemployment","housing_starts":"housing",
    "avg_hourly_wage":"wages","nonfarm_payroll":"gdp_pc","labor_force_part":"labor_part"
}

def build_trend_chart(df, indicator, hl_range=None):
    x, y = df["date"], df[indicator]
    unit_map = {"unemployment":"%","gdp":"B USD","inflation_cpi":"Index",
                "federal_funds_rate":"%","housing_starts":"K","avg_hourly_wage":"USD",
                "nonfarm_payroll":"K","labor_force_part":"%"}
    unit = unit_map.get(indicator, "")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines",
                             line=dict(color="#d0daea", width=2, shape="spline"),
                             showlegend=False))
    if hl_range:
        y0, y1 = hl_range
        mask = (df["date"].dt.year >= y0) & (df["date"].dt.year <= y1)
        dh = df[mask]
        if not dh.empty:
            fig.add_trace(go.Scatter(x=dh["date"], y=dh[indicator], mode="lines",
                                     line=dict(color="#1a6fd4", width=3, shape="spline"),
                                     showlegend=False))
            pk = dh[indicator].idxmax()
            fig.add_trace(go.Scatter(
                x=[dh.loc[pk,"date"]], y=[dh.loc[pk,indicator]],
                mode="markers+text",
                marker=dict(size=8, color="#1a6fd4"),
                text=[f"{dh.loc[pk,indicator]:.1f} {unit}"],
                textposition="top center",
                textfont=dict(size=10, color="#1a6fd4", family="Inter"),
                showlegend=False))
        fig.add_vrect(x0=f"{y0}-01-01", x1=f"{y1}-12-31",
                      fillcolor="rgba(26,111,212,0.06)",
                      line=dict(color="rgba(26,111,212,0.25)", width=1))

    fig.update_layout(
        margin=dict(l=48, r=16, t=16, b=44), height=400,
        xaxis=dict(showgrid=False, showline=True, linecolor="#e2e6ea",
                   tickfont=dict(size=10, family="Inter", color="#8a96a3"),
                   tickformat="%Y", dtick="M12"),
        yaxis=dict(showgrid=True, gridcolor="#f0f3f6", zeroline=False,
                   tickfont=dict(size=10, family="Inter", color="#8a96a3"),
                   title=dict(text=unit, font=dict(size=10, color="#8a96a3"))),
        plot_bgcolor="#fff", paper_bgcolor="#fff",
        hovermode="x unified", font=dict(family="Inter"),
    )
    return fig


def build_map_chart(state_df, metric, mode, year, highlight_states=None):
    hl = set(highlight_states or [])
    yr_df   = state_df[state_df["year"] == year].set_index("state")
    prev_yr = max(year - 1, YEARS[0])
    prev_df = state_df[state_df["year"] == prev_yr].set_index("state")

    if mode == "current":
        z = [yr_df.loc[s, metric] if s in yr_df.index else 0 for s in STATES]
        title = f"{MAP_LABELS[metric]}  ·  {year}"
        cscale_key = f"current_{metric}"
    elif mode == "change":
        z = [round(yr_df.loc[s, metric] - prev_df.loc[s, metric], 2)
             if s in yr_df.index and s in prev_df.index else 0 for s in STATES]
        title = f"{MAP_LABELS[metric]}  —  Change {prev_yr}→{year}"
        cscale_key = "change"
    else:  # rank
        raw = {s: yr_df.loc[s, metric] for s in STATES if s in yr_df.index}
        sorted_vals = sorted(raw.values())
        good_low = metric in ("unemployment",)
        z = [sorted_vals.index(raw[s]) + 1 if good_low
             else len(sorted_vals) - sorted_vals.index(raw[s])
             for s in STATES]
        title = f"{MAP_LABELS[metric]}  ·  State Ranking  ·  {year}  (1 = Best)"
        cscale_key = "rank"

    cscale = COLORSCALES.get(cscale_key, COLORSCALES["current_unemployment"])
    hover = [f"<b>{STATE_NAMES.get(s,s)}</b><br>{z[i]}" for i,s in enumerate(STATES)]
    opacity_base = 0.35 if hl else 1.0

    fig = go.Figure()
    fig.add_trace(go.Choropleth(
        locations=STATES, locationmode="USA-states", z=z,
        colorscale=cscale, showscale=True,
        colorbar=dict(thickness=10, len=.55, outlinewidth=0,
                      tickfont=dict(size=9, family="Inter")),
        marker=dict(line=dict(color="#fff", width=.8), opacity=opacity_base),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover
    ))

    if hl:
        hl_idx  = [i for i,s in enumerate(STATES) if s in hl]
        fig.add_trace(go.Choropleth(
            locations=[STATES[i] for i in hl_idx],
            locationmode="USA-states",
            z=[z[i] for i in hl_idx],
            colorscale=cscale, showscale=False,
            marker=dict(line=dict(color="#fff", width=2), opacity=1.0),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=[hover[i] for i in hl_idx]
        ))

    fig.update_layout(
        geo=dict(scope="usa", showlakes=True, lakecolor="#eef2f7",
                 bgcolor="#fff", subunitcolor="#fff"),
        margin=dict(t=36, b=4, l=0, r=0), height=400,
        paper_bgcolor="#fff", font=dict(family="Inter"),
        title=dict(text=title, font=dict(size=12, color="#0d1f35", family="Inter"),
                   x=0.01, y=.97, xanchor="left", yanchor="top"),
    )
    return fig

# ── Session state ─────────────────────────────────────────────
defaults = dict(
    messages=[], indicator="unemployment", hl_range=None,
    map_metric="unemployment", map_mode="current",
    map_year=2024, highlight_states=[], agent=None
)
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Load data ─────────────────────────────────────────────────
df, live = load_national_data()
state_df = build_state_df()
latest   = df.iloc[-1]

# ── Lazy-load agent ───────────────────────────────────────────
if st.session_state.agent is None:
    with st.spinner("Connecting to Snowflake & loading agent…"):
        try:
            st.session_state.agent = build_agent()
        except Exception:
            st.session_state.agent = "unavailable"

# ── METRIC CARDS ──────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, cls, lbl, val, delta, dcls in [
    (c1, "",  "GDP",              f"${latest.gdp/1000:.1f}T", "↑ 2.3% YoY",      "du"),
    (c2, "g", "Unemployment",     f"{latest.unemployment:.1f}%","↓ 0.2 pts",       "du"),
    (c3, "r", "Inflation (CPI)",  f"{latest.inflation_cpi:.1f}", "↑ 0.1 pts",      "dd"),
    (c4, "a", "Fed Funds Rate",   f"{latest.federal_funds_rate:.2f}%","— Held steady","df"),
]:
    col.markdown(f"""
    <div class="mc {cls}">
        <div class="ml">{lbl}</div>
        <div class="mv">{val}</div>
        <div class="md {dcls}">{delta}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

# ── MAIN LAYOUT ───────────────────────────────────────────────
left, right = st.columns([2.1, 1], gap="large")

# ────────────────────────────────────────────────────────
# LEFT — visualizations
# ────────────────────────────────────────────────────────
with left:
    tab_trend, tab_map = st.tabs(["📈  Trend Chart", "🗺️  State Map"])

    # ── TREND TAB ─────────────────────────────────────────
    with tab_trend:
        ind_opts = {
            "Unemployment Rate":       "unemployment",
            "GDP (Billions USD)":      "gdp",
            "Inflation (CPI)":         "inflation_cpi",
            "Fed Funds Rate":          "federal_funds_rate",
            "Housing Starts":          "housing_starts",
            "Avg Hourly Wage":         "avg_hourly_wage",
            "Nonfarm Payroll":         "nonfarm_payroll",
            "Labor Force Participation":"labor_force_part",
        }
        sel = st.selectbox("Indicator", list(ind_opts.keys()),
                           index=list(ind_opts.values()).index(st.session_state.indicator),
                           label_visibility="collapsed")
        st.session_state.indicator = ind_opts[sel]

        fig_trend = build_trend_chart(df, st.session_state.indicator, st.session_state.hl_range)
        st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})

        if st.session_state.hl_range:
            y0, y1 = st.session_state.hl_range
            st.markdown(f"""<span class="pill pill-b">● AI Highlighted: {y0}–{y1}</span>""",
                        unsafe_allow_html=True)

    # ── MAP TAB ───────────────────────────────────────────
    with tab_map:
        mc1, mc2, mc3 = st.columns([1, 1, 1])
        with mc1:
            map_metric_label = st.selectbox(
                "Map metric",
                ["Unemployment", "GDP per Capita", "Wages", "Housing Starts", "Labor Participation"],
                index=["unemployment","gdp_pc","wages","housing","labor_part"].index(st.session_state.map_metric),
                label_visibility="collapsed"
            )
            st.session_state.map_metric = {
                "Unemployment":"unemployment","GDP per Capita":"gdp_pc",
                "Wages":"wages","Housing Starts":"housing","Labor Participation":"labor_part"
            }[map_metric_label]
        with mc2:
            map_mode_label = st.selectbox("Mode", ["Current","YoY Change","Ranking"],
                                          index=["current","change","rank"].index(st.session_state.map_mode),
                                          label_visibility="collapsed")
            st.session_state.map_mode = {"Current":"current","YoY Change":"change","Ranking":"rank"}[map_mode_label]
        with mc3:
            map_year = st.select_slider("Year", options=YEARS,
                                        value=st.session_state.map_year, label_visibility="collapsed")
            st.session_state.map_year = map_year

        fig_map = build_map_chart(
            state_df, st.session_state.map_metric,
            st.session_state.map_mode, st.session_state.map_year,
            st.session_state.highlight_states
        )
        st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

        if st.session_state.highlight_states:
            hl_names = ", ".join(STATE_NAMES.get(s,s) for s in st.session_state.highlight_states[:5])
            st.markdown(f"""<span class="pill pill-g">● Focus states: {hl_names}</span>""",
                        unsafe_allow_html=True)
        if not live:
            st.caption("⚠️ Snowflake unreachable — showing synthetic data. Check .env credentials.")

# ────────────────────────────────────────────────────────
# RIGHT — AI chat
# ────────────────────────────────────────────────────────
with right:
    st.markdown('<div class="sec-hdr">AI Economic Agent</div>', unsafe_allow_html=True)

    # Chat history
    chat_html = ""
    if not st.session_state.messages:
        chat_html += """<div class="msg-a">Ask me anything about US economic data 2015–2024.<br><br>
I'll update the <strong>trend chart</strong> and <strong>state map</strong> based on your query — choosing the right mode, year, and metric automatically.</div>"""
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_html += f'<div class="msg-u">{msg["content"]}</div>'
        else:
            pills = ""
            if msg.get("indicator_pill"):
                pills += f'<div class="pill pill-b">{msg["indicator_pill"]}</div>'
            if msg.get("map_pill"):
                pills += f'<div class="pill pill-a">{msg["map_pill"]}</div>'
            if msg.get("state_pill"):
                pills += f'<div class="pill pill-g">{msg["state_pill"]}</div>'
            chat_html += f'<div class="msg-a">{msg["content"]}{pills}</div>'

    st.markdown(
        f'<div style="height:440px;overflow-y:auto;padding:4px 2px;">{chat_html}</div>',
        unsafe_allow_html=True
    )

    question = st.chat_input("Ask about any indicator or state…")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("Thinking…"):
            if st.session_state.agent == "unavailable":
                # Pure fallback
                from agent import fallback_response
                result = fallback_response(question)
            else:
                result = ask_structured(st.session_state.agent, question)

        # Apply national indicator & highlight
        if result.get("indicator") and result["indicator"] in [
            "unemployment","gdp","inflation_cpi","federal_funds_rate",
            "housing_starts","avg_hourly_wage","nonfarm_payroll","labor_force_part"
        ]:
            st.session_state.indicator = result["indicator"]

        if result.get("year_range"):
            st.session_state.hl_range = result["year_range"]

        # Apply map spec
        ms = result.get("map_spec", {})
        if ms.get("metric") and ms["metric"] in ["unemployment","gdp_pc","wages","housing","labor_part"]:
            st.session_state.map_metric = ms["metric"]
        if ms.get("mode") in ["current","change","rank"]:
            st.session_state.map_mode = ms["mode"]
        if ms.get("year") and ms["year"] in YEARS:
            st.session_state.map_year = ms["year"]
        st.session_state.highlight_states = ms.get("highlight_states", [])

        # Build pills for message
        ind_label = {
            "unemployment":"Unemployment","gdp":"GDP","inflation_cpi":"Inflation",
            "federal_funds_rate":"Fed Rate","housing_starts":"Housing Starts",
            "avg_hourly_wage":"Avg Wage","nonfarm_payroll":"Payrolls","labor_force_part":"Labor Part."
        }.get(st.session_state.indicator, st.session_state.indicator)
        yr_label = f" · {st.session_state.hl_range[0]}–{st.session_state.hl_range[1]}" if st.session_state.hl_range else ""
        mode_label = {"current":"Snapshot","change":"YoY Change","rank":"Ranking"}.get(st.session_state.map_mode,"")

        st.session_state.messages.append({
            "role": "assistant",
            "content": result.get("answer", "I encountered an issue processing that request."),
            "indicator_pill": f"{ind_label}{yr_label}",
            "map_pill": f"Map: {MAP_LABELS.get(st.session_state.map_metric,'')} · {mode_label}",
            "state_pill": (f"Focus: {' '.join(st.session_state.highlight_states[:4])}"
                           if st.session_state.highlight_states else ""),
        })
        st.rerun()
