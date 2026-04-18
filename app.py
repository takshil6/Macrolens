# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os, re, time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from sqlalchemy import create_engine

load_dotenv()

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Economic Indicators AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  #MainMenu, footer, header {visibility: hidden;}
  .block-container {padding: 1rem 1.5rem !important;}
  .metric-card {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
  }
  .metric-label { font-size: 12px; color: #6c757d; margin-bottom: 4px; }
  .metric-value { font-size: 22px; font-weight: 600; color: #212529; }
  .metric-delta { font-size: 12px; margin-top: 2px; }
  .delta-up   { color: #0F6E56; }
  .delta-down { color: #A32D2D; }
  .chat-msg-user {
    background: #E1F5EE;
    color: #085041;
    border-radius: 12px 12px 2px 12px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 14px;
    max-width: 80%;
    margin-left: auto;
  }
  .chat-msg-ai {
    background: #f1f3f5;
    color: #212529;
    border-radius: 12px 12px 12px 2px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 14px;
    max-width: 90%;
  }
  .highlight-pill {
    display: inline-block;
    background: #FAEEDA;
    color: #633806;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 12px;
    font-weight: 600;
    margin-top: 6px;
  }
  .stTextInput > div > div > input {
    border-radius: 20px !important;
    border: 1.5px solid #dee2e6 !important;
    padding: 10px 16px !important;
  }
</style>
""", unsafe_allow_html=True)

# ── Load data from Snowflake ──────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    engine = create_engine(
        f"snowflake://{os.getenv('SNOWFLAKE_USER')}:{os.getenv('SNOWFLAKE_PASSWORD')}"
        f"@{os.getenv('SNOWFLAKE_ACCOUNT')}/{os.getenv('SNOWFLAKE_DATABASE')}"
        f"/{os.getenv('SNOWFLAKE_SCHEMA')}?warehouse={os.getenv('SNOWFLAKE_WAREHOUSE')}"
    )
    df = pd.read_sql("SELECT * FROM ECONOMIC_DATA ORDER BY DATE", engine)
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    return df

# ── Build LangChain agent ─────────────────────────────────────
@st.cache_resource
def build_agent():
    engine = create_engine(
        f"snowflake://{os.getenv('SNOWFLAKE_USER')}:{os.getenv('SNOWFLAKE_PASSWORD')}"
        f"@{os.getenv('SNOWFLAKE_ACCOUNT')}/{os.getenv('SNOWFLAKE_DATABASE')}"
        f"/{os.getenv('SNOWFLAKE_SCHEMA')}?warehouse={os.getenv('SNOWFLAKE_WAREHOUSE')}"
    )
    db  = SQLDatabase(engine, include_tables=["economic_data"], sample_rows_in_table_info=3)
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile", temperature=0)
    return create_sql_agent(llm=llm, db=db, agent_type="openai-tools",
                            verbose=False, handle_parsing_errors=True)

# ── Chart highlight detector ──────────────────────────────────
def detect_highlight(question, answer):
    """Guess which indicator and year range to highlight from the question."""
    q = question.lower()
    a = answer.lower()

    indicator = "unemployment"
    if any(w in q+a for w in ["gdp","gross domestic"]):       indicator = "gdp"
    elif any(w in q+a for w in ["inflation","cpi","price"]):  indicator = "inflation_cpi"
    elif any(w in q+a for w in ["fed","interest","rate"]):    indicator = "federal_funds_rate"
    elif any(w in q+a for w in ["housing","house","starts"]): indicator = "housing_starts"
    elif any(w in q+a for w in ["wage","earning","pay"]):     indicator = "avg_hourly_wage"
    elif any(w in q+a for w in ["payroll","jobs","employ"]):  indicator = "nonfarm_payroll"
    elif any(w in q+a for w in ["labor","participation"]):    indicator = "labor_force_part"

    years = re.findall(r'\b(20\d{2})\b', question + " " + answer)
    year_range = (int(min(years)), int(max(years))) if len(years) >= 2 else \
                 (int(years[0]), int(years[0])) if years else (2015, 2024)

    return indicator, year_range

# ── Build Plotly chart ────────────────────────────────────────
def build_chart(df, indicator, highlight_range=None, title=None):
    col_labels = {
        "gdp"              : ("GDP", "Billions USD"),
        "inflation_cpi"    : ("Inflation (CPI)", "Index"),
        "federal_funds_rate": ("Fed Funds Rate", "%"),
        "unemployment"     : ("Unemployment Rate", "%"),
        "housing_starts"   : ("Housing Starts", "Thousands"),
        "avg_hourly_wage"  : ("Avg Hourly Wage", "USD"),
        "nonfarm_payroll"  : ("Nonfarm Payroll", "Thousands"),
        "labor_force_part" : ("Labor Force Participation", "%"),
    }
    label, unit = col_labels.get(indicator, (indicator, ""))

    fig = go.Figure()

    # Base line
    fig.add_trace(go.Scatter(
        x=df["date"], y=df[indicator],
        mode="lines",
        line=dict(color="#CBD5E0", width=1.5),
        name=label,
        showlegend=False,
    ))

    # Highlighted region
    if highlight_range:
        y0, y1 = highlight_range
        mask = (df["date"].dt.year >= y0) & (df["date"].dt.year <= y1)
        df_hl = df[mask]
        if not df_hl.empty:
            fig.add_trace(go.Scatter(
                x=df_hl["date"], y=df_hl[indicator],
                mode="lines+markers",
                line=dict(color="#1D9E75", width=3),
                marker=dict(size=5, color="#1D9E75"),
                name="Highlighted",
                showlegend=False,
            ))
            # Peak marker
            peak_idx = df_hl[indicator].idxmax()
            fig.add_annotation(
                x=df_hl.loc[peak_idx, "date"],
                y=df_hl.loc[peak_idx, indicator],
                text=f"  {df_hl.loc[peak_idx, indicator]:.1f} {unit}",
                showarrow=True, arrowhead=2,
                arrowcolor="#1D9E75", font=dict(size=12, color="#1D9E75"),
                bgcolor="white", bordercolor="#1D9E75", borderwidth=1,
            )

    fig.update_layout(
        title=dict(text=title or label, font=dict(size=15, color="#212529"), x=0),
        xaxis=dict(showgrid=False, tickfont=dict(size=11)),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", tickfont=dict(size=11),
                   title=dict(text=unit, font=dict(size=11))),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=50, r=20, t=50, b=40),
        height=340,
        hovermode="x unified",
    )
    return fig

# ── Session state ─────────────────────────────────────────────
if "messages"   not in st.session_state: st.session_state.messages   = []
if "indicator"  not in st.session_state: st.session_state.indicator  = "unemployment"
if "hl_range"   not in st.session_state: st.session_state.hl_range   = None
if "chart_title"not in st.session_state: st.session_state.chart_title= None

# ── Load data ─────────────────────────────────────────────────
with st.spinner("Connecting to Snowflake..."):
    df = load_data()

latest = df.iloc[-1]

# ══════════════════════════════════════════════════════════════
# LAYOUT — two columns
# ══════════════════════════════════════════════════════════════
left, right = st.columns([1.6, 1], gap="large")

# ── LEFT PANEL ────────────────────────────────────────────────
with left:
    st.markdown("### Economic indicators dashboard")
    st.caption(f"Live from Snowflake · {len(df)} data points · 2015–2024")

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">GDP (latest)</div>
            <div class="metric-value">${latest.gdp/1000:.1f}T</div>
            <div class="metric-delta delta-up">+2.3% YoY</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Unemployment</div>
            <div class="metric-value">{latest.unemployment:.1f}%</div>
            <div class="metric-delta delta-up">-0.3 pts</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Inflation (CPI)</div>
            <div class="metric-value">{latest.inflation_cpi:.1f}</div>
            <div class="metric-delta delta-down">+3.2% YoY</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Fed funds rate</div>
            <div class="metric-value">{latest.federal_funds_rate:.2f}%</div>
            <div class="metric-delta delta-down">Held steady</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Indicator selector
    indicator_options = {
        "Unemployment rate"       : "unemployment",
        "GDP"                     : "gdp",
        "Inflation (CPI)"         : "inflation_cpi",
        "Fed funds rate"          : "federal_funds_rate",
        "Housing starts"          : "housing_starts",
        "Avg hourly wage"         : "avg_hourly_wage",
        "Nonfarm payroll"         : "nonfarm_payroll",
        "Labor force participation": "labor_force_part",
    }
    selected_label = st.selectbox(
        "Chart indicator",
        list(indicator_options.keys()),
        index=list(indicator_options.values()).index(st.session_state.indicator),
        label_visibility="collapsed",
    )
    st.session_state.indicator = indicator_options[selected_label]

    # Chart
    fig = build_chart(
        df,
        st.session_state.indicator,
        st.session_state.hl_range,
        st.session_state.chart_title,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    if st.session_state.hl_range:
        st.info(f"Highlighted: {st.session_state.hl_range[0]}–{st.session_state.hl_range[1]} · Ask another question to update")

# ── RIGHT PANEL ───────────────────────────────────────────────
with right:
    st.markdown("### AI agent")
    st.caption("Ask anything · powered by Groq + LangChain")

    # Chat history
    chat_container = st.container(height=420)
    with chat_container:
        if not st.session_state.messages:
            st.markdown("""<div class="chat-msg-ai">
                Hi! Ask me anything about US economic data from 2015–2024.
                The chart on the left will highlight the answer automatically.
                <br><br>Try: <i>"What happened to unemployment in 2020?"</i>
            </div>""", unsafe_allow_html=True)
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-msg-user">{msg["content"]}</div>',
                            unsafe_allow_html=True)
            else:
                pill = f'<br><span class="highlight-pill">{msg.get("pill","")}</span>' \
                       if msg.get("pill") else ""
                st.markdown(f'<div class="chat-msg-ai">{msg["content"]}{pill}</div>',
                            unsafe_allow_html=True)

    # Input
    question = st.chat_input("Ask about the data...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("Querying Snowflake..."):
            agent = build_agent()
            try:
                result  = agent.invoke({"input": question})
                answer  = result["output"]
            except Exception as e:
                answer = f"Sorry, I ran into an issue: {str(e)}"

        # Update chart highlight
        indicator, hl_range = detect_highlight(question, answer)
        st.session_state.indicator   = indicator
        st.session_state.hl_range    = hl_range
        st.session_state.chart_title = f"{selected_label} — highlighted by AI"

        # Build pill summary
        pill = f"{indicator.replace('_',' ').title()} · {hl_range[0]}–{hl_range[1]}"

        st.session_state.messages.append({
            "role"   : "assistant",
            "content": answer,
            "pill"   : pill,
        })
        st.rerun()