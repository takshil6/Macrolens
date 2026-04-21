# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os, re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from sqlalchemy import create_engine

load_dotenv()

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="US Economic Pulse",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Apple-style Custom CSS ───────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@400;500;600&display=swap');
    
    :root {
        --apple-blue: #0071e3;
    }
    
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding: 2rem 3rem !important;}
    
    body {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Helvetica Neue", sans-serif;
        background: #f5f5f7;
    }
    
    /* Premium Header */
    .apple-header {
        background: rgba(255,255,255,0.92);
        backdrop-filter: blur(20px);
        border-bottom: 1px solid #d2d2d7;
        padding: 1rem 2rem;
        margin: -2rem -3rem 2rem;
        box-shadow: 0 1px 0 rgba(0,0,0,0.1);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .logo {
        font-size: 28px;
        font-weight: 600;
        color: #1d1d1f;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Metric Cards - Apple style */
    .metric-card {
        background: white;
        border-radius: 20px;
        padding: 28px 32px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        text-align: center;
        transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.12);
    }
    .metric-label { 
        font-size: 15px; 
        color: #6e6e73; 
        font-weight: 500; 
        margin-bottom: 6px; 
    }
    .metric-value { 
        font-size: 42px; 
        font-weight: 600; 
        letter-spacing: -1px; 
        color: #1d1d1f; 
    }
    .metric-delta {
        font-size: 15px;
        margin-top: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 4px;
    }
    .delta-up { color: #00b36b; }
    .delta-down { color: #ff3b30; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: white;
        border-radius: 9999px;
        padding: 6px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        width: fit-content;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 9999px;
        font-weight: 600;
        font-size: 17px;
        padding: 12px 32px;
    }
    .stTabs [aria-selected="true"] {
        background: #0071e3 !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(0,113,227,0.3) !important;
    }
    
    /* Chart container */
    .plotly-chart {
        border-radius: 28px;
        overflow: hidden;
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
    }
    
    /* Chat styling */
    .chat-msg-user {
        background: #0071e3;
        color: white;
        border-radius: 20px 20px 4px 20px;
        padding: 14px 20px;
        margin: 12px 0;
        max-width: 80%;
        margin-left: auto;
        font-size: 16px;
    }
    .chat-msg-ai {
        background: #f2f2f7;
        color: #1d1d1f;
        border-radius: 20px 20px 20px 4px;
        padding: 14px 20px;
        margin: 12px 0;
        max-width: 80%;
        font-size: 16px;
    }
    
    .stChatInput input {
        border-radius: 9999px !important;
        padding: 18px 24px !important;
        border: 1.5px solid #d2d2d7 !important;
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
    db = SQLDatabase(engine, include_tables=["economic_data"], sample_rows_in_table_info=3)
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile", temperature=0)
    return create_sql_agent(llm=llm, db=db, agent_type="openai-tools",
                            verbose=False, handle_parsing_errors=True)

# ── Chart highlight detector ──────────────────────────────────
def detect_highlight(question, answer):
    q = question.lower()
    a = answer.lower()

    indicator = "unemployment"
    if any(w in q + a for w in ["gdp", "gross domestic"]):       indicator = "gdp"
    elif any(w in q + a for w in ["inflation", "cpi", "price"]):  indicator = "inflation_cpi"
    elif any(w in q + a for w in ["fed", "interest", "rate"]):    indicator = "federal_funds_rate"
    elif any(w in q + a for w in ["housing", "house", "starts"]): indicator = "housing_starts"
    elif any(w in q + a for w in ["wage", "earning", "pay"]):     indicator = "avg_hourly_wage"
    elif any(w in q + a for w in ["payroll", "jobs", "employ"]):  indicator = "nonfarm_payroll"
    elif any(w in q + a for w in ["labor", "participation"]):     indicator = "labor_force_part"

    years = re.findall(r'\b(20\d{2})\b', question + " " + answer)
    year_range = (int(min(years)), int(max(years))) if len(years) >= 2 else \
                 (int(years[0]), int(years[0])) if years else (2015, 2024)

    return indicator, year_range

# ── Updated Apple-style Plotly Chart ──────────────────────────
def build_chart(df, indicator, highlight_range=None, title=None):
    col_labels = {
        "gdp": ("GDP", "Billions USD"),
        "inflation_cpi": ("Inflation (CPI)", "Index"),
        "federal_funds_rate": ("Fed Funds Rate", "%"),
        "unemployment": ("Unemployment Rate", "%"),
        "housing_starts": ("Housing Starts", "Thousands"),
        "avg_hourly_wage": ("Avg Hourly Wage", "USD"),
        "nonfarm_payroll": ("Nonfarm Payroll", "Thousands"),
        "labor_force_part": ("Labor Force Participation", "%"),
    }
    label, unit = col_labels.get(indicator, (indicator, ""))

    fig = go.Figure()

    # Clean Apple-style base line
    fig.add_trace(go.Scatter(
        x=df["date"], y=df[indicator],
        mode="lines",
        line=dict(color="#0071e3", width=3.5, shape="spline"),
        name=label,
        showlegend=False,
    ))

    # Highlight region
    if highlight_range:
        y0, y1 = highlight_range
        mask = (df["date"].dt.year >= y0) & (df["date"].dt.year <= y1)
        df_hl = df[mask]
        if not df_hl.empty:
            fig.add_trace(go.Scatter(
                x=df_hl["date"], y=df_hl[indicator],
                mode="lines+markers",
                line=dict(color="#00b36b", width=4.5),
                marker=dict(size=6, color="#00b36b"),
                name="Highlighted",
                showlegend=False,
            ))
            # Peak annotation
            peak_idx = df_hl[indicator].idxmax()
            fig.add_annotation(
                x=df_hl.loc[peak_idx, "date"],
                y=df_hl.loc[peak_idx, indicator],
                text=f"  {df_hl.loc[peak_idx, indicator]:.1f} {unit}",
                showarrow=True,
                arrowhead=2,
                arrowcolor="#00b36b",
                font=dict(size=13, color="#00b36b"),
                bgcolor="white",
                bordercolor="#00b36b",
                borderwidth=1.5,
            )

    fig.update_layout(
        title=dict(text=title or label, font=dict(size=24, color="#1d1d1f", family="SF Pro Display"), x=0),
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0", tickfont=dict(size=13)),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", tickfont=dict(size=13),
                   title=dict(text=unit, font=dict(size=13))),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=50, r=30, t=30, b=60),
        height=460,
        hovermode="x unified",
        font=dict(family="SF Pro Display")
    )
    return fig

# ── Choropleth Map (US States) ────────────────────────────────
def build_choropleth():
    # Sample 2024 state unemployment data (you can replace with real Snowflake query if you have a state table)
    locations = ["CA","TX","NY","FL","IL","PA","OH","GA","NC","MI","NJ","VA","WA","AZ","MA","TN","IN","MO","MD","WI","CO","MN","SC","AL","LA","KY","OR","OK","CT","UT","IA","NV","AR","MS","KS","NM","NE","ID","WV","HI","NH","ME","MT","RI","DE","SD","ND","AK","VT","WY"]
    unemp = [4.8,4.1,4.3,3.9,4.7,4.0,4.2,3.8,3.7,4.5,4.4,3.6,4.6,4.0,3.5,3.9,3.8,3.7,3.4,3.6,3.9,3.5,3.8,3.6,4.3,4.1,4.2,3.7,3.9,3.2,3.4,5.1,3.5,4.0,3.3,4.4,3.1,3.9,4.8,3.2,2.9,3.5,3.8,3.6,3.4,2.8,2.7,4.9,2.6,3.0]

    fig = go.Figure(go.Choropleth(
        locations=locations,
        locationmode="USA-states",
        z=unemp,
        colorscale=[[0, '#e3f0ff'], [0.5, '#0071e3'], [1, '#003d80']],
        colorbar=dict(title="Unemployment %", thickness=15),
        hovertemplate="<b>%{location}</b><br>Unemployment: %{z}%<extra></extra>"
    ))

    fig.update_layout(
        geo=dict(scope="usa", showlakes=True, lakecolor="#f5f5f7"),
        height=460,
        margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor="white",
        plot_bgcolor="white"
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

# ── Apple-style Header ────────────────────────────────────────
st.markdown("""
<div class="apple-header">
    <div class="logo">
        <span>Pulse</span>
        <span style="width:8px;height:8px;background:#0071e3;border-radius:50%;"></span>
    </div>
    <div style="font-size:24px;font-weight:600;color:#1d1d1f;">US Economic Pulse</div>
    <div style="display:flex;align-items:center;gap:8px;font-size:15px;color:#0071e3;font-weight:500;">
        <span style="display:inline-block;width:8px;height:8px;background:#0071e3;border-radius:50%;animation:pulse 2s infinite;"></span>
        LIVE • Snowflake • 2024
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;font-size:52px;font-weight:700;letter-spacing:-1.5px;color:#1d1d1f;margin-bottom:8px;'>US Economic Pulse</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;font-size:24px;color:#6e6e73;margin-bottom:40px;'>Real-time macro data. Beautifully designed. Powered by Groq + Snowflake.</p>", unsafe_allow_html=True)

# ── Metrics ───────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">GDP (Latest)</div>
        <div class="metric-value">${latest.gdp/1000:.1f}T</div>
        <div class="metric-delta delta-up">↑ 2.3% YoY</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Unemployment Rate</div>
        <div class="metric-value">{latest.unemployment:.1f}%</div>
        <div class="metric-delta delta-up">↓ 0.2 pts</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Inflation (CPI)</div>
        <div class="metric-value">{latest.inflation_cpi:.1f}</div>
        <div class="metric-delta delta-down">↑ 0.1 pts</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Fed Funds Rate</div>
        <div class="metric-value">{latest.federal_funds_rate:.2f}%</div>
        <div class="metric-delta">Held steady</div>
    </div>
    """, unsafe_allow_html=True)

# ── Two-column layout ─────────────────────────────────────────
left, right = st.columns([2.1, 1], gap="large")

# ── LEFT PANEL (Visualizations) ───────────────────────────────
with left:
    tab1, tab2 = st.tabs(["📈 Trends", "🗺️ US Map"])

    # Trends Tab
    with tab1:
        # Indicator selector
        indicator_options = {
            "Unemployment rate": "unemployment",
            "GDP": "gdp",
            "Inflation (CPI)": "inflation_cpi",
            "Fed funds rate": "federal_funds_rate",
            "Housing starts": "housing_starts",
            "Avg hourly wage": "avg_hourly_wage",
            "Nonfarm payroll": "nonfarm_payroll",
            "Labor force participation": "labor_force_part",
        }
        selected_label = st.selectbox(
            "Choose indicator",
            list(indicator_options.keys()),
            index=list(indicator_options.values()).index(st.session_state.indicator),
            label_visibility="collapsed"
        )
        st.session_state.indicator = indicator_options[selected_label]

        # Chart
        fig = build_chart(
            df,
            st.session_state.indicator,
            st.session_state.hl_range,
            st.session_state.chart_title
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        if st.session_state.hl_range:
            st.info(f"✅ Highlighted: {st.session_state.hl_range[0]}–{st.session_state.hl_range[1]}")

    # US Map Tab
    with tab2:
        st.markdown("### Unemployment Rate by State • 2024")
        map_fig = build_choropleth()
        st.plotly_chart(map_fig, use_container_width=True, config={"displayModeBar": False})
        st.caption("Sample data — connect a state-level table in Snowflake for live sync")

# ── RIGHT PANEL (AI Agent) ────────────────────────────────────
with right:
    st.markdown("### AI Economic Agent")
    st.caption("Powered by Groq • LangChain • Snowflake")

    # Chat container
    chat_container = st.container(height=480)
    with chat_container:
        if not st.session_state.messages:
            st.markdown("""
            <div class="chat-msg-ai">
                Hello. Ask me anything about the US economy from 2015–2024.<br><br>
                The chart on the left will automatically highlight the key insight.
                <br><br><em>Try: "What happened to unemployment in 2020?"</em>
            </div>
            """, unsafe_allow_html=True)

        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-msg-user">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                pill = f'<br><span style="display:inline-block;background:#e3f0ff;color:#0071e3;border-radius:6px;padding:2px 8px;font-size:12px;font-weight:600;">{msg.get("pill","")}</span>' if msg.get("pill") else ""
                st.markdown(f'<div class="chat-msg-ai">{msg["content"]}{pill}</div>', unsafe_allow_html=True)

    # Chat input
    question = st.chat_input("Ask about the data...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("Thinking..."):
            agent = build_agent()
            try:
                result = agent.invoke({"input": question})
                answer = result["output"]
            except Exception as e:
                answer = f"Sorry, I ran into an issue: {str(e)}"

        # Auto-update chart highlight
        indicator, hl_range = detect_highlight(question, answer)
        st.session_state.indicator = indicator
        st.session_state.hl_range = hl_range
        st.session_state.chart_title = f"{selected_label} — highlighted by AI"

        pill = f"{indicator.replace('_',' ').title()} · {hl_range[0]}–{hl_range[1]}"

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "pill": pill,
        })
        st.rerun()