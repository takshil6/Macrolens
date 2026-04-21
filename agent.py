"""
agent.py  —  LangChain SQL agent that returns structured JSON
             so app.py can update both the trend chart and state map.
"""

import os, json, re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from sqlalchemy import create_engine

load_dotenv()

# ── System prompt ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are a US economic data analyst AI with access to a Snowflake database
containing monthly US macro indicators (GDP, unemployment, inflation_cpi, federal_funds_rate,
housing_starts, avg_hourly_wage, nonfarm_payroll, labor_force_part) from 2015–2024.

After querying the database (when needed), respond ONLY with a single JSON object.
No markdown, no explanation outside the JSON.

JSON schema:
{
  "answer": "<2–3 sentence analytical insight with specific numbers. Reference real events.>",
  "indicator": "<one of: unemployment | gdp | inflation_cpi | federal_funds_rate | housing_starts | avg_hourly_wage | nonfarm_payroll | labor_force_part>",
  "year_range": [<startYear>, <endYear>],
  "map_spec": {
    "metric":           "<one of: unemployment | gdp_pc | wages | housing | labor_part>",
    "mode":             "<one of: current | change | rank>",
    "year":             <integer 2015–2024>,
    "highlight_states": ["<up to 6 two-letter state abbreviations most relevant to the answer>"]
  }
}

Map mode guidelines:
- "current"  → single-year snapshot (use when asking about a specific year or latest values)
- "change"   → year-over-year delta (use when asking about what changed, spikes, crashes)
- "rank"     → state rankings 1–50 best-to-worst (use when asking which states lead/lag)

Be precise: cite actual data values from the database query results wherever possible.
"""

PREFIX = """You are a friendly economic data assistant with access to US economic indicators
(GDP, inflation, unemployment, etc.) from 2015–2024 in a Snowflake database.
Query the database for specific numbers, then respond with valid JSON only (see system prompt).
"""

def get_engine():
    return create_engine(
        f"snowflake://{os.getenv('SNOWFLAKE_USER')}:{os.getenv('SNOWFLAKE_PASSWORD')}"
        f"@{os.getenv('SNOWFLAKE_ACCOUNT')}/{os.getenv('SNOWFLAKE_DATABASE')}"
        f"/{os.getenv('SNOWFLAKE_SCHEMA')}?warehouse={os.getenv('SNOWFLAKE_WAREHOUSE')}"
    )


def build_agent():
    """Build and return the LangChain SQL agent + LLM handle."""
    print("  Connecting to Snowflake…")
    engine = get_engine()
    db = SQLDatabase(engine, include_tables=["economic_data"], sample_rows_in_table_info=3)
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0,
    )
    sql_agent = create_sql_agent(
        llm=llm, db=db, agent_type="openai-tools",
        verbose=False, handle_parsing_errors=True,
        agent_kwargs={"system_message": SYSTEM_PROMPT},
        prefix=PREFIX,
    )
    print("  Agent ready!\n")
    return {"sql_agent": sql_agent, "llm": llm}


def _is_data_question(llm, text: str) -> bool:
    """Route: does this need a DB lookup?"""
    resp = llm.invoke(
        f'Reply with one word only — DATA or CHAT.\n'
        f'DATA = asks for specific numbers, trends, comparisons, statistics.\n'
        f'CHAT = greeting, thanks, farewell, general knowledge.\n\n'
        f'Message: "{text}"\nAnswer:'
    ).content.strip().upper()
    return resp.startswith("DATA")


def _extract_json(text: str) -> dict:
    """Pull the first {...} block out of the model response."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError("No JSON found in response")


def ask_structured(agent: dict, question: str) -> dict:
    """
    Ask the agent a question and return a structured dict:
    {
        "answer":           str,
        "indicator":        str,
        "year_range":       [int, int] | None,
        "map_spec": {
            "metric":           str,
            "mode":             str,
            "year":             int,
            "highlight_states": list[str],
        }
    }
    Falls back to fallback_response() on any error.
    """
    llm       = agent["llm"]
    sql_agent = agent["sql_agent"]

    try:
        if _is_data_question(llm, question):
            raw = sql_agent.invoke({"input": question})["output"]
        else:
            raw = llm.invoke(
                f"{SYSTEM_PROMPT}\n\nUser: {question}\n\nRespond with JSON only."
            ).content.strip()

        parsed = _extract_json(raw)
        return _normalize(parsed)

    except Exception as e:
        print(f"  [agent] fallback triggered: {e}")
        return fallback_response(question)


def _normalize(d: dict) -> dict:
    """Ensure all expected keys exist with sensible defaults."""
    ms = d.get("map_spec") or {}
    return {
        "answer":     d.get("answer", ""),
        "indicator":  d.get("indicator", "unemployment"),
        "year_range": d.get("year_range") or None,
        "map_spec": {
            "metric":           ms.get("metric", "unemployment"),
            "mode":             ms.get("mode", "current"),
            "year":             ms.get("year", 2024),
            "highlight_states": ms.get("highlight_states") or [],
        }
    }


# ── Keyword fallback (no DB / no LLM) ────────────────────────
def fallback_response(question: str) -> dict:
    q = question.lower()

    if "unemploy" in q and "2020" in q:
        return _normalize({
            "answer": "In April 2020, US unemployment surged to 14.7%—the highest since WWII—as COVID-19 eliminated 22M jobs in two months. Hawaii (+18 pts) and Nevada (+15 pts) were hit hardest due to tourism collapse. By year-end it had recovered to 6.7%.",
            "indicator": "unemployment", "year_range": [2019, 2021],
            "map_spec": {"metric":"unemployment","mode":"change","year":2020,
                         "highlight_states":["HI","NV","NY","CA","MI"]},
        })
    if "wage" in q or "earn" in q:
        return _normalize({
            "answer": "Average hourly wages grew 5–6% annually in 2021–2022, the fastest pace in 40 years, driven by post-COVID labor shortages. Massachusetts, Washington, and Connecticut lead at $33–38/hr; Mississippi and Arkansas trail at $21–23/hr.",
            "indicator": "avg_hourly_wage", "year_range": [2021, 2023],
            "map_spec": {"metric":"wages","mode":"current","year":2023,
                         "highlight_states":["MA","WA","CT","CA","NJ"]},
        })
    if "housing" in q:
        return _normalize({
            "answer": "Housing starts peaked at ~1.8M in late 2021 before Fed rate hikes cut demand sharply through 2023. Utah, Idaho, and Texas lead per-capita construction activity; Rhode Island and New York are at the bottom.",
            "indicator": "housing_starts", "year_range": [2021, 2023],
            "map_spec": {"metric":"housing","mode":"current","year":2022,
                         "highlight_states":["TX","FL","UT","ID","NC"]},
        })
    if "fed" in q or ("rate" in q and "interest" in q) or "2022" in q:
        return _normalize({
            "answer": "The Fed raised rates 11 times from 0.25% to 5.5% between March 2022 and July 2023—the fastest tightening cycle since Volcker in 1980—to combat 9.1% peak CPI in June 2022.",
            "indicator": "federal_funds_rate", "year_range": [2022, 2023],
            "map_spec": {"metric":"unemployment","mode":"rank","year":2022,
                         "highlight_states":[]},
        })
    if "gdp" in q or ("state" in q and "richest" in q) or "per capita" in q:
        return _normalize({
            "answer": "New York ($109K/capita), Massachusetts ($102K), and Connecticut ($98K) lead state GDP per capita, reflecting finance and biotech concentration. Mississippi ($40K) and West Virginia ($43K) trail significantly.",
            "indicator": "gdp", "year_range": [2020, 2024],
            "map_spec": {"metric":"gdp_pc","mode":"current","year":2024,
                         "highlight_states":["NY","MA","CT","WA","NJ"]},
        })
    if "recover" in q or "covid" in q:
        return _normalize({
            "answer": "Sun Belt states (TX, FL, AZ, UT) recovered pre-COVID employment by mid-2021, nearly a year ahead of Northeast/Midwest states. Hawaii was last, not reaching pre-COVID unemployment until late 2022.",
            "indicator": "unemployment", "year_range": [2020, 2022],
            "map_spec": {"metric":"unemployment","mode":"change","year":2021,
                         "highlight_states":["TX","FL","UT","AZ","HI"]},
        })
    if "labor" in q or "participation" in q:
        return _normalize({
            "answer": "Labor force participation dropped from 63.3% to 61.7% in 2020 and never fully recovered, partly driven by early retirements. North Dakota (70%), Nebraska (68%), and Iowa (67%) lead; West Virginia (55%) and Mississippi (58%) trail.",
            "indicator": "labor_force_part", "year_range": [2019, 2022],
            "map_spec": {"metric":"labor_part","mode":"rank","year":2024,
                         "highlight_states":["ND","NE","IA","SD","MN"]},
        })
    if "inflation" in q or "cpi" in q:
        return _normalize({
            "answer": "US CPI inflation peaked at 9.1% in June 2022, the highest in 40 years, driven by supply-chain disruptions and energy prices post-COVID. The Fed's aggressive tightening brought it back to ~2.8% by 2024.",
            "indicator": "inflation_cpi", "year_range": [2021, 2023],
            "map_spec": {"metric":"unemployment","mode":"current","year":2022,
                         "highlight_states":[]},
        })
    # Generic fallback
    return _normalize({
        "answer": "The data shows notable disruption around 2020 (COVID shock) and 2022 (Fed tightening cycle). The state map highlights regional variation — Sun Belt states generally show stronger economic metrics while Appalachian and Rust Belt states lag.",
        "indicator": "unemployment", "year_range": [2020, 2022],
        "map_spec": {"metric":"unemployment","mode":"change","year":2020,
                     "highlight_states":["NV","HI","NY","CA","MI"]},
    })
