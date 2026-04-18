# agent.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from sqlalchemy import create_engine

load_dotenv()

def get_snowflake_engine():
    account   = os.getenv("SNOWFLAKE_ACCOUNT")
    user      = os.getenv("SNOWFLAKE_USER")
    password  = os.getenv("SNOWFLAKE_PASSWORD")
    database  = os.getenv("SNOWFLAKE_DATABASE")
    schema    = os.getenv("SNOWFLAKE_SCHEMA")
    warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
    return create_engine(
        f"snowflake://{user}:{password}@{account}/{database}/{schema}?warehouse={warehouse}"
    )

def _is_data_question(llm, text):
    """Returns True if the message needs a database lookup."""
    prompt = f"""You are a router. Decide if this message requires querying an economic database.
Reply with only one word: DATA or CHAT.

- DATA: asks about specific numbers, trends, years, comparisons, statistics
- CHAT: greetings, thanks, farewells, general knowledge questions, small talk

Message: "{text}"
Answer:"""
    response = llm.invoke(prompt).content.strip().upper()
    return response.startswith("DATA")

def build_agent():
    print("  Connecting to Snowflake...")
    engine = get_snowflake_engine()

    db = SQLDatabase(
        engine,
        include_tables=["economic_data"],
        sample_rows_in_table_info=3,
    )

    llm = ChatGroq(
        api_key    = os.getenv("GROQ_API_KEY"),
        model_name = "llama-3.3-70b-versatile",
        temperature= 0,
    )

    prefix = """You are a friendly economic data assistant with access to US economic indicators (GDP, inflation, unemployment, etc.) from 2015–2024.
Only use SQL tools when the user asks about specific data or trends. Explain results in plain English."""

    sql_agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="openai-tools",
        verbose=False,
        handle_parsing_errors=True,
        prefix=prefix,
    )

    print("  Agent ready!\n")
    return {"sql_agent": sql_agent, "llm": llm}

def ask(agent, question):
    llm       = agent["llm"]
    sql_agent = agent["sql_agent"]

    if _is_data_question(llm, question):
        result = sql_agent.invoke({"input": question})
        reply  = result["output"]
    else:
        # pure conversational — no DB needed
        prompt = f"""You are a friendly economic data assistant. Respond naturally to this message.
If it's a farewell, wish them well. If it's thanks, acknowledge warmly. Keep it short.

User: {question}"""
        reply = llm.invoke(prompt).content.strip()

    print(f"\nAssistant: {reply}\n")
    return reply