# load_snowflake.py
import os
import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    return snowflake.connector.connect(
        account   = os.getenv("SNOWFLAKE_ACCOUNT"),
        user      = os.getenv("SNOWFLAKE_USER"),
        password  = os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse = os.getenv("SNOWFLAKE_WAREHOUSE"),
        database  = os.getenv("SNOWFLAKE_DATABASE"),
        schema    = os.getenv("SNOWFLAKE_SCHEMA"),
    )

def load_to_snowflake(df):
    print("\n--- LOAD TO SNOWFLAKE ---")

    # Snowflake expects uppercase column names
    df.columns = [c.upper() for c in df.columns]

    # Convert date column to proper type
    df["DATE"] = pd.to_datetime(df["DATE"]).dt.date

    conn = get_connection()
    print("  Connected to Snowflake successfully")

    success, nchunks, nrows, _ = write_pandas(
        conn,
        df,
        table_name        = "ECONOMIC_DATA",
        database          = os.getenv("SNOWFLAKE_DATABASE"),
        schema            = os.getenv("SNOWFLAKE_SCHEMA"),
        overwrite         = True,
        use_logical_type  = True,
    )

    if success:
        print(f"  Loaded {nrows} rows across {nchunks} chunk(s)")
        print(f"  Table: ECONOMIC_DB.INDICATORS.ECONOMIC_DATA")
    else:
        print("  Load failed — check credentials and table name")

    conn.close()