# main.py
from extract import extract_all
from transform import merge_all, clean
from load_local import save
from load_snowflake import load_to_snowflake
import pandas as pd

if __name__ == "__main__":
    print("=== Economic Indicators Pipeline ===")
    raw_frames = extract_all()
    merged     = merge_all(raw_frames)
    clean_df   = clean(merged)
    save(clean_df)

    # Load to Snowflake
    load_to_snowflake(clean_df.copy())

    print("\nPipeline complete!")