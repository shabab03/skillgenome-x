# backend/app/services/ingest.py

import pandas as pd
from datetime import datetime

REQUIRED_COLUMNS = [
    "user_id",
    "region",
    "timestamp",
    "source",
    "raw_text",
    "skill_tags",
    "engagement"
]

def ingest_csv(file_path: str) -> pd.DataFrame:
    """
    Ingest raw CSV data into SkillGenome X.

    Responsibilities:
    - Load raw data
    - Validate schema
    - Normalize formats
    - Add ingestion metadata

    This layer is isolated so that new data sources
    (APIs, streaming, surveys) can be added later
    without touching ML or graph logic.
    """

    # 1. Load data
    df = pd.read_csv(file_path)

    # 2. Validate schema
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # 3. Normalize fields
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["source"] = df["source"].str.lower().str.strip()
    df["region"] = df["region"].str.strip()
    df["skill_tags"] = df["skill_tags"].fillna("").astype(str)

    # Remove rows with invalid timestamps
    df = df.dropna(subset=["timestamp"])

    # 4. Add ingestion metadata
    df["ingestion_time"] = datetime.utcnow()
    df["ingestion_type"] = "csv"

    return df