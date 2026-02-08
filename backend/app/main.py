# backend/app/main.py
from app.services.bot_filter import apply_bot_filter
import app.config as config

from fastapi import FastAPI
import pandas as pd

from app.config import APP_NAME, DATA_PATH
from app.schemas import OverviewResponse
from app.services.ingest import ingest_csv

app = FastAPI(title=APP_NAME)

# In-memory dataset (MVP design)
df: pd.DataFrame | None = None


@app.on_event("startup")
def startup_event():
    """
    Load and ingest dataset at startup.
    In production, this can be replaced by
    database or streaming ingestion.
    """
    global df
    df = ingest_csv(DATA_PATH)
    print(f"[Startup] Ingested {len(df)} records")


@app.get("/")
def root():
    return {"message": "SkillGenome X backend is running"}


@app.get("/dashboard/overview", response_model=OverviewResponse)
def get_overview():
    """
    High-level summary for dashboard.
    """
    return {
        "total_records": len(df),
        "total_users": df["user_id"].nunique(),
        "total_regions": df["region"].nunique(),
        "total_skills": df["skill_tags"]
            .str.split(";")
            .explode()
            .nunique()
    }