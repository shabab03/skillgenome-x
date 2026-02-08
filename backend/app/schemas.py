# app/schemas.py

from pydantic import BaseModel
from typing import List, Dict, Any

class OverviewResponse(BaseModel):
    total_records: int
    total_users: int
    total_regions: int
    total_skills: int

class HeatmapResponse(BaseModel):
    region: str
    skill_count: int

class ClusterResponse(BaseModel):
    region: str
    cluster_id: int
    top_skills: List[str]