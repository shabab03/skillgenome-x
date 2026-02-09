"""Simple skill demand forecasting using weekly counts and linear trend."""

import numpy as np
import pandas as pd


def forecast_skill(df: pd.DataFrame, skill: str, horizon_weeks: int = 12) -> dict:
    """
    Forecast weekly counts for a skill using a linear trend on historical data.

    Args:
        df: DataFrame with columns "timestamp" (datetime) and "skill_tags" (semicolon-separated).
        skill: Skill to filter and forecast (exact match within tags).
        horizon_weeks: Number of weeks to extrapolate (default 12).

    Returns:
        JSON-serializable dict with "historical", "forecast", and "trend".
    """
    if df.empty or "timestamp" not in df.columns or "skill_tags" not in df.columns:
        return _empty_result()

    skill = skill.strip()
    if not skill:
        return _empty_result()

    # Filter rows where skill appears in skill_tags (exact match in list)
    def has_skill(tags):
        if pd.isna(tags):
            return False
        return skill in [s.strip() for s in str(tags).split(";") if s.strip()]

    mask = df["skill_tags"].apply(has_skill)
    filtered = df.loc[mask].copy()
    if filtered.empty:
        return _empty_result()

    filtered["timestamp"] = pd.to_datetime(filtered["timestamp"])
    weekly = (
        filtered.set_index("timestamp")
        .resample("W", label="left")
        .size()
        .reset_index(name="count")
    )
    weekly["week"] = weekly["timestamp"].dt.strftime("%Y-%m-%d")

    if len(weekly) < 2:
        hist = [{"week": r["week"], "count": int(r["count"])} for _, r in weekly.iterrows()]
        return {
            "historical": hist,
            "forecast": [],
            "trend": "stable",
        }

    # Linear trend: x = 0,1,2,... vs count
    x = np.arange(len(weekly), dtype=float)
    y = weekly["count"].values.astype(float)
    slope, intercept = np.polyfit(x, y, 1)
    mean_count = float(np.mean(y))
    slope_threshold = max(0.1, mean_count * 0.02)
    if slope > slope_threshold:
        trend = "rising"
    elif slope < -slope_threshold:
        trend = "declining"
    else:
        trend = "stable"

    # Historical output
    historical = [{"week": w, "count": int(c)} for w, c in zip(weekly["week"], weekly["count"])]

    # Extrapolate for horizon_weeks
    last_ts = weekly["timestamp"].iloc[-1]
    forecast = []
    for i in range(1, horizon_weeks + 1):
        pred_x = len(weekly) - 1 + i
        pred_count = slope * pred_x + intercept
        pred_count = max(0.0, round(pred_count, 1))
        week_ts = last_ts + pd.Timedelta(weeks=i)
        forecast.append({
            "week": week_ts.strftime("%Y-%m-%d"),
            "predicted_count": float(pred_count),
        })

    return {
        "historical": historical,
        "forecast": forecast,
        "trend": trend,
    }


def _empty_result() -> dict:
    return {
        "historical": [],
        "forecast": [],
        "trend": "stable",
    }
