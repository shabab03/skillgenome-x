"""
Bot detection for SkillGenome X.
Identify and remove adversarial or bot-like users using explainable rules.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _normalize_text(series: pd.Series) -> pd.Series:
    """Lowercase, strip, and collapse repeated whitespace."""
    return (
        series.astype("string")
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def apply_bot_filter(
    df: pd.DataFrame, config: Any
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """
    Detect bot-like users and return a cleaned dataframe plus stats.

    Bot detection rules (explainable):
    1) posts_per_day = total_posts / max(active_days, 1) per user.
    2) duplicate_text_ratio = 1 - (unique_texts / total_posts) per user.
    3) User is marked bot if posts_per_day or duplicate_text_ratio exceeds
       config thresholds.

    Adds columns:
    - is_bot (bool)
    - trust_score (float in [0, 1], bots have lower score)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: user_id, timestamp, raw_text, engagement.
    config : object
        Must have attributes:
        - BOT_POSTS_PER_DAY_THRESHOLD
        - BOT_DUPLICATE_TEXT_THRESHOLD

    Returns
    -------
    cleaned_df : pd.DataFrame
        Copy of df with bot rows removed.
    stats : dict
        total_users, bots_detected, percent_removed (user-level).
    """
    required_cols = {"user_id", "timestamp", "raw_text", "engagement"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()

    threshold_ppd = float(getattr(config, "BOT_POSTS_PER_DAY_THRESHOLD", 40))
    threshold_dup = float(getattr(config, "BOT_DUPLICATE_TEXT_THRESHOLD", 0.75))

    # Parse timestamps safely (supports string timestamps)
    ts = pd.to_datetime(df["timestamp"], errors="coerce")

    # Active day = unique day on which user posted (keep datetime64 for speed)
    df["_day"] = ts.dt.floor("D")

    # Normalize text for duplicate detection
    df["_norm_text"] = _normalize_text(df["raw_text"])

    # ---------- Per-user metrics (vectorized) ----------
    g = df.groupby("user_id", sort=False)

    total_posts = g.size()  # Series indexed by user_id
    active_days = g["_day"].nunique().clip(lower=1)  # avoid division by 0
    posts_per_day = total_posts / active_days

    unique_texts = g["_norm_text"].nunique()
    duplicate_text_ratio = 1.0 - (unique_texts / total_posts.clip(lower=1))

    user_metrics = pd.DataFrame(
        {
            "posts_per_day": posts_per_day,
            "duplicate_text_ratio": duplicate_text_ratio,
        }
    )

    # ---------- Map metrics back to rows ----------
    df["posts_per_day"] = df["user_id"].map(user_metrics["posts_per_day"])
    df["duplicate_text_ratio"] = df["user_id"].map(
        user_metrics["duplicate_text_ratio"]
    )

    # ---------- Bot decision ----------
    df["is_bot"] = (df["posts_per_day"] > threshold_ppd) | (
        df["duplicate_text_ratio"] > threshold_dup
    )

    # Simple explainable trust score
    df["trust_score"] = np.where(df["is_bot"], 0.2, 1.0).astype(float)

    # Remove helper columns
    df.drop(columns=["_day", "_norm_text"], inplace=True)

    # ---------- Stats ----------
    total_users = df["user_id"].nunique()
    bots_detected = df.loc[df["is_bot"], "user_id"].nunique()
    percent_removed = (100.0 * bots_detected / total_users) if total_users else 0.0

    cleaned_df = df.loc[~df["is_bot"]].copy()
    stats = {
        "total_users": int(total_users),
        "bots_detected": int(bots_detected),
        "percent_removed": round(float(percent_removed), 2),
    }

    return cleaned_df, stats