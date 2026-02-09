"""Cluster regions by skill profiles and return top skills per cluster."""

import pandas as pd
from sklearn.cluster import KMeans


def cluster_regions(df: pd.DataFrame, n_clusters: int = 3) -> list:
    """
    Cluster regions by skill frequency and return region, cluster_id, and cluster top skills.

    Args:
        df: DataFrame with columns "region" and "skill_tags" (semicolon-separated).
        n_clusters: Number of KMeans clusters (default 3).

    Returns:
        JSON-serializable list of dicts: {"region", "cluster_id", "top_skills"}.
    """
    if df.empty or "region" not in df.columns or "skill_tags" not in df.columns:
        return []

    # Expand to one row per (region, skill)
    exp = df[["region", "skill_tags"]].dropna(subset=["skill_tags"]).copy()
    exp["skill"] = exp["skill_tags"].astype(str).str.split(";")
    exp = exp.explode("skill")
    exp["skill"] = exp["skill"].str.strip()
    exp = exp[exp["skill"] != ""][["region", "skill"]]

    if exp.empty:
        return []

    # Region × skill frequency matrix
    mat = exp.groupby(["region", "skill"]).size().unstack(fill_value=0)

    # Cap n_clusters if fewer regions
    k = min(n_clusters, len(mat))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(mat)

    # Top 5 skills per cluster (sum skill counts over regions in cluster)
    mat = mat.assign(_cluster=labels)
    cluster_skill_counts = mat.groupby("_cluster").sum().drop(columns=["_cluster"], errors="ignore")

    top_5_per_cluster = {}
    for c in range(k):
        if c in cluster_skill_counts.index:
            top = cluster_skill_counts.loc[c].nlargest(5)
            top_5_per_cluster[c] = top[top > 0].index.astype(str).tolist()[:5]
        else:
            top_5_per_cluster[c] = []

    # One dict per region: region, cluster_id, that cluster's top_skills
    result = []
    for region in mat.index:
        cid = int(mat.loc[region, "_cluster"])
        result.append({
            "region": str(region),
            "cluster_id": cid,
            "top_skills": top_5_per_cluster[cid],
        })

    return result
