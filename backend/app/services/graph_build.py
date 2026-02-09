"""Build a skill co-occurrence graph from skill_tags column."""

import itertools
from collections import Counter

import networkx as nx
import pandas as pd


def build_skill_graph(df: pd.DataFrame):
    """
    Build an undirected graph where nodes are skills and edge weight is co-occurrence count.

    Args:
        df: DataFrame with a column "skill_tags" of semicolon-separated skill strings.

    Returns:
        Tuple of (graph, top_10_skills_by_degree, top_10_pairs_by_weight).
        - graph: networkx.Graph
        - top_10_skills: list of dicts with keys "skill", "degree"
        - top_10_pairs: list of dicts with keys "skill_1", "skill_2", "weight"
    """
    pair_counts: Counter = Counter()

    for tags in df["skill_tags"].dropna().astype(str):
        skills = [s.strip() for s in tags.split(";") if s.strip()]
        skills = sorted(set(skills))
        for a, b in itertools.combinations(skills, 2):
            pair_counts[(a, b)] += 1

    G = nx.Graph()
    for (a, b), w in pair_counts.items():
        G.add_edge(a, b, weight=w)

    top_skills = [
        {"skill": s, "degree": d}
        for s, d in sorted(G.degree(), key=lambda x: -x[1])[:10]
    ]

    top_pairs = [
        {"skill_1": a, "skill_2": b, "weight": int(w)}
        for a, b, w in sorted(
            ((u, v, G.edges[u, v]["weight"]) for u, v in G.edges()),
            key=lambda x: -x[2],
        )[:10]
    ]

    return G, top_skills, top_pairs
