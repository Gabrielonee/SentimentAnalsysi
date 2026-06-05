from __future__ import annotations
import argparse
import json
import logging
from collections import Counter
from itertools import combinations
from pathlib import Path

import networkx as nx
import pandas as pd

from .config import DATA_DIR, OUTPUT_DIR, EXEC_BY_KEY

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def build_cooccurrence_graph(df_mentions: pd.DataFrame) -> nx.Graph:
    """df_mentions deve avere colonna 'executives' (lista di chiavi)."""
    edge_counter: Counter[tuple[str, str]] = Counter()
    node_counter: Counter[str] = Counter()
    for execs in df_mentions["executives"]:
        if isinstance(execs, str):
            # Riga proveniente da parquet 'long': comportarsi di conseguenza
            continue
        execs = sorted(set(execs))
        for e in execs:
            node_counter[e] += 1
        for a, b in combinations(execs, 2):
            edge_counter[(a, b)] += 1

    G = nx.Graph()
    for k, w in node_counter.items():
        info = EXEC_BY_KEY.get(k)
        G.add_node(
            k,
            label=info.display_name if info else k,
            role=info.role if info else "",
            mentions=int(w),
        )
    for (a, b), w in edge_counter.items():
        G.add_edge(a, b, weight=int(w))
    return G


def compute_metrics(G: nx.Graph) -> pd.DataFrame:
    if G.number_of_nodes() == 0:
        return pd.DataFrame()
    deg  = nx.degree_centrality(G)
    btw  = nx.betweenness_centrality(G, weight="weight", normalized=True)
    try:
        eig  = nx.eigenvector_centrality(G, weight="weight", max_iter=500)
    except nx.PowerIterationFailedConvergence:
        eig = {n: 0.0 for n in G.nodes}
    clu  = nx.clustering(G, weight="weight")

    # Community detection con greedy modularity (no dipendenze esterne)
    communities = list(nx.community.greedy_modularity_communities(G, weight="weight"))
    comm_map = {n: i for i, comm in enumerate(communities) for n in comm}

    rows = []
    for n, attrs in G.nodes(data=True):
        rows.append({
            "executive": n,
            "label": attrs.get("label", n),
            "role": attrs.get("role", ""),
            "mentions": attrs.get("mentions", 0),
            "degree": deg.get(n, 0.0),
            "betweenness": btw.get(n, 0.0),
            "eigenvector": eig.get(n, 0.0),
            "clustering": clu.get(n, 0.0),
            "community": comm_map.get(n, -1),
        })
    return pd.DataFrame(rows).sort_values("mentions", ascending=False)


def export_json(G: nx.Graph, metrics: pd.DataFrame, out_path: Path):
    m_by_key = {r["executive"]: r for _, r in metrics.iterrows()}
    nodes = []
    for n, attrs in G.nodes(data=True):
        m = m_by_key.get(n, {})
        nodes.append({
            "id": n,
            "label": attrs.get("label", n),
            "role": attrs.get("role", ""),
            "mentions": attrs.get("mentions", 0),
            "degree": float(m.get("degree", 0.0)),
            "betweenness": float(m.get("betweenness", 0.0)),
            "eigenvector": float(m.get("eigenvector", 0.0)),
            "community": int(m.get("community", -1)),
        })
    edges = [{"source": u, "target": v, "weight": int(d["weight"])}
             for u, v, d in G.edges(data=True)]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"nodes": nodes, "edges": edges}, f, ensure_ascii=False, indent=2)


def main(in_path: Path | None = None,
         out_gexf: Path | None = None,
         out_json: Path | None = None,
         out_metrics: Path | None = None):
    in_path     = in_path     or DATA_DIR  / "mentions.parquet"
    out_gexf    = out_gexf    or OUTPUT_DIR / "graph.gexf"
    out_json    = out_json    or OUTPUT_DIR / "graph.json"
    out_metrics = out_metrics or OUTPUT_DIR / "metrics.csv"

    df = pd.read_parquet(in_path)
    logger.info("Letto %s (%d frasi)", in_path, len(df))
    G = build_cooccurrence_graph(df)
    logger.info("Grafo: %d nodi, %d archi", G.number_of_nodes(), G.number_of_edges())
    metrics = compute_metrics(G)
    metrics.to_csv(out_metrics, index=False)
    nx.write_gexf(G, out_gexf)
    export_json(G, metrics, out_json)
    logger.info("Salvati %s, %s, %s", out_gexf, out_json, out_metrics)
    return G, metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path, default=None)
    args = ap.parse_args()
    main(in_path=args.in_path)
