from __future__ import annotations
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, OUTPUT_DIR, EXEC_BY_KEY

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _explode_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    if "executives" in df.columns and len(df) \
       and isinstance(df["executives"].iloc[0], (list, np.ndarray)):
        df = df.explode("executives").rename(columns={"executives": "executive"})
    return df


def load_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    v = pd.read_parquet(DATA_DIR / "sentiment.parquet")
    a = pd.read_parquet(DATA_DIR / "sentiment_absa.parquet")
    return _explode_if_needed(v), _explode_if_needed(a)


def merge_for(executive: str, v: pd.DataFrame, a: pd.DataFrame) -> pd.DataFrame:
    """Join vanilla e ABSA sulla coppia (sentence_id, executive)."""
    v = v[v["executive"] == executive].copy()
    a = a[a["executive"] == executive].copy()
    keys = ["sentence_id", "executive"]
    cols_v = keys + ["sentence", "sentiment_score", "sentiment_label",
                     "created_utc", "lang"]
    cols_a = keys + ["sentiment_score", "sentiment_label"]
    cols_v = [c for c in cols_v if c in v.columns]
    cols_a = [c for c in cols_a if c in a.columns]
    m = v[cols_v].merge(a[cols_a], on=keys, suffixes=("_van", "_absa"))
    m["delta"] = m["sentiment_score_absa"] - m["sentiment_score_van"]
    return m


def aggregate_table(v: pd.DataFrame, a: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, info in EXEC_BY_KEY.items():
        vs = v[v["executive"] == key]["sentiment_score"]
        as_ = a[a["executive"] == key]["sentiment_score"]
        if len(vs) == 0 or len(as_) == 0: continue
        rows.append({
            "executive": info.display_name,
            "n_van":  len(vs),
            "n_absa": len(as_),
            "mean_van":   round(vs.mean(),   3),
            "mean_absa":  round(as_.mean(),  3),
            "delta_mean": round(as_.mean() - vs.mean(), 3),
            "median_van":  round(vs.median(),  3),
            "median_absa": round(as_.median(), 3),
        })
    return pd.DataFrame(rows).sort_values("delta_mean", ascending=False)


def print_section(title: str):
    print(f"\n{'='*78}\n{title}\n{'='*78}")


def main(executive: str, top_k: int = 20, min_delta: float = 0.5,
         export_dir: Path | None = None):
    if executive not in EXEC_BY_KEY:
        raise SystemExit(f"Dirigente '{executive}' non trovato.")
    info = EXEC_BY_KEY[executive]

    v, a = load_pair()

    print_section("Confronto aggregato Vanilla vs ABSA — tutti i dirigenti")
    print(aggregate_table(v, a).to_string(index=False))

    m = merge_for(executive, v, a)
    if m.empty:
        print(f"\nNessuna coppia merge-abile per {info.display_name}.")
        return

    print_section(f"{info.display_name} — frasi in cui ABSA è PIÙ POSITIVO del Vanilla "
                  f"(top {top_k} per delta)")
    pos = m.sort_values("delta", ascending=False).head(top_k)
    for _, r in pos.iterrows():
        print(f"  van={r['sentiment_score_van']:+.2f} -> absa={r['sentiment_score_absa']:+.2f} "
              f"(Δ={r['delta']:+.2f}) | ({r.get('lang','?')}) {r['sentence'][:120]}")

    print_section(f"{info.display_name} — frasi in cui ABSA è PIÙ NEGATIVO del Vanilla "
                  f"(top {top_k} per delta)")
    neg = m.sort_values("delta", ascending=True).head(top_k)
    for _, r in neg.iterrows():
        print(f"  van={r['sentiment_score_van']:+.2f} -> absa={r['sentiment_score_absa']:+.2f} "
              f"(Δ={r['delta']:+.2f}) | ({r.get('lang','?')}) {r['sentence'][:120]}")

    print_section(f"Distribuzione delta sentiment ({info.display_name})")
    print(m["delta"].describe().round(3).to_string())
    quartile_agreement = ((m["sentiment_label_van"] == m["sentiment_label_absa"]).mean())
    print(f"\nLabel agreement vanilla vs ABSA: {quartile_agreement:.1%}")
    print(f"Frasi con |delta| > {min_delta}: {(m['delta'].abs() > min_delta).sum()} "
          f"({(m['delta'].abs() > min_delta).mean():.1%})")

    out_dir = export_dir or OUTPUT_DIR / f"compare_{executive}"
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate_table(v, a).to_csv(out_dir / "aggregate.csv", index=False)
    m.to_csv(out_dir / "all_pairs.csv", index=False)
    m[m["delta"].abs() > min_delta].to_csv(out_dir / "high_disagreement.csv", index=False)
    print(f"\nCSV salvati in {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--executive", required=True,
                    help="Chiave del dirigente (es. maldini, cardinale)")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--min-delta", type=float, default=0.5,
                    help="Soglia |delta| per esportare il CSV delle disagreement")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    main(executive=args.executive, top_k=args.top_k,
         min_delta=args.min_delta, export_dir=args.out)
