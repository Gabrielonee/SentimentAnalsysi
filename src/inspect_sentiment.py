from __future__ import annotations
import argparse
import logging
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

from .config import DATA_DIR, OUTPUT_DIR, EXEC_BY_KEY

# Download stopwords se non già disponibili
try:
    stopwords.words('italian')
except LookupError:
    nltk.download('stopwords', quiet=True)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

STOP = set(stopwords.words('italian')) | set(stopwords.words('english'))
TOKEN_RE = re.compile(r"[A-Za-zÀ-Ýà-ý']{3,}")


def load_for_executive(key: str, sent_path: Path | None = None) -> pd.DataFrame:
    sent_path = sent_path or DATA_DIR / "sentiment.parquet"
    df = pd.read_parquet(sent_path)
    if "executives" in df.columns and len(df) and isinstance(df["executives"].iloc[0], (list, np.ndarray)):
        df = df.explode("executives").rename(columns={"executives": "executive"})
    df = df[df["executive"] == key].copy()
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True)
    return df


def describe(df: pd.DataFrame) -> dict:
    s = df["sentiment_score"]
    return {
        "n_sentences": int(len(df)),
        "mean":   float(s.mean()),
        "median": float(s.median()),
        "std":    float(s.std()),
        "q25":    float(s.quantile(0.25)),
        "q75":    float(s.quantile(0.75)),
        "share_positive": float((df["sentiment_label"] == "positive").mean()),
        "share_neutral":  float((df["sentiment_label"] == "neutral").mean()),
        "share_negative": float((df["sentiment_label"] == "negative").mean()),
        "share_strong_pos": float((s > +0.5).mean()),
        "share_strong_neg": float((s < -0.5).mean()),
    }


def top_sentences(df: pd.DataFrame, k: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["created_utc", "lang", "sentiment_score", "sentiment_label",
            "sentence", "doc_id"]
    cols = [c for c in cols if c in df.columns]
    most_pos = df.sort_values("sentiment_score", ascending=False).head(k)[cols]
    most_neg = df.sort_values("sentiment_score", ascending=True).head(k)[cols]
    return most_pos, most_neg


def monthly_distribution(df: pd.DataFrame) -> pd.DataFrame:
    g = df.copy()
    g["month"] = g["created_utc"].dt.tz_convert("UTC").dt.to_period("M").dt.to_timestamp()
    return (g.groupby("month")
             .agg(n=("sentiment_score", "size"),
                  mean=("sentiment_score", "mean"),
                  share_pos=("sentiment_label", lambda s: (s == "positive").mean()),
                  share_neg=("sentiment_label", lambda s: (s == "negative").mean()))
             .reset_index())


def before_after(df: pd.DataFrame, event_date: datetime,
                 window_days: int = 60) -> pd.DataFrame:
    pre  = df[(df["created_utc"] >= event_date - pd.Timedelta(days=window_days)) &
              (df["created_utc"] <  event_date)]
    post = df[(df["created_utc"] >  event_date) &
              (df["created_utc"] <= event_date + pd.Timedelta(days=window_days))]
    rows = [{
        "fase":      "PRIMA (-%dgg)" % window_days,
        "n":         len(pre),
        "mean":      pre["sentiment_score"].mean() if len(pre) else np.nan,
        "median":    pre["sentiment_score"].median() if len(pre) else np.nan,
        "share_pos": (pre["sentiment_label"] == "positive").mean() if len(pre) else np.nan,
        "share_neg": (pre["sentiment_label"] == "negative").mean() if len(pre) else np.nan,
    }, {
        "fase":      "DOPO (+%dgg)" % window_days,
        "n":         len(post),
        "mean":      post["sentiment_score"].mean() if len(post) else np.nan,
        "median":    post["sentiment_score"].median() if len(post) else np.nan,
        "share_pos": (post["sentiment_label"] == "positive").mean() if len(post) else np.nan,
        "share_neg": (post["sentiment_label"] == "negative").mean() if len(post) else np.nan,
    }]
    return pd.DataFrame(rows)


def keyness(df: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    def tok(text):
        return [t.lower() for t in TOKEN_RE.findall(text or "") if t.lower() not in STOP]
    pos = df[df["sentiment_label"] == "positive"]["sentence"]
    neg = df[df["sentiment_label"] == "negative"]["sentence"]
    cnt_pos, cnt_neg = Counter(), Counter()
    for s in pos: cnt_pos.update(tok(s))
    for s in neg: cnt_neg.update(tok(s))
    N_pos = sum(cnt_pos.values()) or 1
    N_neg = sum(cnt_neg.values()) or 1
    vocab = set(cnt_pos) | set(cnt_neg)
    rows = []
    for w in vocab:
        a, b = cnt_pos[w], cnt_neg[w]
        if a + b < 5: continue
        log_odds = np.log((a + 0.5) / (N_pos - a + 0.5)) - \
                   np.log((b + 0.5) / (N_neg - b + 0.5))
        rows.append({"token": w, "count_pos": a, "count_neg": b,
                     "log_odds_pos_vs_neg": log_odds})
    keys = pd.DataFrame(rows)
    if keys.empty: return keys
    return pd.concat([
        keys.sort_values("log_odds_pos_vs_neg", ascending=False).head(top_n)
            .assign(direction="POS"),
        keys.sort_values("log_odds_pos_vs_neg", ascending=True).head(top_n)
            .assign(direction="NEG"),
    ])


def print_section(title: str):
    print(f"\n{'='*78}\n{title}\n{'='*78}")


def main(executive_key: str, k: int = 20,
         event: datetime | None = None, window_days: int = 60,
         export_dir: Path | None = None,
         sentiment_path: Path | None = None):
    if executive_key not in EXEC_BY_KEY:
        raise SystemExit(f"Dirigente '{executive_key}' non trovato. "
                         f"Validi: {sorted(EXEC_BY_KEY)}")
    info = EXEC_BY_KEY[executive_key]
    df = load_for_executive(executive_key, sent_path=sentiment_path)
    if df.empty:
        raise SystemExit(f"Nessuna riga per {info.display_name}.")

    print_section(f"Diagnostica sentiment — {info.display_name} ({info.role})")
    stats = describe(df)
    for k_, v in stats.items():
        if isinstance(v, float): print(f"  {k_:<18}: {v:+.3f}")
        else: print(f"  {k_:<18}: {v}")

    print_section("Top frasi MOLTO POSITIVE (sort by sentiment_score desc)")
    most_pos, most_neg = top_sentences(df, k=k)
    for _, r in most_pos.iterrows():
        print(f"  [{r['sentiment_score']:+.2f}] ({r['lang']}, {r['created_utc'].date()}) "
              f"{r['sentence'][:140]}")

    print_section("Top frasi MOLTO NEGATIVE (sort by sentiment_score asc)")
    for _, r in most_neg.iterrows():
        print(f"  [{r['sentiment_score']:+.2f}] ({r['lang']}, {r['created_utc'].date()}) "
              f"{r['sentence'][:140]}")

    print_section("Distribuzione mensile del sentiment")
    mdist = monthly_distribution(df)
    print(mdist.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    if event is not None:
        print_section(f"Confronto pre / post evento {event.date()} "
                      f"(±{window_days} giorni)")
        ba = before_after(df, event, window_days=window_days)
        print(ba.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    print_section("Token più caratteristici di frasi POS vs NEG (log-odds)")
    keys = keyness(df, top_n=20)
    if keys.empty:
        print("  (volume insufficiente)")
    else:
        print(keys.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    # Export
    out = export_dir or OUTPUT_DIR / f"inspect_{executive_key}"
    out.mkdir(parents=True, exist_ok=True)
    most_pos.to_csv(out / "top_positive.csv", index=False)
    most_neg.to_csv(out / "top_negative.csv", index=False)
    mdist.to_csv(out / "monthly.csv", index=False)
    if not keys.empty:
        keys.to_csv(out / "keyness.csv", index=False)
    if event is not None:
        before_after(df, event, window_days).to_csv(out / "before_after.csv", index=False)
    print(f"\nFile CSV salvati in {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--executive", required=True,
                    help="Chiave del dirigente (es. maldini, furlani, cardinale)")
    ap.add_argument("--top-k", type=int, default=20,
                    help="Quante frasi estreme mostrare per polarità")
    ap.add_argument("--event", type=lambda s: datetime.fromisoformat(s).replace(tzinfo=pd.Timestamp.utcnow().tzinfo),
                    default=None,
                    help="Data evento per confronto pre/post (YYYY-MM-DD)")
    ap.add_argument("--window-days", type=int, default=60,
                    help="Finestra ± attorno all'evento")
    ap.add_argument("--out", type=Path, default=None,
                    help="Cartella in cui salvare i CSV diagnostici")
    ap.add_argument("--sentiment", type=Path, default=None,
                    help="File sentiment di input (default data/sentiment.parquet, "
                         "passa data/sentiment_absa.parquet per analizzare ABSA)")
    args = ap.parse_args()
    main(executive_key=args.executive,
         k=args.top_k,
         event=args.event,
         window_days=args.window_days,
         export_dir=args.out,
         sentiment_path=args.sentiment)
