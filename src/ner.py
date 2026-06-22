from __future__ import annotations
import argparse
import logging
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .config import DATA_DIR, EXECUTIVES

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SENTENCE_SPLIT = re.compile(r"(?<=[\.!?])\s+(?=[A-ZÀ-Ý\"'])")


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = SENTENCE_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def build_alias_index() -> list[tuple[str, str, re.Pattern]]:
    """Ritorna lista di (key, alias, regex word-boundary case-insensitive)."""
    out = []
    for e in EXECUTIVES:
        for a in e.aliases:
            # Escape regex; word boundary per evitare match parziali (es. "Ibra" in "library")
            pattern = re.compile(rf"(?<!\w){re.escape(a)}(?!\w)", flags=re.IGNORECASE)
            out.append((e.key, a, pattern))
    # Ordina dal più lungo al più corto per matchare prima le forme complete
    out.sort(key=lambda x: len(x[1]), reverse=True)
    return out


ALIAS_INDEX = build_alias_index()


def annotate_sentence(sentence: str) -> list[str]:
    """Ritorna l'elenco di chiavi dirigenti citati nella frase (deduplicato)."""
    found: set[str] = set()
    for key, _alias, pat in ALIAS_INDEX:
        if pat.search(sentence):
            found.add(key)
    return sorted(found)


def annotate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Espande il dataframe a granularità (riga, frase) con executive list."""
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="NER"):
        sentences = split_sentences(r["clean_text"])
        for sent_idx, sent in enumerate(sentences):
            keys = annotate_sentence(sent)
            if not keys:
                continue
            rows.append({
                "doc_id": r["id"],
                "sentence_id": f"{r['id']}#s{sent_idx}",
                "created_utc": r["created_utc"],
                "lang": r["lang"],
                "score": r["score"],
                "sentence": sent,
                "executives": keys,                  # lista
                "n_executives": len(keys),
            })
    return pd.DataFrame(rows)


def main(in_path: Path | None = None, out_path: Path | None = None):
    in_path = in_path or DATA_DIR / "clean_text.parquet"
    out_path = out_path or DATA_DIR / "mentions.parquet"
    df = pd.read_parquet(in_path)
    logger.info("Letto %s (%d righe)", in_path, len(df))
    out = annotate_dataframe(df)
    out.to_parquet(out_path, index=False)
    logger.info("Salvato %s (%d frasi con menzioni)", out_path, len(out))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    main(in_path=args.in_path, out_path=args.out)
