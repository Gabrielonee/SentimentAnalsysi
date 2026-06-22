
from __future__ import annotations
import argparse
import glob as glob_mod
import logging
import re
from pathlib import Path

import pandas as pd
from langdetect import detect, DetectorFactory, LangDetectException
from tqdm import tqdm

from .config import DATA_DIR, TARGET_LANGUAGES

DetectorFactory.seed = 42  # detect è non deterministico di default
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Pattern di pulizia
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MARKDOWN_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")           # [text](url) -> text
QUOTE_RE = re.compile(r"^&gt;.*$", flags=re.MULTILINE)       # quote markdown reddit
MULTI_WS = re.compile(r"\s+")
USER_RE = re.compile(r"/u/\w+|u/\w+")
SUB_RE = re.compile(r"/r/\w+|r/\w+")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = QUOTE_RE.sub(" ", text)
    text = MARKDOWN_RE.sub(r"\1", text)
    text = URL_RE.sub(" ", text)
    text = USER_RE.sub(" ", text)
    text = SUB_RE.sub(" ", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = MULTI_WS.sub(" ", text).strip()
    return text


def detect_lang(text: str) -> str:
    if not text or len(text) < 12:
        return "und"
    try:
        return detect(text)
    except LangDetectException:
        return "und"


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["full_text"] = (df["title"].fillna("") + " " + df["body"].fillna("")).str.strip()
    tqdm.pandas(desc="cleaning")
    df["clean_text"] = df["full_text"].progress_apply(clean_text)
    df = df[df["clean_text"].str.len() >= 12].copy()
    tqdm.pandas(desc="lang detect")
    df["lang"] = df["clean_text"].progress_apply(detect_lang)
    df = df[df["lang"].isin(TARGET_LANGUAGES)].copy()
    # Filtra bot/automoderator
    df = df[~df["author"].fillna("").str.lower().isin({"automoderator", "[deleted]"})]
    # Filtra outlier toxicity-driven
    df = df[df["score"].fillna(0) > -5]
    return df.reset_index(drop=True)


def _resolve_inputs(in_paths: list[Path] | None,
                    glob_pattern: str | None) -> list[Path]:
    """Risolve l'elenco di file di input dall'argomentazione utente."""
    paths: list[Path] = list(in_paths or [])
    if glob_pattern:
        paths.extend(sorted(Path(p) for p in glob_mod.glob(glob_pattern)))
    if not paths:
        paths = [DATA_DIR / "raw_reddit.parquet"]
    # deduplica preservando ordine
    seen, uniq = set(), []
    for p in paths:
        sp = str(p.resolve())
        if sp in seen: continue
        seen.add(sp); uniq.append(p)
    return uniq


def main(in_paths: list[Path] | None = None,
         out_path: Path | None = None,
         glob_pattern: str | None = None):
    paths = _resolve_inputs(in_paths, glob_pattern)
    out_path = out_path or DATA_DIR / "clean_text.parquet"

    dfs = []
    for p in paths:
        d = pd.read_parquet(p)
        logger.info("Letto %s (%d righe)", p, len(d))
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    before = len(df)
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    if before != len(df):
        logger.info("Deduplica: %d -> %d righe (%d duplicati rimossi)",
                    before, len(df), before - len(df))

    out = preprocess_dataframe(df)
    out.to_parquet(out_path, index=False)
    logger.info("Salvato %s (%d righe filtrate)", out_path, len(out))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_paths", type=Path, action="append",
                    default=None, help="File di input parquet (ripetibile).")
    ap.add_argument("--glob", dest="glob_pattern", type=str, default=None,
                    help="Pattern glob, es. 'data/raw_20*.parquet'")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    main(in_paths=args.in_paths, out_path=args.out,
         glob_pattern=args.glob_pattern)
