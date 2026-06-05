from __future__ import annotations
import argparse
import logging
from pathlib import Path

import pandas as pd

from .config import DATA_DIR
from . import preprocess, ner, sentiment

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


def per_season_paths(season: str) -> dict[str, Path]:
    """File path per la pipeline isolata di una singola stagione."""
    return {
        "raw":       DATA_DIR / f"raw_{season}.parquet",
        "clean":     DATA_DIR / f"clean_{season}.parquet",
        "mentions":  DATA_DIR / f"mentions_{season}.parquet",
        "sentiment": DATA_DIR / f"sentiment_{season}.parquet",
        "absa":      DATA_DIR / f"sentiment_absa_{season}.parquet",
    }


STANDARD = {
    "clean":     DATA_DIR / "clean_text.parquet",
    "mentions":  DATA_DIR / "mentions.parquet",
    "sentiment": DATA_DIR / "sentiment.parquet",
    "absa":      DATA_DIR / "sentiment_absa.parquet",
}


def process_season(season: str, with_absa: bool = True,
                   dry_run: bool = False) -> dict[str, Path]:
    paths = per_season_paths(season)
    if not paths["raw"].exists():
        raise FileNotFoundError(
            f"File raw mancante: {paths['raw']}\n"
            f"  Scaricalo prima con:\n"
            f"  python -m src.scraper --start {season[:4]}-08-01 "
            f"--end {2000 + int(season[5:])}-06-30 --out {paths['raw']}"
        )

    logger.info("=" * 70)
    logger.info("Stagione %s  —  raw: %s", season, paths["raw"].name)
    logger.info("=" * 70)

    if dry_run:
        for step, src, dst in [
            ("preprocess",      paths["raw"],      paths["clean"]),
            ("ner",             paths["clean"],    paths["mentions"]),
            ("sentiment",       paths["mentions"], paths["sentiment"]),
        ] + ([("aspect_sentiment", paths["mentions"], paths["absa"])] if with_absa else []):
            existing = "[skip, esiste]" if dst.exists() else "[da fare]"
            logger.info("  %-18s %-40s %s", step, dst.name, existing)
        return paths

    # 1. preprocess
    if paths["clean"].exists():
        logger.info("Skip preprocess (esiste già: %s)", paths["clean"].name)
    else:
        preprocess.main(in_paths=[paths["raw"]], out_path=paths["clean"])

    # 2. ner
    if paths["mentions"].exists():
        logger.info("Skip ner (esiste già: %s)", paths["mentions"].name)
    else:
        ner.main(in_path=paths["clean"], out_path=paths["mentions"])

    # 3. sentiment vanilla
    if paths["sentiment"].exists():
        logger.info("Skip sentiment vanilla (esiste già: %s)", paths["sentiment"].name)
    else:
        sentiment.main(in_path=paths["mentions"], out_path=paths["sentiment"])

    # 4. ABSA (lazy import: ~1 GB di modelli, evita di caricarli se --skip-absa)
    if with_absa:
        if paths["absa"].exists():
            logger.info("Skip ABSA (esiste già: %s)", paths["absa"].name)
        else:
            from . import aspect_sentiment
            aspect_sentiment.main(in_path=paths["mentions"], out_path=paths["absa"])

    return paths


DEDUP_KEY = {
    "clean": "id",          # post/commenti Reddit, chiave univoca
    "mentions": "sentence_id",
    "sentiment": "sentence_id",
    "absa": "sentence_id",  # ABSA: una riga per (sentence, executive), ma sentence_id è unico per frase
}


def merge_into_standard(seasons: list[str], with_absa: bool = True,
                        dry_run: bool = False):
    keys = ["clean", "mentions", "sentiment"]
    if with_absa:
        keys.append("absa")

    for key in keys:
        std = STANDARD[key]
        dedup = DEDUP_KEY[key]
        new_dfs = []
        for season in seasons:
            p = per_season_paths(season)[key]
            if not p.exists():
                logger.warning("  manca: %s", p.name)
                continue
            new_dfs.append(pd.read_parquet(p))
        if not new_dfs:
            logger.warning("Nessun nuovo dato da mergiare per '%s'", key)
            continue
        new_df = pd.concat(new_dfs, ignore_index=True)

        if std.exists():
            old = pd.read_parquet(std)
            # Per ABSA il dedup va fatto su (sentence_id, executive) — coppia
            if key == "absa" and "executive" in old.columns:
                merged = (pd.concat([old, new_df], ignore_index=True)
                            .drop_duplicates(subset=[dedup, "executive"])
                            .reset_index(drop=True))
            else:
                merged = (pd.concat([old, new_df], ignore_index=True)
                            .drop_duplicates(subset=[dedup])
                            .reset_index(drop=True))
            dup = len(old) + len(new_df) - len(merged)
        else:
            merged = new_df
            dup = 0

        msg = (f"{std.name}: {'[DRY] ' if dry_run else ''}"
               f"{len(new_df):,} nuove + esistenti -> {len(merged):,} totali "
               f"({dup:,} duplicati rimossi)")
        if dry_run:
            logger.info(msg)
        else:
            merged.to_parquet(std, index=False)
            logger.info(msg)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seasons", nargs="+",
                    help="Stagioni da aggiungere, formato YYYY-YY (es. 2021-22 2025-26)")
    ap.add_argument("--skip-absa", action="store_true",
                    help="Salta ABSA (più rapido)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Mostra il piano senza eseguirlo")
    ap.add_argument("--force-rebuild", action="store_true",
                    help="Cancella e ri-genera i file per-stagione esistenti")
    args = ap.parse_args()

    # Validazione formato
    for s in args.seasons:
        if len(s) != 7 or s[4] != "-":
            raise SystemExit(f"Formato stagione non valido: '{s}'. Usa YYYY-YY")

    # Force rebuild: cancella i file per-stagione (non i raw)
    if args.force_rebuild and not args.dry_run:
        logger.info("--force-rebuild: cancello i file per-stagione esistenti")
        for season in args.seasons:
            for key in ("clean", "mentions", "sentiment", "absa"):
                p = per_season_paths(season)[key]
                if p.exists():
                    p.unlink()
                    logger.info("  rimosso: %s", p.name)

    # Processa stagione per stagione
    for season in args.seasons:
        process_season(season, with_absa=not args.skip_absa,
                       dry_run=args.dry_run)

    # Merge nei file standard
    logger.info("\n%s\nMerge nei file standard\n%s", "=" * 70, "=" * 70)
    merge_into_standard(args.seasons, with_absa=not args.skip_absa,
                        dry_run=args.dry_run)

    if not args.dry_run:
        logger.info("\nFatto. Passi successivi consigliati:")
        logger.info("  python -m src.network")
        logger.info("  python -m src.results --sentiment data/sentiment_absa.parquet")
        logger.info("  streamlit run src/dashboard.py")


if __name__ == "__main__":
    main()
