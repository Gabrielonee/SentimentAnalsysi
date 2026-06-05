from __future__ import annotations
import argparse
import io
import logging
from pathlib import Path

import pandas as pd
import requests

from .config import DATA_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SEASONS = [
    ("2021-22", "https://www.football-data.co.uk/mmz4281/2122/I1.csv"),
    ("2022-23", "https://www.football-data.co.uk/mmz4281/2223/I1.csv"),
    ("2023-24", "https://www.football-data.co.uk/mmz4281/2324/I1.csv"),
    ("2024-25", "https://www.football-data.co.uk/mmz4281/2425/I1.csv"),
    ("2025-26", "https://www.football-data.co.uk/mmz4281/2526/I1.csv"),
]
MILAN = "Milan"


def fetch_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=60, headers={
        "User-Agent": "milan-sentiment-research/0.3 (academic)"
    })
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def extract_milan(df: pd.DataFrame, season: str) -> pd.DataFrame:
    df = df[(df["HomeTeam"] == MILAN) | (df["AwayTeam"] == MILAN)].copy()
    rows = []
    for _, r in df.iterrows():
        date = pd.to_datetime(r["Date"], dayfirst=True, errors="coerce")
        if pd.isna(date):
            continue
        date_utc = pd.Timestamp(date.year, date.month, date.day, tz="UTC")
        home = (r["HomeTeam"] == MILAN)
        opponent = r["AwayTeam"] if home else r["HomeTeam"]
        gf = int(r["FTHG"] if home else r["FTAG"])
        ga = int(r["FTAG"] if home else r["FTHG"])
        ftr = r["FTR"]
        if ftr == "D":
            points = 1
        elif (ftr == "H" and home) or (ftr == "A" and not home):
            points = 3
        else:
            points = 0
        rows.append({
            "date": date_utc.isoformat(),
            "competition": "Serie A",
            "opponent": opponent,
            "home_away": "H" if home else "A",
            "goals_for": gf,
            "goals_against": ga,
            "points": points,
            "season": season,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("date").reset_index(drop=True)
    return out


def main(out_path: Path | None = None, only_season: str | None = None):
    out_path = out_path or DATA_DIR / "matches.csv"
    all_rows: list[pd.DataFrame] = []
    for season, url in SEASONS:
        if only_season and season != only_season:
            continue
        logger.info("Scarico %s -> %s", season, url)
        try:
            raw = fetch_csv(url)
            milan = extract_milan(raw, season)
            logger.info("  -> %d partite Milan", len(milan))
            all_rows.append(milan)
        except Exception as e:
            logger.warning("  ERRORE su %s (%s). Stagione saltata.", season, e)

    if not all_rows:
        raise SystemExit("Nessuna stagione scaricata correttamente.")

    final = pd.concat(all_rows, ignore_index=True).sort_values("date")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(out_path, index=False)
    logger.info("Salvato %s (%d partite totali)", out_path, len(final))

    print("\nRiepilogo per stagione:")
    for season, group in final.groupby("season"):
        w = int((group["points"] == 3).sum())
        d = int((group["points"] == 1).sum())
        l = int((group["points"] == 0).sum())
        pts = int(group["points"].sum())
        gf = int(group["goals_for"].sum())
        ga = int(group["goals_against"].sum())
        print(f"  {season}: {len(group):2d} partite | "
              f"{w}V {d}P {l}S | {pts} pts | {gf} GF {ga} GA")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--season", type=str, default=None)
    args = ap.parse_args()
    main(out_path=args.out, only_season=args.season)
