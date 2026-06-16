from __future__ import annotations
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, OUTPUT_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def weekly_sentiment(df_sent: pd.DataFrame) -> pd.DataFrame:
    """Aggregazione settimanale (lunedì) per executive."""
    df = df_sent.copy()
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True)
    df["week"] = df["created_utc"].dt.tz_convert("UTC").dt.to_period("W-MON").dt.start_time
    # Esplodi per executive (se la riga ha lista di execs)
    if "executives" in df.columns and df["executives"].dtype == object \
       and df["executives"].iloc[0:1].apply(lambda x: isinstance(x, (list, np.ndarray))).any():
        df = df.explode("executives").rename(columns={"executives": "executive"})
    agg = (df.groupby(["week", "executive"], as_index=False)
             .agg(mean_score=("sentiment_score", "mean"),
                  n=("sentiment_score", "count")))
    agg_all = (df.groupby(["week"], as_index=False)
                 .agg(mean_score=("sentiment_score", "mean"),
                      n=("sentiment_score", "count")))
    agg_all["executive"] = "__all__"
    return pd.concat([agg, agg_all], ignore_index=True)


def weekly_results(df_matches: pd.DataFrame) -> pd.DataFrame:
    """Aggrega i risultati settimanalmente (solo Serie A per i punti)."""
    df = df_matches.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["week"] = df["date"].dt.to_period("W-MON").dt.start_time
    serie_a = df[df["competition"].str.lower() == "serie a"]
    agg = (serie_a.groupby("week", as_index=False)
                  .agg(points=("points", "sum"),
                       goals_for=("goals_for", "sum"),
                       goals_against=("goals_against", "sum"),
                       n_matches=("points", "count")))
    agg["goal_diff"] = agg["goals_for"] - agg["goals_against"]
    return agg


def correlate(sent_weekly: pd.DataFrame, res_weekly: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import pearsonr, spearmanr
    rows = []
    merged_full = res_weekly.merge(sent_weekly, on="week", how="inner")
    for exec_key, g in merged_full.groupby("executive"):
        g = g.dropna(subset=["mean_score", "points"])
        if len(g) < 5:
            continue
        try:
            pr, pp = pearsonr(g["mean_score"], g["points"])
        except Exception:
            pr, pp = np.nan, np.nan
        try:
            sr, sp = spearmanr(g["mean_score"], g["points"])
        except Exception:
            sr, sp = np.nan, np.nan
        rows.append({
            "executive": exec_key,
            "n_weeks": len(g),
            "pearson_r": pr, "pearson_p": pp,
            "spearman_r": sr, "spearman_p": sp,
        })
    return pd.DataFrame(rows).sort_values("pearson_p")


def correlate_robust(sent_weekly: pd.DataFrame, res_weekly: pd.DataFrame,
                     hac_maxlags: int = 4) -> pd.DataFrame:
    """Correlazione sentiment-punti corretta per autocorrelazione.

    Il p-value 'naive' di Pearson assume osservazioni indipendenti, ipotesi
    violata da serie temporali persistenti come sentiment e forma: l'n effettivo
    è < n e la significatività risulta sovrastimata. Questa funzione riporta,
    oltre all'r di Pearson, due p-value robusti:

    - p_neff: ricalcolato sull'effective sample size con la correzione AR(1)
      n_eff = n * (1 - phi_x*phi_y) / (1 + phi_x*phi_y), dove phi sono le
      autocorrelazioni lag-1 delle due serie;
    - p_hac:  p-value dello slope di una regressione OLS (punti ~ sentiment)
      con standard error Newey-West (HAC), robusti ad autocorrelazione.
    """
    from scipy.stats import t as student_t
    import statsmodels.api as sm

    def lag1_autocorr(v: np.ndarray) -> float:
        v = np.asarray(v, dtype=float)
        v = v - v.mean()
        denom = float(np.sum(v * v))
        return float(np.sum(v[1:] * v[:-1]) / denom) if denom > 0 else 0.0

    rows = []
    merged_full = res_weekly.merge(sent_weekly, on="week", how="inner")
    for exec_key, g in merged_full.groupby("executive"):
        g = g.sort_values("week").dropna(subset=["mean_score", "points"])
        n = len(g)
        if n < 8:
            continue
        x = g["mean_score"].to_numpy()
        y = g["points"].to_numpy()
        r = float(np.corrcoef(x, y)[0, 1])

        # 1) effective sample size (correzione AR(1) su entrambe le serie)
        phix, phiy = lag1_autocorr(x), lag1_autocorr(y)
        factor = (1.0 - phix * phiy) / (1.0 + phix * phiy)
        n_eff = max(3.0, n * factor)
        if abs(r) < 1.0:
            t_eff = r * np.sqrt((n_eff - 2.0) / (1.0 - r ** 2))
            p_neff = float(2.0 * student_t.sf(abs(t_eff), df=n_eff - 2.0))
        else:
            p_neff = 0.0

        # 2) Newey-West / HAC sullo slope OLS (points ~ const + sentiment)
        X = sm.add_constant(x)
        ols = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_maxlags})
        p_hac = float(ols.pvalues[1])

        rows.append({
            "executive": exec_key, "n": n, "pearson_r": r,
            "lag1_sentiment": round(phix, 3), "lag1_points": round(phiy, 3),
            "n_eff": round(n_eff, 1), "p_neff": p_neff, "p_hac": p_hac,
        })
    return pd.DataFrame(rows).sort_values("p_hac")


def granger_test(sent_weekly: pd.DataFrame, res_weekly: pd.DataFrame,
                 max_lag: int = 4) -> pd.DataFrame:
    """Granger causality: sentiment → punti? e viceversa."""
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        logger.warning("statsmodels non disponibile; skip Granger.")
        return pd.DataFrame()
    rows = []
    merged = (res_weekly.merge(sent_weekly[sent_weekly["executive"] == "__all__"],
                                on="week", how="inner")
                        .sort_values("week"))
    if len(merged) < max_lag * 3:
        logger.info("Serie troppo corta per Granger (%d punti). Skip.", len(merged))
        return pd.DataFrame()
    series = merged[["points", "mean_score"]].dropna().to_numpy()
    try:
        res = grangercausalitytests(series, maxlag=max_lag, verbose=False)
        for lag, r in res.items():
            f_test = r[0]["ssr_ftest"]
            rows.append({"direction": "sentiment->points", "lag": lag,
                         "F": f_test[0], "p": f_test[1]})
    except Exception as e:
        logger.warning("Granger sentiment->points fallito: %s", e)
    try:
        res = grangercausalitytests(series[:, ::-1], maxlag=max_lag, verbose=False)
        for lag, r in res.items():
            f_test = r[0]["ssr_ftest"]
            rows.append({"direction": "points->sentiment", "lag": lag,
                         "F": f_test[0], "p": f_test[1]})
    except Exception as e:
        logger.warning("Granger points->sentiment fallito: %s", e)
    return pd.DataFrame(rows)


def main(sent_path: Path | None = None,
         matches_path: Path | None = None,
         out_dir: Path | None = None):
    sent_path = sent_path or DATA_DIR / "sentiment.parquet"
    matches_path = matches_path or DATA_DIR / "matches.csv"
    out_dir = out_dir or OUTPUT_DIR

    df_sent = pd.read_parquet(sent_path)
    df_matches = pd.read_csv(matches_path)
    logger.info("Sentiment %d righe; matches %d righe", len(df_sent), len(df_matches))

    sw = weekly_sentiment(df_sent)
    rw = weekly_results(df_matches)
    sw.to_csv(out_dir / "weekly_sentiment.csv", index=False)
    rw.to_csv(out_dir / "weekly_results.csv", index=False)

    corr = correlate(sw, rw)
    corr.to_csv(out_dir / "correlations.csv", index=False)
    logger.info("Correlazioni salvate (%d righe)", len(corr))

    corr_robust = correlate_robust(sw, rw)
    if not corr_robust.empty:
        corr_robust.to_csv(out_dir / "correlations_robust.csv", index=False)
        logger.info("Correlazioni robuste (autocorrelazione) salvate (%d righe)",
                    len(corr_robust))

    granger = granger_test(sw, rw)
    if not granger.empty:
        granger.to_csv(out_dir / "granger.csv", index=False)
        logger.info("Granger salvato (%d righe)", len(granger))
    return corr, granger


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentiment", type=Path, default=None)
    ap.add_argument("--matches", type=Path, default=None)
    args = ap.parse_args()
    main(sent_path=args.sentiment, matches_path=args.matches)
