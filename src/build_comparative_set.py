"""
Estrae dal dataset ABSA i candidati *comparativi / con negazione / multi-target*
— i casi in cui il modello yangheng/deberta-v3-base-absa-v1.1 tende a sbagliare,
perché confonde il sentiment del target con quello dell'entità contrapposta.

Produce un CSV pronto per l'annotazione manuale:
    data/comparative_to_label.csv

Colonne:
    doc_id, sentence_id, aspect_target, sentence_en, n_executives,
    cue            -> quale pattern ha fatto scattare il candidato
    model_label    -> etichetta attuale del modello (da correggere se sbagliata)
    model_score    -> p_pos - p_neg attuale
    gold_label     -> VUOTA: la riempi tu a mano (negative/neutral/positive)
    note           -> VUOTA: opzionale, per appunti di annotazione

Linea guida di annotazione (IMPORTANTISSIMA per la coerenza):
    Il gold_label è il sentiment ESPRESSO VERSO `aspect_target`, non verso la
    frase nel suo complesso. In "X è scarso a differenza di Maldini che era top",
    con aspect=Maldini il gold è POSITIVE anche se la frase contiene parole negative.

Uso:
    python -m src.build_comparative_set                # default: 300 esempi
    python -m src.build_comparative_set --n 400 --seed 7
"""
from __future__ import annotations
import argparse
import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATA_IN = Path("data/sentiment_absa.parquet")
DATA_OUT = Path("data/comparative_to_label.csv")

# Pattern lessicali che segnalano confronto / negazione / nostalgia.
# Ognuno è (nome_cue, regex case-insensitive). L'ordine definisce la priorità
# con cui viene assegnata l'etichetta `cue` se più pattern combaciano.
CUES: list[tuple[str, str]] = [
    ("comparison", r"\b(compared to|compared with|rather than|instead of|as opposed to|unlike|versus|\bvs\b|better than|worse than|more than|less than|other than)\b"),
    ("than",       r"\b\w+er than\b"),                       # "better/worse/cheaper than"
    ("negation",   r"\b(not|isn'?t|aren'?t|wasn'?t|never|no longer|nothing|hardly)\b"),
    ("absence",    r"\b(without|miss(?:es|ed|ing)?|gone|left|fired|sacked|bring back|come back|return)\b"),
    ("nostalgia",  r"\b(used to|back then|those days|remember when|the old|real director|like \w+)\b"),
]
COMPILED = [(name, re.compile(pat, re.IGNORECASE)) for name, pat in CUES]


def _first_cue(text: str) -> str | None:
    for name, rx in COMPILED:
        if rx.search(text):
            return name
    return None


def build(n: int = 300, seed: int = 42) -> pd.DataFrame:
    df = pd.read_parquet(DATA_IN)
    # Lavoriamo sulla versione inglese (input effettivo del modello).
    text_col = "sentence_en" if "sentence_en" in df.columns else "sentence"
    df = df.dropna(subset=[text_col, "aspect_target"]).copy()

    # 1) Candidati per pattern lessicale.
    df["cue"] = df[text_col].astype(str).map(_first_cue)
    lexical = df[df["cue"].notna()].copy()

    # 2) Candidati multi-target (più dirigenti nella stessa frase): qui il
    #    rischio di "contaminazione" del sentiment fra entità è massimo.
    multi = df[df["n_executives"] >= 2].copy()
    multi["cue"] = multi["cue"].fillna("multi_target")

    cand = pd.concat([lexical, multi]).drop_duplicates(
        subset=["sentence_id", "aspect_target"]
    )
    logger.info("Candidati grezzi: %d (lessicali=%d, multi-target=%d)",
                len(cand), len(lexical), len(multi))

    # 3) Campionamento bilanciato: per aspect e per label predetta, così il set
    #    da annotare non è dominato da Maldini/negativi.
    cand["model_score"] = (cand.get("p_pos", 0) - cand.get("p_neg", 0)).round(3)
    cand = cand.rename(columns={"sentiment_label": "model_label"})

    rng = seed
    per_group = max(1, n // (cand["aspect_target"].nunique() * 3))
    sampled = (
        cand.groupby(["aspect_target", "model_label"], group_keys=False)
        .apply(lambda g: g.sample(min(len(g), per_group), random_state=rng))
    )
    # Completa fino a n con un campione casuale dei rimanenti.
    if len(sampled) < n:
        rest = cand.drop(sampled.index)
        extra = rest.sample(min(len(rest), n - len(sampled)), random_state=rng)
        sampled = pd.concat([sampled, extra])
    sampled = sampled.sample(min(len(sampled), n), random_state=rng).reset_index(drop=True)

    out = sampled[[
        "doc_id", "sentence_id", "aspect_target", text_col,
        "n_executives", "cue", "model_label", "model_score",
    ]].rename(columns={text_col: "sentence_en"})
    out["gold_label"] = ""      # da riempire a mano: negative / neutral / positive
    out["note"] = ""
    return out


def main(n: int, seed: int, out_path: Path):
    out = build(n=n, seed=seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    logger.info("Scritti %d esempi da annotare in %s", len(out), out_path)
    logger.info("Distribuzione per cue:\n%s", out["cue"].value_counts().to_string())
    logger.info("Distribuzione per aspect:\n%s", out["aspect_target"].value_counts().to_string())
    print(f"\nProssimo passo: apri {out_path}, riempi la colonna 'gold_label' "
          f"(negative/neutral/positive) seguendo la linea guida nell'header dello script.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="Numero di esempi da estrarre")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=DATA_OUT)
    args = ap.parse_args()
    main(n=args.n, seed=args.seed, out_path=args.out)
