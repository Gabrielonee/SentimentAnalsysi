"""
Valutazione MIRATA del modello fine-tuned vs modello base.

L'accuratezza aggregata sul test set comparativo può restare piatta, ma la
domanda di tesi è diversa: il fine-tuning corregge il BIAS SPECIFICO su Maldini
(frasi nostalgiche/comparative classificate erroneamente come negative)?

Questo script fa due cose:
  1. PROBE qualitativo: una lista di frasi-chiave (i casi che ci interessano)
     valutate da base vs fine-tuned, affiancate.
  2. ANALISI sul dataset: per ogni dirigente, ricalcola il sentiment con il
     modello fine-tuned sulle frasi "a rischio" (comparazione/negazione/
     nostalgia/assenza) e confronta il tasso di NEGATIVE base vs fine-tuned.
     Per Maldini ci aspettiamo che i negativi scendano (bias corretto).

Uso:
    python -m src.eval_finetuned                      # probe + Maldini
    python -m src.eval_finetuned --all-aspects        # tutti i dirigenti
    python -m src.eval_finetuned --max-per-aspect 800 # limita per velocità
"""
from __future__ import annotations
import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_MODEL = "yangheng/deberta-v3-base-absa-v1.1"
FT_DIR = Path("output/absa_finetuned")
DATA = Path("data/sentiment_absa.parquet")
LABELS = ["negative", "neutral", "positive"]

CUES = [
    r"\b(compared to|compared with|rather than|instead of|as opposed to|unlike|versus|\bvs\b|better than|worse than|other than)\b",
    r"\b\w+er than\b",
    r"\b(not|isn'?t|aren'?t|wasn'?t|never|no longer|nothing|hardly)\b",
    r"\b(without|miss(?:es|ed|ing)?|gone|left|fired|sacked|bring back|come back|return)\b",
    r"\b(used to|back then|those days|remember when|the old|real director|like \w+)\b",
]
CUE_RE = [re.compile(p, re.I) for p in CUES]

# Frasi-sonda: i casi emblematici del problema (target = aspect).
PROBES = [
    ("Maldini", "Zlatan is not Maldini, I love King Z but he's all talk compared to Maldini who walked the walk."),
    ("Maldini", "Why are we not bringing players like Maldini promised?"),
    ("Maldini", "Is it really so hard to bring back maldini and let him do his thing"),
    ("Maldini", "We have become a selling club since Cardinale arrived and Maldini was sacked."),
    ("Maldini", "A dismissal can never erase the history of Paolo Maldini, a great captain and example of Milanism."),
    ("Cardinale", "Cardinale destroyed the club by firing Maldini, what a clown."),
    ("Furlani", "Furlani is doing a great job, the budget is balanced and the team is competitive."),
    ("Ibrahimovic", "You can't not love Zlatan."),
]


class Scorer:
    def __init__(self, model_name_or_path: str):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tok = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.model.eval()
        self.device = ("mps" if torch.backends.mps.is_available()
                       else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}

    def predict(self, aspect: str, sentence: str) -> tuple[str, float]:
        import torch
        enc = self.tok(str(aspect), str(sentence), truncation=True,
                       max_length=256, return_tensors="pt").to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(**enc).logits, dim=-1)[0].cpu().numpy()
        d = {self.id2label[i]: float(probs[i]) for i in range(len(probs))}
        neg, pos = d.get("negative", 0.0), d.get("positive", 0.0)
        label = max(d, key=d.get)
        return label, pos - neg

    def predict_batch(self, aspects, sentences, batch=32) -> list[str]:
        import torch
        out = []
        for i in range(0, len(sentences), batch):
            a = list(aspects[i:i + batch]); s = list(sentences[i:i + batch])
            enc = self.tok(a, s, truncation=True, max_length=256,
                           padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits
                idx = logits.argmax(-1).cpu().numpy()
            out.extend(self.id2label[int(j)] for j in idx)
        return out


def has_cue(t: str) -> bool:
    t = str(t)
    return any(rx.search(t) for rx in CUE_RE)


def main(all_aspects: bool, max_per_aspect: int | None):
    if not FT_DIR.exists():
        raise SystemExit(f"Modello fine-tuned non trovato in {FT_DIR}. "
                         "Esegui prima src.finetune_absa.")

    logger.info("Carico modello BASE…")
    base = Scorer(BASE_MODEL)
    logger.info("Carico modello FINE-TUNED…")
    ft = Scorer(str(FT_DIR))

    # ---- 1. PROBE qualitativo ----
    print("\n" + "=" * 78)
    print("PROBE: frasi-chiave  (target | base → fine-tuned)")
    print("=" * 78)
    for asp, sent in PROBES:
        bl, bs = base.predict(asp, sent)
        fl, fs = ft.predict(asp, sent)
        flag = "  <-- CAMBIATO" if bl != fl else ""
        print(f"\n[{asp}] {sent[:90]}")
        print(f"   base: {bl:>8} ({bs:+.2f})   →   FT: {fl:>8} ({fs:+.2f}){flag}")

    # ---- 2. ANALISI sul dataset (frasi a rischio) ----
    df = pd.read_parquet(DATA)
    tc = "sentence_en" if "sentence_en" in df.columns else "sentence"
    df = df[df[tc].map(has_cue)].copy()       # solo frasi a rischio
    aspects = ["Maldini"] if not all_aspects else \
        ["Maldini", "Cardinale", "Furlani", "Ibrahimovic", "Moncada", "Massara", "Gazidis"]

    print("\n" + "=" * 78)
    print("BIAS sulle frasi A RISCHIO (comparazione/negazione/nostalgia/assenza)")
    print("Tasso di NEGATIVE: base vs fine-tuned  (Δ negativo = bias corretto)")
    print("=" * 78)
    print(f"{'dirigente':<14}{'n':>6}{'%neg base':>12}{'%neg FT':>10}{'Δ':>9}")
    for asp in aspects:
        sub = df[df["aspect_target"] == asp]
        if max_per_aspect:
            sub = sub.head(max_per_aspect)
        if sub.empty:
            continue
        base_neg = (sub["sentiment_label"] == "negative").mean()   # già = base nel parquet
        ft_labels = ft.predict_batch(sub["aspect_target"].tolist(), sub[tc].tolist())
        ft_neg = np.mean([l == "negative" for l in ft_labels])
        print(f"{asp:<14}{len(sub):>6}{base_neg*100:>11.1f}%{ft_neg*100:>9.1f}%{(ft_neg-base_neg)*100:>+8.1f}")
    print("\nNota: %neg base = etichette già salvate nel parquet (modello base).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-aspects", action="store_true",
                    help="Analizza tutti i dirigenti (default: solo Maldini)")
    ap.add_argument("--max-per-aspect", type=int, default=None,
                    help="Limita le frasi per dirigente (più veloce su MPS/CPU)")
    args = ap.parse_args()
    main(all_aspects=args.all_aspects, max_per_aspect=args.max_per_aspect)
