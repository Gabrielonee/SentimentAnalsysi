from __future__ import annotations
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import DATA_DIR, SENTIMENT_MODEL

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _auto_device():
    try:
        import torch
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    return -1


class TransformerSentiment:
    LABELS = ["negative", "neutral", "positive"]

    def __init__(self, model_name: str = SENTIMENT_MODEL,
                 device: int | str | None = None, batch_size: int = 32):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
        if device is None:
            device = _auto_device()
        logger.info("Sentiment device: %s", device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.pipe = pipeline(
            task="sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
            top_k=None,
            truncation=True,
        )
        self.batch_size = batch_size

    @staticmethod
    def _to_class_list(r) -> list[dict]:
        if isinstance(r, dict):
            return [r]
        if isinstance(r, list) and r and isinstance(r[0], list):
            return r[0]
        return r

    def score(self, texts: list[str]) -> list[dict]:
        out = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="sentiment"):
            chunk = texts[i:i + self.batch_size]
            res = self.pipe(chunk)
            for r in res:
                class_list = self._to_class_list(r)
                probs = {d["label"].lower(): float(d["score"]) for d in class_list}
                neg = probs.get("negative") or probs.get("label_0") or 0.0
                neu = probs.get("neutral")  or probs.get("label_1") or 0.0
                pos = probs.get("positive") or probs.get("label_2") or 0.0
                label = self.LABELS[int(np.argmax([neg, neu, pos]))]
                out.append({
                    "sentiment_label": label,
                    "p_neg": neg,
                    "p_neu": neu,
                    "p_pos": pos,
                    "sentiment_score": pos - neg,
                })
        return out


class LexiconSentiment:
    """Fallback rapido VADER (EN) + lessico minimale (IT)."""
    LABELS = ["negative", "neutral", "positive"]

    POS_IT = {"forte", "bravo", "grande", "vittoria", "gol", "vinciamo",
              "ottimo", "miglior", "campione", "scudetto", "geniale",
              "sopra le righe", "applausi", "top", "fenomeno"}
    NEG_IT = {"esonero", "out", "scarso", "scandalo", "vergogna", "disastro",
              "delusione", "errore", "peggior", "tragedia", "incompetente",
              "via", "dimettiti"}

    def __init__(self):
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            import nltk
            try:
                self.vader = SentimentIntensityAnalyzer()
            except LookupError:
                nltk.download("vader_lexicon", quiet=True)
                self.vader = SentimentIntensityAnalyzer()
        except ImportError:
            self.vader = None

    def _score_it(self, text: str) -> float:
        t = text.lower()
        pos = sum(1 for w in self.POS_IT if w in t)
        neg = sum(1 for w in self.NEG_IT if w in t)
        if pos + neg == 0: return 0.0
        return (pos - neg) / (pos + neg)

    def _score_en(self, text: str) -> float:
        if self.vader is None: return 0.0
        return self.vader.polarity_scores(text)["compound"]

    def score(self, texts: list[str], langs: list[str] | None = None) -> list[dict]:
        out = []
        langs = langs or ["en"] * len(texts)
        for text, lang in zip(texts, langs):
            s = self._score_it(text) if lang == "it" else self._score_en(text)
            if   s >  0.15: label, p_neg, p_neu, p_pos = "positive", 0.1, 0.2, 0.7
            elif s < -0.15: label, p_neg, p_neu, p_pos = "negative", 0.7, 0.2, 0.1
            else:           label, p_neg, p_neu, p_pos = "neutral",  0.25, 0.5, 0.25
            out.append({
                "sentiment_label": label,
                "p_neg": p_neg, "p_neu": p_neu, "p_pos": p_pos,
                "sentiment_score": float(s),
            })
        return out


def main(in_path: Path | None = None,
         out_path: Path | None = None,
         backend: str = "transformer"):
    in_path = in_path or DATA_DIR / "mentions.parquet"
    out_path = out_path or DATA_DIR / "sentiment.parquet"
    df = pd.read_parquet(in_path)
    logger.info("Letto %s (%d frasi)", in_path, len(df))

    if backend == "transformer":
        engine = TransformerSentiment()
        scores = engine.score(df["sentence"].tolist())
    else:
        engine = LexiconSentiment()
        scores = engine.score(df["sentence"].tolist(), df["lang"].tolist())

    scores_df = pd.DataFrame(scores)
    out = pd.concat([df.reset_index(drop=True), scores_df], axis=1)
    out.to_parquet(out_path, index=False)
    logger.info("Salvato %s (%d righe)", out_path, len(out))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--backend", choices=["transformer", "lexicon"],
                    default="transformer",
                    help="Backend di sentiment. 'lexicon' è il fallback rapido.")
    args = ap.parse_args()
    main(in_path=args.in_path, out_path=args.out, backend=args.backend)
