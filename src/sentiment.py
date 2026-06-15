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
