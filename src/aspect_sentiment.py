from __future__ import annotations
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import DATA_DIR, EXEC_BY_KEY, EXECUTIVES
from .sentiment import _auto_device

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ABSA_MODEL = "yangheng/deberta-v3-base-absa-v1.1"
TRANSLATOR_MODEL = "Helsinki-NLP/opus-mt-it-en"


class TranslatorIT_EN:
    def __init__(self, batch_size: int = 16, max_length: int = 256):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.tokenizer = AutoTokenizer.from_pretrained(TRANSLATOR_MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATOR_MODEL)
        self.device = _auto_device()
        self.dev_str = self._device_string(self.device)
        if self.device != -1:
            self.model = self.model.to(self.dev_str)
        self.model.eval()
        self.batch_size = batch_size
        self.max_length = max_length
        logger.info("Translator caricato (%s, device=%s)", TRANSLATOR_MODEL, self.dev_str)

    @staticmethod
    def _device_string(device) -> str:
        if device == -1: return "cpu"
        if isinstance(device, str): return device
        return f"cuda:{device}"

    def translate(self, texts: list[str]) -> list[str]:
        import torch
        out: list[str] = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="IT->EN"):
            chunk = texts[i:i + self.batch_size]
            enc = self.tokenizer(chunk, return_tensors="pt", padding=True,
                                 truncation=True, max_length=self.max_length)
            if self.device != -1:
                enc = {k: v.to(self.dev_str) for k, v in enc.items()}
            with torch.no_grad():
                generated = self.model.generate(**enc,
                                                max_length=self.max_length,
                                                num_beams=2,
                                                early_stopping=True)
            decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            out.extend(decoded)
        return out


class AspectSentimentAnalyzer:
    LABELS = ["negative", "neutral", "positive"]

    def __init__(self, batch_size: int = 32):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tokenizer = AutoTokenizer.from_pretrained(ABSA_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(ABSA_MODEL)
        self.device = _auto_device()
        if self.device != -1:
            import torch
            dev_str = self.device if isinstance(self.device, str) else f"cuda:{self.device}"
            self.model = self.model.to(dev_str)
        self.model.eval()
        self.batch_size = batch_size
        self.id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        logger.info("ABSA caricato (%s) – label map: %s", ABSA_MODEL, self.id2label)

    @staticmethod
    def normalize_target(executive_key: str) -> str:
        import unicodedata
        info = EXEC_BY_KEY[executive_key]
        last = info.display_name.split()[-1]
        nfkd = unicodedata.normalize("NFKD", last)
        return "".join(c for c in nfkd if not unicodedata.combining(c))

    def score_pairs(self, sentences: list[str], aspects: list[str]) -> list[dict]:
        import torch
        out = []
        N = len(sentences)
        for i in tqdm(range(0, N, self.batch_size), desc="absa"):
            sent_chunk = sentences[i:i + self.batch_size]
            asp_chunk  = aspects[i:i + self.batch_size]
            enc = self.tokenizer(asp_chunk, sent_chunk,
                                 padding=True, truncation=True,
                                 max_length=256, return_tensors="pt")
            if self.device != -1:
                dev_str = self.device if isinstance(self.device, str) else f"cuda:{self.device}"
                enc = {k: v.to(dev_str) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits
                probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            for p in probs:
                d = {self.id2label[j]: float(p[j]) for j in range(len(p))}
                neg = d.get("negative", 0.0)
                neu = d.get("neutral",  0.0)
                pos = d.get("positive", 0.0)
                label = self.LABELS[int(np.argmax([neg, neu, pos]))]
                out.append({
                    "sentiment_label": label,
                    "p_neg": neg, "p_neu": neu, "p_pos": pos,
                    "sentiment_score": pos - neg,
                })
        return out


def main(in_path: Path | None = None,
         out_path: Path | None = None,
         translate_it: bool = True,
         max_rows: int | None = None):
    in_path  = in_path or DATA_DIR / "mentions.parquet"
    out_path = out_path or DATA_DIR / "sentiment_absa.parquet"

    df = pd.read_parquet(in_path)
    if max_rows:
        df = df.head(max_rows)
    logger.info("Letto %s (%d frasi)", in_path, len(df))

    long = df.explode("executives").rename(columns={"executives": "executive"})
    long = long.reset_index(drop=True)
    logger.info("Coppie (frase, dirigente) da analizzare: %d", len(long))

    sentences = long["sentence"].astype(str).tolist()
    if translate_it:
        it_mask = (long["lang"] == "it").to_numpy()
        n_it = int(it_mask.sum())
        if n_it:
            logger.info("Traduco %d frasi IT -> EN", n_it)
            trans = TranslatorIT_EN()
            it_sentences = [sentences[i] for i in np.where(it_mask)[0]]
            it_translated = trans.translate(it_sentences)
            for j, idx in enumerate(np.where(it_mask)[0]):
                sentences[idx] = it_translated[j]
            del trans
            try:
                import torch
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass

    aspects = [AspectSentimentAnalyzer.normalize_target(k) for k in long["executive"]]
    analyzer = AspectSentimentAnalyzer()
    scores = analyzer.score_pairs(sentences, aspects)
    scores_df = pd.DataFrame(scores)

    long["sentence_en"] = sentences
    long["aspect_target"] = aspects
    out = pd.concat([long.reset_index(drop=True), scores_df], axis=1)
    out.to_parquet(out_path, index=False)
    logger.info("Salvato %s (%d righe)", out_path, len(out))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-translate", action="store_true",
                    help="Salta la traduzione IT->EN (le frasi IT verranno valutate "
                         "dal modello in EN: performance ridotte ma più veloce).")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="Limita il numero di righe (utile per smoke test).")
    args = ap.parse_args()
    main(in_path=args.in_path, out_path=args.out,
         translate_it=not args.no_translate, max_rows=args.max_rows)
