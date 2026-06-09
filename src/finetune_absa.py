"""
Fine-tuning locale (Mac MPS / CPU) di yangheng/deberta-v3-base-absa-v1.1 sui
casi comparativi annotati a mano, con valutazione *before / after* su un test set
tenuto da parte.

Pipeline:
  1. Carica data/comparative_to_label.csv e tiene solo le righe con gold_label.
  2. (Opzionale, consigliato) aggiunge un "anchor set" anti-forgetting: esempi
     su cui il modello base è molto sicuro (|p_pos-p_neg|>soglia), usando la sua
     stessa predizione come pseudo-gold. Serve a non far dimenticare l'ABSA
     generale mentre si specializza sui comparativi.
  3. Split stratificato train/val/test.
  4. Valuta il modello BASE sul test (baseline).
  5. Fine-tune (full oppure LoRA con --lora se 'peft' è installato).
  6. Rivaluta sul test e stampa il confronto + matrice di confusione.
  7. Salva il modello in output/absa_finetuned/.

L'input al modello è sempre la coppia (aspect, sentence) — identico a
src/aspect_sentiment.AspectSentimentAnalyzer.score_pairs, così il fine-tuning
è coerente con come usi il modello in produzione.

Uso tipico:
    python -m src.finetune_absa --epochs 4 --anchors 600
    python -m src.finetune_absa --lora            # se hai installato peft
    python -m src.finetune_absa --no-train        # solo baseline sul test set

NB: con poche centinaia di esempi l'obiettivo non è "risolvere" negazione e
comparazione, ma mostrare un miglioramento MISURABILE sul test comparativo
senza degradare i casi generali (controllati via anchor set nel report).
"""
from __future__ import annotations
import argparse
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ABSA_MODEL = "yangheng/deberta-v3-base-absa-v1.1"
LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

LABELED_CSV = Path("data/comparative_to_label.csv")
FULL_PARQUET = Path("data/sentiment_absa.parquet")
OUT_DIR = Path("output/absa_finetuned")


# --------------------------------------------------------------------------- #
# Dati
# --------------------------------------------------------------------------- #
def _normalize_label(x: str) -> str | None:
    if not isinstance(x, str):
        return None
    x = x.strip().lower()
    aliases = {"neg": "negative", "neu": "neutral", "pos": "positive",
               "negative": "negative", "neutral": "neutral", "positive": "positive"}
    return aliases.get(x)


def load_labeled() -> pd.DataFrame:
    df = pd.read_csv(LABELED_CSV)
    df["gold"] = df["gold_label"].map(_normalize_label)
    df = df.dropna(subset=["gold", "sentence_en", "aspect_target"]).copy()
    if df.empty:
        raise SystemExit(
            f"Nessuna riga con gold_label valida in {LABELED_CSV}.\n"
            "Apri il CSV e riempi la colonna 'gold_label' (negative/neutral/positive)."
        )
    df = df.rename(columns={"sentence_en": "sentence", "aspect_target": "aspect"})
    df["source"] = "labeled"
    logger.info("Esempi annotati: %d  (distribuzione: %s)",
                len(df), Counter(df["gold"]))
    return df[["sentence", "aspect", "gold", "source"]]


def load_anchors(n: int) -> pd.DataFrame:
    """Esempi ad alta confidenza dal dataset completo: pseudo-gold = label del
    modello base. Bilanciati per classe. Servono come ancora anti-forgetting."""
    if n <= 0:
        return pd.DataFrame(columns=["sentence", "aspect", "gold", "source"])
    df = pd.read_parquet(FULL_PARQUET)
    df["conf"] = (df["p_pos"] - df["p_neg"]).abs()
    # alta confidenza e label netta
    strong = df[(df["conf"] > 0.8)].copy()
    per_class = max(1, n // 3)
    picks = []
    for lab in LABELS:
        sub = strong[strong["sentiment_label"] == lab]
        picks.append(sub.sample(min(len(sub), per_class), random_state=0))
    anc = pd.concat(picks)
    # Seleziono PRIMA le 3 colonne giuste: il parquet ha già una colonna
    # 'sentence' (testo originale), quindi rinominare 'sentence_en'->'sentence'
    # creerebbe due colonne omonime e romperebbe il concat a valle.
    anc = anc[["sentence_en", "aspect_target", "sentiment_label"]].rename(
        columns={"sentence_en": "sentence",
                 "aspect_target": "aspect",
                 "sentiment_label": "gold"})
    anc["source"] = "anchor"
    anc = anc.reset_index(drop=True)
    logger.info("Anchor set: %d  (distribuzione: %s)",
                len(anc), Counter(anc["gold"]))
    return anc[["sentence", "aspect", "gold", "source"]]


def stratified_split(df: pd.DataFrame, seed: int = 42):
    """Split 70/15/15 stratificato per (gold). Gli anchor restano nel train."""
    labeled = df[df["source"] == "labeled"]
    anchors = df[df["source"] == "anchor"]
    train_parts, val_parts, test_parts = [], [], []
    rng = np.random.default_rng(seed)
    for lab, g in labeled.groupby("gold"):
        idx = rng.permutation(g.index.to_numpy())
        n = len(idx)
        n_test = max(1, int(0.15 * n))
        n_val = max(1, int(0.15 * n))
        test_parts.append(g.loc[idx[:n_test]])
        val_parts.append(g.loc[idx[n_test:n_test + n_val]])
        train_parts.append(g.loc[idx[n_test + n_val:]])
    train = pd.concat(train_parts + [anchors]).sample(frac=1, random_state=seed)
    val = pd.concat(val_parts)
    test = pd.concat(test_parts)
    logger.info("Split -> train=%d (incl. %d anchor)  val=%d  test=%d",
                len(train), len(anchors), len(val), len(test))
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Torch dataset
# --------------------------------------------------------------------------- #
def make_dataset(df: pd.DataFrame, tokenizer, max_len: int):
    import torch

    class DS(torch.utils.data.Dataset):
        def __len__(self): return len(df)

        def __getitem__(self, i):
            row = df.iloc[i]
            enc = tokenizer(str(row["aspect"]), str(row["sentence"]),
                            truncation=True, max_length=max_len,
                            padding="max_length", return_tensors="pt")
            item = {k: v.squeeze(0) for k, v in enc.items()}
            item["labels"] = torch.tensor(LABEL2ID[row["gold"]], dtype=torch.long)
            return item

    return DS()


# --------------------------------------------------------------------------- #
# Metriche
# --------------------------------------------------------------------------- #
def macro_f1(y_true, y_pred) -> tuple[float, dict]:
    f1s = {}
    for c, name in enumerate(LABELS):
        tp = sum((p == c and t == c) for t, p in zip(y_true, y_pred))
        fp = sum((p == c and t != c) for t, p in zip(y_true, y_pred))
        fn = sum((p != c and t == c) for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s[name] = round(f1, 3)
    return round(sum(f1s.values()) / len(f1s), 3), f1s


def confusion(y_true, y_pred) -> str:
    m = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[t, p] += 1
    hdr = "true\\pred  " + "  ".join(f"{l[:3]:>4}" for l in LABELS)
    rows = [hdr]
    for i, l in enumerate(LABELS):
        rows.append(f"{l:>9}  " + "  ".join(f"{m[i, j]:>4}" for j in range(3)))
    return "\n".join(rows)


def evaluate(model, tokenizer, test_df, device, max_len) -> dict:
    import torch
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for _, row in test_df.iterrows():
            enc = tokenizer(str(row["aspect"]), str(row["sentence"]),
                            truncation=True, max_length=max_len,
                            return_tensors="pt").to(device)
            logits = model(**enc).logits
            y_pred.append(int(logits.argmax(-1).item()))
            y_true.append(LABEL2ID[row["gold"]])
    acc = round(sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true), 3)
    mf1, per = macro_f1(y_true, y_pred)
    return {"accuracy": acc, "macro_f1": mf1, "per_class_f1": per,
            "confusion": confusion(y_true, y_pred)}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main(epochs: int, anchors: int, lr: float, batch: int, max_len: int,
         use_lora: bool, do_train: bool, seed: int):
    import torch
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              TrainingArguments, Trainer)

    device = _device()
    logger.info("Device: %s", device)

    data = pd.concat([load_labeled(), load_anchors(anchors)], ignore_index=True)
    train_df, val_df, test_df = stratified_split(data, seed=seed)

    tokenizer = AutoTokenizer.from_pretrained(ABSA_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(ABSA_MODEL).to(device)

    # Allinea la mappa label del modello al nostro ordine [neg, neu, pos].
    base_id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    logger.info("Label map del modello base: %s", base_id2label)
    if [base_id2label.get(i) for i in range(3)] != LABELS:
        logger.warning("ATTENZIONE: l'ordine label del modello (%s) differisce da "
                       "%s. Verifica la mappatura prima di interpretare i numeri.",
                       base_id2label, LABELS)

    logger.info("=== Baseline (modello base) sul test set ===")
    base_metrics = evaluate(model, tokenizer, test_df, device, max_len)
    logger.info("BASE  acc=%.3f  macroF1=%.3f  per-classe=%s",
                base_metrics["accuracy"], base_metrics["macro_f1"],
                base_metrics["per_class_f1"])
    print("\nMatrice di confusione BASE:\n" + base_metrics["confusion"] + "\n")

    if not do_train:
        logger.info("--no-train: mi fermo alla baseline.")
        return

    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(task_type="SEQ_CLS", r=8, lora_alpha=16,
                             lora_dropout=0.1, target_modules=["query_proj", "value_proj"])
            model = get_peft_model(model, cfg)
            model.print_trainable_parameters()
        except ImportError:
            logger.warning("peft non installato: passo al full fine-tuning. "
                           "(pip install peft per usare LoRA)")
            use_lora = False

    train_ds = make_dataset(train_df, tokenizer, max_len)
    val_ds = make_dataset(val_df, tokenizer, max_len)

    args = TrainingArguments(
        output_dir=str(OUT_DIR / "_checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_steps=int(0.1 * (len(train_ds) / batch) * epochs),
        eval_strategy="epoch",
        # Salva a ogni epoca e tieni il MIGLIOR checkpoint (per macro-F1 sul val),
        # non l'ultimo: con dataset piccoli il modello overfitta e le epoche
        # finali peggiorano. Senza questo si salverebbe il modello peggiore.
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=10,
        seed=seed,
        # MPS è auto-rilevato dalle versioni recenti di transformers
        # (l'argomento use_mps_device è stato rimosso).
        fp16=False,        # MPS non gestisce bene fp16
        report_to=[],
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        mf1, _ = macro_f1(list(labels), list(preds))
        acc = float((preds == labels).mean())
        return {"accuracy": round(acc, 3), "macro_f1": mf1}

    from transformers import EarlyStoppingCallback
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=val_ds, compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    logger.info("=== Fine-tuning (%s) ===", "LoRA" if use_lora else "full")
    trainer.train()

    logger.info("=== Modello fine-tuned sul test set ===")
    ft_metrics = evaluate(model, tokenizer, test_df, device, max_len)
    logger.info("FT    acc=%.3f  macroF1=%.3f  per-classe=%s",
                ft_metrics["accuracy"], ft_metrics["macro_f1"],
                ft_metrics["per_class_f1"])
    print("\nMatrice di confusione FINE-TUNED:\n" + ft_metrics["confusion"] + "\n")

    # Report di confronto.
    print("=" * 56)
    print(f"{'metrica':<14}{'BASE':>10}{'FINE-TUNED':>14}{'Δ':>10}")
    for k in ("accuracy", "macro_f1"):
        b, f = base_metrics[k], ft_metrics[k]
        print(f"{k:<14}{b:>10.3f}{f:>14.3f}{f - b:>+10.3f}")
    print("=" * 56)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    logger.info("Modello salvato in %s", OUT_DIR)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--anchors", type=int, default=600,
                    help="N esempi anti-forgetting ad alta confidenza (0 per disattivare)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--lora", action="store_true", help="Usa LoRA (richiede peft)")
    ap.add_argument("--no-train", dest="do_train", action="store_false",
                    help="Calcola solo la baseline sul test set")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(epochs=args.epochs, anchors=args.anchors, lr=args.lr, batch=args.batch,
         max_len=args.max_len, use_lora=args.lora, do_train=args.do_train,
         seed=args.seed)
