from __future__ import annotations
import argparse
import html
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from .config import OUTPUT_DIR, SENTIMENT_MODEL, EXEC_BY_KEY
from .aspect_sentiment import ABSA_MODEL, AspectSentimentAnalyzer
from .ner import annotate_sentence
from .sentiment import _auto_device

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

LABELS_VAN  = ["negative", "neutral", "positive"]
LABELS_ABSA = ["negative", "neutral", "positive"]

DEFAULT_EXAMPLES = [
    {"aspect": "Maldini",
     "sentence": "Zlatan is not Maldini, I love King Z but he's all talk compared to Maldini who walked the walk."},
    {"aspect": "Maldini",
     "sentence": "And some of the players that have arrived since Maldini left, omg the team just looks worse and worse."},
    {"aspect": "Maldini",
     "sentence": "Bro they took out Paolo Maldini which is all you need to know."},
    {"aspect": "Maldini",
     "sentence": "It's a direct straight line to the bottom right on a graph from the day Maldini got fired."},
    {"aspect": "Cardinale",
     "sentence": "Cardinale destroyed the club by firing Maldini, what a clown."},
    {"aspect": "Furlani",
     "sentence": "Furlani is doing a great job, the budget is balanced and the team is competitive."},
    {"aspect": "Ibrahimovic",
     "sentence": "Ibra is all talk and nothing else, he doesn't really decide anything in the club."},
    # --- Casi multi-aspect: 2 dirigenti con polarità opposte (estratti reali da r/ACMilan) ---
    {"aspect": "Furlani",
     "sentence": "Furlani really is the Anti-Maldini."},
    {"aspect": "Ibrahimovic",
     "sentence": "We gave up Maldini basically for Ibra"},
    {"aspect": "Moncada",
     "sentence": "I have faith in Moncada but not in Furlani."},
    {"aspect": "Maldini",
     "sentence": "Maldini is 10x the leader Ibra is off the pitch."},
    {"aspect": "Cardinale",
     "sentence": "He and cardinale must leave, I want maldini back."},
]


@dataclass
class Attribution:
    tokens: list[str]
    scores: list[float]
    pred_label: str
    pred_probs: dict[str, float]
    target_class: str


def _normalize_scores(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    mx = max(abs(s) for s in scores) or 1.0
    return [s / mx for s in scores]


def _color_for(score: float) -> str:
    """Rosso per negativo, blu per positivo, intensità proporzionale."""
    a = min(abs(score), 1.0)
    if score >= 0:
        return f"rgba(31, 119, 180, {a:.3f})"    # blu
    return f"rgba(214, 39, 40, {a:.3f})"         # rosso


def render_html_block(title: str, subtitle: str, attr: Attribution) -> str:
    norm = _normalize_scores(attr.scores)
    spans = []
    for tok, s in zip(attr.tokens, norm):
        text = html.escape(tok.replace("▁", " ").replace("Ġ", " "))
        spans.append(
            f'<span style="background-color:{_color_for(s)};'
            f'padding:1px 2px;border-radius:3px;" title="{s:+.3f}">{text}</span>'
        )
    body = "".join(spans)
    probs_html = " · ".join(
        f"<b>{lbl}</b>={p:.3f}" for lbl, p in attr.pred_probs.items()
    )
    return (
        f'<div style="margin:14px 0;padding:10px;border:1px solid #ccc;'
        f'border-radius:6px;font-family:sans-serif;font-size:14px;line-height:1.9;">'
        f'<div style="font-weight:600;margin-bottom:4px;">{html.escape(title)}</div>'
        f'<div style="color:#666;margin-bottom:8px;">{html.escape(subtitle)}</div>'
        f'<div style="margin-bottom:6px;">{body}</div>'
        f'<div style="color:#444;font-size:12px;">Pred: {probs_html} → '
        f'<b>{attr.pred_label}</b> (target classe: <i>{attr.target_class}</i>)</div>'
        f'</div>'
    )


class _BaseIGExplainer:
    LABELS = LABELS_VAN
    MAX_LEN = 256
    STEPS = 32

    def __init__(self, model_name: str):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.device = _auto_device()
        self.dev_str = (self.device if isinstance(self.device, str)
                        else (f"cuda:{self.device}" if self.device != -1 else "cpu"))
        self.model.to(self.dev_str)
        self.id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}

    def _label_index(self, label: str) -> int:
        for idx, lbl in self.id2label.items():
            if lbl == label:
                return idx
        return 2 if label == "positive" else (0 if label == "negative" else 1)

    def _embedding_layer(self):
        for name in ("embeddings", "embed_tokens"):
            mod = getattr(self.model, "roberta", None) or \
                  getattr(self.model, "deberta", None) or \
                  getattr(self.model, "bert", None) or self.model
            if hasattr(mod, "embeddings"):
                return mod.embeddings
            if hasattr(mod, "embed_tokens"):
                return mod.embed_tokens
        raise RuntimeError("Embedding layer non trovato")

    def _predict(self, input_ids, attention_mask):
        import torch
        with torch.no_grad():
            logits = self.model(input_ids=input_ids,
                                attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
        ordered = {self.id2label[i]: float(probs[i]) for i in range(len(probs))}
        ordered = {k: ordered.get(k, 0.0) for k in self.LABELS}
        label = max(ordered, key=ordered.get)
        return label, ordered

    def _build_inputs(self, *args, **kwargs):
        raise NotImplementedError

    def explain(self, *args, target: str | None = None, **kwargs) -> Attribution:
        import torch

        inputs = self._build_inputs(*args, **kwargs)
        input_ids = inputs["input_ids"].to(self.dev_str)
        attention_mask = inputs["attention_mask"].to(self.dev_str)

        pred_label, pred_probs = self._predict(input_ids, attention_mask)
        target_label = target or pred_label
        target_idx = self._label_index(target_label)

        # Word-embedding layer (gestito da HF in modo uniforme per RoBERTa/DeBERTa)
        word_emb = self.model.get_input_embeddings()
        baseline_ids = torch.full_like(input_ids, self.tokenizer.pad_token_id)

        with torch.no_grad():
            input_emb = word_emb(input_ids)          # (1, L, H)
            baseline_emb = word_emb(baseline_ids)     # (1, L, H)
        delta = input_emb - baseline_emb

        total_grad = torch.zeros_like(input_emb)
        for step in range(1, self.STEPS + 1):
            alpha = step / float(self.STEPS)
            interp = (baseline_emb + alpha * delta).detach().requires_grad_(True)
            logits = self.model(inputs_embeds=interp,
                                attention_mask=attention_mask).logits
            prob = torch.softmax(logits, dim=-1)[0, target_idx]
            grad = torch.autograd.grad(prob, interp)[0]
            total_grad = total_grad + grad

        avg_grad = total_grad / self.STEPS
        ig = (delta * avg_grad).sum(dim=-1).squeeze(0)   # attribuzione per token
        scores = ig.detach().cpu().tolist()

        ids_list = input_ids.squeeze(0).cpu().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(ids_list)
        # rimuovi i token PAD finali
        keep = [i for i, _ in enumerate(tokens)
                if ids_list[i] != self.tokenizer.pad_token_id]
        tokens = [tokens[i] for i in keep]
        scores = [scores[i] for i in keep]
        return Attribution(tokens=tokens, scores=scores,
                           pred_label=pred_label, pred_probs=pred_probs,
                           target_class=target_label)


class VanillaExplainer(_BaseIGExplainer):
    def __init__(self):
        super().__init__(SENTIMENT_MODEL)

    def _build_inputs(self, text: str, **_):
        return self.tokenizer(text, return_tensors="pt",
                              truncation=True, max_length=self.MAX_LEN)


class ABSAExplainer(_BaseIGExplainer):
    def __init__(self):
        super().__init__(ABSA_MODEL)

    def _build_inputs(self, sentence: str, aspect: str, **_):
        return self.tokenizer(aspect, sentence, return_tensors="pt",
                              truncation=True, max_length=self.MAX_LEN)


def _targets_for_sentence(sentence: str, fallback_aspect: str = "") -> list[tuple[str, str]]:
    """Dirigenti citati nella frase, come lista di (display_name, aspect_target).

    Usa la stessa NER del resto della pipeline (`annotate_sentence`) così che il
    numero di blocchi ABSA coincida con il numero di dirigenti effettivamente
    menzionati. Se la NER non rileva nessun dirigente si ricade sull'aspect
    fornito nell'esempio (utile per target generici come "the management")."""
    targets: list[tuple[str, str]] = []
    for key in annotate_sentence(sentence):
        display = EXEC_BY_KEY[key].display_name
        target = AspectSentimentAnalyzer.normalize_target(key)
        targets.append((display, target))
    if not targets and fallback_aspect:
        targets.append((fallback_aspect, fallback_aspect))
    return targets


def explain_examples(examples: list[dict], out_path: Path,
                     translate_it: bool = False):
    """Calcola attribuzioni per i due modelli e scrive un HTML side-by-side."""
    logger.info("Carico modelli (può richiedere qualche minuto al primo run)…")
    van = VanillaExplainer()
    absa = ABSAExplainer()

    html_parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Explainability Vanilla vs ABSA</title></head><body>",
        "<h1 style='font-family:sans-serif;'>Integrated Gradients — Vanilla vs ABSA</h1>",
        "<p style='font-family:sans-serif;color:#555;max-width:800px;'>"
        "Token <span style='background:rgba(31,119,180,0.6);padding:1px 4px;'>blu</span> "
        "spingono la predizione verso <b>positivo</b>; "
        "token <span style='background:rgba(214,39,40,0.6);padding:1px 4px;'>rossi</span> "
        "verso <b>negativo</b>. Intensità del colore proporzionale a |attribuzione|.</p>",
        "<p style='font-family:sans-serif;color:#555;max-width:800px;'>"
        "Il modello <b>Vanilla</b> assegna <b>un solo sentiment all'intera frase</b>. "
        "Il modello <b>ABSA</b> produce invece <b>un blocco per ogni dirigente citato</b> "
        "(N&nbsp;=&nbsp;numero di dirigenti nella frase): questo mostra come l'ABSA serva "
        "ad assegnare un sentiment all'<i>aspect</i> specifico, anche quando nella stessa "
        "frase compaiono più persone con polarità opposte.</p>",
    ]
    for i, ex in enumerate(examples, 1):
        sentence = ex["sentence"]
        fallback_aspect = ex.get("aspect", "")
        targets = _targets_for_sentence(sentence, fallback_aspect)
        html_parts.append(f"<hr><h2 style='font-family:sans-serif;'>Esempio {i}</h2>")
        html_parts.append(f"<p style='font-family:sans-serif;'>"
                          f"<i>{html.escape(sentence)}</i></p>")
        detected = ", ".join(html.escape(d) for d, _ in targets) or "—"
        html_parts.append(f"<p style='font-family:sans-serif;'><b>Dirigenti rilevati "
                          f"(N={len(targets)}):</b> <code>{detected}</code></p>")

        # --- Vanilla: un solo blocco sull'intera frase ---
        try:
            a_van = van.explain(sentence)
            html_parts.append(render_html_block(
                "Vanilla (XLM-RoBERTa)",
                "Sentence-level — un solo sentiment per l'intera frase",
                a_van))
        except Exception as e:
            html_parts.append(f"<p>Vanilla error: {html.escape(str(e))}</p>")

        # --- ABSA: un blocco per ciascun dirigente citato ---
        if not targets:
            html_parts.append("<p style='font-family:sans-serif;color:#a00;'>"
                              "Nessun dirigente rilevato nella frase: ABSA non eseguito.</p>")
        for display, target in targets:
            try:
                a_absa = absa.explain(sentence, aspect=target)
                html_parts.append(render_html_block(
                    f"ABSA (DeBERTa-v3) — target = {display}",
                    f"Target-aware — input: [{target}] [SEP] [sentence]",
                    a_absa))
            except Exception as e:
                html_parts.append(f"<p>ABSA error ({html.escape(display)}): "
                                  f"{html.escape(str(e))}</p>")

    html_parts.append("</body></html>")
    out_path.write_text("\n".join(html_parts), encoding="utf-8")
    logger.info("HTML salvato: %s", out_path)


def _resolve_aspect_from_executive(exec_key: str) -> str:
    """Deriva l'aspect ASCII dal display_name del dirigente."""
    from .config import EXEC_BY_KEY
    from .aspect_sentiment import AspectSentimentAnalyzer
    if exec_key in EXEC_BY_KEY:
        return AspectSentimentAnalyzer.normalize_target(exec_key)
    return exec_key


def main(input_csv: Path | None = None,
         text: str | None = None,
         aspect: str | None = None,
         out_path: Path | None = None,
         limit: int | None = None):
    if input_csv:
        df = pd.read_csv(input_csv)
        if "sentence" not in df.columns:
            raise SystemExit("Il CSV deve avere almeno la colonna 'sentence'.")
        if "aspect" not in df.columns:
            if "executive" in df.columns:
                df["aspect"] = df["executive"].apply(_resolve_aspect_from_executive)
                logger.info("Colonna 'aspect' derivata da 'executive'.")
            else:
                raise SystemExit(
                    "Il CSV deve avere 'aspect' oppure 'executive' "
                    "(chiave config: maldini, cardinale, furlani, ...).")
        if limit:
            df = df.head(limit)
        examples = df[["sentence", "aspect"]].to_dict(orient="records")
    elif text:
        examples = [{"sentence": text, "aspect": aspect or "the management"}]
    else:
        examples = DEFAULT_EXAMPLES

    out_path = out_path or (OUTPUT_DIR / "explain_sentiment.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    explain_examples(examples, out_path)
    print(f"\nApri il report con:")
    print(f"  open {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=None,
                    help="CSV con colonna 'sentence' e 'aspect' "
                         "(oppure 'executive' da cui aspect viene derivato)")
    ap.add_argument("--text", type=str, default=None,
                    help="Frase singola (alternativa a --input)")
    ap.add_argument("--aspect", type=str, default=None,
                    help="Target ABSA (richiesto se --text)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limita il numero di esempi processati dal CSV")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    main(input_csv=args.input, text=args.text, aspect=args.aspect,
         out_path=args.out, limit=args.limit)
