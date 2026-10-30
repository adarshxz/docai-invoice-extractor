"""``docai-eval``: score a fine-tuned model on labelled JSONL pages.

Two views:

* **Entity F1** (seqeval) on the model's word tags, per entity type.
* **Field accuracy** of the final JSON, i.e. after BIO decoding, confidence
  filtering and value parsing: a field counts as correct when the extracted
  value equals the value parsed from the gold-labelled words. Only fields the
  page's labels contain are scored, so a dataset that doesn't label tax (SROIE)
  isn't penalised when the model finds tax anyway.

The gold words are used in place of Tesseract, so this measures the model and
post-processing, not OCR quality.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from docai.data.common import LabelledWord, reading_order
from docai.labels import split_tag
from docai.postprocessing import PageResult, build_output, clean_vendor_name, parse_amount, parse_date
from docai.model import TokenClassifier
from docai.schemas import OCRWord, WordPrediction
from docai.training.dataset import AnnotatedPage, read_jsonl

logger = logging.getLogger(__name__)

FIELDS = {
    "VENDOR_NAME": "vendor_name",
    "DATE": "invoice_date",
    "TOTAL_AMOUNT": "total_amount",
    "TAX_AMOUNT": "tax_amount",
    "INVOICE_NUMBER": "invoice_number",
}


def page_words(page: AnnotatedPage, image: Image.Image) -> list[OCRWord]:
    """Gold words as :class:`OCRWord` s, with pixel boxes for the image and row-based line numbers."""
    w, h = image.size
    if page.box_format == "normalized":
        norm = list(page.boxes)
        pixel = [(x0 * w // 1000, y0 * h // 1000, x1 * w // 1000, y1 * h // 1000) for x0, y0, x1, y1 in norm]
    else:
        pixel = list(page.boxes)
        norm = [(min(1000, x0 * 1000 // w), min(1000, y0 * 1000 // h), min(1000, x1 * 1000 // w), min(1000, y1 * 1000 // h))
                for x0, y0, x1, y1 in pixel]

    # Line numbers stand in for Tesseract's (block, par, line); post-processing uses them to merge and pair.
    lw = [LabelledWord(text=str(i), box=b) for i, b in enumerate(pixel)]
    line_of: dict[int, int] = {}
    line, last_center = 0, None
    for item in reading_order(lw):
        center = (item.box[1] + item.box[3]) / 2
        height = max(item.box[3] - item.box[1], 1)
        if last_center is None or abs(center - last_center) > height / 2:
            line += 1
            last_center = center
        line_of[int(item.text)] = line
    return [
        OCRWord(text=t, pixel_box=tuple(p), norm_box=tuple(n), ocr_confidence=100.0, line_num=line_of[i])  # type: ignore[arg-type]
        for i, (t, p, n) in enumerate(zip(page.words, pixel, norm, strict=True))
    ]


def gold_entities(page: AnnotatedPage) -> dict[str, list[str]]:
    """Gold entity texts per type, decoded from the page's BIO labels."""
    out: dict[str, list[str]] = {}
    current: list[str] = []
    ctype: str | None = None
    for word, tag in [*zip(page.words, page.labels, strict=True), ("", "O")]:
        prefix, etype = split_tag(tag)
        if ctype and (etype != ctype or prefix == "B"):
            out.setdefault(ctype, []).append(" ".join(current))
            current, ctype = [], None
        if etype:
            current.append(word)
            ctype = etype
    return out


def _norm_text(s: str | None) -> str:
    return re.sub(r"[^A-Z0-9]+", " ", (s or "").upper()).strip()


def _gold_has_value(etype: str, text: str, day_first: bool) -> bool:
    if etype in ("TOTAL_AMOUNT", "TAX_AMOUNT"):
        return parse_amount(text) is not None
    if etype == "DATE":
        return parse_date(text, day_first=day_first) is not None
    return bool(_norm_text(text))


def field_correct(etype: str, predicted: Any, gold_texts: Sequence[str], *, day_first: bool) -> bool:
    """Whether the extracted value matches any of the gold entity texts for that field."""
    if predicted is None:
        return False
    for gold in gold_texts:
        if etype in ("TOTAL_AMOUNT", "TAX_AMOUNT"):
            value = parse_amount(gold)
            if value is not None and abs(value - float(predicted)) < 0.005:
                return True
        elif etype == "DATE":
            d = parse_date(gold, day_first=day_first)
            if d is not None and d.isoformat() == predicted:
                return True
        elif etype == "VENDOR_NAME":
            if _norm_text(clean_vendor_name(gold)) == _norm_text(predicted):
                return True
        elif _norm_text(gold).replace(" ", "") == _norm_text(predicted).replace(" ", ""):
            return True
    return False


class OracleClassifier:
    """Predicts the gold labels with full confidence: measures the post-processing ceiling."""

    def __init__(self) -> None:
        self.labels: list[str] = []

    def predict(self, image: Image.Image, words: Sequence[OCRWord]) -> list[WordPrediction]:
        return [WordPrediction(word=w, label=t, confidence=1.0) for w, t in zip(words, self.labels, strict=True)]


def evaluate(classifier: TokenClassifier, files: Sequence[Path], *, limit: int | None = None,
             threshold: float = 0.70, day_first: bool = False) -> dict[str, Any]:
    """Score ``classifier`` on each JSONL file; returns a JSON-serialisable report."""
    from seqeval.metrics import classification_report

    clf = classifier
    report: dict[str, Any] = {}
    for path in files:
        pages = read_jsonl(path)[:limit]
        gold_tags: list[list[str]] = []
        pred_tags: list[list[str]] = []
        hits: Counter[str] = Counter()
        totals: Counter[str] = Counter()
        for page in pages:
            image = Image.open(page.image_path).convert("RGB")
            words = page_words(page, image)
            if isinstance(clf, OracleClassifier):
                clf.labels = list(page.labels)
            preds = clf.predict(image, words)
            gold_tags.append(list(page.labels))
            pred_tags.append([p.label for p in preds])

            entities, _ = build_output([PageResult(page=0, words=words, predictions=preds)],
                                       confidence_threshold=threshold, day_first=day_first)
            gold = gold_entities(page)
            for etype, attr in FIELDS.items():
                if etype not in gold or not any(_gold_has_value(etype, g, day_first) for g in gold[etype]):
                    continue  # unlabelled, or a placeholder such as CORD's "-" for "no tax"
                totals[etype] += 1
                hits[etype] += field_correct(etype, getattr(entities, attr).value, gold[etype], day_first=day_first)

        seq = classification_report(gold_tags, pred_tags, output_dict=True, zero_division=0)
        report[str(path)] = {
            "pages": len(pages),
            "entity_f1": {k: round(float(v["f1-score"]), 3) for k, v in seq.items() if k.isupper() or k == "micro avg"},
            "field_accuracy": {k: round(hits[k] / totals[k], 3) for k in totals},
            "field_counts": dict(totals),
        }
        logger.info("%s: %s", path, json.dumps(report[str(path)]))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="fine-tuned checkpoint directory")
    parser.add_argument("--oracle", action="store_true", help="use gold labels instead of a model (post-processing ceiling)")
    parser.add_argument("--data", nargs="+", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--day-first", action="store_true", help="read ambiguous dates as DD/MM (e.g. SROIE)")
    parser.add_argument("--output", type=Path, help="also write the report as JSON")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    if args.oracle:
        clf: TokenClassifier = OracleClassifier()
    elif args.model:
        from docai.model import LayoutLMv3TokenClassifier

        clf = LayoutLMv3TokenClassifier(args.model, device=args.device)
    else:
        parser.error("give --model or --oracle")
    report = evaluate(clf, args.data, limit=args.limit, threshold=args.threshold, day_first=args.day_first)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text)


if __name__ == "__main__":
    main()
