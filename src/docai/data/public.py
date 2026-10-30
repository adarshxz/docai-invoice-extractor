"""Converters for public receipt datasets on the Hugging Face hub.

* **CORD v2** (``naver-clova-ix/cord-v2``, CC-BY-4.0): 1,000 Indonesian
  receipts with word boxes and a category per text line. Provides
  ``ITEM_DESCRIPTION`` (``menu.nm``), ``TOTAL_AMOUNT`` (``total.total_price``)
  and ``TAX_AMOUNT`` (``sub_total.tax_price``). Store names are blurred out,
  so it has no vendor or date labels.
* **SROIE 2019** (``jsdnrs/ICDAR2019-SROIE``, CC-BY-4.0): ~990 Malaysian
  receipts with line boxes and four document-level strings. Provides
  ``VENDOR_NAME`` (company), ``DATE`` and ``TOTAL_AMOUNT``. The strings are
  matched back onto the words heuristically; pages where a string cannot be
  found are skipped rather than mislabelled.

Neither dataset labels invoice numbers, and each leaves some entity types
present but unlabelled (SROIE's line items are tagged ``O``). Use them for a
first training stage and finish on fully labelled invoices of your own.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docai.data.common import JsonlWriter, LabelledWord, decode_image, reading_order, split_line

logger = logging.getLogger(__name__)

CORD_REPO = "naver-clova-ix/cord-v2"
SROIE_REPO = "jsdnrs/ICDAR2019-SROIE"

CORD_CATEGORY_MAP: dict[str, str] = {
    "menu.nm": "ITEM_DESCRIPTION",
    "total.total_price": "TOTAL_AMOUNT",
    "sub_total.tax_price": "TAX_AMOUNT",
}
SROIE_ENTITY_MAP: dict[str, str] = {"company": "VENDOR_NAME", "date": "DATE", "total": "TOTAL_AMOUNT"}


@dataclass
class ConversionStats:
    written: int = 0
    skipped: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1


def iter_parquet_rows(repo: str, split: str, columns: Sequence[str]) -> Iterator[dict[str, Any]]:
    """Stream rows of one split of a parquet-backed hub dataset (downloads on first use)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    files = sorted(
        f
        for f in HfApi().list_repo_files(repo, repo_type="dataset")
        if f.endswith(".parquet") and Path(f).name.startswith(f"{split}-")
    )
    if not files:
        raise ValueError(f"{repo} has no split {split!r}")
    for name in files:
        table = pq.ParquetFile(hf_hub_download(repo, name, repo_type="dataset"))
        for batch in table.iter_batches(columns=list(columns), batch_size=16):
            yield from batch.to_pylist()


# --------------------------------------------------------------------------
# CORD
# --------------------------------------------------------------------------


def cord_words(ground_truth: dict[str, Any]) -> list[LabelledWord]:
    """Labelled words in reading order from one CORD ``ground_truth`` object.

    Each ``valid_line`` is one entity instance. Key words such as "TOTAL"
    (``is_key == 1``) are left as ``O``.
    """
    words: list[LabelledWord] = []
    for line_idx, line in enumerate(ground_truth.get("valid_line", [])):
        etype = CORD_CATEGORY_MAP.get(line.get("category", ""))
        for w in line.get("words", []):
            text = str(w.get("text", "")).strip()
            if not text:
                continue
            q = w["quad"]
            xs, ys = (q["x1"], q["x2"], q["x3"], q["x4"]), (q["y1"], q["y2"], q["y3"], q["y4"])
            labelled = etype is not None and not w.get("is_key", 0)
            words.append(
                LabelledWord(
                    text=text,
                    box=(min(xs), min(ys), max(xs), max(ys)),
                    entity_type=etype if labelled else None,
                    entity_id=line_idx if labelled else None,
                )
            )
    return reading_order(words)


def convert_cord(out_dir: Path, splits: Sequence[str] = ("train", "validation", "test"), *, limit: int | None = None,
                 normalize_images: bool = True) -> dict[str, ConversionStats]:
    """Write ``out_dir/<split>.jsonl`` and ``out_dir/images/`` for each CORD split."""
    results: dict[str, ConversionStats] = {}
    for split in splits:
        stats = ConversionStats()
        with JsonlWriter(out_dir / f"{split}.jsonl", out_dir / "images", normalize_images=normalize_images) as writer:
            for i, row in enumerate(iter_parquet_rows(CORD_REPO, split, ["image", "ground_truth"])):
                if limit is not None and i >= limit:
                    break
                words = cord_words(json.loads(row["ground_truth"]))
                if not words:
                    stats.skip("no words")
                    continue
                writer.write(f"cord_{split}_{i:04d}", decode_image(row["image"]["bytes"]), words)
            stats.written = writer.count
        results[split] = stats
        logger.info("CORD %s: %d written, %d skipped", split, stats.written, stats.skipped)
    return results


# --------------------------------------------------------------------------
# SROIE
# --------------------------------------------------------------------------

_EDGE_PUNCT = re.compile(r"^[^\w]+|[^\w]+$")
_TOTAL_EXCLUDE = ("SUB", "EXCL", "GST", "TAX", "ROUND", "QTY", "ITEM", "DISC", "SAVING")


def _norm(token: str) -> str:
    return _EDGE_PUNCT.sub("", token.upper())


def _find_sequence(tokens: Sequence[str], target: Sequence[str]) -> list[list[int]]:
    """All index runs where ``target`` occurs in ``tokens`` (both already normalised, blanks skipped)."""
    idx = [i for i, t in enumerate(tokens) if t]
    seq = [tokens[i] for i in idx]
    n = len(target)
    return [idx[s : s + n] for s in range(len(seq) - n + 1) if seq[s : s + n] == list(target)]


def _token_similarity(found: str, wanted: str) -> float:
    """1.0 for equal tokens or a glued suffix ("BHD(285560-H" vs "BHD"), else a difflib ratio."""
    if found == wanted or (len(wanted) >= 3 and found.startswith(wanted)):
        return 1.0
    return difflib.SequenceMatcher(None, found, wanted).ratio()


def _find_fuzzy_sequence(tokens: Sequence[str], target: Sequence[str]) -> list[int] | None:
    """Best window where every token is similar (ratio >= 0.6) and at least half match exactly.

    Absorbs the typos found in SROIE's ground truth ("KOPITAM SND" for "KOPITIAM SDN").
    """
    idx = [i for i, t in enumerate(tokens) if t]
    n = len(target)
    best: tuple[float, list[int]] | None = None
    for s in range(len(idx) - n + 1):
        window = idx[s : s + n]
        sims = [_token_similarity(tokens[i], t) for i, t in zip(window, target, strict=True)]
        if min(sims) < 0.6 or sum(x == 1.0 for x in sims) * 2 < n:
            continue
        if best is None or sum(sims) > best[0]:
            best = (sum(sims), window)
    return best[1] if best else None


def _row_text(words: Sequence[LabelledWord], i: int) -> str:
    """Upper-cased text of every word whose vertical centre falls in word ``i``'s line."""
    y0, y1 = words[i].box[1], words[i].box[3]
    return " ".join(w.text.upper() for w in words if y0 <= (w.box[1] + w.box[3]) / 2 <= y1)


def _total_score(words: Sequence[LabelledWord], run: list[int]) -> int:
    row = _row_text(words, run[0])
    if "TOTAL" not in row:
        return 0
    return 1 if any(x in row for x in _TOTAL_EXCLUDE) else 2


def locate_entity(words: Sequence[LabelledWord], key: str, value: str) -> list[int] | None:
    """Indices of the words that spell ``value``, or ``None`` if it can't be found.

    Exact token-sequence match first; for single-token values (dates,
    amounts) a token *containing* the value is accepted ("DATE:15/01/2019",
    "RM9.00"); multi-token values fall back to a strict fuzzy match. For ``total`` the occurrence on a row reading "TOTAL" (but not
    "SUBTOTAL", "TOTAL EXCL. GST" ...) wins; otherwise the first occurrence.
    """
    target = [t for t in (_norm(t) for t in value.split()) if t]
    if not target:
        return None
    tokens = [_norm(w.text) for w in words]
    runs = _find_sequence(tokens, target)
    if len(target) == 1:  # exact hits first, then glued ones ("RM9.00"); total scoring may prefer the latter
        runs += [[i] for i, t in enumerate(tokens) if t and t != target[0] and target[0] in t]
    if not runs and len(target) >= 2 and (fuzzy := _find_fuzzy_sequence(tokens, target)):
        runs = [fuzzy]
    if not runs:
        return None
    if key == "total":
        return max(runs, key=lambda r: _total_score(words, r))  # max() keeps the first of equal scores
    return runs[0]


def sroie_words(lines: Sequence[str], boxes: Sequence[Sequence[int]], entities: dict[str, str | None]
                ) -> tuple[list[LabelledWord], list[str]]:
    """Split SROIE lines into words and label them.

    Returns ``(words, missing)`` where ``missing`` names entities whose value
    could not be located.
    """
    words = [LabelledWord(text=t, box=b) for line, box in zip(lines, boxes, strict=True) for t, b in split_line(line, box)]
    missing: list[str] = []
    for entity_id, (key, etype) in enumerate(SROIE_ENTITY_MAP.items()):
        value = (entities.get(key) or "").strip()
        if not value:
            continue
        hit = locate_entity(words, key, value)
        if hit is None:
            missing.append(key)
            continue
        # Fill gaps: punctuation-only tokens such as "&" are skipped while matching but belong to the entity.
        for i in range(hit[0], hit[-1] + 1):
            if words[i].entity_type is None:  # never overwrite an earlier entity
                words[i].entity_type, words[i].entity_id = etype, entity_id
    return words, missing


def convert_sroie(out_dir: Path, splits: Sequence[str] = ("train", "test"), *, limit: int | None = None,
                  normalize_images: bool = True) -> dict[str, ConversionStats]:
    """Write ``out_dir/<split>.jsonl`` and ``out_dir/images/`` for each SROIE split."""
    results: dict[str, ConversionStats] = {}
    for split in splits:
        stats = ConversionStats()
        with JsonlWriter(out_dir / f"{split}.jsonl", out_dir / "images", normalize_images=normalize_images) as writer:
            rows = iter_parquet_rows(SROIE_REPO, split, ["image", "key", "entities", "words", "bboxes"])
            for i, row in enumerate(rows):
                if limit is not None and i >= limit:
                    break
                words, missing = sroie_words(row["words"], row["bboxes"], row["entities"] or {})
                if missing:
                    stats.skip("unmatched " + "+".join(missing))
                    continue
                writer.write(f"sroie_{split}_{row['key']}", decode_image(row["image"]["bytes"]), words)
            stats.written = writer.count
        results[split] = stats
        logger.info("SROIE %s: %d written, %d skipped %s", split, stats.written, stats.skipped, stats.reasons)
    return results
