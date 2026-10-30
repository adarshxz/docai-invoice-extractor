"""Merge converted JSONL files into one train/validation split, and summarise label counts."""

from __future__ import annotations

import json
import os
import random
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from docai.data.common import write_records

_PAGE_SUFFIX = re.compile(r"_p\d+$")  # ls-prepare names pages "<document>_p<n>"


def _load(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for rec in records:  # re-anchor image paths so they stay valid from any output location
        rec["image"] = str((path.parent / rec["image"]).resolve())
    return records


def merge_and_split(
    inputs: Sequence[Path],
    out_dir: Path,
    *,
    val_fraction: float = 0.1,
    seed: int = 42,
    repeat: Mapping[Path, int] | None = None,
) -> dict[str, Counter[str]]:
    """Write ``out_dir/train.jsonl`` and ``out_dir/val.jsonl``.

    Each input is split separately (stratified by source) so every source is
    represented in validation, and by document rather than by page. ``repeat`` oversamples a source's *training*
    pages, e.g. ``{Path("data/invoices/labelled.jsonl"): 3}`` to weight your own
    invoices above the public receipts.

    Returns per-split page counts by source file name.
    """
    rng = random.Random(seed)
    repeat = {p.resolve(): n for p, n in (repeat or {}).items()}
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    counts: dict[str, Counter[str]] = {"train": Counter(), "val": Counter()}

    for path in inputs:
        groups: dict[str, list[dict[str, Any]]] = {}
        for rec in _load(path):  # pages of one document stay together, so val never sees a train document
            groups.setdefault(_PAGE_SUFFIX.sub("", str(rec.get("id", id(rec)))), []).append(rec)
        docs = list(groups.values())
        rng.shuffle(docs)
        n_val = round(len(docs) * val_fraction)
        if val_fraction > 0 and n_val == 0 and len(docs) >= 2:
            n_val = 1
        val_recs = [r for d in docs[:n_val] for r in d]
        train_recs = [r for d in docs[n_val:] for r in d]

        source = f"{path.parent.name}/{path.name}"
        reps = max(1, repeat.get(path.resolve(), 1))
        val.extend(val_recs)
        train.extend(dict(rec) for rec in train_recs for _ in range(reps))  # copies: paths are rewritten below
        counts["val"][source] += len(val_recs)
        counts["train"][source] += len(train_recs) * reps

    rng.shuffle(train)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, recs in (("train", train), ("val", val)):
        target = out_dir / f"{name}.jsonl"
        for rec in recs:
            rec["image"] = os.path.relpath(rec["image"], out_dir)
        write_records(target, recs)
    return counts


def label_stats(path: Path) -> dict[str, int]:
    """Pages, words, and entity-instance count (``B-`` tags) per type in a JSONL file."""
    pages = words = 0
    entities: Counter[str] = Counter()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        pages += 1
        words += len(rec["words"])
        entities.update(tag[2:] for tag in rec["labels"] if tag.startswith("B-"))
    return {"pages": pages, "words": words, **dict(sorted(entities.items()))}
