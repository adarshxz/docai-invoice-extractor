"""Checks against the real ``microsoft/layoutlmv3-base`` processor and weights.

Run with ``pytest -m slow``; the first run downloads ~500 MB from the HF hub.
"""

from __future__ import annotations

import pytest
from PIL import Image

from conftest import INVOICE, PAGE_H, PAGE_W, word
from docai.labels import LABELS

pytestmark = pytest.mark.slow

BASE = "microsoft/layoutlmv3-base"


@pytest.fixture(scope="module")
def classifier():
    from docai.model import LayoutLMv3TokenClassifier

    return LayoutLMv3TokenClassifier(BASE, device="cpu", max_seq_length=128, stride=32)


def test_subtokens_duplicate_their_word_box():
    from transformers import LayoutLMv3Processor

    processor = LayoutLMv3Processor.from_pretrained(BASE, apply_ocr=False)
    words = [w for w, _ in INVOICE]
    enc = processor(
        Image.new("RGB", (PAGE_W, PAGE_H), "white"),
        [w.text for w in words],
        boxes=[list(w.norm_box) for w in words],
        return_tensors="pt",
    )
    word_ids = enc.word_ids(0)
    boxes = enc["bbox"][0].tolist()
    assert len(word_ids) > len(words)  # "INV-2024-0042" etc. split into several BPE pieces
    for pos, wid in enumerate(word_ids):
        if wid is not None:
            assert boxes[pos] == list(words[wid].norm_box)


def test_predict_one_label_per_word(classifier):
    words = [w for w, _ in INVOICE]
    preds = classifier.predict(Image.new("RGB", (PAGE_W, PAGE_H), "white"), words)
    assert [p.word for p in preds] == words
    assert all(p.label in LABELS and 0 <= p.confidence <= 1 for p in preds)


def test_long_page_is_windowed(classifier):
    # ~300 words at max_seq_length=128 forces several overlapping windows.
    words = [word(f"item{i}", (i % 20) * 50, (i // 20) * 90, (i % 20) * 50 + 45, (i // 20) * 90 + 30) for i in range(300)]
    preds = classifier.predict(Image.new("RGB", (PAGE_W, PAGE_H), "white"), words)
    assert len(preds) == 300


def test_empty_page(classifier):
    assert classifier.predict(Image.new("RGB", (100, 100)), []) == []


def test_training_encoding_labels_first_subtoken_only(tmp_path):
    import json

    from transformers import LayoutLMv3Processor

    from docai.labels import IGNORE_INDEX, LABEL2ID
    from docai.training.dataset import InvoiceTokenDataset, read_jsonl

    Image.new("RGB", (PAGE_W, PAGE_H), "white").save(tmp_path / "p.png")
    rec = {
        "image": "p.png",
        "words": [w.text for w, _ in INVOICE],
        "boxes": [list(w.pixel_box) for w, _ in INVOICE],
        "labels": [label for _, label in INVOICE],
    }
    (tmp_path / "train.jsonl").write_text(json.dumps(rec) + "\n")

    processor = LayoutLMv3Processor.from_pretrained(BASE, apply_ocr=False)
    item = InvoiceTokenDataset(read_jsonl(tmp_path / "train.jsonl"), processor, max_length=128)[0]

    labelled = [int(x) for x in item["labels"] if x != IGNORE_INDEX]
    assert labelled == [LABEL2ID[label] for _, label in INVOICE]
    assert item["pixel_values"].shape == (3, 224, 224)
    assert item["input_ids"].shape == item["labels"].shape == item["bbox"].shape[:1]
    assert item["bbox"].max() <= 1000


def test_collate_batch_pads_to_longest():
    import torch

    from docai.labels import IGNORE_INDEX
    from docai.training.dataset import collate_batch

    def feat(n):
        return {"input_ids": torch.arange(n), "attention_mask": torch.ones(n, dtype=torch.long),
                "labels": torch.zeros(n, dtype=torch.long), "bbox": torch.ones(n, 4, dtype=torch.long),
                "pixel_values": torch.zeros(3, 224, 224)}

    batch = collate_batch([feat(3), feat(5)], pad_token_id=1)
    assert batch["input_ids"].shape == (2, 5) and batch["bbox"].shape == (2, 5, 4)
    assert batch["input_ids"][0, 3:].tolist() == [1, 1]
    assert batch["labels"][0, 3:].tolist() == [IGNORE_INDEX] * 2
    assert batch["attention_mask"][0].tolist() == [1, 1, 1, 0, 0]
    assert batch["bbox"][0, 4].tolist() == [0, 0, 0, 0]
