from __future__ import annotations

import json
import random
from collections import Counter

from docai.data.common import bio_tags, reading_order
from docai.data.synthetic import MoneyFormat, degrade, discover_fonts, generate, render_invoice
from docai.labels import ENTITY_TYPES
from docai.postprocessing import parse_amount, parse_date
from docai.training.dataset import read_jsonl


def _entities(words):
    out: dict[int, list] = {}
    for w in words:
        if w.entity_id is not None:
            out.setdefault(w.entity_id, []).append(w)
    return list(out.values())


def test_money_format_round_trips_through_parser():
    for args in [("$", "USD", ".", ",", False, False), ("€", "EUR", ",", ".", True, True), ("CHF", "CHF", ".", "'", False, True)]:
        fmt = MoneyFormat(*args)
        assert parse_amount(fmt(12345.67)) == 12345.67
        assert parse_amount(fmt(-3.5)) == -3.5


def test_rendered_pages_have_consistent_labels():
    fonts = discover_fonts()
    seen: Counter[str] = Counter()
    for seed in range(40):
        canvas = render_invoice(random.Random(seed), fonts)
        groups = _entities(canvas.words)
        types = Counter(g[0].entity_type for g in groups)
        seen.update(types.keys())
        for single in ("VENDOR_NAME", "INVOICE_NUMBER", "DATE", "TOTAL_AMOUNT"):
            assert types[single] == 1, (seed, single, types)
        assert types["TAX_AMOUNT"] <= 1
        assert types["ITEM_DESCRIPTION"] >= 1
        for g in groups:  # every labelled value is parseable by the post-processor
            text = " ".join(w.text for w in g)
            if g[0].entity_type == "DATE":
                assert parse_date(text, day_first=True) or parse_date(text), text
            if g[0].entity_type in ("TOTAL_AMOUNT", "TAX_AMOUNT"):
                assert parse_amount(text) is not None, text
        for w in canvas.words:
            x0, y0, x1, y1 = w.box
            assert 0 <= x0 < x1 <= canvas.image.width and 0 <= y0 < y1 <= canvas.image.height, w
    assert set(seen) == set(ENTITY_TYPES)


def test_degrade_keeps_boxes_on_page():
    rng = random.Random(3)
    canvas = render_invoice(rng, discover_fonts())
    for _ in range(5):
        img = degrade(canvas.image, canvas.words, rng)
        assert all(0 <= w.box[0] < w.box[2] <= img.width + 1 and 0 <= w.box[1] < w.box[3] <= img.height + 1
                   for w in canvas.words)


def test_bio_from_rendered_page_is_well_formed():
    words = reading_order(render_invoice(random.Random(11), discover_fonts()).words)
    tags = bio_tags(words)
    for prev, cur in zip(["O", *tags], tags):
        if cur.startswith("I-"):
            assert prev[2:] == cur[2:]


def test_generate_writes_training_ready_jsonl(tmp_path):
    assert generate(tmp_path, 3, seed=5) == 3
    pages = read_jsonl(tmp_path / "synth.jsonl")
    assert len(pages) == 3 and all(p.image_path.exists() for p in pages)
    rec = json.loads((tmp_path / "synth.jsonl").read_text().splitlines()[0])
    assert all(0 <= v <= 1000 for box in rec["boxes"] for v in box)


def test_single_value_fields_stay_one_entity_after_reading_order():
    fonts = discover_fonts()
    for seed in range(60):
        words = reading_order(render_invoice(random.Random(seed), fonts).words)
        starts = Counter(t[2:] for t in bio_tags(words) if t.startswith("B-"))
        for field in ("VENDOR_NAME", "INVOICE_NUMBER", "DATE", "TOTAL_AMOUNT"):
            assert starts[field] == 1, (seed, field, starts)
