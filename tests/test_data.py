from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from conftest import INVOICE, PAGE_H, PAGE_W
from docai.data.common import JsonlWriter, LabelledWord, bio_tags, reading_order, split_line
from docai.data.labelstudio import (
    assign_labels,
    convert,
    label_config_xml,
    latest_annotation,
    page_id_from_task,
    rectangles_from_annotation,
)
from docai.data.public import cord_words, locate_entity, sroie_words
from docai.data.split import label_stats, merge_and_split
from docai.labels import ENTITY_TYPES
from docai.training.dataset import read_jsonl

# --- common ----------------------------------------------------------------


def test_split_line_apportions_width_by_characters():
    assert split_line("AB  CDEF", (0, 10, 80, 30)) == [("AB", (0, 10, 20, 30)), ("CDEF", (40, 10, 80, 30))]
    assert split_line("   ", (0, 0, 10, 10)) == []


def test_reading_order_groups_rows():
    words = [LabelledWord("b", (50, 12, 60, 22)), LabelledWord("c", (0, 40, 10, 50)), LabelledWord("a", (0, 10, 10, 20))]
    assert [w.text for w in reading_order(words)] == ["a", "b", "c"]


def test_bio_tags_split_adjacent_instances_of_same_type():
    words = [
        LabelledWord("Bolt", (0, 0, 1, 1), "ITEM_DESCRIPTION", 1),
        LabelledWord("M6", (0, 0, 1, 1), "ITEM_DESCRIPTION", 1),
        LabelledWord("Nut", (0, 0, 1, 1), "ITEM_DESCRIPTION", 2),
        LabelledWord("x", (0, 0, 1, 1)),
        LabelledWord("9.00", (0, 0, 1, 1), "TOTAL_AMOUNT", 3),
    ]
    assert bio_tags(words) == ["B-ITEM_DESCRIPTION", "I-ITEM_DESCRIPTION", "B-ITEM_DESCRIPTION", "O", "B-TOTAL_AMOUNT"]


def test_jsonl_writer_output_is_readable_by_training_dataset(tmp_path):
    words = [LabelledWord("Total", (100, 100, 200, 140)), LabelledWord("9.00", (300, 100, 400, 140), "TOTAL_AMOUNT", 0)]
    with JsonlWriter(tmp_path / "x.jsonl", tmp_path / "images", max_side=500) as w:
        w.write("p1", Image.new("RGB", (2000, 1000), "white"), words)

    rec = json.loads((tmp_path / "x.jsonl").read_text())
    assert rec["boxes"] == [[50, 100, 100, 140], [150, 100, 200, 140]]  # normalised against the original size
    assert Image.open(tmp_path / rec["image"]).size == (500, 250)
    page = read_jsonl(tmp_path / "x.jsonl")[0]
    assert page.labels == ["O", "B-TOTAL_AMOUNT"] and page.box_format == "normalized"


# --- CORD ------------------------------------------------------------------


def _quad(x0, y0, x1, y1):
    return {"x1": x0, "y1": y0, "x2": x1, "y2": y0, "x3": x1, "y3": y1, "x4": x0, "y4": y1}


def test_cord_words_maps_categories_and_ignores_key_words():
    gt = {
        "valid_line": [
            {"category": "total.total_price", "words": [
                {"text": "TOTAL", "quad": _quad(10, 300, 80, 320), "is_key": 1},
                {"text": "45,500", "quad": _quad(200, 300, 260, 320), "is_key": 0},
            ]},
            {"category": "menu.nm", "words": [
                {"text": "EGG", "quad": _quad(10, 100, 40, 120), "is_key": 0},
                {"text": "TART", "quad": _quad(45, 100, 90, 120), "is_key": 0},
            ]},
            {"category": "menu.cnt", "words": [{"text": "1", "quad": _quad(0, 100, 5, 120), "is_key": 0}]},
        ]
    }
    words = cord_words(gt)
    assert [(w.text, t) for w, t in zip(words, bio_tags(words))] == [
        ("1", "O"),
        ("EGG", "B-ITEM_DESCRIPTION"),
        ("TART", "I-ITEM_DESCRIPTION"),
        ("TOTAL", "O"),
        ("45,500", "B-TOTAL_AMOUNT"),
    ]


# --- SROIE -----------------------------------------------------------------

SROIE_LINES = [
    ("OLD TOWN KOPITIAM SDN BHD(716269-X)", (10, 10, 400, 30)),
    ("DATE : 15/01/2019 11:05 AM", (10, 40, 300, 60)),
    ("ITEM 193.00", (10, 70, 300, 90)),
    ("TOTAL EXCL GST:", (10, 100, 200, 120)),
    ("193.00", (250, 100, 300, 120)),
    ("TOTAL:", (10, 130, 100, 150)),
    ("RM193.00", (250, 130, 330, 150)),
]


def test_sroie_words_matches_entities_with_typos_and_glued_text():
    lines, boxes = zip(*SROIE_LINES)
    words, missing = sroie_words(lines, boxes, {"company": "OLD TOWN KOPITAM SND BHD", "date": "15/01/2019", "total": "193.00",
                                                "address": "ignored"})
    assert missing == []
    labelled = [(w.text, t) for w, t in zip(words, bio_tags(words)) if t != "O"]
    assert labelled == [
        ("OLD", "B-VENDOR_NAME"),
        ("TOWN", "I-VENDOR_NAME"),
        ("KOPITIAM", "I-VENDOR_NAME"),
        ("SDN", "I-VENDOR_NAME"),
        ("BHD(716269-X)", "I-VENDOR_NAME"),
        ("15/01/2019", "B-DATE"),
        ("RM193.00", "B-TOTAL_AMOUNT"),  # the "TOTAL:" row, not the item or "TOTAL EXCL GST" rows
    ]


def test_sroie_reports_unmatched_entities():
    lines, boxes = zip(*SROIE_LINES)
    _, missing = sroie_words(lines, boxes, {"company": "COMPLETELY DIFFERENT NAME", "date": "", "total": "193.00"})
    assert missing == ["company"]


def test_locate_entity_fuzzy_needs_mostly_exact_tokens():
    words = [LabelledWord(t, (i * 10, 0, i * 10 + 9, 9)) for i, t in enumerate("ABC TRADING SDN BHD".split())]
    assert locate_entity(words, "company", "ABC TRAOING SDN BHD") == [0, 1, 2, 3]
    assert locate_entity(words, "company", "XYZ TRAOING SND BHD") is None


# --- Label Studio ----------------------------------------------------------


def test_label_config_lists_every_entity_type():
    xml = label_config_xml()
    assert all(f'value="{e}"' in xml for e in ENTITY_TYPES)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"page_id": "inv_p1", "image": "whatever"}, "inv_p1"),
        ({"image": "/data/local-files/?d=images/inv%201_p2.png"}, "inv 1_p2"),
        ({"image": "/data/upload/3/a1b2c3d4-inv_p3.png"}, "inv_p3"),
    ],
)
def test_page_id_from_task(data, expected):
    assert page_id_from_task({"data": data}) == expected


def test_latest_annotation_skips_cancelled():
    task = {"annotations": [
        {"id": 1, "updated_at": "2026-01-02", "result": []},
        {"id": 2, "updated_at": "2026-01-03", "was_cancelled": True},
        {"id": 3, "updated_at": "2026-01-01"},
    ]}
    assert latest_annotation(task)["id"] == 1
    assert latest_annotation({"annotations": []}) is None


def test_assign_labels_prefers_smallest_enclosing_rectangle():
    words = [w for w, _ in INVOICE]
    rects = [
        ("VENDOR_NAME", (40, 30, 380, 80)),
        ("ITEM_DESCRIPTION", (0, 380, 990, 500)),  # sloppy box covering both item rows
        ("ITEM_DESCRIPTION", (40, 440, 200, 490)),  # tight box on "Gadget"
        ("DATE", (0, 1300, 50, 1350)),  # empty: no text there
    ]
    labelled, empty = assign_labels(words, rects)
    tags = dict(zip([w.text for w in labelled], bio_tags(labelled)))
    assert tags["Acme"] == "B-VENDOR_NAME" and tags["Ltd."] == "I-VENDOR_NAME"
    assert tags["Widget"] == "B-ITEM_DESCRIPTION" and tags["$120.00"] == "I-ITEM_DESCRIPTION"
    assert tags["Gadget"] == "B-ITEM_DESCRIPTION"
    assert tags["Invoice"] == "O"
    assert empty == 1


def test_ls_convert_end_to_end(tmp_path):
    workdir = tmp_path / "work"
    (workdir / "ocr").mkdir(parents=True)
    (workdir / "images").mkdir()
    Image.new("RGB", (PAGE_W, PAGE_H), "white").save(workdir / "images" / "inv_p1.png")
    (workdir / "ocr" / "inv_p1.json").write_text(json.dumps({
        "page_id": "inv_p1", "width": PAGE_W, "height": PAGE_H,
        "words": [w.model_dump(mode="json") for w, _ in INVOICE],
    }))

    def rect(label, x0, y0, x1, y1):
        return {"type": "rectanglelabels", "from_name": "label", "to_name": "image", "original_width": PAGE_W,
                "original_height": PAGE_H, "value": {"x": x0 / PAGE_W * 100, "y": y0 / PAGE_H * 100,
                "width": (x1 - x0) / PAGE_W * 100, "height": (y1 - y0) / PAGE_H * 100, "rotation": 0,
                "rectanglelabels": [label]}}

    export = [
        {"id": 1, "data": {"image": "/data/local-files/?d=images/inv_p1.png", "page_id": "inv_p1"},
         "annotations": [{"result": [rect("INVOICE_NUMBER", 755, 35, 955, 75), rect("TOTAL_AMOUNT", 795, 645, 925, 685)]}]},
        {"id": 2, "data": {"image": "/data/local-files/?d=images/other_p1.png"}, "annotations": []},
    ]
    (tmp_path / "export.json").write_text(json.dumps(export))

    stats = convert(tmp_path / "export.json", workdir, tmp_path / "out" / "labelled.jsonl")
    assert (stats.pages, stats.entities, stats.unlabelled_tasks, stats.empty_boxes) == (1, 2, 1, 0)

    page = read_jsonl(tmp_path / "out" / "labelled.jsonl")[0]
    assert page.image_path == (workdir / "images" / "inv_p1.png").resolve()
    assert [label for label in page.labels if label != "O"] == ["B-INVOICE_NUMBER", "B-TOTAL_AMOUNT"]


def test_rectangles_from_annotation_converts_percent_to_pixels():
    ann = {"result": [
        {"type": "rectanglelabels", "value": {"x": 10, "y": 20, "width": 50, "height": 5, "rectanglelabels": ["DATE"]}},
        {"type": "rectanglelabels", "value": {"x": 0, "y": 0, "width": 1, "height": 1, "rectanglelabels": ["SHIP_TO"]}},
        {"type": "textarea", "value": {"text": ["note"]}},
    ]}
    assert rectangles_from_annotation(ann, 1000, 2000) == [("DATE", (100.0, 400.0, 600.0, 500.0))]


# --- split -----------------------------------------------------------------


def _write_source(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / "img.png").write_bytes(b"")
    path.write_text("".join(
        json.dumps({"id": f"{path.stem}{i}", "image": "img.png", "words": ["a", "b"], "boxes": [[0, 0, 1, 1]] * 2,
                    "labels": ["B-DATE", "O"]}) + "\n" for i in range(n)))


def test_merge_and_split_stratifies_and_oversamples(tmp_path):
    a, b = tmp_path / "a" / "train.jsonl", tmp_path / "b" / "labelled.jsonl"
    _write_source(a, 20)
    _write_source(b, 10)
    counts = merge_and_split([a, b], tmp_path / "final", val_fraction=0.1, repeat={b: 3})

    assert counts["val"] == {"a/train.jsonl": 2, "b/labelled.jsonl": 1}
    assert counts["train"] == {"a/train.jsonl": 18, "b/labelled.jsonl": 27}
    train = [json.loads(x) for x in (tmp_path / "final" / "train.jsonl").read_text().splitlines()]
    assert len(train) == 45
    assert all((tmp_path / "final" / r["image"]).resolve().exists() for r in train)
    val_ids = {json.loads(x)["id"] for x in (tmp_path / "final" / "val.jsonl").read_text().splitlines()}
    assert not val_ids & {r["id"] for r in train}

    assert label_stats(tmp_path / "final" / "val.jsonl") == {"pages": 3, "words": 6, "DATE": 3}


def test_ls_prepare_writes_images_ocr_and_tasks(tmp_path, monkeypatch):
    import docai.data.labelstudio as ls

    raw = tmp_path / "raw" / "acme co"
    raw.mkdir(parents=True)
    Image.new("RGB", (PAGE_W, PAGE_H), "white").save(raw / "inv 1.png")
    (raw / "notes.txt").write_text("ignored")
    (raw / "broken.jpg").write_bytes(b"not a jpeg")
    monkeypatch.setattr(ls, "run_ocr", lambda img, **_: [w for w, _ in INVOICE])

    stats = ls.prepare(tmp_path / "raw", tmp_path / "work")
    assert (stats.documents, stats.pages, len(stats.failed)) == (1, 1, 1)

    tasks = json.loads((tmp_path / "work" / "tasks.json").read_text())
    assert tasks == [{"data": {"image": "/data/local-files/?d=images/acme_co_inv_1_p1.png", "page_id": "acme_co_inv_1_p1"}}]
    sidecar = json.loads((tmp_path / "work" / "ocr" / "acme_co_inv_1_p1.json").read_text())
    assert (sidecar["width"], sidecar["height"]) == Image.open(tmp_path / "work" / "images" / "acme_co_inv_1_p1.png").size
    assert len(sidecar["words"]) == len(INVOICE)
    assert (tmp_path / "work" / "label_config.xml").exists()


def test_merge_and_split_keeps_document_pages_together(tmp_path):
    src = tmp_path / "inv" / "labelled.jsonl"
    src.parent.mkdir()
    src.write_text("".join(
        json.dumps({"id": f"doc{d}_p{p}", "image": "x.png", "words": [], "boxes": [], "labels": []}) + "\n"
        for d in range(10) for p in (1, 2, 3)))
    merge_and_split([src], tmp_path / "final", val_fraction=0.2)
    val_docs = {json.loads(x)["id"].split("_")[0] for x in (tmp_path / "final" / "val.jsonl").read_text().splitlines()}
    train_docs = {json.loads(x)["id"].split("_")[0] for x in (tmp_path / "final" / "train.jsonl").read_text().splitlines()}
    assert len(val_docs) == 2 and not val_docs & train_docs


def test_sroie_punctuation_inside_entity_is_labelled():
    words, missing = sroie_words(["TEO HENG STATIONERY & BOOKS", "TOTAL: 5.00"], [(0, 0, 300, 20), (0, 40, 200, 60)],
                                 {"company": "TEO HENG STATIONERY & BOOKS", "total": "5.00"})
    assert missing == []
    assert [t for t in bio_tags(words)][:5] == ["B-VENDOR_NAME"] + ["I-VENDOR_NAME"] * 4
