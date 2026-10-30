from __future__ import annotations

import datetime as dt

import pytest

from conftest import INVOICE, pred, word
from docai.postprocessing import (
    PageResult,
    build_output,
    clean_invoice_number,
    clean_vendor_name,
    decode_bio,
    detect_currency,
    filter_by_confidence,
    looks_like_money,
    merge_adjacent_entities,
    parse_amount,
    parse_date,
)

# --- BIO decoding ----------------------------------------------------------


def test_decode_bio_well_formed():
    a, b, c, d = (word(t, 10 * i, 0, 10 * i + 9, 9) for i, t in enumerate("abcd"))
    ents = decode_bio([pred(a, "B-DATE", 0.9), pred(b, "I-DATE", 0.7), pred(c, "O"), pred(d, "B-TOTAL_AMOUNT")])
    assert [(e.entity_type, e.text) for e in ents] == [("DATE", "a b"), ("TOTAL_AMOUNT", "d")]
    assert ents[0].confidence == pytest.approx(0.8)


def test_decode_bio_lenient_on_malformed_sequences():
    a, b, c, d = (word(t, 10 * i, 0, 10 * i + 9, 9) for i, t in enumerate("abcd"))
    ents = decode_bio([pred(a, "I-DATE"), pred(b, "I-TAX_AMOUNT"), pred(c, "B-TAX_AMOUNT"), pred(d, "I-TAX_AMOUNT")])
    assert [(e.entity_type, e.text) for e in ents] == [("DATE", "a"), ("TAX_AMOUNT", "b"), ("TAX_AMOUNT", "c d")]


def test_merge_adjacent_same_line_only():
    jan, d15, y24 = word("Jan", 0, 0, 30, 10), word("15", 35, 0, 50, 10), word("2024", 55, 0, 90, 10, line=2)
    ents = decode_bio([pred(jan, "B-DATE", 0.9), pred(d15, "B-DATE", 0.6), pred(y24, "B-DATE")])
    merged = merge_adjacent_entities(ents)
    assert [e.text for e in merged] == ["Jan 15", "2024"]
    assert merged[0].confidence == pytest.approx(0.75)


def test_merge_keeps_line_items_separate():
    a, b = word("Bolt", 0, 0, 30, 10), word("Nut", 35, 0, 60, 10)
    ents = decode_bio([pred(a, "B-ITEM_DESCRIPTION"), pred(b, "B-ITEM_DESCRIPTION")])
    assert len(merge_adjacent_entities(ents)) == 2


def test_filter_by_confidence_is_strict():
    a, b = word("a", 0, 0, 1, 1), word("b", 5, 0, 6, 1)
    ents = decode_bio([pred(a, "B-DATE", 0.70), pred(b, "B-TOTAL_AMOUNT", 0.71)])
    assert [e.entity_type for e in filter_by_confidence(ents, 0.70)] == ["TOTAL_AMOUNT"]


# --- value parsing ---------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("$1,234.56", 1234.56),
        ("1.234,56 €", 1234.56),
        ("EUR 1 234,50", 1234.50),
        ("1'234.50", 1234.50),
        ("1,234", 1234.0),
        ("12,5", 12.5),
        ("1.234.567", 1234567.0),
        ("(45.00)", -45.0),
        ("- 10.00", -10.0),
        ("Total: 99", 99.0),
        ("n/a", None),
    ],
)
def test_parse_amount(text, expected):
    assert parse_amount(text) == expected


@pytest.mark.parametrize(
    ("text", "money"),
    [("$120.00", True), ("1,200.50", True), ("99,90€", True), ("2", False), ("2024", False), ("INV-10.00X1", False)],
)
def test_looks_like_money(text, money):
    assert looks_like_money(text) is money


@pytest.mark.parametrize(
    ("text", "expected"),
    [("Total USD 10", "USD"), ("€ 5,00", "EUR"), ("CA$ 12", "CAD"), ("£3", "GBP"), ("Rs. 500", "INR"), ("10.00", None)],
)
def test_detect_currency(text, expected):
    assert detect_currency(text) == expected


@pytest.mark.parametrize(
    ("text", "day_first", "expected"),
    [
        ("2024-01-15", False, dt.date(2024, 1, 15)),
        ("15/01/2024", False, dt.date(2024, 1, 15)),
        ("01/15/2024", False, dt.date(2024, 1, 15)),
        ("03/04/2024", False, dt.date(2024, 3, 4)),
        ("03/04/2024", True, dt.date(2024, 4, 3)),
        ("15.01.24", False, dt.date(2024, 1, 15)),
        ("15 Jan 2024", False, dt.date(2024, 1, 15)),
        ("5th September, 2023", False, dt.date(2023, 9, 5)),
        ("March 5, 2024", False, dt.date(2024, 3, 5)),
        ("Sept. 30 2022", False, dt.date(2022, 9, 30)),
        ("31/02/2024", False, None),
        ("Due on receipt", False, None),
    ],
)
def test_parse_date(text, day_first, expected):
    assert parse_date(text, day_first=day_first) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [("INV-2024-0042", "INV-2024-0042"), ("Invoice No.: 10023", "10023"), ("# A-77", "A-77"), ("Invoice", None)],
)
def test_clean_invoice_number(text, expected):
    assert clean_invoice_number(text) == expected


def test_clean_vendor_name():
    assert clean_vendor_name("  Acme   Supplies Ltd. :") == "Acme Supplies Ltd"
    assert clean_vendor_name("123 - ") is None


# --- assembly --------------------------------------------------------------


def test_build_output_end_to_end():
    words = [w for w, _ in INVOICE]
    preds = [pred(w, label) for w, label in INVOICE]
    entities, items = build_output([PageResult(page=0, words=words, predictions=preds)])

    assert entities.vendor_name.value == "Acme Supplies Ltd"
    assert entities.invoice_number.value == "INV-2024-0042"
    assert entities.invoice_date.value == "2024-03-05"
    assert entities.total_amount.value == 165.55
    assert entities.total_amount.currency == "USD"
    assert entities.tax_amount.value == 15.05
    # Line total is the right-most money cell, not the unit price or quantity.
    assert [(i.description, i.amount) for i in items] == [("Widget A", 120.0), ("Gadget", 30.5)]


def test_build_output_drops_low_confidence_and_invalid_values():
    low = word("99.00", 800, 650, 920, 680, line=7)  # no symbol anywhere -> default currency
    bad_date = word("someday", 690, 90, 780, 120, line=3)
    entities, _ = build_output(
        [PageResult(page=0, words=[low, bad_date], predictions=[pred(low, "B-TOTAL_AMOUNT", 0.5), pred(bad_date, "B-DATE")])],
        default_currency="EUR",
    )
    assert entities.total_amount.value is None
    assert entities.total_amount.currency == "EUR"
    assert entities.total_amount.confidence == 0.0
    assert entities.invoice_date.value is None


def test_build_output_picks_most_confident_candidate_across_pages():
    p0 = word("$10.00", 800, 650, 920, 680)
    p1 = word("€250.00", 800, 650, 920, 680)
    entities, _ = build_output(
        [
            PageResult(page=0, words=[p0], predictions=[pred(p0, "B-TOTAL_AMOUNT", 0.8)]),
            PageResult(page=1, words=[p1], predictions=[pred(p1, "B-TOTAL_AMOUNT", 0.9)]),
        ]
    )
    assert entities.total_amount.value == 250.0
    assert entities.total_amount.currency == "EUR"
