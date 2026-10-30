"""Stage 4: BIO decoding, entity merging, confidence filtering and value validation."""

from __future__ import annotations

import datetime as dt
import logging
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TypeVar

from docai.labels import split_tag
from docai.schemas import (
    DateField,
    Entity,
    ExtractedEntities,
    LineItem,
    MoneyField,
    OCRWord,
    TextField,
    WordPrediction,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Entity types that may legitimately appear several times next to each other
# (two line items on one row are two items, not one).
_NON_MERGEABLE: frozenset[str] = frozenset({"ITEM_DESCRIPTION"})


# --------------------------------------------------------------------------
# BIO decoding
# --------------------------------------------------------------------------


def decode_bio(predictions: Sequence[WordPrediction], page: int = 0) -> list[Entity]:
    """Group word-level BIO tags into entities.

    Decoding is lenient, as model output often is not well-formed BIO:

    * ``B-X`` always starts a new entity.
    * ``I-X`` continues the open entity if it is also ``X``; otherwise (an
      orphan ``I-``, or one following another type) it starts a new entity.
    * ``O`` closes the open entity.

    An entity's confidence is the mean of its words' confidences.
    """
    entities: list[Entity] = []
    words: list[OCRWord] = []
    confs: list[float] = []
    current: str | None = None

    def close() -> None:
        nonlocal words, confs, current
        if current is not None and words:
            entities.append(Entity(entity_type=current, words=words, confidence=sum(confs) / len(confs), page=page))
        words, confs, current = [], [], None

    for pred in predictions:
        prefix, etype = split_tag(pred.label)
        if etype is None:
            close()
            continue
        if prefix == "B" or etype != current:
            close()
            current = etype
        words.append(pred.word)
        confs.append(pred.confidence)
    close()
    return entities


def merge_adjacent_entities(entities: Sequence[Entity]) -> list[Entity]:
    """Join consecutive same-type entities that sit on the same text line.

    Repairs output like ``B-DATE B-DATE B-DATE`` for "Jan 15 2024". Merged
    confidence is the word-weighted mean. Types in ``_NON_MERGEABLE`` are left
    as they are.
    """
    merged: list[Entity] = []
    for ent in entities:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and prev.entity_type == ent.entity_type
            and ent.entity_type not in _NON_MERGEABLE
            and prev.page == ent.page
            and prev.words[-1].line_key == ent.words[0].line_key
        ):
            n_prev, n_new = len(prev.words), len(ent.words)
            merged[-1] = Entity(
                entity_type=prev.entity_type,
                words=[*prev.words, *ent.words],
                confidence=(prev.confidence * n_prev + ent.confidence * n_new) / (n_prev + n_new),
                page=prev.page,
            )
        else:
            merged.append(ent)
    return merged


def filter_by_confidence(entities: Iterable[Entity], threshold: float) -> list[Entity]:
    """Keep entities whose confidence is strictly above ``threshold``."""
    return [e for e in entities if e.confidence > threshold]


# --------------------------------------------------------------------------
# Value parsing & validation
# --------------------------------------------------------------------------

_CURRENCY_SYMBOLS: dict[str, str] = {
    "US$": "USD",
    "CA$": "CAD",
    "C$": "CAD",
    "A$": "AUD",
    "AU$": "AUD",
    "S$": "SGD",
    "HK$": "HKD",
    "R$": "BRL",
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "₹": "INR",
    "₩": "KRW",
    "₽": "RUB",
    "Rs.": "INR",
    "Rs": "INR",
}
_ISO_CODES = (
    "USD EUR GBP INR JPY CNY CAD AUD CHF SGD HKD NZD SEK NOK DKK PLN CZK ZAR AED SAR BRL MXN KRW RUB TRY"
).split()
_ISO_RE = re.compile(r"\b(" + "|".join(_ISO_CODES) + r")\b")
_SYMBOL_RE = re.compile("|".join(re.escape(s) for s in sorted(_CURRENCY_SYMBOLS, key=len, reverse=True)))

_NUMBER_RE = re.compile(r"\d(?:[\d.,'   ]*\d)?")
_MONEY_TOKEN_RE = re.compile(r"^[^\d]{0,4}\(?-?\d{1,3}(?:[,.' ]?\d{3})*[.,]\d{2}\)?[^\d]{0,4}$")


def detect_currency(text: str) -> str | None:
    """Return an ISO-4217 code found in ``text`` (codes win over symbols), or ``None``."""
    if m := _ISO_RE.search(text.upper()):
        return m.group(1)
    if m := _SYMBOL_RE.search(text):
        return _CURRENCY_SYMBOLS[m.group(0)]
    return None


def parse_amount(text: str) -> float | None:
    """Parse a monetary string into a float, handling US and European formats.

    Examples: ``"$1,234.56"`` -> 1234.56, ``"1.234,56 €"`` -> 1234.56,
    ``"(45.00)"`` -> -45.0, ``"1 234,5"`` -> 1234.5. Returns ``None`` if no
    number is present.
    """
    m = _NUMBER_RE.search(text)
    if not m:
        return None
    number = re.sub(r"[\s'  ]", "", m.group(0))
    before, after = text[: m.start()], text[m.end() :]
    negative = ("(" in before and ")" in after) or before.strip().endswith("-")

    last_dot, last_comma = number.rfind("."), number.rfind(",")
    if last_dot >= 0 and last_comma >= 0:
        decimal = "." if last_dot > last_comma else ","
    elif last_comma >= 0:
        # "1,234" / "1,234,567" are thousands; "12,5" / "12,50" are decimals.
        decimal = None if re.fullmatch(r"\d{1,3}(,\d{3})+", number) else ","
    elif number.count(".") > 1:
        decimal = None  # "1.234.567" thousands separators
    else:
        decimal = "." if last_dot >= 0 else None

    thousands = {".", ","} - ({decimal} if decimal else set())
    for sep in thousands:
        number = number.replace(sep, "")
    if decimal == ",":
        number = number.replace(",", ".")
    try:
        value = float(number)
    except ValueError:
        return None
    return -value if negative else value


def looks_like_money(text: str) -> bool:
    """True for tokens such as ``"$120.00"``, ``"1,200.50"``, ``"99,90€"``: a number with two decimals."""
    return bool(_MONEY_TOKEN_RE.match(text.strip()))


_MONTHS: dict[str, int] = {
    name: i
    for i, names in enumerate(
        [
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "sept", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        ],
        start=1,
    )
    for name in names
}
_ISO_DATE_RE = re.compile(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b")
_NUMERIC_DATE_RE = re.compile(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{4}|\d{2})\b")
_DAY_MONTH_YEAR_RE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?[\s\-.,/]*([A-Za-z]{3,9})\.?[\s\-.,/]*(\d{4}|\d{2})\b")
_MONTH_DAY_YEAR_RE = re.compile(r"\b([A-Za-z]{3,9})\.?[\s\-.,/]*(\d{1,2})(?:st|nd|rd|th)?[\s\-.,/]*(\d{4}|\d{2})\b")


def _make_date(year: int, month: int, day: int) -> dt.date | None:
    if year < 100:
        year += 2000
    if not 1900 <= year <= 2100:
        return None
    try:
        return dt.date(year, month, day)
    except ValueError:
        return None


def parse_date(text: str, *, day_first: bool = False) -> dt.date | None:
    """Parse common invoice date formats.

    Supported: ``2024-01-15``, ``15/01/2024``, ``01/15/24``, ``15.01.2024``,
    ``15 Jan 2024``, ``15th January, 2024``, ``Jan 15, 2024``. For numeric dates
    where both parts are <= 12, ``day_first`` decides the reading.
    """
    if m := _ISO_DATE_RE.search(text):
        return _make_date(int(m[1]), int(m[2]), int(m[3]))

    if m := _NUMERIC_DATE_RE.search(text):
        a, b, year = int(m[1]), int(m[2]), int(m[3])
        if a > 12:
            day, month = a, b
        elif b > 12:
            month, day = a, b
        else:
            day, month = (a, b) if day_first else (b, a)
        return _make_date(year, month, day)

    if (m := _DAY_MONTH_YEAR_RE.search(text)) and (month := _MONTHS.get(m[2].lower())):
        return _make_date(int(m[3]), month, int(m[1]))

    if (m := _MONTH_DAY_YEAR_RE.search(text)) and (month := _MONTHS.get(m[1].lower())):
        return _make_date(int(m[3]), month, int(m[2]))

    return None


# Label words are only stripped when a separator follows, so "INV-2024-001" survives intact.
_INVOICE_LABEL_RE = re.compile(r"^(?:(?:invoice|inv|bill|no|number|num|nr)(?=[\s.:#]|$)|[#:.\s])+", re.IGNORECASE)
_INVOICE_NUMBER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-/_.#]*$")


def clean_invoice_number(text: str) -> str | None:
    """Strip label words ("Invoice No.:") and validate an invoice number.

    A valid number is one alphanumeric token (``-/_.#`` allowed) containing at
    least one digit.
    """
    candidate = _INVOICE_LABEL_RE.sub("", text.strip()).replace(" ", "").strip(".:,;")
    if not candidate or not any(c.isdigit() for c in candidate):
        return None
    return candidate if _INVOICE_NUMBER_RE.match(candidate) else None


def clean_vendor_name(text: str) -> str | None:
    """Collapse whitespace and strip stray punctuation from the edges."""
    cleaned = re.sub(r"\s+", " ", text).strip(" \t:;,.|-_")
    return cleaned if any(c.isalpha() for c in cleaned) else None


# --------------------------------------------------------------------------
# Line items
# --------------------------------------------------------------------------


def _row_amount(description: Entity, page_words: Sequence[OCRWord]) -> float | None:
    """Find the amount for a line item: the right-most money-like word on its row.

    The row is the description's first text line; a word belongs to it when its
    vertical centre falls inside that line's band (with a 25 % tolerance) and it
    starts to the right of the description.
    """
    first_line = [w for w in description.words if w.line_key == description.words[0].line_key]
    top = min(w.pixel_box[1] for w in first_line)
    bottom = max(w.pixel_box[3] for w in first_line)
    right = max(w.pixel_box[2] for w in first_line)
    tol = 0.25 * (bottom - top)
    desc_ids = {id(w) for w in description.words}

    best: tuple[int, float] | None = None
    for w in page_words:
        if id(w) in desc_ids or w.pixel_box[0] < right:
            continue
        center = (w.pixel_box[1] + w.pixel_box[3]) / 2
        if not top - tol <= center <= bottom + tol or not looks_like_money(w.text):
            continue
        value = parse_amount(w.text)
        if value is not None and (best is None or w.pixel_box[0] > best[0]):
            best = (w.pixel_box[0], value)
    return best[1] if best else None


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


@dataclass
class PageResult:
    """Everything stage 4 needs from one page."""

    page: int
    words: list[OCRWord]
    predictions: list[WordPrediction]
    entities: list[Entity] = field(default_factory=list)


def _best(entities: Iterable[Entity], parse: Callable[[Entity], T | None]) -> tuple[T, Entity] | None:
    """Highest-confidence entity whose value parses; ties go to the earlier one."""
    best: tuple[T, Entity] | None = None
    for ent in entities:
        value = parse(ent)
        if value is None:
            logger.debug("Rejected %s candidate %r", ent.entity_type, ent.text)
            continue
        if best is None or ent.confidence > best[1].confidence:
            best = (value, ent)
    return best


def build_output(
    pages: Sequence[PageResult],
    *,
    confidence_threshold: float = 0.70,
    default_currency: str = "USD",
    day_first: bool = False,
) -> tuple[ExtractedEntities, list[LineItem]]:
    """Run BIO decoding, merging, filtering and validation over all pages.

    Single-valued fields take the highest-confidence valid candidate across
    the document. Currency comes from the amount's own text, then from anywhere
    on the document, then ``default_currency``. Line items keep page order and
    are only emitted when an amount is found on their row.
    """
    by_type: dict[str, list[Entity]] = {}
    for page in pages:
        page.entities = filter_by_confidence(
            merge_adjacent_entities(decode_bio(page.predictions, page=page.page)), confidence_threshold
        )
        for ent in page.entities:
            by_type.setdefault(ent.entity_type, []).append(ent)

    doc_currency = detect_currency(" ".join(w.text for p in pages for w in p.words)) or default_currency

    def text_field(etype: str, clean: Callable[[str], str | None]) -> TextField:
        hit = _best(by_type.get(etype, []), lambda e: clean(e.text))
        return TextField(value=hit[0], confidence=round(hit[1].confidence, 4)) if hit else TextField()

    def money_field(etype: str) -> MoneyField:
        hit = _best(by_type.get(etype, []), lambda e: parse_amount(e.text))
        if not hit:
            return MoneyField(currency=doc_currency)
        value, ent = hit
        return MoneyField(
            value=round(value, 2),
            currency=detect_currency(ent.text) or doc_currency,
            confidence=round(ent.confidence, 4),
        )

    date_hit = _best(by_type.get("DATE", []), lambda e: parse_date(e.text, day_first=day_first))
    entities = ExtractedEntities(
        invoice_number=text_field("INVOICE_NUMBER", clean_invoice_number),
        vendor_name=text_field("VENDOR_NAME", clean_vendor_name),
        invoice_date=(
            DateField(value=date_hit[0].isoformat(), confidence=round(date_hit[1].confidence, 4))
            if date_hit
            else DateField()
        ),
        total_amount=money_field("TOTAL_AMOUNT"),
        tax_amount=money_field("TAX_AMOUNT"),
    )

    line_items: list[LineItem] = []
    for page in pages:
        for ent in page.entities:
            if ent.entity_type != "ITEM_DESCRIPTION":
                continue
            amount = _row_amount(ent, page.words)
            description = re.sub(r"\s+", " ", ent.text).strip()
            if amount is None or not description:
                logger.debug("Dropped line item without amount: %r", description)
                continue
            line_items.append(LineItem(description=description, amount=round(amount, 2)))

    return entities, line_items
