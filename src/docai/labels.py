"""BIO label schema shared by training and inference."""

from __future__ import annotations

from typing import Final

ENTITY_TYPES: Final[tuple[str, ...]] = (
    "INVOICE_NUMBER",
    "VENDOR_NAME",
    "DATE",
    "TOTAL_AMOUNT",
    "TAX_AMOUNT",
    "ITEM_DESCRIPTION",
)

OUTSIDE: Final[str] = "O"

LABELS: Final[list[str]] = [OUTSIDE] + [f"{p}-{e}" for e in ENTITY_TYPES for p in ("B", "I")]
LABEL2ID: Final[dict[str, int]] = {label: i for i, label in enumerate(LABELS)}
ID2LABEL: Final[dict[int, str]] = {i: label for label, i in LABEL2ID.items()}

# Value used by PyTorch's CrossEntropyLoss to skip a position.
IGNORE_INDEX: Final[int] = -100


def split_tag(tag: str) -> tuple[str, str | None]:
    """Split a BIO tag into ``(prefix, entity_type)``.

    >>> split_tag("B-DATE")
    ('B', 'DATE')
    >>> split_tag("O")
    ('O', None)
    """
    if tag == OUTSIDE or "-" not in tag:
        return OUTSIDE, None
    prefix, entity = tag.split("-", 1)
    return prefix, entity
