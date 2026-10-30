"""Synthetic invoices with exact word-level labels.

Each page is drawn from randomised parts - layout, fonts, currency and number
format, date format, invoice-number style, 1-12 line items, optional tax,
discount and shipping rows - and every word's box and BIO tag is recorded as
it is drawn. Decoys the model must learn to ignore are always present: the
customer's company in "Bill to", due dates, PO numbers, subtotals, tax IDs
and bank details.

Scans are imitated with small rotations, blur, noise, JPEG artefacts,
fax-style resampling and uneven lighting.

Synthetic pages teach layout and field semantics cheaply, and they are the
only automatically-labelled source of ``INVOICE_NUMBER``. They are not a
substitute for real invoices from your own suppliers.
"""

from __future__ import annotations

import datetime as dt
import io
import logging
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from docai.data.common import JsonlWriter, LabelledWord, reading_order

logger = logging.getLogger(__name__)

PAGE_W, PAGE_H = 1700, 2200  # US Letter at 200 DPI

# --------------------------------------------------------------------------
# Fonts
# --------------------------------------------------------------------------

_FONT_DIRS = ("/System/Library/Fonts/Supplemental", "/System/Library/Fonts", "/Library/Fonts",
              "/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation", "C:/Windows/Fonts")
_FONT_PAIRS = (  # (regular, bold) file names
    ("Arial.ttf", "Arial Bold.ttf"),
    ("Verdana.ttf", "Verdana Bold.ttf"),
    ("Tahoma.ttf", "Tahoma Bold.ttf"),
    ("Georgia.ttf", "Georgia Bold.ttf"),
    ("Times New Roman.ttf", "Times New Roman Bold.ttf"),
    ("Trebuchet MS.ttf", "Trebuchet MS Bold.ttf"),
    ("Courier New.ttf", "Courier New Bold.ttf"),
    ("Arial Narrow.ttf", "Arial Narrow Bold.ttf"),
    ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"),
    ("DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf"),
    ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf"),
    ("LiberationSerif-Regular.ttf", "LiberationSerif-Bold.ttf"),
    ("LiberationMono-Regular.ttf", "LiberationMono-Bold.ttf"),
)


def discover_fonts() -> list[tuple[Path, Path]]:
    """Regular/bold TrueType pairs installed on this machine."""
    found: list[tuple[Path, Path]] = []
    for regular, bold in _FONT_PAIRS:
        for d in _FONT_DIRS:
            r, b = Path(d) / regular, Path(d) / bold
            if r.exists() and b.exists():
                found.append((r, b))
                break
    return found


@dataclass(frozen=True)
class FontSet:
    regular: Callable[[int], ImageFont.FreeTypeFont]
    bold: Callable[[int], ImageFont.FreeTypeFont]


def _font_set(pair: tuple[Path, Path] | None) -> FontSet:
    cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}

    def loader(path: Path | None, key: str) -> Callable[[int], ImageFont.FreeTypeFont]:
        def get(size: int) -> ImageFont.FreeTypeFont:
            if (key, size) not in cache:
                cache[(key, size)] = (
                    ImageFont.truetype(str(path), size) if path else ImageFont.load_default(size)  # type: ignore[assignment]
                )
            return cache[(key, size)]

        return get

    return FontSet(loader(pair[0] if pair else None, "r"), loader(pair[1] if pair else None, "b"))


# --------------------------------------------------------------------------
# Vocabulary (all invented; any resemblance to real businesses is accidental)
# --------------------------------------------------------------------------

_NAME_A = ("Blue", "Summit", "Harbor", "Pine", "Atlas", "Crescent", "Granite", "Silver", "Maple", "Orion", "Redwood",
           "Lakeside", "Northgate", "Brightline", "Ironwood", "Cedar", "Falcon", "Meridian", "Copper", "Juniper",
           "Keystone", "Lumen", "Quarry", "Riverbend", "Sterling", "Tidewater", "Vantage", "Willow", "Zenith", "Aurora")
_NAME_B = ("Supply", "Logistics", "Consulting", "Print", "Electrical", "Office Solutions", "Foods", "Software",
           "Engineering", "Hardware", "Design Studio", "Medical Supplies", "Packaging", "Textiles", "Analytics",
           "Catering", "Builders", "Freight", "Labs", "Media", "Plumbing", "Security", "Traders", "Distributors")
_SURNAMES = ("Patel", "Garcia", "Nguyen", "Okafor", "Schmidt", "Rossi", "Kowalski", "Tanaka", "Haddad", "Silva",
             "Andersen", "Moreau", "Ivanova", "Mensah", "Kaur", "O'Brien", "Fischer", "Costa", "Novak", "Reyes")
_SUFFIXES = ("Ltd", "Ltd.", "LLC", "Inc.", "GmbH", "Pvt. Ltd.", "Co.", "S.A.", "& Sons", "Limited", "Corp.", "B.V.", "")
_STREETS = ("Oak", "Market", "Station", "Mill", "Church", "Park", "High", "King", "Bridge", "Elm", "Industrial",
            "Commerce", "Harbour", "Victoria", "Lincoln", "Riverside")
_STREET_TYPES = ("Street", "St.", "Road", "Rd", "Avenue", "Ave", "Lane", "Way", "Boulevard", "Drive", "Park")
_CITIES = ("Springfield", "Riverton", "Lakewood", "Fairview", "Kingston", "Ashford", "Milton", "Bristol", "Dover",
           "Georgetown", "Franklin", "Clayton", "Salem", "Oakridge", "Brookfield", "Westport")
_PRODUCTS = ("Consulting services", "Web development", "A4 copy paper (box of 5 reams)", "Laser toner cartridge",
             "Ergonomic office chair", "USB-C docking station", "Annual software licence", "Cloud hosting - monthly",
             "Installation and setup", "Maintenance contract", "Steel fasteners M8 (100 pcs)", "Safety gloves",
             "LED panel light 60x60", "Network switch 24-port", "Freight charges", "Graphic design - logo package",
             "Catering for 40 guests", "Printed brochures A5", "Cleaning services", "Legal advisory",
             "Laptop stand aluminium", "Wireless keyboard and mouse set", "Hard hat with visor", "Copper cable 2.5mm",
             "PVC pipe 40mm x 3m", "Coffee beans 1kg", "Training workshop (half day)", "Data migration",
             "Security audit", "Shipping pallets", "Engineering hours", "Site survey", "Replacement filter",
             "Paint - white 10L", "Monitor 27 inch", "Packing tape 48mm", "Stationery bundle", "Courier delivery",
             "Software support - premium tier", "Translation services (per page)", "Photography session",
             "Warehouse storage (per pallet/week)", "Electrical inspection", "Hydraulic pump repair")

# (symbol, code, decimal separator, thousands separator, symbol after amount?, space between symbol and number?)
_CURRENCIES = (
    ("$", "USD", ".", ",", False, False),
    ("$", "USD", ".", ",", False, True),
    ("USD", "USD", ".", ",", False, True),
    ("€", "EUR", ",", ".", True, True),
    ("€", "EUR", ".", ",", False, False),
    ("EUR", "EUR", ",", ".", False, True),
    ("£", "GBP", ".", ",", False, False),
    ("Rs.", "INR", ".", ",", False, True),
    ("INR", "INR", ".", ",", False, True),
    ("CHF", "CHF", ".", "'", False, True),
    ("CA$", "CAD", ".", ",", False, False),
    ("A$", "AUD", ".", ",", False, False),
    ("", "", ".", ",", False, False),  # bare numbers, currency only in a caption
)
_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d %B %Y", "%B %d, %Y", "%d-%b-%Y", "%d.%m.%Y", "%b %d, %Y",
                 "%d %b %Y")
_TAX_LABELS = ("Tax", "Sales Tax", "VAT", "GST", "Tax ({rate}%)", "VAT {rate}%", "GST @ {rate}%", "Sales Tax ({rate}%)")
_TOTAL_LABELS = ("Total", "TOTAL", "Total Due", "Amount Due", "Grand Total", "Balance Due", "Total Amount",
                 "TOTAL DUE", "Invoice Total")
_INVNO_LABELS = ("Invoice No:", "Invoice #", "Invoice Number:", "Inv. No.", "Invoice No.", "Bill No:", "No.", "Invoice:")
_DATE_LABELS = ("Date:", "Invoice Date:", "Issue Date:", "Dated:", "Date of Issue:", "Invoice date")


# --------------------------------------------------------------------------
# Canvas that records what it draws
# --------------------------------------------------------------------------


class Canvas:
    """A white page that records every word it draws, with its box and entity."""

    def __init__(self, width: int, height: int) -> None:
        self.image = Image.new("L", (width, height), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.words: list[LabelledWord] = []
        self._next_entity = 0

    def new_entity(self) -> int:
        self._next_entity += 1
        return self._next_entity

    def text(
        self,
        x: float,
        y: float,
        s: str,
        font: ImageFont.FreeTypeFont,
        *,
        entity: str | None = None,
        entity_id: int | None = None,
        align: str = "left",
        fill: int = 0,
    ) -> float:
        """Draw ``s`` with its top-left (or top-right if ``align='right'``) at ``(x, y)``.

        Returns the x coordinate just past the text. Words get ``entity`` and a
        shared instance id (a fresh one unless ``entity_id`` is given).
        """
        width = font.getlength(s)
        if align == "right":
            x -= width
        elif align == "center":
            x -= width / 2
        self.draw.text((x, y), s, font=font, fill=fill, anchor="la")
        eid = (entity_id if entity_id is not None else self.new_entity()) if entity else None

        pos = 0
        for token in s.split():
            start = s.index(token, pos)
            pos = start + len(token)
            left, top, right, bottom = font.getbbox(token, anchor="la")
            x0 = x + font.getlength(s[:start]) + left
            self.words.append(
                LabelledWord(
                    text=token,
                    box=(int(x0), int(y + top), int(math.ceil(x0 + right - left)), int(math.ceil(y + bottom))),
                    entity_type=entity,
                    entity_id=eid,
                )
            )
        return x + width

    def line(self, x0: float, y0: float, x1: float, y1: float, width: int = 2, fill: int = 0) -> None:
        self.draw.line((x0, y0, x1, y1), fill=fill, width=width)

    def rect(self, x0: float, y0: float, x1: float, y1: float, fill: int) -> None:
        self.draw.rectangle((x0, y0, x1, y1), fill=fill)


# --------------------------------------------------------------------------
# Content generators
# --------------------------------------------------------------------------


def company_name(rng: random.Random) -> str:
    if rng.random() < 0.3:
        base = f"{rng.choice(_SURNAMES)} {rng.choice(('& Associates', 'Traders', 'Brothers', 'Enterprises', 'Group'))}"
    else:
        base = f"{rng.choice(_NAME_A)} {rng.choice(_NAME_B)}"
    suffix = rng.choice(_SUFFIXES)
    return f"{base} {suffix}".strip()


def address_lines(rng: random.Random) -> list[str]:
    lines = [f"{rng.randint(1, 980)} {rng.choice(_STREETS)} {rng.choice(_STREET_TYPES)}"]
    if rng.random() < 0.3:
        lines.append(f"Unit {rng.randint(1, 40)}")
    lines.append(f"{rng.choice(_CITIES)} {rng.randint(10000, 99999)}")
    return lines


def invoice_number(rng: random.Random, issued: dt.date) -> str:
    n = rng.randint(1, 99999)
    return rng.choice((
        f"INV-{issued.year}-{n:04d}",
        f"{rng.randint(1000, 999999)}",
        f"{rng.choice('ABCDEFGHKLMNPRST')}-{rng.randint(100, 9999)}",
        f"{issued.year}/{n:04d}",
        f"INV{n:05d}",
        f"{rng.choice(('SI', 'TI', 'BL', 'IN'))}{issued.strftime('%y%m')}{rng.randint(10, 999):03d}",
    ))


@dataclass(frozen=True)
class MoneyFormat:
    symbol: str
    code: str
    decimal: str
    thousands: str
    after: bool
    space: bool

    def __call__(self, value: float, with_symbol: bool = True) -> str:
        whole, frac = f"{abs(value):,.2f}".split(".")
        number = whole.replace(",", self.thousands) + self.decimal + frac
        if value < 0:
            number = "-" + number
        if not with_symbol or not self.symbol:
            return number
        sep = " " if self.space else ""
        return f"{number}{sep}{self.symbol}" if self.after else f"{self.symbol}{sep}{number}"


# --------------------------------------------------------------------------
# Page layout
# --------------------------------------------------------------------------


def render_invoice(rng: random.Random, fonts: Sequence[tuple[Path, Path]]) -> Canvas:
    """Draw one invoice page and return the canvas with its labelled words."""
    fs = _font_set(rng.choice(fonts) if fonts else None)
    base = rng.randint(24, 32)
    reg, bold = fs.regular(base), fs.bold(base)
    small = fs.regular(max(18, base - 6))
    lh = int(base * rng.uniform(1.35, 1.6))  # line height
    c = Canvas(PAGE_W, PAGE_H)
    left = rng.randint(90, 170)
    right = PAGE_W - rng.randint(90, 170)
    money = MoneyFormat(*rng.choice(_CURRENCIES))
    issued = dt.date(2018, 1, 1) + dt.timedelta(days=rng.randint(0, 3200))
    date_fmt = rng.choice(_DATE_FORMATS)

    # --- vendor header ------------------------------------------------
    vendor = company_name(rng)
    vfont = fs.bold(int(base * rng.uniform(1.4, 2.3)))
    y = rng.randint(90, 160)
    header_style = rng.choice(("left", "left", "center", "right"))
    hx = {"left": left, "center": PAGE_W / 2, "right": right}[header_style]
    c.text(hx, y, vendor if rng.random() > 0.2 else vendor.upper(), vfont, entity="VENDOR_NAME", align=header_style)
    y += int(vfont.size * 1.4)
    for line in address_lines(rng) + rng.sample(
        [f"Tel: +{rng.randint(1, 99)} {rng.randint(100, 999)} {rng.randint(1000000, 9999999)}",
         f"accounts@{vendor.split()[0].lower()}.example",
         f"VAT Reg No: {rng.choice(('GB', 'DE', 'FR', 'NL'))}{rng.randint(100000000, 999999999)}",
         f"GSTIN: {rng.randint(10, 37)}ABCDE{rng.randint(1000, 9999)}F1Z{rng.randint(1, 9)}"], k=rng.randint(1, 3)):
        c.text(hx, y, line, small, align=header_style)
        y += int(small.size * 1.35)

    # --- title + meta block ------------------------------------------
    title = rng.choice(("INVOICE", "Invoice", "TAX INVOICE", "Tax Invoice", "BILL", "Sales Invoice"))
    tfont = fs.bold(int(base * rng.uniform(1.5, 2.2)))
    meta_right = header_style != "right"
    mx_label, mx_value = (right - 560, right) if meta_right else (left, left + 560)
    ty = rng.randint(90, 160) if header_style != "center" else y + 20
    c.text(mx_value if meta_right else mx_label, ty, title, tfont, align="right" if meta_right else "left")
    my = ty + int(tfont.size * 1.6)

    inv_no = invoice_number(rng, issued)
    rows: list[tuple[str, str, str | None]] = [
        (rng.choice(_INVNO_LABELS), inv_no, "INVOICE_NUMBER"),
        (rng.choice(_DATE_LABELS), issued.strftime(date_fmt), "DATE"),
    ]
    if rng.random() < 0.7:
        due = issued + dt.timedelta(days=rng.choice((7, 14, 15, 30, 45, 60)))
        rows.append((rng.choice(("Due Date:", "Payment Due:", "Due:")), due.strftime(date_fmt), None))
    if rng.random() < 0.5:
        rows.append((rng.choice(("PO Number:", "PO #", "Order No:", "Ref:")), f"PO-{rng.randint(1000, 99999)}", None))
    if rng.random() < 0.4:
        rows.append(("Customer ID:", f"C{rng.randint(1000, 99999)}", None))
    if rng.random() < 0.3:
        rows.append(("Terms:", rng.choice(("Net 30", "Net 15", "Due on receipt", "50% advance")), None))
    rng.shuffle(rows)
    inline = rng.random() < 0.35
    for label, value, ent in rows:
        if inline:
            end = c.text(mx_label, my, label, bold if rng.random() < 0.5 else reg)
            c.text(end + base * 0.5, my, value, reg, entity=ent)
        else:
            c.text(mx_label, my, label, bold)
            c.text(mx_value, my, value, reg, entity=ent, align="right")
        my += lh

    # --- bill to -------------------------------------------------------
    y = max(y, my) + rng.randint(40, 90)
    c.text(left, y, rng.choice(("Bill To:", "BILL TO", "Billed To:", "Customer:", "Invoice To:")), bold)
    y += lh
    c.text(left, y, company_name(rng), reg)  # decoy: the buyer is not the vendor
    y += lh
    for line in address_lines(rng):
        c.text(left, y, line, small)
        y += int(small.size * 1.35)
    if rng.random() < 0.35:
        sx, sy = PAGE_W / 2 + 40, y - lh - 2 * int(small.size * 1.35)
        c.text(sx, sy - lh, "Ship To:", bold)
        for line in address_lines(rng):
            c.text(sx, sy, line, small)
            sy += int(small.size * 1.35)

    # --- items table -------------------------------------------------
    y += rng.randint(50, 100)
    has_sku = rng.random() < 0.3
    cols = {"amount": right, "unit": right - rng.randint(230, 300), "qty": right - rng.randint(420, 500)}
    desc_x = left + (180 if has_sku else 0)
    desc_max = cols["qty"] - 90 - desc_x
    header_band = rng.random() < 0.5
    if header_band:
        c.rect(left - 10, y - 10, right + 10, y + lh, fill=rng.randint(200, 235))
    if has_sku:
        c.text(left, y, rng.choice(("Item", "SKU", "Code")), bold)
    c.text(desc_x, y, rng.choice(("Description", "Item Description", "Particulars", "Product / Service", "Details")), bold)
    c.text(cols["qty"], y, rng.choice(("Qty", "Quantity", "Hours", "Units")), bold, align="right")
    c.text(cols["unit"], y, rng.choice(("Unit Price", "Rate", "Price", "Unit Cost")), bold, align="right")
    c.text(cols["amount"], y, rng.choice(("Amount", "Total", "Line Total", f"Amount ({money.code})" if money.code else "Amount")),
           bold, align="right")
    y += lh + 8
    if not header_band:
        c.line(left, y - 6, right, y - 6)

    subtotal = 0.0
    rule_rows = rng.random() < 0.4
    for _ in range(rng.choices(range(1, 13), weights=[8, 9, 9, 8, 7, 6, 5, 4, 3, 2, 2, 1])[0]):
        qty = rng.choice((1, 1, 1, 2, 3, 4, 5, 10, 12, 20, 25, 50))
        unit = round(rng.choice((rng.uniform(1, 50), rng.uniform(20, 400), rng.uniform(100, 3000))), 2)
        amount = round(qty * unit, 2)
        subtotal += amount
        if has_sku:
            c.text(left, y, f"{rng.choice('ABCDEFGHJK')}{rng.randint(100, 9999)}", reg)
        c.text(cols["qty"], y, str(qty), reg, align="right")
        c.text(cols["unit"], y, money(unit, with_symbol=rng.random() < 0.3), reg, align="right")
        c.text(cols["amount"], y, money(amount, with_symbol=rng.random() < 0.5), reg, align="right")
        # description, wrapped to the column; all its lines are one entity
        desc = rng.choice(_PRODUCTS)
        if rng.random() < 0.25:
            desc += " - " + rng.choice(("per hour", "incl. delivery", "as per quote", f"batch {rng.randint(1, 99)}"))
        eid = c.new_entity()
        lines, current = [], ""
        for word in desc.split():
            trial = f"{current} {word}".strip()
            if current and reg.getlength(trial) > desc_max:
                lines.append(current)
                current = word
            else:
                current = trial
        lines.append(current)
        for i, text in enumerate(lines):
            c.text(desc_x, y + i * lh, text, reg, entity="ITEM_DESCRIPTION", entity_id=eid)
        y += lh * len(lines) + rng.randint(4, 14)
        if rule_rows:
            c.line(left, y - 6, right, y - 6, width=1, fill=150)
    c.line(left, y, right, y)
    y += 24

    # --- totals ------------------------------------------------------
    label_x = cols["unit"] - rng.randint(0, 120)
    totals: list[tuple[str, float, str | None, bool]] = [("Subtotal", subtotal, None, False)]
    running = subtotal
    if rng.random() < 0.2:
        disc = round(subtotal * rng.choice((0.05, 0.1, 0.15)), 2)
        totals.append(("Discount", -disc, None, False))
        running -= disc
    if rng.random() < 0.85:
        rate = rng.choice((5, 7, 8.25, 10, 12, 15, 18, 19, 20, 21))
        tax = round(running * rate / 100, 2)
        rate_txt = f"{rate:g}"
        totals.append((rng.choice(_TAX_LABELS).format(rate=rate_txt), tax, "TAX_AMOUNT", False))
        running += tax
    if rng.random() < 0.25:
        ship = round(rng.uniform(5, 120), 2)
        totals.append((rng.choice(("Shipping", "Freight", "Delivery")), ship, None, False))
        running += ship
    totals.append((rng.choice(_TOTAL_LABELS), round(running, 2), "TOTAL_AMOUNT", True))
    if rng.random() < 0.2:
        totals.append(("Amount Paid", 0.0, None, False))

    for label, value, ent, is_total in totals:
        font = fs.bold(int(base * 1.15)) if is_total else reg
        if is_total and rng.random() < 0.4:
            c.rect(label_x - 20, y - 8, right + 10, y + font.size + 10, fill=rng.randint(215, 240))
        c.text(label_x, y, label + (":" if rng.random() < 0.4 else ""), font if is_total else reg, align="right")
        c.text(right, y, money(value, with_symbol=is_total or rng.random() < 0.5), font, entity=ent, align="right")
        y += int(lh * (1.25 if is_total else 1.0))

    # --- footer ------------------------------------------------------
    fy = max(y + 80, PAGE_H - rng.randint(220, 380))
    footer = rng.sample(
        ["Thank you for your business!",
         f"Please pay within {rng.choice((7, 14, 30))} days.",
         f"Bank: {rng.choice(_NAME_A)} Bank  IBAN: GB{rng.randint(10, 99)} ABCD {rng.randint(1000, 9999)} {rng.randint(1000, 9999)}",
         f"Account No: {rng.randint(10000000, 99999999)}  Sort code: {rng.randint(10, 99)}-{rng.randint(10, 99)}-{rng.randint(10, 99)}",
         "Goods remain our property until paid in full.",
         f"Questions? Contact billing on +{rng.randint(1, 99)} {rng.randint(1000, 9999)} {rng.randint(100000, 999999)}"],
        k=rng.randint(1, 3),
    )
    for line in footer:
        if fy > PAGE_H - 80:
            break
        c.text(left, fy, line, small)
        fy += int(small.size * 1.5)
    if rng.random() < 0.5 and fy < PAGE_H - 60:
        c.text(right, PAGE_H - 90, "Page 1 of 1", small, align="right")
    return c


# --------------------------------------------------------------------------
# Scan degradation
# --------------------------------------------------------------------------


def degrade(image: Image.Image, words: list[LabelledWord], rng: random.Random) -> Image.Image:
    """Make a clean render look scanned or photographed; boxes are moved to match any rotation."""
    if rng.random() < 0.6:
        angle = rng.uniform(-2.0, 2.0)
        w, h = image.size
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        cos, sin = abs(m[0, 0]), abs(m[0, 1])
        nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
        m[0, 2] += nw / 2 - w / 2
        m[1, 2] += nh / 2 - h / 2
        arr = cv2.warpAffine(np.asarray(image), m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=255)
        image = Image.fromarray(arr)
        for word in words:
            x0, y0, x1, y1 = word.box
            pts = np.array([[x0, y0, 1], [x1, y0, 1], [x1, y1, 1], [x0, y1, 1]], dtype=float) @ m.T
            word.box = (int(pts[:, 0].min()), int(pts[:, 1].min()), int(math.ceil(pts[:, 0].max())),
                        int(math.ceil(pts[:, 1].max())))

    arr = np.asarray(image, dtype=np.float32)
    if rng.random() < 0.4:  # uneven lighting
        h, w = arr.shape
        gx = np.linspace(rng.uniform(-25, 0), rng.uniform(0, 25), w)[None, :]
        gy = np.linspace(rng.uniform(-25, 0), rng.uniform(0, 25), h)[:, None]
        arr = arr + gx + gy
    if rng.random() < 0.5:  # paper tone / contrast
        arr = arr * rng.uniform(0.75, 1.0) + rng.uniform(0, 40)
    if rng.random() < 0.5:
        arr = arr + np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(3, 14), arr.shape)
    image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    if rng.random() < 0.4:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.3)))
    if rng.random() < 0.2:  # fax: down- then up-sample
        f = rng.uniform(0.4, 0.6)
        size = image.size
        image = image.resize((int(size[0] * f), int(size[1] * f)), Image.Resampling.NEAREST).resize(size)
    if rng.random() < 0.5:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=rng.randint(25, 75))
        image = Image.open(io.BytesIO(buf.getvalue()))
        image.load()
    return image.convert("RGB")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def generate(out_dir: Path, n: int, *, seed: int = 0, clean_fraction: float = 0.2) -> int:
    """Write ``n`` synthetic pages to ``out_dir/synth.jsonl`` + ``out_dir/images/``.

    Args:
        out_dir: Output directory.
        n: Number of pages.
        seed: Makes the set reproducible.
        clean_fraction: Share of pages left as clean digital renders (no scan effects).
    """
    fonts = discover_fonts()
    if not fonts:
        logger.warning("No TrueType fonts found; falling back to Pillow's default font (less variety)")
    rng = random.Random(seed)
    with JsonlWriter(out_dir / "synth.jsonl", out_dir / "images") as writer:
        for i in range(n):
            canvas = render_invoice(rng, fonts)
            # Order on the untilted layout: after rotation a long line drifts across rows.
            words = reading_order([w for w in canvas.words if w.box[2] > w.box[0] and w.box[3] > w.box[1]])
            image = canvas.image.convert("RGB") if rng.random() < clean_fraction else degrade(canvas.image, words, rng)
            writer.write(f"synth_{seed}_{i:05d}", image, words)
            if (i + 1) % 200 == 0:
                logger.info("synthetic: %d/%d", i + 1, n)
        return writer.count
