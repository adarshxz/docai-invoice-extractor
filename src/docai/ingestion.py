"""Stage 1a: turn an uploaded file (PDF, PNG, JPEG) into RGB page images."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageOps

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".png", ".jpg", ".jpeg"})
_PDF_MAGIC = b"%PDF"


class UnsupportedDocumentError(ValueError):
    """Raised when the input is not a PDF, PNG or JPEG, or cannot be decoded."""


def is_pdf(data: bytes) -> bool:
    """Detect a PDF by its magic bytes rather than trusting the file name."""
    return data[:1024].lstrip().startswith(_PDF_MAGIC)


def load_pages(
    data: bytes, *, dpi: int = 200, max_pages: int = 20, poppler_path: str | None = None
) -> tuple[list[Image.Image], int]:
    """Decode a document into RGB page images.

    Args:
        data: Raw file bytes.
        dpi: Rasterisation resolution for PDFs.
        max_pages: Upper bound on the number of pages returned.
        poppler_path: Directory with Poppler's binaries when they are not on ``PATH``.

    Returns:
        ``(pages, total_pages)`` where ``total_pages`` counts every page in the
        source, including any dropped by ``max_pages``.

    Raises:
        UnsupportedDocumentError: If the bytes are neither a PDF nor a readable image.
    """
    if not data:
        raise UnsupportedDocumentError("Empty file.")

    if is_pdf(data):
        return _load_pdf(data, dpi=dpi, max_pages=max_pages, poppler_path=poppler_path)

    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.format not in {"PNG", "JPEG", "MPO"}:
                raise UnsupportedDocumentError(f"Unsupported image format: {img.format}")
            page = ImageOps.exif_transpose(img).convert("RGB")
    except UnsupportedDocumentError:
        raise
    except Exception as exc:  # PIL raises a variety of errors for corrupt files
        raise UnsupportedDocumentError(f"Cannot decode image: {exc}") from exc
    return [page], 1


def load_pages_from_path(
    path: str | Path, *, dpi: int = 200, max_pages: int = 20, poppler_path: str | None = None
) -> tuple[list[Image.Image], int]:
    """Convenience wrapper around :func:`load_pages` for files on disk."""
    return load_pages(Path(path).read_bytes(), dpi=dpi, max_pages=max_pages, poppler_path=poppler_path)


def _load_pdf(data: bytes, *, dpi: int, max_pages: int, poppler_path: str | None) -> tuple[list[Image.Image], int]:
    # Imported lazily so image-only deployments don't need poppler bindings at import time.
    from pdf2image import convert_from_bytes, pdfinfo_from_bytes
    from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError

    try:
        total_pages = int(pdfinfo_from_bytes(data, poppler_path=poppler_path)["Pages"])
        pages = convert_from_bytes(
            data, dpi=dpi, first_page=1, last_page=min(total_pages, max_pages), poppler_path=poppler_path
        )
    except PDFInfoNotInstalledError as exc:
        raise RuntimeError("poppler-utils is not installed; it is required for PDF input.") from exc
    except (PDFPageCountError, PDFSyntaxError) as exc:
        raise UnsupportedDocumentError(f"Cannot read PDF: {exc}") from exc

    return [p.convert("RGB") for p in pages], total_pages
