"""End-to-end orchestration of the five pipeline stages for one document."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from PIL import Image

from docai.config import Settings, get_settings
from docai.ingestion import load_pages
from docai.model import LayoutLMv3TokenClassifier, TokenClassifier
from docai.ocr import run_ocr
from docai.postprocessing import PageResult, build_output
from docai.preprocessing import preprocess_page
from docai.schemas import DocumentMetadata, ExtractionResult, OCRWord, ProcessingStatus

logger = logging.getLogger(__name__)

OCRFunction = Callable[[Image.Image], list[OCRWord]]


class InvoiceExtractionPipeline:
    """Ingest -> preprocess -> OCR -> LayoutLMv3 -> post-process.

    The classifier and OCR function are injectable so tests (or alternative
    OCR engines) can replace them.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        classifier: TokenClassifier | None = None,
        ocr: OCRFunction | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._classifier = classifier
        self._ocr: OCRFunction = ocr or (
            lambda img: run_ocr(
                img,
                lang=self.settings.tesseract_lang,
                config=self.settings.tesseract_config,
                min_confidence=self.settings.ocr_min_confidence,
                tesseract_cmd=self.settings.tesseract_cmd,
            )
        )

    @property
    def classifier(self) -> TokenClassifier:
        """The model, loaded on first use."""
        if self._classifier is None:
            self._classifier = LayoutLMv3TokenClassifier.from_settings(self.settings)
        return self._classifier

    def process_page(self, page: Image.Image, index: int) -> PageResult:
        """Run stages 1b-3 on one page image."""
        s = self.settings
        cleaned = preprocess_page(
            page,
            min_angle=s.deskew_min_angle,
            max_angle=s.deskew_max_angle,
            clip_limit=s.clahe_clip_limit,
            tile_grid=s.clahe_tile_grid,
        )
        words = self._ocr(cleaned.image)
        predictions = self.classifier.predict(cleaned.image, words)
        logger.debug("page %d: skew %.2f deg, %d words", index, cleaned.skew_angle, len(words))
        return PageResult(page=index, words=words, predictions=predictions)

    def process_bytes(self, data: bytes, file_name: str) -> ExtractionResult:
        """Extract entities from a document's raw bytes.

        Never raises for bad input or processing errors: those yield a result
        with ``status = FAILED`` (the cause is logged).
        """
        s = self.settings
        start = time.perf_counter()
        total_pages = 0
        try:
            pages, total_pages = load_pages(data, dpi=s.pdf_dpi, max_pages=s.max_pages, poppler_path=s.poppler_path)
            if total_pages > len(pages):
                logger.warning("%s: only the first %d of %d pages processed", file_name, len(pages), total_pages)
            page_results = [self.process_page(page, i) for i, page in enumerate(pages)]
            entities, line_items = build_output(
                page_results,
                confidence_threshold=s.confidence_threshold,
                default_currency=s.default_currency,
                day_first=s.day_first_dates,
            )
        except Exception:
            logger.exception("Extraction failed for %s", file_name)
            return ExtractionResult.failed(file_name, total_pages, time.perf_counter() - start, s.default_currency)

        return ExtractionResult(
            document_metadata=DocumentMetadata(
                file_name=file_name,
                total_pages=total_pages,
                processing_time_seconds=round(time.perf_counter() - start, 3),
                status=ProcessingStatus.SUCCESS,
            ),
            extracted_entities=entities,
            line_items=line_items,
        )

    def process_path(self, path: str | Path, file_name: str | None = None) -> ExtractionResult:
        """Extract entities from a file on disk."""
        path = Path(path)
        return self.process_bytes(path.read_bytes(), file_name or path.name)


@lru_cache(maxsize=1)
def get_pipeline() -> InvoiceExtractionPipeline:
    """Process-wide pipeline, so the model is loaded once per worker process."""
    return InvoiceExtractionPipeline(get_settings())
