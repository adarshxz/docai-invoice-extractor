"""Pydantic models for the pipeline's intermediate data and its JSON output."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

BBox = tuple[int, int, int, int]


# --------------------------------------------------------------------------
# Intermediate structures
# --------------------------------------------------------------------------


class OCRWord(BaseModel):
    """One word from Tesseract with both its pixel box and its [0, 1000] box."""

    model_config = ConfigDict(frozen=True)

    text: str
    pixel_box: BBox
    norm_box: BBox
    ocr_confidence: float
    block_num: int = 0
    par_num: int = 0
    line_num: int = 0

    @property
    def line_key(self) -> tuple[int, int, int]:
        """Tesseract's (block, paragraph, line) identity of the word's line."""
        return (self.block_num, self.par_num, self.line_num)


class WordPrediction(BaseModel):
    """Model output for one OCR word."""

    word: OCRWord
    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class Entity(BaseModel):
    """A contiguous span of words that the BIO decoder grouped together."""

    entity_type: str
    words: list[OCRWord]
    confidence: float = Field(ge=0.0, le=1.0)
    page: int = 0

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


# --------------------------------------------------------------------------
# Output payload
# --------------------------------------------------------------------------


class ProcessingStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class DocumentMetadata(BaseModel):
    file_name: str
    total_pages: int
    processing_time_seconds: float
    status: ProcessingStatus


class TextField(BaseModel):
    value: str | None = None
    confidence: float = 0.0


class DateField(BaseModel):
    value: str | None = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="YYYY-MM-DD")
    confidence: float = 0.0


class MoneyField(BaseModel):
    value: float | None = None
    currency: str
    confidence: float = 0.0


class ExtractedEntities(BaseModel):
    invoice_number: TextField
    vendor_name: TextField
    invoice_date: DateField
    total_amount: MoneyField
    tax_amount: MoneyField


class LineItem(BaseModel):
    description: str
    amount: float


class ExtractionResult(BaseModel):
    """The strict JSON contract returned by the pipeline and the API."""

    document_metadata: DocumentMetadata
    extracted_entities: ExtractedEntities
    line_items: list[LineItem] = Field(default_factory=list)

    @classmethod
    def failed(cls, file_name: str, total_pages: int, elapsed: float, currency: str) -> "ExtractionResult":
        """An empty result with ``status = FAILED``."""
        return cls(
            document_metadata=DocumentMetadata(
                file_name=file_name,
                total_pages=total_pages,
                processing_time_seconds=round(elapsed, 3),
                status=ProcessingStatus.FAILED,
            ),
            extracted_entities=ExtractedEntities(
                invoice_number=TextField(),
                vendor_name=TextField(),
                invoice_date=DateField(),
                total_amount=MoneyField(currency=currency),
                tax_amount=MoneyField(currency=currency),
            ),
            line_items=[],
        )
