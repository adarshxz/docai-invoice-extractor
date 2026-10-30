"""Stage 3: joint visual / text / layout token classification with LayoutLMv3.

The processor is given OCR words and their normalised boxes (``apply_ocr=False``).
LayoutLMv3's fast tokenizer splits each word into BPE sub-tokens and gives every
sub-token a copy of its word's ``[x0, y0, x1, y1]`` box, which is the alignment
the model was pre-trained with. Predictions are read back per word from the
word's first sub-token.

Pages longer than ``max_seq_length`` tokens are split into overlapping windows
(``stride`` tokens of overlap); a word seen in several windows gets the average
of its class probabilities.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

import numpy as np
from PIL import Image

from docai.config import Settings
from docai.labels import ID2LABEL, LABEL2ID, LABELS
from docai.schemas import OCRWord, WordPrediction

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class TokenClassifier(Protocol):
    """Anything that can label OCR words on a page image."""

    def predict(self, image: Image.Image, words: Sequence[OCRWord]) -> list[WordPrediction]: ...


def resolve_device(requested: str) -> "torch.device":
    """Turn ``'auto'`` into the best available accelerator, else honour the request."""
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def first_subtoken_positions(word_ids: Sequence[int | None]) -> dict[int, int]:
    """Map each word index to the position of its first sub-token in one window.

    Special and padding tokens (``word_id is None``) are skipped. A word whose
    first sub-token fell into the previous window is mapped to its first
    sub-token in this one.

    >>> first_subtoken_positions([None, 0, 0, 1, None, None])
    {0: 1, 1: 3}
    """
    positions: dict[int, int] = {}
    for pos, wid in enumerate(word_ids):
        if wid is not None and wid not in positions:
            positions[wid] = pos
    return positions


def aggregate_word_probabilities(
    window_probs: Sequence[np.ndarray],
    window_word_ids: Sequence[Sequence[int | None]],
    num_words: int,
) -> np.ndarray:
    """Average first-sub-token class probabilities for each word over all windows.

    Args:
        window_probs: One ``(seq_len, num_labels)`` softmax array per window.
        window_word_ids: Token-to-word mapping for each window.
        num_words: Number of OCR words on the page.

    Returns:
        A ``(num_words, num_labels)`` array. A word no window covered (only
        possible if the tokenizer dropped it) gets a one-hot ``O`` row.
    """
    num_labels = window_probs[0].shape[-1]
    totals = np.zeros((num_words, num_labels), dtype=np.float64)
    counts = np.zeros(num_words, dtype=np.int64)
    for probs, word_ids in zip(window_probs, window_word_ids, strict=True):
        for wid, pos in first_subtoken_positions(word_ids).items():
            totals[wid] += probs[pos]
            counts[wid] += 1

    missing = counts == 0
    totals[missing, LABEL2ID["O"]] = 1.0
    counts[missing] = 1
    return totals / counts[:, None]


class LayoutLMv3TokenClassifier:
    """Wraps a fine-tuned ``LayoutLMv3ForTokenClassification`` checkpoint.

    Load once per process (see :func:`docai.pipeline.get_pipeline`); the
    instance is safe to reuse across documents but not across threads.
    """

    def __init__(
        self,
        model_path: str,
        processor_path: str | None = None,
        *,
        device: str = "auto",
        max_seq_length: int = 512,
        stride: int = 128,
        batch_size: int = 4,
    ) -> None:
        """Load processor and model.

        Args:
            model_path: Fine-tuned checkpoint directory or hub id. Loading the
                bare ``microsoft/layoutlmv3-base`` works but its classification
                head is randomly initialised, so predictions are meaningless
                until the model is fine-tuned (``docai-train``).
            processor_path: Where to load the processor from; defaults to ``model_path``.
            device: ``'auto'`` or an explicit torch device string.
            max_seq_length: Tokens per window (LayoutLMv3 supports at most 512).
            stride: Token overlap between consecutive windows.
            batch_size: Windows per forward pass.
        """
        import torch
        from transformers import AutoConfig, LayoutLMv3ForTokenClassification, LayoutLMv3Processor

        self._torch = torch
        self.device = resolve_device(device)
        self.max_seq_length = max_seq_length
        self.stride = stride
        self.batch_size = batch_size

        self.processor = LayoutLMv3Processor.from_pretrained(processor_path or model_path, apply_ocr=False)

        config = AutoConfig.from_pretrained(model_path)
        overrides: dict[str, object] = {}
        if set(config.id2label.values()) != set(LABELS):
            overrides = {"num_labels": len(LABELS), "id2label": ID2LABEL, "label2id": LABEL2ID}
        model, info = LayoutLMv3ForTokenClassification.from_pretrained(
            model_path, output_loading_info=True, **overrides
        )
        if any(key.startswith("classifier") for key in info.get("missing_keys", [])):
            logger.warning(
                "Checkpoint %r has no trained classification head; predictions will be random. "
                "Fine-tune with `docai-train` and point DOCAI_MODEL_PATH at the result.",
                model_path,
            )
        self.model = model.to(self.device).eval()
        self.id2label: dict[int, str] = {int(k): v for k, v in self.model.config.id2label.items()}

    @classmethod
    def from_settings(cls, settings: Settings) -> "LayoutLMv3TokenClassifier":
        return cls(
            settings.model_path,
            settings.processor_path,
            device=settings.device,
            max_seq_length=settings.max_seq_length,
            stride=settings.window_stride,
        )

    def predict(self, image: Image.Image, words: Sequence[OCRWord]) -> list[WordPrediction]:
        """Label every OCR word on ``image``.

        Args:
            image: The preprocessed page the words were read from.
            words: OCR words carrying ``norm_box`` in [0, 1000].

        Returns:
            One :class:`WordPrediction` per input word, in input order.
        """
        if not words:
            return []
        torch = self._torch

        encoding = self.processor(
            image.convert("RGB"),
            [w.text for w in words],
            boxes=[list(w.norm_box) for w in words],
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        num_windows = len(encoding["input_ids"])
        window_word_ids = [encoding.word_ids(i) for i in range(num_windows)]

        pixel_values = encoding["pixel_values"]
        if isinstance(pixel_values, list):  # overflow windows come back as a list of per-window tensors
            pixel_values = torch.stack([torch.as_tensor(p) for p in pixel_values])

        window_probs: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, num_windows, self.batch_size):
                batch = slice(start, start + self.batch_size)
                logits = self.model(
                    input_ids=encoding["input_ids"][batch].to(self.device),
                    attention_mask=encoding["attention_mask"][batch].to(self.device),
                    bbox=encoding["bbox"][batch].to(self.device),
                    pixel_values=pixel_values[batch].to(self.device),
                ).logits
                window_probs.extend(torch.softmax(logits.float(), dim=-1).cpu().numpy())

        word_probs = aggregate_word_probabilities(window_probs, window_word_ids, len(words))
        best = word_probs.argmax(axis=-1)
        return [
            WordPrediction(word=word, label=self.id2label[int(idx)], confidence=float(word_probs[i, idx]))
            for i, (word, idx) in enumerate(zip(words, best, strict=True))
        ]
