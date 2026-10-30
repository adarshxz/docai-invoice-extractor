"""``docai-train``: fine-tune ``microsoft/layoutlmv3-base`` on annotated invoices.

Example::

    docai-train --train data/train.jsonl --eval data/val.jsonl --output models/invoice-layoutlmv3

Then serve it with ``DOCAI_MODEL_PATH=models/invoice-layoutlmv3``.
"""

from __future__ import annotations

import argparse
import functools
import inspect
import logging
from typing import Any

import numpy as np

from docai.labels import ID2LABEL, IGNORE_INDEX, LABEL2ID, LABELS
from docai.training.dataset import InvoiceTokenDataset, collate_batch, read_jsonl

logger = logging.getLogger(__name__)


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    """Entity-level precision/recall/F1 via ``seqeval`` (falls back to token accuracy)."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    true_seqs: list[list[str]] = []
    pred_seqs: list[list[str]] = []
    for p_row, l_row in zip(preds, labels, strict=True):
        keep = l_row != IGNORE_INDEX
        true_seqs.append([ID2LABEL[int(i)] for i in l_row[keep]])
        pred_seqs.append([ID2LABEL[int(i)] for i in p_row[keep]])

    try:
        from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
    except ImportError:
        flat_t = [t for s in true_seqs for t in s]
        flat_p = [p for s in pred_seqs for p in s]
        return {"token_accuracy": float(np.mean([t == p for t, p in zip(flat_t, flat_p, strict=True)]))}
    metrics = {
        "precision": precision_score(true_seqs, pred_seqs),
        "recall": recall_score(true_seqs, pred_seqs),
        "f1": f1_score(true_seqs, pred_seqs),
    }
    report = classification_report(true_seqs, pred_seqs, output_dict=True, zero_division=0)
    metrics.update({f"f1_{name}": float(v["f1-score"]) for name, v in report.items() if name.isupper()})
    return metrics


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run the Hugging Face ``Trainer``."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", required=True, help="Training JSONL")
    parser.add_argument("--eval", help="Validation JSONL")
    parser.add_argument("--output", required=True, help="Directory for the fine-tuned checkpoint")
    parser.add_argument("--base-model", default="microsoft/layoutlmv3-base")
    parser.add_argument("--epochs", type=float, default=20)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=-1, help="stop after N optimiser steps (smoke tests)")
    parser.add_argument("--limit", type=int, help="use only the first N train/eval pages (smoke tests)")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader worker processes")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    from transformers import (
        LayoutLMv3ForTokenClassification,
        LayoutLMv3Processor,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    processor = LayoutLMv3Processor.from_pretrained(args.base_model, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        args.base_model, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )

    train_ds = InvoiceTokenDataset(read_jsonl(args.train)[: args.limit], processor, args.max_length)
    eval_ds = InvoiceTokenDataset(read_jsonl(args.eval)[: args.limit], processor, args.max_length) if args.eval else None

    # transformers 5 folded `warmup_ratio` into `warmup_steps` (a float below 1 is a ratio).
    warmup = (
        {"warmup_ratio": 0.1}
        if "warmup_ratio" in inspect.signature(TrainingArguments.__init__).parameters
        else {"warmup_steps": 0.1}
    )
    training_args = TrainingArguments(
        **warmup,
        output_dir=args.output,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        dataloader_num_workers=args.workers,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,

        weight_decay=0.01,
        eval_strategy="epoch" if eval_ds else "no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="f1" if eval_ds else None,
        logging_steps=10,
        remove_unused_columns=False,
        seed=args.seed,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics if eval_ds else None,
        data_collator=functools.partial(collate_batch, pad_token_id=processor.tokenizer.pad_token_id),
    )
    trainer.train()
    trainer.save_model(args.output)
    processor.save_pretrained(args.output)
    logger.info("Saved fine-tuned model to %s", args.output)


if __name__ == "__main__":
    main()
