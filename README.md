# docai — Multi-Modal Document & Invoice AI

Pulls structured invoice fields out of scanned PDFs, PNGs and JPEGs with `microsoft/layoutlmv3-base`.

| Stage | Module | What it does |
|---|---|---|
| 1. Ingestion & preprocessing | `ingestion.py`, `preprocessing.py` | `pdf2image` rasterisation, deskew via `cv2.minAreaRect` on text-line blobs, CLAHE contrast |
| 2. OCR & spatial normalisation | `ocr.py` | `pytesseract.image_to_data()`, boxes → `min(1000, max(0, ⌊1000·x/W⌋))` |
| 3. Token classification | `model.py` | LayoutLMv3 over image patches + BPE tokens + boxes; each sub-token copies its word's box; overlapping 512-token windows for long pages |
| 4. Post-processing | `postprocessing.py` | Lenient BIO decoding, same-line merging, confidence `> 0.70`, regex validation of amounts / dates / invoice numbers, line-item ↔ amount pairing by row |
| 5. Serving | `api/`, `Dockerfile`, `docker-compose.yml` | FastAPI + Celery + Redis, multi-stage image |

## The model must be fine-tuned first

`microsoft/layoutlmv3-base` is a pre-trained backbone with **no** invoice classification head. If you load it directly, the head is initialised at random and the predictions mean nothing (a warning is logged). Fine-tune it on labelled pages first:

```bash
pip install -e ".[train]"
docai-train --train data/train.jsonl --eval data/val.jsonl --output models/invoice-layoutlmv3
```

Each JSONL line is one page: `{"image": "...", "words": [...], "boxes": [[x0,y0,x1,y1], ...], "labels": ["B-VENDOR_NAME", ...]}`. Boxes are in pixels by default; add `"box_format": "normalized"` if they are already on the 0–1000 scale. See `training/dataset.py`.

### Building the training data

`docai-data` produces that JSONL from three kinds of source.

**Synthetic invoices.** These are rendered locally with exact labels for all six entity types, including `INVOICE_NUMBER`:

```bash
docai-data synth data/synth --n 1500
```

Every page varies the layout, fonts, currency and number format (`$1,234.56`, `1.234,56 €`, `CHF 1'234.50`), date format, invoice-number style, 1–12 line items (descriptions wrap), and the tax, discount and shipping rows. Each page also carries decoys the model must learn to ignore: the "Bill to" company, due dates, PO numbers, subtotals, tax IDs and bank details. 80% get scan effects (rotation, blur, noise, JPEG artefacts, fax resampling, uneven lighting). All vendor and customer names are invented.

**Public receipts.** These download from the Hugging Face hub and convert in one command each:

```bash
docai-data sroie data/sroie
```

```bash
docai-data cord data/cord
```

| Dataset | Pages | Labels it provides | Missing |
|---|---|---|---|
| SROIE 2019 | ~970 | `VENDOR_NAME`, `DATE`, `TOTAL_AMOUNT` | invoice no., tax, items (present on the page but tagged `O`) |
| CORD v2 | 1,000 | `ITEM_DESCRIPTION`, `TOTAL_AMOUNT`, `TAX_AMOUNT` | invoice no.; vendor and date are blurred out |

SROIE only provides document-level strings, so the converter has to find them on the page. Values that sit next to other text ("RM9.00") or that have typos in SROIE's own labels are handled. The total is taken from the row reading "TOTAL", not the subtotal row. Pages where a value can't be found are skipped (about 1.5%) so they don't teach the model wrong labels. Images are converted to grayscale with boosted contrast, as at inference.

**A first model from synthetic + public data only.** This needs no labelling:

```bash
docai-data split data/synth/synth.jsonl data/sroie/train.jsonl data/cord/train.jsonl --out data/final
```

```bash
docai-train --train data/final/train.jsonl --eval data/final/val.jsonl --output models/invoice-layoutlmv3
```

**Evaluate** on test sets the model never trained on:

```bash
docai-eval --model models/invoice-layoutlmv3 --data data/sroie/test.jsonl data/cord/test.jsonl --day-first
```

The report has two kinds of score:
- **Entity F1:** how well the model tags each word.
- **Field accuracy:** whether the final JSON value is exactly right after post-processing (e.g. `total_amount == 193.00`). This is the number that matters.

`--oracle` feeds the correct labels instead of model predictions. It measures the ceiling of post-processing alone, which is currently 100% on both test sets. So any remaining errors belong to the model, not the parsing.

**Results of the first model** (`models/invoice-layoutlmv3`): 5 epochs on 2,619 synthetic + SROIE + CORD pages, 1 h 55 min on an M2 MacBook. Field accuracy on test pages it never trained on:

| Field | SROIE test (356 receipts) | CORD test (100 receipts) |
|---|---|---|
| Vendor name | 95.8% | – |
| Date | 99.2% | – |
| Total amount | 92.1% (94.1% at threshold 0.5) | 98.9% (100% at threshold 0.5) |
| Tax amount | – | 100% |

Validation F1 for the invoice number is 1.00, but that is on synthetic invoices only; real invoices will score lower. Setting `DOCAI_CONFIDENCE_THRESHOLD=0.5` improved totals with no losses on these test sets.

**Your own invoices.** Label these in Label Studio and convert the export with `ls-prepare` / `ls-convert`. See **[docs/labelling-guide.md](docs/labelling-guide.md)** for setup, labelling rules and the merge/train commands. These pages matter most: they are the only source of `INVOICE_NUMBER`, and the only data that looks like your production documents.

Caveat: the public sets label different subsets of entities. For example, SROIE's line items are tagged `O` while CORD's are tagged `ITEM_DESCRIPTION`, which sends the model mixed signals. If item extraction stays weak once your own invoices are in, retrain without SROIE, or train on public data first and then continue on your invoices alone (`--base-model models/stage1`).

## Run locally

Needs `tesseract` and `poppler`.

**macOS on Apple Silicon (M1–M4):** you need the Apple Silicon Homebrew in `/opt/homebrew`. An Intel Homebrew in `/usr/local` fails with "Bad CPU type" on Macs without Rosetta. Install it (this asks for your password), then the tools:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

```bash
eval "$(/opt/homebrew/bin/brew shellenv)" && brew install tesseract poppler
```

**No admin rights?** Use micromamba, which installs into your home folder:

```bash
curl -Ls https://micro.mamba.pm/api/micromamba/osx-arm64/latest | tar -xvj -C ~ bin/micromamba
```

```bash
~/bin/micromamba create -y -p ~/docai-tools -c conda-forge tesseract poppler
```

Then tell docai where they are (in a `.env` file in the project root, or as environment variables):

```bash
printf 'DOCAI_TESSERACT_CMD=%s\nDOCAI_POPPLER_PATH=%s\n' ~/docai-tools/bin/tesseract ~/docai-tools/bin >> .env
```

Then run the pipeline:

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
DOCAI_MODEL_PATH=models/invoice-layoutlmv3 .venv/bin/docai-extract invoice.pdf
```

## Run the service

```bash
docker compose up --build
```

```bash
curl -F file=@invoice.pdf http://localhost:8000/v1/documents
```

```bash
curl http://localhost:8000/v1/documents/<task_id>
```

| Endpoint | Purpose |
|---|---|
| `POST /v1/documents` | Upload a document. Returns `202 {task_id, status_url}` |
| `GET /v1/documents/{task_id}` | `PENDING` / `STARTED` / `SUCCESS` (with `result`) / `FAILURE` |
| `POST /v1/extract` | Synchronous extraction in the API process (dev / low volume) |
| `GET /health` | Liveness and Redis reachability |

Mount the fine-tuned checkpoint at `./models/invoice-layoutlmv3`, or set `DOCAI_MODEL_PATH`. All settings are in `config.py` and can be overridden with `DOCAI_*` environment variables.

## Output decisions worth knowing

- **Status**: bad input or a processing error gives `status: "FAILED"` with null fields. It does not raise an HTTP error.
- **Single-value fields**: each one takes the highest-confidence candidate that passes validation, across all pages. A missing field is `{"value": null, "confidence": 0.0}`.
- **Currency** comes from the amount's own text first, then from anywhere on the document, then `DOCAI_DEFAULT_CURRENCY` (USD).
- **Ambiguous dates** like `03/04/2024` are read month-first. Set `DOCAI_DAY_FIRST_DATES=true` to read them day-first.
- **Line items**: each one's amount is the right-most money-like value (two decimals) on the description's row. Items with no amount on their row are dropped, so `amount` stays a float.

## Tests

```bash
.venv/bin/pytest
```

```bash
.venv/bin/pytest -m slow
```

The first command runs the fast unit tests. The second runs against the real LayoutLMv3 processor and weights, and downloads them on the first run.
