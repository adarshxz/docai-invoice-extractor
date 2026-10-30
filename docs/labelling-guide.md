# Labelling your invoices

The model only learns what the labels teach it, and inconsistent labels hurt more than having fewer of them. Every annotator should follow these rules.

## 1. Prepare the pages

Put your invoices (PDF, PNG, JPEG) in a folder, e.g. `raw_invoices/`. Subfolders are fine. Then run:

```bash
docai-data ls-prepare raw_invoices/ data/invoices
```

For each page, this saves the same straightened, contrast-enhanced image the production pipeline will see (`images/`) and Tesseract's words for it (`ocr/`). It also writes `tasks.json` and `label_config.xml`. Needs `tesseract` and `poppler` installed.

## 2. Set up Label Studio (once)

```bash
pip install label-studio
```

```bash
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$PWD/data/invoices" label-studio start
```

1. Create a project. Under **Labeling Setup → Code**, paste the contents of `data/invoices/label_config.xml`.
2. Go to **Settings → Cloud Storage → Add Source Storage → Local files** and set the absolute path to `…/data/invoices/images`. This lets Label Studio serve the images. **Don't sync it**, or you'll get duplicate tasks.
3. Use **Import** to load `data/invoices/tasks.json`.

## 3. Draw boxes

Draw **one rectangle per entity instance** and pick its label. You never type text: a word belongs to a box when the word's centre is inside it. Loose boxes are fine, as long as they don't take in the centre of a neighbouring word.

| Label | Box this | Do not box |
|---|---|---|
| `VENDOR_NAME` | The seller's name as printed, usually in the header or logo text | The buyer / "Bill to", addresses, phone numbers, the legal suffix if it's on a separate line |
| `INVOICE_NUMBER` | The number itself: `INV-2024-0042` | The caption "Invoice #", order or PO numbers, customer IDs |
| `DATE` | The invoice / issue date | Due date, delivery date, period ranges, timestamps |
| `TOTAL_AMOUNT` | The final amount payable, including tax (the grand total) | Subtotal, amount paid, balance brought forward, the "Total" caption |
| `TAX_AMOUNT` | The total tax amount (VAT / GST / sales tax) | The rate ("20%"), per-line tax, tax registration numbers |
| `ITEM_DESCRIPTION` | Each line item's description, **one box per item**, including lines that wrap | Quantity, unit price, line amount, SKU columns, headers |

Rules of thumb:

- **When one value appears twice** (e.g. the total in both a summary box and a footer), label the most prominent one only.
- **When a field isn't on the page**, don't label anything for it. A page with no entities at all should still be **submitted**, because it teaches the model what "nothing" looks like.
- **When the OCR misread the text**, still box it. The model learns from position and appearance as well as the text.
- **When you're unsure**, leave a comment on the task and **skip** it rather than guessing. Skipped tasks are not exported as training data.

## 4. How many pages, and which

- **150–200 pages** is a reasonable first target, and more always helps. A good rule is at least 50 examples of each label. `docai-data stats` counts them.
- **Spread pages across vendors and layouts.** 200 invoices from 60 vendors beat 200 from 5. Cap any one vendor at roughly 10 pages.
- **Include the hard cases you see in production:** phone photos, faxes, rotated scans, multi-page invoices, non-English text.

## 5. Export and convert

In Label Studio, choose **Export → JSON** (not "JSON-MIN"), then:

```bash
docai-data ls-convert ~/Downloads/project-1-at-….json data/invoices
```

This writes `data/invoices/labelled.jsonl`. If it warns that some boxes contained no OCR words, Tesseract missed that text. Check those pages: the scan may be too poor to use.

## 6. Merge with public data and train

```bash
docai-data split data/synth/synth.jsonl data/sroie/train.jsonl data/cord/train.jsonl data/invoices/labelled.jsonl --out data/final --repeat data/invoices/labelled.jsonl=3
```

```bash
docai-data stats data/final/train.jsonl data/final/val.jsonl
```

```bash
docai-train --train data/final/train.jsonl --eval data/final/val.jsonl --output models/invoice-layoutlmv3
```

`--repeat …=3` makes your own invoices count three times as much as the synthetic and receipt pages, since they match production best. The split keeps all pages of one document on the same side, so the validation score is honest.
