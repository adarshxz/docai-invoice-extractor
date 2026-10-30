"""``docai-data``: build training data.

Commands::

    docai-data cord   data/cord                   # download + convert CORD v2
    docai-data sroie  data/sroie                  # download + convert SROIE 2019
    docai-data synth  data/synth --n 1500         # render labelled synthetic invoices
    docai-data ls-prepare raw_invoices/ data/invoices
    docai-data ls-convert export.json data/invoices
    docai-data split data/cord/train.jsonl data/sroie/train.jsonl data/invoices/labelled.jsonl \\
                     --out data/final --repeat data/invoices/labelled.jsonl=3
    docai-data stats data/final/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path


def _repeat_arg(value: str) -> tuple[Path, int]:
    path, _, n = value.rpartition("=")
    if not path or not n.isdigit():
        raise argparse.ArgumentTypeError("expected PATH=N, e.g. data/invoices/labelled.jsonl=3")
    return Path(path), int(n)


def main(argv: list[str] | None = None) -> int:
    """Dispatch a sub-command; returns a process exit code."""
    parser = argparse.ArgumentParser(prog="docai-data", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    for name, default_splits in (("cord", ["train", "validation", "test"]), ("sroie", ["train", "test"])):
        p = sub.add_parser(name, help=f"download and convert {name.upper()}")
        p.add_argument("out_dir", type=Path)
        p.add_argument("--splits", nargs="+", default=default_splits)
        p.add_argument("--limit", type=int, help="pages per split (for a quick look)")
        p.add_argument("--raw-images", action="store_true", help="keep colour images (skip grayscale + CLAHE)")

    p = sub.add_parser("synth", help="render synthetic invoices with exact labels")
    p.add_argument("out_dir", type=Path)
    p.add_argument("--n", type=int, default=1500, help="number of pages")
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("ls-prepare", help="preprocess + OCR your invoices for Label Studio")
    p.add_argument("input_dir", type=Path)
    p.add_argument("workdir", type=Path)

    p = sub.add_parser("ls-convert", help="Label Studio JSON export -> training JSONL")
    p.add_argument("export", type=Path)
    p.add_argument("workdir", type=Path)
    p.add_argument("--output", type=Path)

    p = sub.add_parser("split", help="merge JSONL files into train/val")
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repeat", type=_repeat_arg, action="append", default=[], metavar="PATH=N")

    p = sub.add_parser("stats", help="entity counts per JSONL file")
    p.add_argument("files", nargs="+", type=Path)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.command in ("cord", "sroie"):
        from docai.data.public import convert_cord, convert_sroie

        fn = convert_cord if args.command == "cord" else convert_sroie
        results = fn(args.out_dir, args.splits, limit=args.limit, normalize_images=not args.raw_images)
        for split, st in results.items():
            print(f"{split}: {st.written} pages written, {st.skipped} skipped {st.reasons or ''}")
    elif args.command == "synth":
        from docai.data.synthetic import generate

        print(f"{generate(args.out_dir, args.n, seed=args.seed)} synthetic pages -> {args.out_dir / 'synth.jsonl'}")
    elif args.command == "ls-prepare":
        from docai.data.labelstudio import prepare

        st = prepare(args.input_dir, args.workdir)
        print(f"{st.documents} documents, {st.pages} pages -> {args.workdir}")
        if st.failed:
            print(f"failed: {st.failed}")
            return 1
    elif args.command == "ls-convert":
        from docai.data.labelstudio import convert

        st = convert(args.export, args.workdir, args.output)
        print(f"{st.pages} labelled pages, {st.entities} entities; {st.unlabelled_tasks} tasks not yet labelled")
        if st.empty_boxes:
            print(f"warning: {st.empty_boxes} boxes contained no OCR word (text the OCR missed?)")
        if st.missing_ocr:
            print(f"error: no OCR sidecar for {st.missing_ocr}; were they prepared in this workdir?")
            return 1
    elif args.command == "split":
        from docai.data.split import merge_and_split

        counts = merge_and_split(args.inputs, args.out, val_fraction=args.val_fraction, seed=args.seed,
                                 repeat=dict(args.repeat))
        for split, c in counts.items():
            print(f"{split}: {sum(c.values())} pages {dict(c)}")
    elif args.command == "stats":
        from docai.data.split import label_stats

        for f in args.files:
            print(f, json.dumps(label_stats(f)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
