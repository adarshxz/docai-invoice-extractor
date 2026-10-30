"""``docai-extract FILE [FILE ...]``: run the pipeline locally and print JSON."""

from __future__ import annotations

import argparse
import logging
import sys

from docai.pipeline import get_pipeline


def main(argv: list[str] | None = None) -> int:
    """Print one JSON result per input file; exit 1 if any extraction failed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="PDF, PNG or JPEG documents")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)

    pipeline = get_pipeline()
    ok = True
    for path in args.files:
        result = pipeline.process_path(path)
        ok &= result.document_metadata.status.value == "SUCCESS"
        print(result.model_dump_json(indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
