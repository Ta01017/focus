#!/usr/bin/env python

import argparse
import json
from pathlib import Path

from PIL import Image


def resolve(path, base):
    path = Path(path)
    if path.is_file() or path.is_absolute():
        return path
    return Path(base) / path


def main():
    parser = argparse.ArgumentParser(description="Verify DOF smoke-test output sizes and metadata fields.")
    parser.add_argument("--reference", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--metadata_results", default=None)
    parser.add_argument("--dataset_base_path", default=".")
    parser.add_argument("--edit_key", default="edit_image")
    parser.add_argument("--result_key", default="generated_image")
    args = parser.parse_args()

    if args.reference and args.output:
        reference_size = Image.open(args.reference).size
        output_size = Image.open(args.output).size
        if output_size != reference_size:
            raise ValueError(f"Output size {output_size} does not match A size {reference_size}: {args.output}")

    if args.metadata_results:
        records = json.loads(Path(args.metadata_results).read_text(encoding="utf-8"))
        if not records:
            raise ValueError("metadata_results.json is empty.")
        for index, record in enumerate(records):
            if args.result_key not in record:
                raise ValueError(f"Result {index} is missing {args.result_key!r}.")
            edits = record.get(args.edit_key)
            if not isinstance(edits, list) or not edits:
                raise ValueError(f"Result {index} has no A image in {args.edit_key!r}.")
            reference = resolve(edits[0], args.dataset_base_path)
            output = resolve(record[args.result_key], args.dataset_base_path)
            if not output.is_file():
                output = Path(record[args.result_key])
            if Image.open(output).size != Image.open(reference).size:
                raise ValueError(f"Result {index} output size does not match A: {output}")


if __name__ == "__main__":
    main()
