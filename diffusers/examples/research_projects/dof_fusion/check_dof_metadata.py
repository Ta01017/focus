#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from dof_utils import add_metadata_args, select_records
from metadata import load_metadata, resolve_data_path


def parse_args():
    parser = argparse.ArgumentParser(description="检查 DiffSynth 景深融合 metadata 并生成预览图。")
    add_metadata_args(parser, metadata_required=True)
    parser.add_argument("--min_edit_images", type=int, default=2)
    parser.add_argument("--require_focus_maps", action="store_true")
    parser.add_argument("--preview_output", default="dof_metadata_preview.jpg")
    parser.add_argument("--report_output", default="dof_metadata_report.json")
    parser.add_argument("--preview_count", type=int, default=16)
    parser.add_argument("--preview_size", type=int, default=256)
    return parser.parse_args()


def focus_range(image):
    array = np.asarray(image)
    if not np.isfinite(array).all():
        raise ValueError("focus map contains NaN or Inf.")
    if np.issubdtype(array.dtype, np.floating):
        normalized = array.astype(np.float32)
    elif np.issubdtype(array.dtype, np.integer):
        max_value = np.iinfo(array.dtype).max
        normalized = array.astype(np.float32) / max_value
    else:
        raise ValueError(f"unsupported focus-map dtype {array.dtype}.")
    minimum = float(normalized.min())
    maximum = float(normalized.max())
    if minimum < 0 or maximum > 1:
        raise ValueError(f"normalized focus-map range is [{minimum}, {maximum}], expected [0, 1].")
    return minimum, maximum


def preview_tile(image, title, size):
    image = ImageOps.contain(image.convert("RGB"), (size, size), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (size, size + 24), "white")
    tile.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    ImageDraw.Draw(tile).text((4, size + 4), title, fill="black")
    return tile


def main():
    args = parse_args()
    records = load_metadata(args.dataset_metadata_path)
    selected = select_records(records, args.start_index, args.max_samples)
    errors = []
    warnings = []
    summaries = []
    previews = []
    required_edits = max(args.min_edit_images, 4 if args.require_focus_maps else 2)

    for index, sample in selected:
        try:
            if args.target_key not in sample:
                raise ValueError(f"missing target field {args.target_key!r}.")
            if args.edit_key not in sample:
                raise ValueError(f"missing edit field {args.edit_key!r}.")
            edits = sample[args.edit_key]
            if not isinstance(edits, list):
                raise ValueError(f"field {args.edit_key!r} must be a list.")
            if len(edits) < required_edits:
                raise ValueError(f"edit_image length {len(edits)} is smaller than required {required_edits}.")
            target_path = resolve_data_path(sample[args.target_key], args.dataset_base_path)
            edit_paths = [resolve_data_path(path, args.dataset_base_path) for path in edits[:4]]
            for path in [target_path, *edit_paths]:
                if not Path(path).is_file():
                    raise FileNotFoundError(path)
            target = Image.open(target_path).convert("RGB")
            conditions = [Image.open(path) for path in edit_paths]
            sizes = [target.size, *(image.size for image in conditions)]
            if len(set(sizes)) > 1:
                warnings.append({"index": index, "message": f"resize 前尺寸不同: {sizes}"})
            ranges = []
            for condition_index, image in enumerate(conditions[2:4], start=2):
                minimum, maximum = focus_range(image)
                ranges.append({"index": condition_index, "min": minimum, "max": maximum, "mode": image.mode})
            prompt = sample.get(args.prompt_key)
            if prompt is not None and not isinstance(prompt, str):
                raise ValueError(f"field {args.prompt_key!r} must be a string.")
            summaries.append(
                {"index": index, "target": str(target_path), "edit_count": len(edits), "sizes": sizes, "focus": ranges}
            )
            if len(previews) < args.preview_count:
                row = [preview_tile(target, f"{index}: GT", args.preview_size)]
                labels = ("A", "B", "focus_a", "focus_b")
                row.extend(
                    preview_tile(image, f"{index}: {labels[position]}", args.preview_size)
                    for position, image in enumerate(conditions)
                )
                previews.append(row)
        except Exception as error:
            errors.append({"index": index, "error": str(error)})
            if not args.continue_on_error:
                break

    if previews:
        columns = max(len(row) for row in previews)
        width = columns * args.preview_size
        height = len(previews) * (args.preview_size + 24)
        grid = Image.new("RGB", (width, height), "#dddddd")
        for row_index, row in enumerate(previews):
            for column_index, tile in enumerate(row):
                grid.paste(tile, (column_index * args.preview_size, row_index * (args.preview_size + 24)))
        preview_path = Path(args.preview_output)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        grid.save(preview_path)

    report = {
        "metadata": args.dataset_metadata_path,
        "checked": len(summaries),
        "errors": errors,
        "warnings": warnings,
        "samples": summaries,
    }
    report_path = Path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"checked={len(summaries)} warnings={len(warnings)} errors={len(errors)}")
    print(f"report={report_path} preview={args.preview_output}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
