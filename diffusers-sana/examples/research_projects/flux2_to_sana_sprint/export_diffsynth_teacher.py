#!/usr/bin/env python3

import argparse
import json
import shlex
import subprocess
from pathlib import Path

from PIL import Image

from metadata_dataset import as_path_list, load_metadata, resolve_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export black-box DiffSynth Studio FLUX.2 responses for cross-architecture distillation."
    )
    parser.add_argument("--dataset_base_path", required=True)
    parser.add_argument("--dataset_metadata_path", required=True)
    parser.add_argument("--condition_key", default="edit_image")
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--seed_key", default="seed")
    parser.add_argument("--id_key", default="id")
    parser.add_argument("--default_prompt", default="")
    parser.add_argument("--teacher_command", required=True)
    parser.add_argument("--command_cwd")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_metadata_path", required=True)
    parser.add_argument("--teacher_target_key", default="teacher_image")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--skip_output_size_check", action="store_true")
    return parser.parse_args()


def command_for_sample(template, values):
    try:
        return [token.format(**values) for token in shlex.split(template)]
    except KeyError as error:
        available = ", ".join(sorted(values))
        raise ValueError(f"Unknown teacher-command placeholder {error}; available: {available}.") from error


def safe_sample_id(value):
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in str(value))


def main():
    args = parse_args()
    if (args.height is None) != (args.width is None):
        raise ValueError("--height and --width must be provided together.")
    records = load_metadata(args.dataset_metadata_path)
    end = None if args.max_samples is None else args.start_index + args.max_samples
    selected = list(enumerate(records[args.start_index:end], start=args.start_index))
    if not selected:
        raise ValueError("Selected metadata range is empty.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    condition_count = None

    for index, record in selected:
        result = dict(record)
        try:
            if args.condition_key not in record:
                raise ValueError(f"Sample {index} is missing {args.condition_key!r}.")
            condition_values = as_path_list(record[args.condition_key], args.condition_key, index)
            if condition_count is None:
                condition_count = len(condition_values)
            elif len(condition_values) != condition_count:
                raise ValueError(
                    f"All samples must have {condition_count} condition images; sample {index} has "
                    f"{len(condition_values)}."
                )
            condition_paths = [resolve_path(value, args.dataset_base_path).resolve() for value in condition_values]
            sizes = []
            for path in condition_paths:
                with Image.open(path) as image:
                    sizes.append(image.size)
            if len(set(sizes)) != 1:
                raise ValueError(f"Sample {index} condition images must have equal sizes, found {sizes}.")
            source_width, source_height = sizes[0]
            height = args.height or source_height
            width = args.width or source_width
            if height % 16 or width % 16:
                raise ValueError(
                    f"Teacher output size must be divisible by 16, got {width}x{height}. "
                    "Preprocess the dataset or pass --height/--width."
                )

            sample_id = safe_sample_id(record.get(args.id_key, index))
            destination = output_dir / f"{index:08d}_{sample_id}.png"
            result[args.teacher_target_key] = str(destination)
            if not (args.skip_existing and destination.is_file()):
                prompt = record.get(args.prompt_key) or args.default_prompt
                if not isinstance(prompt, str):
                    raise ValueError(f"Sample {index} prompt must be a string.")
                seed = int(record.get(args.seed_key, args.seed + index))
                values = {
                    "condition_images_json": json.dumps([str(path) for path in condition_paths]),
                    "condition_images_csv": ",".join(str(path) for path in condition_paths),
                    "output": str(destination),
                    "prompt": prompt,
                    "seed": seed,
                    "height": height,
                    "width": width,
                    "index": index,
                    "id": sample_id,
                }
                values.update({f"condition_{i}": str(path) for i, path in enumerate(condition_paths)})
                subprocess.run(
                    command_for_sample(args.teacher_command, values),
                    cwd=args.command_cwd,
                    check=True,
                )
            if not destination.is_file():
                raise FileNotFoundError(f"Teacher command did not create {destination}.")
            if not args.skip_output_size_check:
                with Image.open(destination) as image:
                    if image.size != (width, height):
                        raise ValueError(
                            f"Teacher output {destination} has size {image.size}, expected {(width, height)}."
                        )
            print(destination)
        except Exception as error:
            if not args.continue_on_error:
                raise
            result["teacher_error"] = str(error)
        results.append(result)

    metadata_path = Path(args.output_metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} records to {metadata_path}")


if __name__ == "__main__":
    main()
