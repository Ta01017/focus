import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_REPAIR_PROMPT = (
    "Repair the artifact or missing bottom surface in image 1. Preserve geometry, shape, viewpoint, "
    "and spatial structure."
)


def load_metadata(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Metadata file is empty: {path}")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "samples", "items"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError("Metadata must be a JSON list, JSONL, or a dict containing data/samples/items.")


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_path(base, path):
    return resolve_dataset_path(path, base, must_exist=False)




def resolve_dataset_path(
    value,
    dataset_base_path=None,
    *,
    record_index=None,
    field_name=None,
    must_exist=True,
):
    if value is None or str(value) == "":
        raise ValueError(
            f"Empty dataset path: record index={record_index}, field={field_name}, "
            f"dataset_base_path={dataset_base_path!r}."
        )
    raw = Path(value)
    base = Path(dataset_base_path) if dataset_base_path not in (None, "") else Path(".")
    attempted = []

    def remember(candidate):
        candidate = Path(candidate)
        if candidate not in attempted:
            attempted.append(candidate)
        return candidate

    if raw.is_absolute():
        final = remember(raw)
    else:
        remember(raw)
        direct = remember(base / raw)
        final = direct
        base_parts = base.parts
        raw_parts = raw.parts
        for overlap in range(min(len(base_parts), len(raw_parts)), 0, -1):
            if base_parts[-overlap:] == raw_parts[:overlap]:
                final = remember(base.joinpath(*raw_parts[overlap:]))
                break
        for candidate in attempted:
            if candidate.exists():
                final = candidate
                break
    try:
        resolved = final.resolve(strict=False)
    except OSError:
        resolved = final.absolute()
    if must_exist and not resolved.exists():
        attempted_text = "\n".join(f"  - {path}" for path in attempted)
        raise FileNotFoundError(
            "Dataset path does not exist.\n"
            f"record index: {record_index}\n"
            f"field name: {field_name}\n"
            f"raw metadata value: {value}\n"
            f"dataset_base_path: {dataset_base_path}\n"
            f"attempted paths:\n{attempted_text}\n"
            f"final resolved path: {resolved}"
        )
    return resolved


def load_rgb(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image path does not exist: {path}")
    return Image.open(path).convert("RGB")


def add_pretrained_args(parser):
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--revision", default=None)
    return parser


def pretrained_kwargs(args):
    return {"local_files_only": args.local_files_only, "cache_dir": args.cache_dir, "revision": args.revision}


def select_records(records, start_index=0, max_samples=None):
    if start_index < 0:
        raise ValueError("--start_index must be non-negative.")
    if max_samples is not None and max_samples < 1:
        raise ValueError("--max_samples must be positive when provided.")
    end = None if max_samples is None else start_index + max_samples
    return list(enumerate(records[start_index:end], start=start_index))


def artifact_output_name(sample, index):
    stem = str(sample.get("id", sample.get("test_uid", f"{index:06d}")))
    if sample.get("obj_name") is not None and sample.get("test_uid") is not None:
        stem = f"{sample['obj_name']}_{sample['test_uid']}"
    return f"{stem.replace('/', '_').replace('\\\\', '_')}.png"


def get_artifact_repair_paths(sample, base_path, index=None, target_key="image", prompt_key="prompt"):
    if target_key not in sample:
        raise ValueError(f"Sample {index} missing target field {target_key!r}.")
    edits = sample.get("edit_image")
    if not isinstance(edits, list) or len(edits) < 1:
        raise ValueError(f"Sample {index} must provide edit_image[0] as src.")
    src = edits[0]
    ref = edits[1] if len(edits) > 1 else edits[0]
    return {
        "gt": resolve_path(base_path, sample[target_key]),
        "src": resolve_path(base_path, src),
        "ref": resolve_path(base_path, ref),
        "prompt": sample.get(prompt_key) or DEFAULT_REPAIR_PROMPT,
    }


def dynamic_image_size(reference_size, max_pixels=None, size_divisor=32, downscale_if_exceeds_max_pixels=False):
    width, height = reference_size
    if width < 1 or height < 1:
        raise ValueError(f"Invalid image size: {reference_size}")
    content_w, content_h = width, height
    if max_pixels is not None and width * height > max_pixels:
        if not downscale_if_exceeds_max_pixels:
            raise ValueError(
                f"Input {width}x{height} exceeds --max_pixels={max_pixels}; pass --downscale_if_exceeds_max_pixels."
            )
        scale = math.sqrt(max_pixels / float(width * height))
        content_w = max(1, int(math.floor(width * scale)))
        content_h = max(1, int(math.floor(height * scale)))
        while content_w * content_h > max_pixels:
            if content_w >= content_h:
                content_w -= 1
            else:
                content_h -= 1
    canvas_w = ((content_w + size_divisor - 1) // size_divisor) * size_divisor
    canvas_h = ((content_h + size_divisor - 1) // size_divisor) * size_divisor
    return (content_w, content_h), (canvas_w, canvas_h)


def _resize_pad(image, content_size, canvas_size):
    image = image.convert("RGB")
    if image.size != content_size:
        image = image.resize(content_size, Image.Resampling.BICUBIC)
    pad_w = canvas_size[0] - content_size[0]
    pad_h = canvas_size[1] - content_size[1]
    if pad_w or pad_h:
        arr = np.asarray(image)
        arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
        image = Image.fromarray(arr, mode="RGB")
    return image


def preprocess_triplet(gt, src, ref, max_pixels=None, size_divisor=32, downscale_if_exceeds_max_pixels=False):
    original_size = src.size
    content_size, canvas_size = dynamic_image_size(
        original_size, max_pixels=max_pixels, size_divisor=size_divisor, downscale_if_exceeds_max_pixels=downscale_if_exceeds_max_pixels
    )
    return (
        {"gt": _resize_pad(gt, content_size, canvas_size), "src": _resize_pad(src, content_size, canvas_size), "ref": _resize_pad(ref, content_size, canvas_size)},
        {"original_size": original_size, "content_size": content_size, "canvas_size": canvas_size},
    )


def preprocess_pair(src, ref, max_pixels=None, size_divisor=32, downscale_if_exceeds_max_pixels=False):
    prepared, info = preprocess_triplet(src, src, ref, max_pixels, size_divisor, downscale_if_exceeds_max_pixels)
    return {"src": prepared["src"], "ref": prepared["ref"]}, info


def restore_output_size(image, size_info, restore_to_original_size=True):
    content_w, content_h = size_info["content_size"]
    image = image.crop((0, 0, content_w, content_h))
    if restore_to_original_size and image.size != size_info["original_size"]:
        image = image.resize(size_info["original_size"], Image.Resampling.LANCZOS)
    return image
