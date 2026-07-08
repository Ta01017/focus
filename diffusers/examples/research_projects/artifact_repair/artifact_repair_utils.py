import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image


DEFAULT_REPAIR_PROMPT = (
    "Repair the artifact or missing bottom surface in image 1. Preserve the geometry, shape, "
    "viewpoint, and spatial structure of image 1. Use image 2 only as an appearance and color "
    "reference for the repaired bottom surface."
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


def resolve_path(base, path):
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(base) / path


def load_rgb(path):
    return Image.open(path).convert("RGB")


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def tensor_stats(x):
    x = x.detach().float().cpu()
    return {
        "shape": list(x.shape),
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "dtype": str(x.dtype),
    }


def add_pretrained_args(parser):
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--revision", default=None)
    return parser


def pretrained_kwargs(args):
    return {
        "local_files_only": args.local_files_only,
        "cache_dir": args.cache_dir,
        "revision": args.revision,
    }


def select_records(records, start_index=0, max_samples=None):
    if start_index < 0:
        raise ValueError("--start_index must be non-negative.")
    if max_samples is not None and max_samples < 1:
        raise ValueError("--max_samples must be positive when provided.")
    end = None if max_samples is None else start_index + max_samples
    return list(enumerate(records[start_index:end], start=start_index))


def dynamic_image_size(reference_size, max_pixels=None, size_divisor=32, downscale_if_exceeds_max_pixels=False):
    width, height = reference_size
    if width < 1 or height < 1:
        raise ValueError(f"Invalid image size: {reference_size}")
    if size_divisor < 1:
        raise ValueError("--size_divisor must be positive.")
    content_w, content_h = width, height
    if max_pixels is not None and width * height > max_pixels:
        if not downscale_if_exceeds_max_pixels:
            raise ValueError(
                f"Input {width}x{height} has {width * height} pixels, exceeding --max_pixels={max_pixels}. "
                "Pass --downscale_if_exceeds_max_pixels to enable aspect-preserving downscale."
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


def _resize_and_pad(image, content_size, canvas_size, label):
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


def preprocess_pair_or_triplet(
    gt,
    src,
    ref,
    max_pixels=None,
    size_divisor=32,
    downscale_if_exceeds_max_pixels=False,
):
    """Prepare GT/src/ref without cropping.

    The src image defines the geometry. If gt/ref have the same aspect ratio but a different
    size, they are resized to src's content size. If src exceeds max_pixels and downscale is
    enabled, all three images are downscaled by the same geometry target, then right/bottom
    padded to the requested divisor.
    """
    original_size = src.size
    content_size, canvas_size = dynamic_image_size(
        original_size,
        max_pixels=max_pixels,
        size_divisor=size_divisor,
        downscale_if_exceeds_max_pixels=downscale_if_exceeds_max_pixels,
    )
    prepared = {
        "gt": _resize_and_pad(gt, content_size, canvas_size, "gt"),
        "src": _resize_and_pad(src, content_size, canvas_size, "src"),
        "ref": _resize_and_pad(ref, content_size, canvas_size, "ref"),
    }
    size_info = {
        "original_size": original_size,
        "content_size": content_size,
        "canvas_size": canvas_size,
    }
    return prepared, size_info


def preprocess_triplet(gt, src, ref, max_pixels=None, size_divisor=32, downscale_if_exceeds_max_pixels=False):
    return preprocess_pair_or_triplet(
        gt,
        src,
        ref,
        max_pixels=max_pixels,
        size_divisor=size_divisor,
        downscale_if_exceeds_max_pixels=downscale_if_exceeds_max_pixels,
    )


def preprocess_pair(src, ref, max_pixels=None, size_divisor=32, downscale_if_exceeds_max_pixels=False):
    prepared, size_info = preprocess_pair_or_triplet(
        src,
        src,
        ref,
        max_pixels=max_pixels,
        size_divisor=size_divisor,
        downscale_if_exceeds_max_pixels=downscale_if_exceeds_max_pixels,
    )
    return {"src": prepared["src"], "ref": prepared["ref"]}, size_info


def build_control_condition(src_tensor, ref_tensor):
    """Build [src_rgb, ref_rgb] condition. Inputs are SANA image tensors in [-1, 1]."""
    if src_tensor.ndim == 3:
        src_tensor = src_tensor.unsqueeze(0)
    if ref_tensor.ndim == 3:
        ref_tensor = ref_tensor.unsqueeze(0)
    return torch.cat([src_tensor, ref_tensor], dim=1)


def restore_output_size(image, size_info, restore_to_original_size=True):
    content_w, content_h = size_info["content_size"]
    image = image.crop((0, 0, content_w, content_h))
    if restore_to_original_size and image.size != size_info["original_size"]:
        image = image.resize(size_info["original_size"], Image.Resampling.LANCZOS)
    return image


def image_tensor_to_pil(image_processor, tensor):
    return image_processor.postprocess(tensor, output_type="pil")[0]


def save_debug_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def artifact_paths(sample, base_path, target_key="image", prompt_key="prompt"):
    if target_key not in sample:
        raise ValueError(f"Sample missing target field {target_key!r}.")
    edits = sample.get("edit_image")
    src = sample.get("src")
    ref = sample.get("ref")
    if src is None:
        if not isinstance(edits, list) or len(edits) < 1:
            raise ValueError("Sample must provide src or edit_image[0].")
        src = edits[0]
    if ref is None:
        if not isinstance(edits, list) or len(edits) < 2:
            raise ValueError("Sample must provide ref or edit_image[1].")
        ref = edits[1]
    return {
        "gt": resolve_path(base_path, sample[target_key]),
        "src": resolve_path(base_path, src),
        "ref": resolve_path(base_path, ref),
        "prompt": sample.get(prompt_key) or DEFAULT_REPAIR_PROMPT,
    }


def get_artifact_repair_paths(item, dataset_base_path, index=None, target_key="image", prompt_key="prompt"):
    paths = artifact_paths(item, dataset_base_path, target_key=target_key, prompt_key=prompt_key)
    paths["index"] = index
    paths["obj_name"] = item.get("obj_name")
    paths["test_uid"] = item.get("test_uid")
    paths["output_name"] = artifact_output_name(item, index if index is not None else 0)
    return paths


def artifact_output_name(sample, index):
    obj_name = sample.get("obj_name")
    test_uid = sample.get("test_uid")
    if obj_name is not None and test_uid is not None:
        stem = f"{obj_name}_{test_uid}"
    else:
        stem = str(sample.get("id", f"{index:06d}"))
    stem = stem.replace("/", "_").replace("\\", "_")
    return f"{stem}.png"
