#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from metadata import load_metadata, require_keys, resolve_data_path


def parse_args():
    parser = argparse.ArgumentParser(description="独立的 DOF 融合结果色差后处理，不参与模型推理。")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--dataset_metadata_path", help="可传原始 metadata 或 metadata_results.json。")
    source_group.add_argument("--input_dir", help="无 JSON 时扫描包含 A/B/生成结果的样本目录。")
    parser.add_argument("--dataset_base_path", default=".")
    parser.add_argument("--generated_dir", default=None, help="原始 metadata 不含 result_key 时的生成结果目录。")
    parser.add_argument("--generated_pattern", default="{index:06d}_{id}.png")
    parser.add_argument("--a_filename", default="a.png")
    parser.add_argument("--b_filename", default="b.png")
    parser.add_argument("--focus_a_filename", default="focus_a.png")
    parser.add_argument("--generated_filename", default="generated.png")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_metadata_path", default=None)
    parser.add_argument("--target_key", default="image")
    parser.add_argument("--edit_key", default="edit_image")
    parser.add_argument("--id_key", default="id")
    parser.add_argument("--result_key", default="generated_image")
    parser.add_argument("--output_key", default="color_corrected_image")
    parser.add_argument(
        "--reference_mode",
        choices=("auto", "a", "b", "mean_ab", "focus_composite", "target"),
        default="auto",
        help="target 仅用于有 GT 的离线评测，部署时不要使用。",
    )
    parser.add_argument("--channels", choices=("chroma", "lab"), default="chroma")
    parser.add_argument(
        "--method",
        choices=("paired_offset", "reinhard", "local_chroma"),
        default="paired_offset",
        help="paired_offset 最适合几何对齐的 A/B 景深融合；reinhard 为传统统计匹配。",
    )
    parser.add_argument("--match_std", action="store_true", help="除均值外也匹配 Lab 标准差。")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--trim_percent", type=float, default=1.0)
    parser.add_argument("--max_chroma_shift", type=float, default=20.0)
    parser.add_argument("--local_radius", type=float, default=32.0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    return parser.parse_args()


def records_from_folder(args):
    root = Path(args.input_dir)
    candidates = [root, *(path for path in sorted(root.iterdir()) if path.is_dir())]
    records = []
    for directory in candidates:
        image_a = directory / args.a_filename
        image_b = directory / args.b_filename
        generated = directory / args.generated_filename
        if not image_a.is_file() or not image_b.is_file():
            continue
        if not generated.is_file() and args.generated_dir is None:
            raise FileNotFoundError(f"Sample directory {directory} is missing {args.generated_filename}.")
        edits = [str(image_a), str(image_b)]
        focus_a = directory / args.focus_a_filename
        if focus_a.is_file():
            edits.append(str(focus_a))
        record = {args.id_key: directory.name, args.edit_key: edits}
        if generated.is_file():
            record[args.result_key] = str(generated)
        records.append(record)
    if not records:
        raise ValueError(
            f"No samples found under {root}; expected {args.a_filename}, {args.b_filename}, and "
            f"{args.generated_filename}."
        )
    return records


def generated_path_for_sample(sample, index, args):
    if sample.get(args.result_key):
        value = sample[args.result_key]
        path = Path(resolve_data_path(value, args.dataset_base_path))
        return path if path.is_file() else Path(value)
    if args.generated_dir is None:
        raise ValueError(
            f"Sample {index} has no {args.result_key!r}; provide --generated_dir or use --input_dir."
        )
    edits = sample.get(args.edit_key, [])
    fallback_id = Path(edits[0]).stem if edits else index
    sample_id = str(sample.get(args.id_key, fallback_id)).replace("/", "_").replace("\\", "_")
    filename = args.generated_pattern.format(index=index, id=sample_id)
    return Path(args.generated_dir) / filename


def srgb_to_lab(rgb):
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    matrix = np.array(
        [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]],
        dtype=np.float32,
    )
    xyz = linear @ matrix.T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    delta = 6 / 29
    f = np.where(xyz > delta**3, np.cbrt(xyz), xyz / (3 * delta**2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def lab_to_srgb(lab):
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    delta = 6 / 29
    xyz = np.stack([fx, fy, fz], axis=-1)
    xyz = np.where(xyz > delta, xyz**3, 3 * delta**2 * (xyz - 4 / 29))
    xyz = xyz * np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    inverse = np.array(
        [[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560], [0.0556434, -0.2040259, 1.0572252]],
        dtype=np.float32,
    )
    linear = xyz @ inverse.T
    srgb = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * np.maximum(linear, 0) ** (1 / 2.4) - 0.055)
    return np.clip(srgb, 0, 1)


def load_rgb(path, base_path, size=None):
    image = Image.open(resolve_data_path(path, base_path)).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def load_reference(sample, args, size):
    if args.reference_mode == "target":
        require_keys(sample, (args.target_key,), 0)
        return load_rgb(sample[args.target_key], args.dataset_base_path, size)
    require_keys(sample, (args.edit_key,), 0)
    edits = sample[args.edit_key]
    if not isinstance(edits, list) or len(edits) < 2:
        raise ValueError(f"{args.edit_key!r} 必须至少包含 A 和 B。")
    image_a = load_rgb(edits[0], args.dataset_base_path, size)
    if args.reference_mode == "a":
        return image_a
    image_b = load_rgb(edits[1], args.dataset_base_path, size)
    if args.reference_mode == "b":
        return image_b
    if args.reference_mode == "mean_ab" or (args.reference_mode == "auto" and len(edits) < 3):
        return (image_a + image_b) * 0.5
    if len(edits) < 3:
        raise ValueError("focus_composite 需要 edit_image[2] = focus_a。")
    focus = Image.open(resolve_data_path(edits[2], args.dataset_base_path)).convert("L")
    focus = focus.resize(size, Image.Resampling.BILINEAR)
    mask = np.asarray(focus, dtype=np.float32)[..., None] / 255.0
    return image_a * mask + image_b * (1 - mask)


def robust_stats(values, trim_percent):
    flat = values.reshape(-1, values.shape[-1])
    if trim_percent > 0:
        lower = np.percentile(flat, trim_percent, axis=0)
        upper = np.percentile(flat, 100 - trim_percent, axis=0)
        clipped = np.clip(flat, lower, upper)
    else:
        clipped = flat
    return clipped.mean(axis=0), clipped.std(axis=0).clip(min=1e-6)


def low_frequency_rgb(rgb, radius):
    image = Image.fromarray(np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8), mode="RGB")
    image = image.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(image, dtype=np.float32) / 255.0


def correct_color(
    generated,
    reference,
    channels,
    method,
    match_std,
    strength,
    trim_percent,
    max_chroma_shift,
    local_radius,
):
    generated_lab = srgb_to_lab(generated)
    reference_lab = srgb_to_lab(reference)
    corrected = generated_lab.copy()
    indices = (1, 2) if channels == "chroma" else (0, 1, 2)
    if method == "paired_offset":
        difference = reference_lab - generated_lab
        flat = difference.reshape(-1, 3)
        if trim_percent > 0:
            lower = np.percentile(flat, trim_percent, axis=0)
            upper = np.percentile(flat, 100 - trim_percent, axis=0)
            flat = np.clip(flat, lower, upper)
        offset = np.median(flat, axis=0)
        offset[1:] = np.clip(offset[1:], -max_chroma_shift, max_chroma_shift)
        for channel in indices:
            corrected[..., channel] = generated_lab[..., channel] + offset[channel]
    elif method == "local_chroma":
        generated_low = srgb_to_lab(low_frequency_rgb(generated, local_radius))
        reference_low = srgb_to_lab(low_frequency_rgb(reference, local_radius))
        difference = reference_low - generated_low
        difference[..., 1:] = np.clip(difference[..., 1:], -max_chroma_shift, max_chroma_shift)
        for channel in indices:
            corrected[..., channel] = generated_lab[..., channel] + difference[..., channel]
    else:
        generated_mean, generated_std = robust_stats(generated_lab, trim_percent)
        reference_mean, reference_std = robust_stats(reference_lab, trim_percent)
        for channel in indices:
            values = generated_lab[..., channel] - generated_mean[channel]
            if match_std:
                values = values * (reference_std[channel] / generated_std[channel])
            corrected[..., channel] = values + reference_mean[channel]
    corrected = generated_lab + strength * (corrected - generated_lab)
    corrected[..., 0] = np.clip(corrected[..., 0], 0, 100)
    corrected[..., 1:] = np.clip(corrected[..., 1:], -128, 127)
    return lab_to_srgb(corrected)


def main():
    args = parse_args()
    if not 0 <= args.strength <= 1:
        raise ValueError("--strength 必须位于 [0,1]。")
    if not 0 <= args.trim_percent < 50:
        raise ValueError("--trim_percent 必须位于 [0,50)。")
    if args.max_chroma_shift <= 0 or args.local_radius <= 0:
        raise ValueError("--max_chroma_shift 和 --local_radius 必须大于 0。")
    records = load_metadata(args.dataset_metadata_path) if args.dataset_metadata_path else records_from_folder(args)
    end = None if args.max_samples is None else args.start_index + args.max_samples
    selected = list(enumerate(records[args.start_index:end], start=args.start_index))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for index, sample in selected:
        result = dict(sample)
        try:
            sample_id = str(sample.get(args.id_key, index)).replace("/", "_").replace("\\", "_")
            destination = output_dir / f"{index:06d}_{sample_id}.png"
            result[args.output_key] = str(destination)
            if not (args.skip_existing and destination.exists()):
                generated_path = generated_path_for_sample(sample, index, args)
                if not generated_path.is_file():
                    raise FileNotFoundError(generated_path)
                generated_image = Image.open(generated_path).convert("RGB")
                generated = np.asarray(generated_image, dtype=np.float32) / 255.0
                reference = load_reference(sample, args, generated_image.size)
                corrected = correct_color(
                    generated,
                    reference,
                    args.channels,
                    args.method,
                    args.match_std,
                    args.strength,
                    args.trim_percent,
                    args.max_chroma_shift,
                    args.local_radius,
                )
                Image.fromarray(np.round(corrected * 255).astype(np.uint8), mode="RGB").save(destination)
            print(destination)
        except Exception as error:
            if not args.continue_on_error:
                raise
            result["color_postprocess_error"] = str(error)
        results.append(result)

    metadata_path = (
        Path(args.output_metadata_path)
        if args.output_metadata_path
        else output_dir / "metadata_color_corrected.json"
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(results)} records to {metadata_path}")


if __name__ == "__main__":
    main()
