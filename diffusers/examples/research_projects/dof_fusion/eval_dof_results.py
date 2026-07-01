#!/usr/bin/env python

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from dof_utils import add_metadata_args, select_records
from metadata import load_metadata, resolve_data_path


def parse_args():
    parser = argparse.ArgumentParser(description="评测景深融合结果的全图、keep 和 blur 区域指标。")
    add_metadata_args(parser, metadata_required=True)
    parser.add_argument("--focus_index", type=int, default=2)
    parser.add_argument("--output_json", default="dof_eval.json")
    parser.add_argument("--output_csv", default="dof_eval.csv")
    return parser.parse_args()


def load_rgb(path, size=None):
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def load_mask(path, size):
    image = Image.open(path).convert("L").resize(size, Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0).clamp(0, 1)


def weighted_mean(value, mask):
    if mask.shape[1] == 1 and value.shape[1] != 1:
        mask = mask.expand(-1, value.shape[1], -1, -1)
    return (value * mask).sum() / mask.sum().clamp_min(1e-8)


def ssim_map(prediction, target):
    kernel_size = 11
    padding = kernel_size // 2
    mu_x = F.avg_pool2d(prediction, kernel_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, kernel_size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(prediction.square(), kernel_size, stride=1, padding=padding) - mu_x.square()
    sigma_y = F.avg_pool2d(target.square(), kernel_size, stride=1, padding=padding) - mu_y.square()
    sigma_xy = F.avg_pool2d(prediction * target, kernel_size, stride=1, padding=padding) - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    return ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    )


def sobel_edges(image):
    gray = image.mean(dim=1, keepdim=True)
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(2, 3)
    edge_x = F.conv2d(gray, kernel_x, padding=1)
    edge_y = F.conv2d(gray, kernel_y, padding=1)
    return torch.sqrt(edge_x.square() + edge_y.square() + 1e-12)


def region_metrics(prediction, target, mask):
    absolute = (prediction - target).abs()
    mse = weighted_mean((prediction - target).square(), mask).item()
    return {
        "psnr": -10 * math.log10(max(mse, 1e-12)),
        "ssim": weighted_mean(ssim_map(prediction, target), mask).item(),
        "l1": weighted_mean(absolute, mask).item(),
        "sobel_l1": weighted_mean((sobel_edges(prediction) - sobel_edges(target)).abs(), mask).item(),
    }


def result_path(record, key, base_path):
    path = Path(record[key])
    if path.is_file() or path.is_absolute():
        return path
    return Path(resolve_data_path(record[key], base_path))


def main():
    args = parse_args()
    records = select_records(load_metadata(args.dataset_metadata_path), args.start_index, args.max_samples)
    rows = []
    for index, record in records:
        try:
            target_path = resolve_data_path(record[args.target_key], args.dataset_base_path)
            prediction_path = result_path(record, args.result_key, args.dataset_base_path)
            edits = record[args.edit_key]
            if len(edits) <= args.focus_index:
                raise ValueError(f"edit_image 缺少 focus index {args.focus_index}。")
            target_image = Image.open(target_path)
            target = load_rgb(target_path)
            prediction = load_rgb(prediction_path, target_image.size)
            keep = load_mask(resolve_data_path(edits[args.focus_index], args.dataset_base_path), target_image.size)
            masks = {"all": torch.ones_like(keep), "keep": keep, "blur": 1 - keep}
            row = {"index": index, "id": record.get(args.id_key, index), "prediction": str(prediction_path)}
            for region, mask in masks.items():
                for metric, value in region_metrics(prediction, target, mask).items():
                    row[f"{region}_{metric}"] = value
            rows.append(row)
            print(f"[{index}] all_psnr={row['all_psnr']:.3f} keep_psnr={row['keep_psnr']:.3f}")
        except Exception as error:
            if not args.continue_on_error:
                raise
            rows.append({"index": index, "id": record.get(args.id_key, index), "error": str(error)})

    metric_keys = sorted({key for row in rows for key in row if key not in {"index", "id", "prediction", "error"}})
    valid_rows = [row for row in rows if "error" not in row]
    summary = {
        key: float(np.mean([row[key] for row in valid_rows])) for key in metric_keys
    } if valid_rows else {}
    output = {"num_samples": len(rows), "num_valid": len(valid_rows), "summary": summary, "samples": rows}
    Path(args.output_json).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    fieldnames = ["index", "id", "prediction", *metric_keys, "error"]
    with Path(args.output_csv).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
