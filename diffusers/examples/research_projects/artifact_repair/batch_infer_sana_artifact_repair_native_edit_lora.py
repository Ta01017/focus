#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from artifact_repair_utils import (
    DEFAULT_REPAIR_PROMPT,
    add_pretrained_args,
    get_artifact_repair_paths,
    load_metadata,
    load_rgb,
    preprocess_pair,
    pretrained_kwargs,
    restore_output_size,
    select_records,
    write_json,
)
from infer_sana_artifact_repair_native_edit_lora import generate, load_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Batch infer SANA native edit LoRA for artifact repair.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset_metadata_path", required=True)
    parser.add_argument("--dataset_base_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init_mode", choices=("noise", "src"), default="src")
    parser.add_argument("--strength", type=float, default=0.15)
    parser.add_argument("--max_pixels", type=int, default=1048576)
    parser.add_argument("--size_divisor", type=int, default=32)
    parser.add_argument("--downscale_if_exceeds_max_pixels", action="store_true")
    parser.add_argument("--restore_to_original_size", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--zero_src", action="store_true")
    parser.add_argument("--zero_ref", action="store_true")
    parser.add_argument("--swap_src_ref", action="store_true")
    add_pretrained_args(parser)
    return parser.parse_args()


def output_name(sample, index):
    if sample.get("obj_name") is not None and sample.get("test_uid") is not None:
        return f"{sample['obj_name']}_{sample['test_uid']}_native_edit.png"
    return f"{index:06d}_native_edit.png"


def main():
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("Dynamic batch inference currently supports --batch_size 1.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipe, native_transformer, _ = load_pipeline(args, dtype)
    records = select_records(load_metadata(args.dataset_metadata_path), args.start_index, args.max_samples)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, sample in records:
        out = output_dir / output_name(sample, index)
        result = {
            "index": index,
            "output": str(out),
            "init_mode": args.init_mode,
            "strength": args.strength,
            "status": "pending",
        }
        try:
            paths = get_artifact_repair_paths(sample, args.dataset_base_path, index)
            src = load_rgb(paths["src"])
            ref = load_rgb(paths["ref"])
            prompt = paths["prompt"] or DEFAULT_REPAIR_PROMPT
            if args.swap_src_ref:
                src, ref = ref, src
            if args.zero_src:
                src = Image.new("RGB", src.size, (0, 0, 0))
            if args.zero_ref:
                ref = Image.new("RGB", ref.size, (0, 0, 0))
            prepared, size_info = preprocess_pair(src, ref, args.max_pixels, args.size_divisor, args.downscale_if_exceeds_max_pixels)
            canvas_w, canvas_h = size_info["canvas_size"]
            generator = torch.Generator(device="cuda").manual_seed(args.seed + index)
            image = generate(
                pipe,
                native_transformer,
                prompt,
                "",
                prepared["src"],
                prepared["ref"],
                canvas_h,
                canvas_w,
                args.steps,
                args.guidance_scale,
                args.init_mode,
                args.strength,
                generator,
            )
            image = restore_output_size(image, size_info, args.restore_to_original_size)
            out.parent.mkdir(parents=True, exist_ok=True)
            image.save(out)
            result.update(
                {
                    "src": str(paths["src"]),
                    "ref": str(paths["ref"]),
                    "gt": str(paths["gt"]),
                    "prompt": prompt,
                    "zero_src": args.zero_src,
                    "zero_ref": args.zero_ref,
                    "swap_src_ref": args.swap_src_ref,
                    "status": "ok",
                }
            )
        except Exception as exc:
            result["status"] = "error"
            result["error"] = repr(exc)
            if not args.continue_on_error:
                results.append(result)
                write_json(output_dir / "metadata_results.json", results)
                raise
        results.append(result)
        print(f"{result['status']}: {out}", flush=True)
    write_json(output_dir / "metadata_results.json", results)


if __name__ == "__main__":
    main()
