#!/usr/bin/env python

import argparse
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from artifact_repair_utils import (  # noqa: E402
    DEFAULT_REPAIR_PROMPT,
    add_pretrained_args,
    artifact_output_name,
    get_artifact_repair_paths,
    load_metadata,
    load_rgb,
    preprocess_pair,
    restore_output_size,
    select_records,
    write_json,
)
from infer_sana_artifact_repair_wan_crossattn import generate, load_pipeline, select_dtype  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Batch infer SANA Artifact Repair Route 3 Wan-style image cross-attention.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset_metadata_path", required=True)
    parser.add_argument("--dataset_base_path", default=".")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_key", default="image")
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16", "auto"), default="auto")
    parser.add_argument("--max_pixels", type=int, default=1048576)
    parser.add_argument("--size_divisor", type=int, default=32)
    parser.add_argument("--downscale_if_exceeds_max_pixels", action="store_true")
    parser.add_argument("--restore_to_original_size", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--init_mode", choices=("pure_noise", "src_latent"), default="pure_noise")
    parser.add_argument("--src_init_strength", type=float, default=0.3)
    parser.add_argument("--disable_image_cross_attention", action="store_true")
    parser.add_argument("--zero_image_tokens", action="store_true")
    parser.add_argument("--swap_image_tokens_path", default=None)
    parser.add_argument("--image_cross_attention_scale", type=float, default=None)
    parser.add_argument("--prompt", default=DEFAULT_REPAIR_PROMPT)
    parser.add_argument("--negative_prompt", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    dtype = select_dtype(args.dtype)
    pipe, image_encoder, image_processor, config = load_pipeline(args, dtype)
    config["_checkpoint"] = str(args.checkpoint)
    records = load_metadata(args.dataset_metadata_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, sample in select_records(records, args.start_index, args.max_samples):
        output = output_dir / artifact_output_name(sample, index)
        try:
            paths = get_artifact_repair_paths(sample, args.dataset_base_path, index, args.target_key, args.prompt_key)
            src = load_rgb(paths["src"])
            ref = load_rgb(paths["ref"])
            token_img = load_rgb(args.swap_image_tokens_path) if args.swap_image_tokens_path else src
            prepared, size_info = preprocess_pair(
                src,
                ref,
                args.max_pixels,
                args.size_divisor,
                args.downscale_if_exceeds_max_pixels,
            )
            token_img = token_img.resize(prepared["src"].size)
            canvas_w, canvas_h = size_info["canvas_size"]
            image, stats = generate(
                pipe,
                image_encoder,
                image_processor,
                config,
                paths["prompt"] or args.prompt,
                args.negative_prompt,
                prepared["src"],
                token_img,
                canvas_h,
                canvas_w,
                args.steps,
                args.guidance_scale,
                args.seed + index,
                args.init_mode,
                args.src_init_strength,
                args.disable_image_cross_attention,
                args.zero_image_tokens,
                args.image_cross_attention_scale,
            )
            image = restore_output_size(image, size_info, args.restore_to_original_size)
            image.save(output)
            stats.update({"index": index, "output": str(output), "status": "success"})
            write_json(output.with_suffix(output.suffix + ".stats.json"), stats)
            results.append(stats)
            print(f"[ROUTE3][BATCH] ok index={index} output={output}", flush=True)
        except Exception as exc:
            record = {"index": index, "output": str(output), "status": "error", "error": repr(exc)}
            results.append(record)
            print(f"[ROUTE3][BATCH] error index={index}: {exc}", flush=True)
            if not args.continue_on_error:
                raise
    write_json(output_dir / "metadata_result.json", results)
    print(f"[ROUTE3][BATCH] metadata_result={output_dir / 'metadata_result.json'}", flush=True)


if __name__ == "__main__":
    main()
