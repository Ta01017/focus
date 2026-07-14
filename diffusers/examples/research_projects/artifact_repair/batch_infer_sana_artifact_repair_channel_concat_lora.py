#!/usr/bin/env python

import argparse
from pathlib import Path

import torch

from artifact_repair_utils import (
    DEFAULT_REPAIR_PROMPT,
    add_pretrained_args,
    artifact_output_name,
    get_artifact_repair_paths,
    load_metadata,
    load_rgb,
    preprocess_pair,
    pretrained_kwargs,
    restore_output_size,
    select_records,
    write_json,
)
from infer_sana_artifact_repair_channel_concat_lora import generate, load_pipeline, select_dtype


def parse_args():
    parser = argparse.ArgumentParser(description="Batch infer SANA Artifact Repair Route 2 channel-concat I2I.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset_metadata_path", required=True)
    parser.add_argument("--dataset_base_path", default=".")
    parser.add_argument("--target_key", default="image")
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--id_key", default="id")
    parser.add_argument("--seed_key", default="seed")
    parser.add_argument("--result_key", default="generated_image")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--prompt", default=DEFAULT_REPAIR_PROMPT)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16", "auto"), default="auto")
    parser.add_argument("--max_pixels", type=int, default=1048576)
    parser.add_argument("--size_divisor", type=int, default=32)
    parser.add_argument("--downscale_if_exceeds_max_pixels", action="store_true")
    parser.add_argument("--restore_to_original_size", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug_latent_dir", default=None)
    parser.add_argument("--save_condition_sensitivity_debug", action="store_true")
    add_pretrained_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("Dynamic Route 2 batch inference currently supports --batch_size 1.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    pipe, _, original_latent_channels = load_pipeline(args, select_dtype(args.dtype))
    records = select_records(load_metadata(args.dataset_metadata_path), args.start_index, args.max_samples)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, sample in records:
        output_path = output_dir / artifact_output_name(sample, index)
        result = {"index": index, "output": str(output_path), "status": "pending"}
        try:
            paths = get_artifact_repair_paths(sample, args.dataset_base_path, index, args.target_key, args.prompt_key)
            for key in ("gt", "src", "ref"):
                if paths.get(key) is not None and not Path(paths[key]).exists():
                    raise FileNotFoundError(f"Resolved {key} path does not exist: {paths[key]}")
            prompt = paths["prompt"] or sample.get(args.prompt_key) or args.prompt
            if args.skip_existing and output_path.exists():
                result.update({"status": "skipped_existing", "src": str(paths["src"]), "target": str(paths["gt"]), args.result_key: str(output_path)})
                results.append(result)
                continue
            src = load_rgb(paths["src"])
            ref = load_rgb(paths["ref"])
            prepared, size_info = preprocess_pair(src, ref, args.max_pixels, args.size_divisor, args.downscale_if_exceeds_max_pixels)
            canvas_w, canvas_h = size_info["canvas_size"]
            seed = int(sample.get(args.seed_key, args.seed))
            image, stats, _, _ = generate(
                pipe,
                prompt,
                args.negative_prompt,
                prepared["src"],
                canvas_h,
                canvas_w,
                args.steps,
                args.guidance_scale,
                seed,
                original_latent_channels,
                args.save_condition_sensitivity_debug,
            )
            stats.update(
                {
                    "index": index,
                    "src": str(paths["src"]),
                    "target": str(paths["gt"]),
                    "prompt": prompt,
                    "seed": seed,
                    "original_size": list(size_info["original_size"]),
                    "inference_size": list(size_info["canvas_size"]),
                }
            )
            if args.debug_latent_dir:
                debug = Path(args.debug_latent_dir) / output_path.stem
                debug.mkdir(parents=True, exist_ok=True)
                src.save(debug / "raw_src.png")
                prepared["src"].save(debug / "resized_src.png")
                image.save(debug / "final_output_before_restore.png")
                write_json(debug / "latent_stats.json", stats)
            image = restore_output_size(image, size_info, args.restore_to_original_size)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)
            stats["restored_size"] = list(image.size)
            write_json(output_path.with_suffix(output_path.suffix + ".stats.json"), stats)
            result.update(
                {
                    "status": "ok",
                    "src": str(paths["src"]),
                    "target": str(paths["gt"]),
                    "prompt": prompt,
                    "seed": seed,
                    args.result_key: str(output_path),
                }
            )
        except Exception as exc:
            result.update({"status": "error", "error": repr(exc)})
            results.append(result)
            write_json(output_dir / "metadata_results.json", results)
            if not args.continue_on_error:
                raise
            print(f"error: index={index} error={exc!r}", flush=True)
            continue
        results.append(result)
        print(f"{result['status']}: {output_path}", flush=True)
    write_json(output_dir / "metadata_results.json", results)


if __name__ == "__main__":
    main()
