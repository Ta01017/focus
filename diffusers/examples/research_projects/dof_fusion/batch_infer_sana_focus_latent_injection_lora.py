#!/usr/bin/env python

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dof_utils import add_metadata_args, add_pretrained_args, restore_output_size, sample_output_path, select_records  # noqa: E402
from infer_sana_focus_latent_injection_lora import generate, load_pipeline_and_model, select_dtype  # noqa: E402
from metadata import load_metadata, resolve_data_path  # noqa: E402


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Batch infer SANA focus latent injection LoRA.")
    parser.add_argument("--model", default="Efficient-Large-Model/Sana_600M_1024px_diffusers")
    parser.add_argument("--condition_mode", choices=("a_only", "ab"), required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--allow_untrained_injector", action="store_true")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--use_a_latent_init", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strength", type=float, default=0.3)
    parser.add_argument("--img2img_schedule_mode", choices=("sliced", "pipeline_full"), default="sliced")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16", "auto"), default="auto")
    parser.add_argument("--condition_scale", type=float, default=1.0)
    parser.add_argument("--zero_a_condition", action="store_true")
    parser.add_argument("--zero_b_condition", action="store_true")
    parser.add_argument("--swap_ab", action="store_true")
    parser.add_argument("--disable_condition_injection", action="store_true")
    parser.add_argument("--max_pixels", type=int, default=1048576)
    parser.add_argument("--size_divisor", type=int, default=32)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--aspect_ratio_tolerance", type=float, default=0.01)
    parser.add_argument("--downscale_if_exceeds_max_pixels", action="store_true")
    parser.add_argument("--restore_to_original_size", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_comparison", action="store_true")
    add_metadata_args(parser, metadata_required=True)
    add_pretrained_args(parser)
    return parser.parse_args()


def concat_images(images):
    widths = [image.width for image in images]
    heights = [image.height for image in images]
    canvas = Image.new("RGB", (sum(widths), max(heights)), "white")
    x = 0
    for image in images:
        canvas.paste(image, (x, 0))
        x += image.width
    return canvas


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    dtype = select_dtype(args.dtype)
    pipe, focus_model, config, _ = load_pipeline_and_model(args, dtype)
    records = load_metadata(args.dataset_metadata_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, sample in select_records(records, args.start_index, args.max_samples):
        output = sample_output_path(sample, index, args.edit_key, args.id_key, output_dir)
        if args.skip_existing and output.exists():
            results.append({"index": index, "output": str(output), "status": "skipped"})
            continue
        try:
            edits = sample.get(args.edit_key)
            if not isinstance(edits, list) or len(edits) < (2 if args.condition_mode == "ab" else 1):
                raise ValueError(f"Sample {index} has insufficient edit_image entries for {args.condition_mode}.")
            image_a_path = resolve_data_path(edits[0], args.dataset_base_path)
            image_b_path = resolve_data_path(edits[1], args.dataset_base_path) if len(edits) > 1 else None
            gt_path = resolve_data_path(sample[args.target_key], args.dataset_base_path)
            image_a = Image.open(image_a_path).convert("RGB")
            image_b = Image.open(image_b_path).convert("RGB") if image_b_path else None
            run_args = argparse.Namespace(**vars(args))
            run_args.seed = int(sample.get(args.seed_key, args.seed + index))
            if run_args.prompt is None and args.prompt_key in sample:
                run_args.prompt = sample.get(args.prompt_key)
            image, stats, prepared, size_info = generate(pipe, focus_model, config, run_args, image_a, image_b)
            image = restore_output_size(image, size_info, args.restore_to_original_size)
            image.save(output)
            comparison_path = None
            if args.save_comparison:
                gt = Image.open(gt_path).convert("RGB").resize(image.size, Image.Resampling.LANCZOS)
                a_vis = image_a.resize(image.size, Image.Resampling.LANCZOS)
                if args.condition_mode == "ab":
                    b_vis = image_b.resize(image.size, Image.Resampling.LANCZOS)
                    comparison = concat_images([a_vis, b_vis, gt, image])
                else:
                    comparison = concat_images([a_vis, gt, image])
                comparison_path = output.with_name(output.stem + "_comparison.png")
                comparison.save(comparison_path)
            row = {
                "index": index,
                "A path": str(image_a_path),
                "B path": str(image_b_path) if image_b_path else None,
                "GT path": str(gt_path),
                "output path": str(output),
                "comparison path": str(comparison_path) if comparison_path else None,
                "status": "success",
                "condition_mode": args.condition_mode,
                "condition_scale": args.condition_scale,
                "use_a_latent_init": args.use_a_latent_init,
                "strength": args.strength,
                "seed": run_args.seed,
                "stats": stats,
            }
            results.append(row)
            print(f"[FOCUS_ROUTE1][BATCH] ok index={index} output={output}", flush=True)
        except Exception as exc:
            row = {"index": index, "output path": str(output), "status": "error", "error": repr(exc)}
            results.append(row)
            print(f"[FOCUS_ROUTE1][BATCH] error index={index}: {exc}", flush=True)
            if not args.continue_on_error:
                raise
    write_json(output_dir / "results.json", results)
    print(f"[FOCUS_ROUTE1][BATCH] results={output_dir / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
