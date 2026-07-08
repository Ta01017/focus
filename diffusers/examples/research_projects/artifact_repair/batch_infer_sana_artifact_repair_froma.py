#!/usr/bin/env python

import argparse
from pathlib import Path

import torch
from PIL import Image

from artifact_repair_utils import (
    DEFAULT_REPAIR_PROMPT,
    add_pretrained_args,
    artifact_output_name,
    build_control_condition,
    get_artifact_repair_paths,
    load_metadata,
    load_rgb,
    preprocess_pair,
    pretrained_kwargs,
    restore_output_size,
    select_records,
    write_json,
)
from infer_sana_artifact_repair_froma import generate, load_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Batch infer SANA artifact repair from-src adapter without ControlNet.")
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
    parser.add_argument("--strength", type=float, default=0.15)
    parser.add_argument("--img2img_schedule_mode", choices=("pipeline_full", "sliced"), default="sliced")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--adapter_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_pixels", type=int, default=1048576)
    parser.add_argument("--size_divisor", type=int, default=32)
    parser.add_argument("--downscale_if_exceeds_max_pixels", action="store_true")
    restore = parser.add_mutually_exclusive_group()
    restore.add_argument("--restore_to_original_size", dest="restore_to_original_size", action="store_true")
    restore.add_argument("--no_restore_to_original_size", dest="restore_to_original_size", action="store_false")
    parser.set_defaults(restore_to_original_size=True)
    parser.add_argument("--debug_latent_dir", default=None)
    parser.add_argument("--zero_ref_condition", action="store_true")
    parser.add_argument("--zero_src_condition", action="store_true")
    add_pretrained_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("This dynamic-size batch script currently supports --batch_size 1.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipe, adapter, _ = load_pipeline(Path(args.checkpoint), args.model, dtype, pretrained_kwargs(args))
    records = load_metadata(args.dataset_metadata_path)
    selected = select_records(records, args.start_index, args.max_samples)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for index, sample in selected:
        output_path = output_dir / artifact_output_name(sample, index)
        try:
            paths = get_artifact_repair_paths(sample, args.dataset_base_path, index, args.target_key, args.prompt_key)
            prompt = paths["prompt"] or args.prompt
            if args.skip_existing and output_path.exists():
                results.append({
                    "index": index,
                    "obj_name": sample.get("obj_name"),
                    "test_uid": sample.get("test_uid"),
                    "src": str(paths["src"]),
                    "ref": str(paths["ref"]),
                    "gt": str(paths["gt"]) if paths.get("gt") else None,
                    "prompt": prompt,
                    "output": str(output_path),
                    "status": "skipped_existing",
                })
                continue
            src = load_rgb(paths["src"])
            ref = load_rgb(paths["ref"])
            prepared, size_info = preprocess_pair(src, ref, args.max_pixels, args.size_divisor, args.downscale_if_exceeds_max_pixels)
            canvas_w, canvas_h = size_info["canvas_size"]
            cond_src = Image.new("RGB", prepared["src"].size, (0, 0, 0)) if args.zero_src_condition else prepared["src"]
            cond_ref = Image.new("RGB", prepared["ref"].size, (0, 0, 0)) if args.zero_ref_condition else prepared["ref"]
            src_tensor = pipe.image_processor.preprocess(cond_src, height=canvas_h, width=canvas_w)
            ref_tensor = pipe.image_processor.preprocess(cond_ref, height=canvas_h, width=canvas_w)
            condition = build_control_condition(src_tensor, ref_tensor)
            seed = int(sample.get(args.seed_key, args.seed))
            generator = torch.Generator(device="cuda").manual_seed(seed)
            image, _, _, _, stats = generate(
                pipe,
                adapter,
                prompt,
                args.negative_prompt,
                prepared["src"],
                condition,
                canvas_h,
                canvas_w,
                args.steps,
                args.guidance_scale,
                args.strength,
                args.img2img_schedule_mode,
                generator,
                args.adapter_scale,
            )
            if args.debug_latent_dir:
                debug = Path(args.debug_latent_dir) / output_path.stem
                debug.mkdir(parents=True, exist_ok=True)
                prepared["src"].save(debug / "raw_src.png")
                prepared["ref"].save(debug / "raw_ref.png")
                image.save(debug / "final_output.png")
                write_json(debug / "latent_stats.json", stats)
            image = restore_output_size(image, size_info, args.restore_to_original_size)
            image.save(output_path)
            results.append({
                "index": index,
                "obj_name": sample.get("obj_name"),
                "test_uid": sample.get("test_uid"),
                "src": str(paths["src"]),
                "ref": str(paths["ref"]),
                "gt": str(paths["gt"]) if paths.get("gt") else None,
                "prompt": prompt,
                args.result_key: str(output_path),
                "output": str(output_path),
                "strength": args.strength,
                "mode": args.img2img_schedule_mode,
                "adapter_scale": args.adapter_scale,
                "seed": seed,
                "status": "ok",
            })
        except Exception as exc:
            results.append({
                "index": index,
                "obj_name": sample.get("obj_name"),
                "test_uid": sample.get("test_uid"),
                "output": str(output_path),
                "status": "error",
                "error": repr(exc),
            })
            write_json(output_dir / "metadata_results.json", results)
            if not args.continue_on_error:
                raise
    write_json(output_dir / "metadata_results.json", results)


if __name__ == "__main__":
    main()
