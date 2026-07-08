#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import torch

from artifact_repair_utils import (
    DEFAULT_REPAIR_PROMPT,
    add_pretrained_args,
    artifact_output_name,
    artifact_paths,
    build_control_condition,
    load_metadata,
    load_rgb,
    preprocess_pair_or_triplet,
    pretrained_kwargs,
    restore_output_size,
    select_records,
)
from infer_sana_artifact_repair_controlnet import generate, load_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Batch infer SANA artifact repair ControlNet.")
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
    parser.add_argument("--conditioning_scale", type=float, default=1.0)
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
    pipe, controlnet, projection, _ = load_pipeline(Path(args.checkpoint), args.model, dtype, pretrained_kwargs(args))
    records = load_metadata(args.dataset_metadata_path)
    selected = select_records(records, args.start_index, args.max_samples)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for index, sample in selected:
        output_path = output_dir / artifact_output_name(sample, index)
        result_record = dict(sample)
        result_record[args.result_key] = str(output_path)
        try:
            if args.skip_existing and output_path.exists():
                result_record["status"] = "skipped_existing"
                results.append(result_record)
                continue
            paths = artifact_paths(sample, args.dataset_base_path, args.target_key, args.prompt_key)
            src = load_rgb(paths["src"])
            ref = load_rgb(paths["ref"])
            prompt = paths["prompt"] or args.prompt
            prepared, size_info = preprocess_pair_or_triplet(
                src,
                src,
                ref,
                max_pixels=args.max_pixels,
                size_divisor=args.size_divisor,
                downscale_if_exceeds_max_pixels=args.downscale_if_exceeds_max_pixels,
            )
            canvas_w, canvas_h = size_info["canvas_size"]
            cond_src = prepared["src"]
            cond_ref = prepared["ref"]
            if args.zero_src_condition:
                from PIL import Image

                cond_src = Image.new("RGB", prepared["src"].size, (0, 0, 0))
            if args.zero_ref_condition:
                from PIL import Image

                cond_ref = Image.new("RGB", prepared["ref"].size, (0, 0, 0))
            src_tensor = pipe.image_processor.preprocess(cond_src, height=canvas_h, width=canvas_w)
            ref_tensor = pipe.image_processor.preprocess(cond_ref, height=canvas_h, width=canvas_w)
            control_condition = build_control_condition(src_tensor, ref_tensor)
            seed = int(sample.get(args.seed_key, args.seed))
            generator = torch.Generator(device="cuda").manual_seed(seed)
            debug_dir = None
            if args.debug_latent_dir:
                debug_dir = Path(args.debug_latent_dir) / output_path.stem
            image, _, _, stats = generate(
                pipe,
                controlnet,
                projection,
                prompt,
                args.negative_prompt,
                prepared["src"],
                control_condition,
                canvas_h,
                canvas_w,
                args.steps,
                args.guidance_scale,
                args.strength,
                args.img2img_schedule_mode,
                generator,
                args.conditioning_scale,
            )
            if debug_dir:
                debug_dir.mkdir(parents=True, exist_ok=True)
                prepared["src"].save(debug_dir / "raw_src.png")
                prepared["ref"].save(debug_dir / "raw_ref.png")
                image.save(debug_dir / "final_output.png")
                (debug_dir / "latent_stats.json").write_text(
                    json.dumps(stats, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            image = restore_output_size(image, size_info, args.restore_to_original_size)
            image.save(output_path)
            result_record["status"] = "ok"
            result_record["seed"] = seed
        except Exception as exc:
            result_record["status"] = "error"
            result_record["error"] = repr(exc)
            results.append(result_record)
            if not args.continue_on_error:
                (output_dir / "metadata_results.json").write_text(
                    json.dumps(results, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                raise
            continue
        results.append(result_record)

    (output_dir / "metadata_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
