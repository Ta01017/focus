#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from dof_utils import (
    add_metadata_args,
    add_pretrained_args,
    prepare_inference_images,
    pretrained_kwargs,
    restore_output_size,
    sample_output_path,
    sample_prompt,
    select_records,
)
from infer_sana_ab_adapter import (
    decode_latents_to_pil,
    encode_image_latents,
    generate_sana_img2img_sliced,
    load_pipeline,
    save_latent_debug,
)
from metadata import load_metadata, require_keys, resolve_data_path
from sana_dof import encode_condition_images


def parse_args():
    parser = argparse.ArgumentParser(description="Batch ordinary SANA + A/B adapter-only DOF fusion.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default=None)
    add_metadata_args(parser, metadata_required=True)
    add_pretrained_args(parser)
    parser.add_argument("--prompt", default="a photorealistic all-in-focus photograph")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=None)
    parser.add_argument("--size_divisor", type=int, default=32)
    parser.add_argument("--aspect_ratio_tolerance", type=float, default=0.01)
    parser.add_argument("--downscale_if_exceeds_max_pixels", action="store_true")
    restore_group = parser.add_mutually_exclusive_group()
    restore_group.add_argument("--restore_to_original_size", dest="restore_to_original_size", action="store_true")
    restore_group.add_argument("--no_restore_to_original_size", dest="restore_to_original_size", action="store_false")
    parser.set_defaults(restore_to_original_size=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_a_latent_init", action="store_true")
    parser.add_argument("--strength", type=float, default=0.6)
    parser.add_argument("--zero_condition_images", action="store_true")
    parser.add_argument("--img2img_schedule_mode", choices=("pipeline_full", "sliced"), default="pipeline_full")
    parser.add_argument("--debug_latent_dir", default=None)
    return parser.parse_args()


def load_condition_pair(record, index, args):
    require_keys(record, (args.edit_key,), index)
    edit_images = record[args.edit_key]
    if not isinstance(edit_images, list) or len(edit_images) < 2:
        raise ValueError(f"Sample {index} field {args.edit_key!r} must be a list containing at least A and B.")
    image_a = Image.open(resolve_data_path(edit_images[0], args.dataset_base_path)).convert("RGB")
    image_b = Image.open(resolve_data_path(edit_images[1], args.dataset_base_path)).convert("RGB")
    if image_a.size != image_b.size:
        print(f"Warning: sample {index} has different A/B sizes before resize: {image_a.size}, {image_b.size}.")
    return image_a, image_b


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch_size must be at least 1.")
    if (args.height is None) != (args.width is None):
        raise ValueError("--height and --width must be provided together or both omitted.")
    if args.height is None and args.batch_size != 1:
        raise ValueError("Dynamic-resolution batch inference requires --batch_size 1.")
    if not torch.cuda.is_available():
        raise RuntimeError("This reference script requires CUDA.")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipe, transformer, config = load_pipeline(Path(args.checkpoint), args.model, dtype, pretrained_kwargs(args))
    model_id = args.model or config["base_model"]
    print(f"[USE_A_LATENT_INIT] {int(args.use_a_latent_init)}", flush=True)
    print(f"[STRENGTH] {args.strength}", flush=True)
    print(f"[ZERO_CONDITION_IMAGES] {int(args.zero_condition_images)}", flush=True)
    print(f"[IMG2IMG_SCHEDULE_MODE] {args.img2img_schedule_mode}", flush=True)
    print(f"[DEBUG_LATENT_DIR] {args.debug_latent_dir}", flush=True)
    records = load_metadata(args.dataset_metadata_path)
    items = select_records(records, args.start_index, args.max_samples)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for index, record in items:
        output = sample_output_path(record, index, args.edit_key, args.id_key, output_dir)
        result = dict(record)
        result[args.result_key] = str(output)
        if args.skip_existing and output.exists():
            results.append(result)
            continue
        try:
            image_a, image_b = load_condition_pair(record, index, args)
            prepared, size_info = prepare_inference_images(
                {"a": image_a, "b": image_b},
                args.height,
                args.width,
                args.max_pixels,
                args.size_divisor,
                args.aspect_ratio_tolerance,
                args.downscale_if_exceeds_max_pixels,
            )
            canvas_width, canvas_height = size_info["canvas_size"]
            condition_a = Image.new("RGB", prepared["a"].size, (0, 0, 0)) if args.zero_condition_images else prepared["a"]
            condition_b = Image.new("RGB", prepared["b"].size, (0, 0, 0)) if args.zero_condition_images else prepared["b"]
            cond_a, cond_b = encode_condition_images(
                pipe, condition_a, condition_b, canvas_height, canvas_width, torch.device("cuda")
            )
            seed = int(record.get(args.seed_key, args.seed + index))
            generator = torch.Generator(device="cuda").manual_seed(seed)
            latents = None
            noise = None
            final_latents = None
            schedule_stats = {}
            if args.use_a_latent_init:
                if args.img2img_schedule_mode == "pipeline_full":
                    a_latents = encode_image_latents(pipe, prepared["a"], canvas_height, canvas_width, torch.device("cuda"))
                    noise = torch.randn(a_latents.shape, generator=generator, device="cuda", dtype=a_latents.dtype)
                    latents = (1 - args.strength) * a_latents + args.strength * noise
                else:
                    image, final_latents, schedule_stats, noise, latents = generate_sana_img2img_sliced(
                        pipe,
                        transformer,
                        sample_prompt(record, args.prompt_key, args.prompt),
                        "",
                        prepared["a"],
                        cond_a,
                        cond_b,
                        canvas_height,
                        canvas_width,
                        args.steps,
                        args.guidance_scale,
                        args.strength,
                        generator,
                    )
            if not (args.use_a_latent_init and args.img2img_schedule_mode == "sliced"):
                with transformer.use_condition(cond_a, cond_b):
                    final_latents = pipe(
                        prompt=sample_prompt(record, args.prompt_key, args.prompt),
                        height=canvas_height,
                        width=canvas_width,
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance_scale,
                        generator=generator,
                        latents=latents,
                        output_type="latent",
                        use_resolution_binning=False,
                    ).images
                image = decode_latents_to_pil(pipe, final_latents)
                pipe.scheduler.set_timesteps(args.steps, device=torch.device("cuda"))
                timesteps = pipe.scheduler.timesteps
                schedule_stats = {
                    "num_inference_steps": args.steps,
                    "init_timestep": None,
                    "t_start": None,
                    "actual_num_denoise_steps": len(timesteps),
                    "selected_sigma": None,
                    "timesteps_first": float(timesteps[0].detach().cpu()),
                    "timesteps_last": float(timesteps[-1].detach().cpu()),
                    "sliced_timesteps_first": None,
                    "sliced_timesteps_last": None,
                }
            if args.debug_latent_dir is not None:
                debug_args = argparse.Namespace(**vars(args))
                debug_args.prompt = sample_prompt(record, args.prompt_key, args.prompt)
                debug_args.checkpoint = str(args.checkpoint)
                save_latent_debug(
                    Path(args.debug_latent_dir) / f"sample_{index:06d}",
                    args=debug_args,
                    model_id=model_id,
                    pipe=pipe,
                    prepared=prepared,
                    size_info=size_info,
                    canvas_width=canvas_width,
                    canvas_height=canvas_height,
                    cond_a=cond_a,
                    cond_b=cond_b,
                    noise=noise,
                    init_latents=latents,
                    final_latents=final_latents,
                    image=image,
                    schedule_stats=schedule_stats,
                )
            image = restore_output_size(image, size_info, args.restore_to_original_size)
            output.parent.mkdir(parents=True, exist_ok=True)
            image.save(output)
        except Exception as error:
            if not args.continue_on_error:
                raise
            result["error"] = str(error)
        results.append(result)
        print(f"[{index + 1}] {output}")

    result_path = output_dir / "metadata_results.json"
    result_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(results)} records to {result_path}")


if __name__ == "__main__":
    main()
