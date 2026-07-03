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
from flux2_controlnet import focus_map_to_tokens
from infer_flux2_klein import DEFAULT_PROMPT
from infer_flux2_klein_controlnet import load_pipeline
from metadata import load_metadata, require_keys, resolve_data_path


def parse_args():
    parser = argparse.ArgumentParser(description="Batch FLUX.2 Klein focus ControlNet inference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default=None)
    add_metadata_args(parser, metadata_required=True)
    add_pretrained_args(parser)
    parser.add_argument("--control_index", type=int, choices=(2, 3), default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=None)
    parser.add_argument("--size_divisor", type=int, default=16)
    parser.add_argument("--aspect_ratio_tolerance", type=float, default=0.01)
    parser.add_argument("--downscale_if_exceeds_max_pixels", action="store_true")
    restore_group = parser.add_mutually_exclusive_group()
    restore_group.add_argument("--restore_to_original_size", dest="restore_to_original_size", action="store_true")
    restore_group.add_argument("--no_restore_to_original_size", dest="restore_to_original_size", action="store_false")
    parser.set_defaults(restore_to_original_size=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--conditioning_scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This reference script requires CUDA.")
    if args.size_divisor % 16:
        raise ValueError("--size_divisor must be a multiple of 16 for FLUX.2 Klein.")
    if args.height is None and args.batch_size != 1:
        raise ValueError("Dynamic-resolution batch inference requires --batch_size 1.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipe, transformer, config = load_pipeline(
        Path(args.checkpoint), args.model, dtype, args.conditioning_scale, pretrained_kwargs(args)
    )
    control_index = config["control_index"] if args.control_index is None else args.control_index
    records = select_records(load_metadata(args.dataset_metadata_path), args.start_index, args.max_samples)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, record in records:
        require_keys(record, (args.edit_key,), index)
        destination = sample_output_path(record, index, args.edit_key, args.id_key, output_dir)
        result = dict(record)
        result[args.result_key] = str(destination)
        if args.skip_existing and destination.exists():
            results.append(result)
            continue
        try:
            edits = record[args.edit_key]
            if not isinstance(edits, list) or len(edits) <= control_index:
                raise ValueError(f"Sample {index} must contain A, B, and control index {control_index}.")
            image_a = Image.open(resolve_data_path(edits[0], args.dataset_base_path)).convert("RGB")
            image_b = Image.open(resolve_data_path(edits[1], args.dataset_base_path)).convert("RGB")
            focus = Image.open(resolve_data_path(edits[control_index], args.dataset_base_path))
            prepared, size_info = prepare_inference_images(
                {"a": image_a, "b": image_b, "focus": focus},
                args.height,
                args.width,
                args.max_pixels,
                args.size_divisor,
                args.aspect_ratio_tolerance,
                args.downscale_if_exceeds_max_pixels,
            )
            canvas_width, canvas_height = size_info["canvas_size"]
            latent_height = canvas_height // (pipe.vae_scale_factor * 2)
            latent_width = canvas_width // (pipe.vae_scale_factor * 2)
            focus_tokens = focus_map_to_tokens(
                prepared["focus"], latent_height, latent_width, torch.device("cuda")
            )
            seed = int(record.get(args.seed_key, args.seed + index))
            generator = torch.Generator(device="cuda").manual_seed(seed)
            with transformer.use_focus_condition(focus_tokens):
                image = pipe(
                    image=[prepared["a"], prepared["b"]],
                    prompt=sample_prompt(record, args.prompt_key, args.prompt),
                    height=canvas_height,
                    width=canvas_width,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                ).images[0]
            restore_output_size(image, size_info, args.restore_to_original_size).save(destination)
        except Exception as error:
            if not args.continue_on_error:
                raise
            result["error"] = str(error)
        results.append(result)
        print(destination)
    (output_dir / "metadata_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
