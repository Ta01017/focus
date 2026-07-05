#!/usr/bin/env python

import argparse
import json
import shlex
import subprocess
from pathlib import Path

import torch
from PIL import Image

from diffusers import Flux2KleinPipeline

from dof_utils import add_metadata_args, add_pretrained_args, prepare_inference_images, pretrained_kwargs, restore_output_size
from infer_flux2_klein import DEFAULT_PROMPT
from metadata import load_metadata, require_keys, resolve_data_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate offline teacher targets for SANA-Sprint response distillation.")
    parser.add_argument("--teacher_backend", choices=("diffusers_flux2", "command"), required=True)
    parser.add_argument("--teacher_model", default="black-forest-labs/FLUX.2-klein-base-4B")
    parser.add_argument(
        "--teacher_command",
        default=None,
        help="Command template with {image_a}, {image_b}, {focus_a}, {focus_b}, {output}, {prompt}, {seed}.",
    )
    parser.add_argument("--teacher_steps", type=int, default=50)
    parser.add_argument("--teacher_guidance_scale", type=float, default=1.0)
    parser.add_argument("--teacher_lora", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_metadata_path", required=True)
    parser.add_argument("--teacher_target_key", default="teacher_image")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_pixels", type=int, default=None)
    parser.add_argument("--size_divisor", type=int, default=16)
    parser.add_argument("--aspect_ratio_tolerance", type=float, default=0.01)
    parser.add_argument("--downscale_if_exceeds_max_pixels", action="store_true")
    add_metadata_args(parser, metadata_required=True)
    add_pretrained_args(parser)
    return parser.parse_args()


def command_for_sample(template, values):
    return [token.format(**values) for token in shlex.split(template)]


def main():
    args = parse_args()
    if args.teacher_backend == "command" and not args.teacher_command:
        raise ValueError("--teacher_backend command requires --teacher_command.")
    if args.teacher_backend == "diffusers_flux2" and args.size_divisor % 16:
        raise ValueError("FLUX2 --size_divisor must be a multiple of 16.")

    records = load_metadata(args.dataset_metadata_path)
    end = None if args.max_samples is None else args.start_index + args.max_samples
    selected = list(enumerate(records[args.start_index:end], start=args.start_index))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = None
    if args.teacher_backend == "diffusers_flux2":
        if not torch.cuda.is_available():
            raise RuntimeError("The built-in FLUX2 teacher requires CUDA.")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        pipe = Flux2KleinPipeline.from_pretrained(
            args.teacher_model, torch_dtype=dtype, **pretrained_kwargs(args)
        ).to("cuda")
        if args.teacher_lora:
            pipe.load_lora_weights(args.teacher_lora)

    results = []
    for index, sample in selected:
        result = dict(sample)
        try:
            require_keys(sample, (args.edit_key,), index)
            edits = sample[args.edit_key]
            if not isinstance(edits, list) or len(edits) < 2:
                raise ValueError(f"Sample {index} requires edit_image[A,B].")
            sample_id = str(sample.get(args.id_key, Path(edits[0]).stem)).replace("/", "_").replace("\\", "_")
            destination = output_dir / f"{index:06d}_{sample_id}.png"
            result[args.teacher_target_key] = str(destination)
            if args.skip_existing and destination.exists():
                results.append(result)
                continue
            prompt = sample.get(args.prompt_key) or args.prompt
            seed = int(sample.get(args.seed_key, args.seed + index))
            image_a_path = resolve_data_path(edits[0], args.dataset_base_path)
            image_b_path = resolve_data_path(edits[1], args.dataset_base_path)

            if pipe is not None:
                prepared, size_info = prepare_inference_images(
                    {
                        "a": Image.open(image_a_path).convert("RGB"),
                        "b": Image.open(image_b_path).convert("RGB"),
                    },
                    max_pixels=args.max_pixels,
                    size_divisor=args.size_divisor,
                    aspect_ratio_tolerance=args.aspect_ratio_tolerance,
                    downscale_if_exceeds_max_pixels=args.downscale_if_exceeds_max_pixels,
                )
                canvas_width, canvas_height = size_info["canvas_size"]
                generator = torch.Generator(device="cuda").manual_seed(seed)
                image = pipe(
                    image=[prepared["a"], prepared["b"]],
                    prompt=prompt,
                    height=canvas_height,
                    width=canvas_width,
                    num_inference_steps=args.teacher_steps,
                    guidance_scale=args.teacher_guidance_scale,
                    generator=generator,
                ).images[0]
                restore_output_size(image, size_info, True).save(destination)
            else:
                values = {
                    "image_a": str(image_a_path),
                    "image_b": str(image_b_path),
                    "focus_a": str(resolve_data_path(edits[2], args.dataset_base_path)) if len(edits) > 2 else "",
                    "focus_b": str(resolve_data_path(edits[3], args.dataset_base_path)) if len(edits) > 3 else "",
                    "output": str(destination),
                    "prompt": prompt,
                    "seed": seed,
                }
                subprocess.run(command_for_sample(args.teacher_command, values), check=True)
                if not destination.is_file():
                    raise FileNotFoundError(f"Teacher command did not create {destination}.")
            print(destination)
        except Exception as error:
            if not args.continue_on_error:
                raise
            result["teacher_error"] = str(error)
        results.append(result)

    metadata_path = Path(args.output_metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(results)} teacher records to {metadata_path}")


if __name__ == "__main__":
    main()
