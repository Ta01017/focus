#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from diffusers import Flux2KleinPipeline

from dof_utils import (
    add_metadata_args,
    add_pretrained_args,
    prepare_inference_images,
    pretrained_kwargs,
    restore_output_size,
)


DEFAULT_PROMPT = (
    "Create one photorealistic all-in-focus image from the two aligned reference photographs. "
    "Use the sharp regions from each reference, preserve the exact scene geometry, identity, color, exposure, and "
    "fine texture. Do not add, remove, move, or restyle anything. Avoid blur, halos, double edges, and ghosting."
)


def parse_args():
    parser = argparse.ArgumentParser(description="FLUX.2 Klein native multi-reference depth-of-field fusion baseline.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--lora", default=None, help="Optional FLUX.2 Klein LoRA directory or weight file.")
    parser.add_argument("--image_a", required=True, help="First differently focused RGB image.")
    parser.add_argument("--image_b", required=True, help="Second differently focused RGB image.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
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
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu_offload", action="store_true")
    add_metadata_args(parser)
    add_pretrained_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.size_divisor % 16:
        raise ValueError("--size_divisor must be a multiple of 16 for FLUX.2 Klein.")

    image_a = Image.open(args.image_a).convert("RGB")
    image_b = Image.open(args.image_b).convert("RGB")
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

    training_config = {}
    if args.lora is not None:
        lora_path = Path(args.lora)
        config_path = (lora_path if lora_path.is_dir() else lora_path.parent) / "training_config.json"
        if config_path.exists():
            training_config = json.loads(config_path.read_text(encoding="utf-8"))
    model = args.model or training_config.get("base_model", "black-forest-labs/FLUX.2-klein-4B")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    pipe = Flux2KleinPipeline.from_pretrained(model, torch_dtype=dtype, **pretrained_kwargs(args))
    if args.lora is not None:
        pipe.load_lora_weights(args.lora)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
        generator_device = "cpu"
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("FLUX.2 Klein inference requires CUDA unless a different placement strategy is added.")
        pipe.to("cuda")
        generator_device = "cuda"

    generator = torch.Generator(device=generator_device).manual_seed(args.seed)
    image = pipe(
        image=[prepared["a"], prepared["b"]],
        prompt=args.prompt,
        height=canvas_height,
        width=canvas_width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images[0]
    image = restore_output_size(image, size_info, args.restore_to_original_size)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


if __name__ == "__main__":
    main()
