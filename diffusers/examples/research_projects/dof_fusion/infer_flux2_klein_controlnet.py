#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file

from diffusers import Flux2KleinPipeline

from dof_utils import (
    add_metadata_args,
    add_pretrained_args,
    prepare_inference_images,
    pretrained_kwargs,
    restore_output_size,
)
from flux2_controlnet import Flux2ControlNetTransformer, Flux2FocusControlNet, focus_map_to_tokens
from infer_flux2_klein import DEFAULT_PROMPT


def parse_args():
    parser = argparse.ArgumentParser(description="FLUX.2 Klein A/B fusion with a focus ControlNet branch.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--image_a", required=True)
    parser.add_argument("--image_b", required=True)
    parser.add_argument("--focus_map", required=True)
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
    parser.add_argument("--conditioning_scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    add_metadata_args(parser)
    add_pretrained_args(parser)
    return parser.parse_args()


def load_pipeline(checkpoint, model_id, dtype, conditioning_scale, pretrained_options):
    config = json.loads((checkpoint / "controlnet_config.json").read_text(encoding="utf-8"))
    pipe = Flux2KleinPipeline.from_pretrained(
        model_id or config["base_model"], torch_dtype=dtype, **pretrained_options
    ).to("cuda")
    controlnet = Flux2FocusControlNet(
        in_channels=config["in_channels"],
        inner_dim=config["inner_dim"],
        hidden_channels=config["hidden_channels"],
        num_layers=config["num_layers"],
        double_block_indices=tuple(config["double_block_indices"]),
        single_block_indices=tuple(config["single_block_indices"]),
    )
    controlnet.load_state_dict(load_file(checkpoint / "controlnet.safetensors"), strict=True)
    controlnet.to(device="cuda", dtype=torch.float32).eval()
    scale = config.get("conditioning_scale", 1.0) if conditioning_scale is None else conditioning_scale
    transformer = Flux2ControlNetTransformer(pipe.transformer, controlnet, scale)
    pipe.transformer = transformer
    return pipe, transformer, config


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This reference script requires CUDA.")
    if args.size_divisor % 16:
        raise ValueError("--size_divisor must be a multiple of 16 for FLUX.2 Klein.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipe, transformer, _ = load_pipeline(
        Path(args.checkpoint), args.model, dtype, args.conditioning_scale, pretrained_kwargs(args)
    )
    image_a = Image.open(args.image_a).convert("RGB")
    image_b = Image.open(args.image_b).convert("RGB")
    focus = Image.open(args.focus_map)
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
    focus_tokens = focus_map_to_tokens(prepared["focus"], latent_height, latent_width, torch.device("cuda"))
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    with transformer.use_focus_condition(focus_tokens):
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
