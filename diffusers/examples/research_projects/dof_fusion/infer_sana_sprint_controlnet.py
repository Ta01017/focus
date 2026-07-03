#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from diffusers import SanaControlNetModel, SanaSprintPipeline

from dof_utils import (
    add_metadata_args,
    add_pretrained_args,
    prepare_inference_images,
    pretrained_kwargs,
    restore_output_size,
)
from sana_dof import DualImageConditionAdapter, encode_condition_images
from sana_sprint_controlnet import SanaSprintFocusControlNetTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="SANA-Sprint A/B fusion with a focus-map ControlNet.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--image_a", required=True)
    parser.add_argument("--image_b", required=True)
    parser.add_argument("--focus_map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="a photorealistic all-in-focus photograph")
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
    parser.add_argument("--steps", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--conditioning_scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    add_metadata_args(parser)
    add_pretrained_args(parser)
    return parser.parse_args()


def load_focus_map(source, height=None, width=None) -> torch.Tensor:
    focus = source if isinstance(source, Image.Image) else Image.open(source)
    focus = focus.convert("L")
    if height is not None and width is not None and focus.size != (width, height):
        focus = focus.resize((width, height), resample=Image.Resampling.BILINEAR)
    focus = torch.from_numpy(np.asarray(focus, dtype=np.float32) / 255.0)
    return focus.unsqueeze(0).unsqueeze(0)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This reference script currently requires CUDA.")

    checkpoint = Path(args.checkpoint)
    config = json.loads((checkpoint / "controlnet_config.json").read_text(encoding="utf-8"))
    model_id = args.model or config["base_model"]
    conditioning_scale = (
        config.get("conditioning_scale", 1.0) if args.conditioning_scale is None else args.conditioning_scale
    )
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipe = SanaSprintPipeline.from_pretrained(
        model_id, torch_dtype=dtype, **pretrained_kwargs(args)
    ).to("cuda")
    pipe.vae.to(dtype=torch.float32)

    controlnet = SanaControlNetModel.from_pretrained(
        checkpoint / "controlnet", torch_dtype=dtype, **pretrained_kwargs(args)
    ).to("cuda")
    adapter = DualImageConditionAdapter(
        latent_channels=pipe.transformer.config.in_channels,
        hidden_channels=config["hidden_channels"],
    )
    adapter.load_state_dict(load_file(checkpoint / "adapter.safetensors"), strict=True)
    adapter.to(device="cuda", dtype=dtype).eval()
    controlnet.eval()
    conditioned_transformer = SanaSprintFocusControlNetTransformer(
        transformer=pipe.transformer,
        controlnet=controlnet,
        adapter=adapter,
        conditioning_scale=conditioning_scale,
    )
    pipe.transformer = conditioned_transformer

    image_a = Image.open(args.image_a).convert("RGB")
    image_b = Image.open(args.image_b).convert("RGB")
    prepared, size_info = prepare_inference_images(
        {"a": image_a, "b": image_b, "focus": Image.open(args.focus_map)},
        args.height,
        args.width,
        args.max_pixels,
        args.size_divisor,
        args.aspect_ratio_tolerance,
        args.downscale_if_exceeds_max_pixels,
    )
    canvas_width, canvas_height = size_info["canvas_size"]
    cond_a_latents, cond_b_latents = encode_condition_images(
        pipe, prepared["a"], prepared["b"], canvas_height, canvas_width, torch.device("cuda")
    )
    focus_map = load_focus_map(prepared["focus"]).to("cuda")

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    with conditioned_transformer.use_conditions(cond_a_latents, cond_b_latents, focus_map):
        image = pipe(
            prompt=args.prompt,
            height=canvas_height,
            width=canvas_width,
            num_inference_steps=args.steps,
            intermediate_timesteps=1.3 if args.steps == 2 else None,
            guidance_scale=args.guidance_scale,
            generator=generator,
            use_resolution_binning=False,
        ).images[0]
    image = restore_output_size(image, size_info, args.restore_to_original_size)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


if __name__ == "__main__":
    main()
