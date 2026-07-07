#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file

from diffusers import SanaPipeline

from dof_utils import add_pretrained_args, prepare_inference_images, pretrained_kwargs, restore_output_size
from sana_dof import ConditionedSanaTransformer, DualImageConditionAdapter, encode_condition_images


def parse_args():
    parser = argparse.ArgumentParser(description="Ordinary SANA + A/B adapter-only DOF fusion inference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--image_a", required=True)
    parser.add_argument("--image_b", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="a photorealistic all-in-focus photograph")
    parser.add_argument("--negative_prompt", default="")
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
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=0)
    add_pretrained_args(parser)
    return parser.parse_args()


def load_pipeline(checkpoint, model_id, dtype, pretrained_options):
    config = json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
    model_id = model_id or config["base_model"]
    pipe = SanaPipeline.from_pretrained(model_id, torch_dtype=dtype, **pretrained_options).to("cuda")
    pipe.vae.to(dtype=torch.float32)
    adapter = DualImageConditionAdapter(pipe.transformer.config.in_channels, config["hidden_channels"])
    adapter.load_state_dict(load_file(checkpoint / "adapter.safetensors"), strict=True)
    adapter.to(device="cuda", dtype=dtype).eval()
    transformer = ConditionedSanaTransformer(pipe.transformer, adapter)
    pipe.transformer = transformer
    return pipe, transformer, config


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This reference script requires CUDA.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipe, transformer, _ = load_pipeline(Path(args.checkpoint), args.model, dtype, pretrained_kwargs(args))
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
    cond_a, cond_b = encode_condition_images(
        pipe, prepared["a"], prepared["b"], canvas_height, canvas_width, torch.device("cuda")
    )
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    with transformer.use_condition(cond_a, cond_b):
        image = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=canvas_height,
            width=canvas_width,
            num_inference_steps=args.steps,
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
