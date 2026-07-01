#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file

from diffusers import SanaSprintImg2ImgPipeline

from sana_dof import ConditionedSanaTransformer, DualImageConditionAdapter, encode_condition_images


def parse_args():
    parser = argparse.ArgumentParser(description="SANA-Sprint img2img A/B fusion inference.")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--image_a", required=True)
    parser.add_argument("--image_b", required=True)
    parser.add_argument("--init_image", default=None, help="Defaults to A; may be a preliminary fusion image.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="a photorealistic all-in-focus photograph")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--strength", type=float, default=0.75)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_pipeline(adapter_path, model_id, dtype):
    config_path = adapter_path.with_name("adapter_config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    pipe = SanaSprintImg2ImgPipeline.from_pretrained(
        model_id or config["base_model"], torch_dtype=dtype
    ).to("cuda")
    pipe.vae.to(dtype=torch.float32)
    adapter = DualImageConditionAdapter(pipe.transformer.config.in_channels, config["hidden_channels"])
    adapter.load_state_dict(load_file(adapter_path), strict=True)
    adapter.to(device="cuda", dtype=dtype).eval()
    transformer = ConditionedSanaTransformer(pipe.transformer, adapter)
    pipe.transformer = transformer
    return pipe, transformer, config


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This reference script requires CUDA.")
    if args.height % 32 or args.width % 32 or not 0 < args.strength <= 1:
        raise ValueError("Dimensions must be divisible by 32 and strength must be in (0, 1].")
    if int(args.steps * args.strength) < 1:
        raise ValueError(
            "steps * strength must select at least one denoising step; one-step img2img needs strength=1."
        )
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipe, transformer, _ = load_pipeline(Path(args.adapter), args.model, dtype)
    image_a = Image.open(args.image_a).convert("RGB")
    image_b = Image.open(args.image_b).convert("RGB")
    init_image = image_a if args.init_image is None else Image.open(args.init_image).convert("RGB")
    cond_a, cond_b = encode_condition_images(
        pipe, image_a, image_b, args.height, args.width, torch.device("cuda")
    )
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    with transformer.use_condition(cond_a, cond_b):
        image = pipe(
            prompt=args.prompt,
            image=init_image,
            strength=args.strength,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            intermediate_timesteps=1.3 if args.steps == 2 else None,
            guidance_scale=args.guidance_scale,
            generator=generator,
            use_resolution_binning=False,
        ).images[0]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


if __name__ == "__main__":
    main()
