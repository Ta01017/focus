#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file

from diffusers import SanaSprintPipeline

from sana_dof import ConditionedSanaTransformer, DualImageConditionAdapter, encode_condition_images


def parse_args():
    parser = argparse.ArgumentParser(
        description="SANA-Sprint depth-of-field fusion with an external dual-image adapter."
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--image_a", required=True)
    parser.add_argument("--image_b", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="a photorealistic all-in-focus photograph")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adapter_hidden_channels", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This reference script currently requires CUDA.")
    if args.height % 32 or args.width % 32:
        raise ValueError("--height and --width must be divisible by 32 for SANA's autoencoder.")

    image_a = Image.open(args.image_a).convert("RGB")
    image_b = Image.open(args.image_b).convert("RGB")
    if image_a.size != image_b.size:
        print(f"Warning: input A/B sizes differ before preprocessing: {image_a.size}, {image_b.size}.")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    config_path = Path(args.adapter).with_name("adapter_config.json")
    adapter_config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    model = args.model or adapter_config.get(
        "base_model", "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers"
    )
    hidden_channels = args.adapter_hidden_channels or adapter_config.get("hidden_channels", 128)
    pipe = SanaSprintPipeline.from_pretrained(model, torch_dtype=dtype).to("cuda")
    pipe.vae.to(dtype=torch.float32)

    adapter = DualImageConditionAdapter(
        latent_channels=pipe.transformer.config.in_channels,
        hidden_channels=hidden_channels,
    )
    adapter.load_state_dict(load_file(args.adapter), strict=True)
    adapter.to(device="cuda", dtype=dtype).eval()

    conditioned_transformer = ConditionedSanaTransformer(pipe.transformer, adapter)
    pipe.transformer = conditioned_transformer
    cond_a_latents, cond_b_latents = encode_condition_images(
        pipe, image_a, image_b, args.height, args.width, torch.device("cuda")
    )

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    with conditioned_transformer.use_condition(cond_a_latents, cond_b_latents):
        image = pipe(
            prompt=args.prompt,
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
