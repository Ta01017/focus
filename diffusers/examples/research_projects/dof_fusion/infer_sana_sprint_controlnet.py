#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from diffusers import SanaControlNetModel, SanaSprintPipeline

from dof_utils import add_metadata_args, add_pretrained_args, pretrained_kwargs
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
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--conditioning_scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    add_metadata_args(parser)
    add_pretrained_args(parser)
    return parser.parse_args()


def load_focus_map(path: str, height: int, width: int) -> torch.Tensor:
    focus = Image.open(path).convert("L")
    focus = focus.resize((width, height), resample=Image.Resampling.BILINEAR)
    focus = torch.from_numpy(np.asarray(focus, dtype=np.float32) / 255.0)
    return focus.unsqueeze(0).unsqueeze(0)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This reference script currently requires CUDA.")
    if args.height % 32 or args.width % 32:
        raise ValueError("--height and --width must be divisible by 32.")

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
    if image_a.size != image_b.size:
        print(f"Warning: input A/B sizes differ before preprocessing: {image_a.size}, {image_b.size}.")
    cond_a_latents, cond_b_latents = encode_condition_images(
        pipe, image_a, image_b, args.height, args.width, torch.device("cuda")
    )
    focus_map = load_focus_map(args.focus_map, args.height, args.width).to("cuda")

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    with conditioned_transformer.use_conditions(cond_a_latents, cond_b_latents, focus_map):
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
