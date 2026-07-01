#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file

from diffusers import SanaControlNetModel, SanaControlNetPipeline

from sana_controlnet_dof import SanaControlNetInferenceTransformer
from sana_dof import DualImageConditionAdapter, encode_condition_images


def parse_args():
    parser = argparse.ArgumentParser(description="Ordinary SANA + focus ControlNet A/B fusion inference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--image_a", required=True)
    parser.add_argument("--image_b", required=True)
    parser.add_argument("--focus_map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="a photorealistic all-in-focus photograph")
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--conditioning_scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_pipeline(checkpoint, model_id, dtype, conditioning_scale):
    config = json.loads((checkpoint / "controlnet_config.json").read_text(encoding="utf-8"))
    model_id = model_id or config["base_model"]
    scale = config.get("conditioning_scale", 1.0) if conditioning_scale is None else conditioning_scale
    controlnet = SanaControlNetModel.from_pretrained(checkpoint / "controlnet", torch_dtype=dtype)
    pipe = SanaControlNetPipeline.from_pretrained(
        model_id, controlnet=controlnet, torch_dtype=dtype
    ).to("cuda")
    pipe.vae.to(dtype=torch.float32)
    adapter = DualImageConditionAdapter(pipe.transformer.config.in_channels, config["hidden_channels"])
    adapter.load_state_dict(load_file(checkpoint / "adapter.safetensors"), strict=True)
    adapter.to(device="cuda", dtype=dtype).eval()
    transformer = SanaControlNetInferenceTransformer(pipe.transformer, adapter)
    pipe.transformer = transformer
    return pipe, transformer, scale, config


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This reference script requires CUDA.")
    if args.height % 32 or args.width % 32:
        raise ValueError("--height and --width must be divisible by 32.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipe, transformer, scale, _ = load_pipeline(
        Path(args.checkpoint), args.model, dtype, args.conditioning_scale
    )
    image_a = Image.open(args.image_a).convert("RGB")
    image_b = Image.open(args.image_b).convert("RGB")
    focus_map = Image.open(args.focus_map).convert("L").convert("RGB")
    cond_a, cond_b = encode_condition_images(
        pipe, image_a, image_b, args.height, args.width, torch.device("cuda")
    )
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    with transformer.use_condition(cond_a, cond_b):
        image = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            control_image=focus_map,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=scale,
            generator=generator,
            use_resolution_binning=False,
        ).images[0]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


if __name__ == "__main__":
    main()
