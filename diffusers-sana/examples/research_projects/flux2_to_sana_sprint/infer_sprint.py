#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file

from diffusers import SanaSprintPipeline

from conditioning import ConditionedSanaTransformer, MultiImageConditionAdapter, encode_condition_batch


def parse_args():
    parser = argparse.ArgumentParser(description="Run a cross-distilled Sana-Sprint adapter.")
    parser.add_argument("--model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--condition_images", nargs="+", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--num_inference_steps", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    config = json.loads((checkpoint / "condition_adapter.json").read_text(encoding="utf-8"))
    expected = config["num_condition_images"]
    if len(args.condition_images) != expected:
        raise ValueError(f"This checkpoint requires {expected} condition images, got {len(args.condition_images)}.")

    pipe = SanaSprintPipeline.from_pretrained(
        args.model or config["base_model"], torch_dtype=torch.bfloat16
    )
    pipe.load_lora_weights(checkpoint)
    adapter = MultiImageConditionAdapter(
        config["latent_channels"], config["num_condition_images"], config["hidden_channels"]
    )
    adapter.load_state_dict(load_file(checkpoint / "condition_adapter.safetensors"), strict=True)
    adapter.to(args.device, dtype=torch.float32).eval()
    pipe.transformer = ConditionedSanaTransformer(pipe.transformer, adapter)
    pipe.to(args.device)
    pipe.vae.to(args.device, dtype=torch.float32)

    height = args.height or config["resolution"]
    width = args.width or config["resolution"]
    images = [Image.open(path).convert("RGB") for path in args.condition_images]
    pixels = [pipe.image_processor.preprocess(image, height=height, width=width)[0] for image in images]
    pixels = torch.stack(pixels).unsqueeze(0).to(args.device, dtype=torch.float32)
    with torch.no_grad():
        condition_latents = encode_condition_batch(pipe.vae, pixels)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    with torch.no_grad(), pipe.transformer.use_condition(condition_latents):
        image = pipe(
            prompt=args.prompt,
            height=height,
            width=width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            use_resolution_binning=False,
        ).images[0]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)


if __name__ == "__main__":
    main()
