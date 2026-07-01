#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from diffusers import Flux2KleinPipeline


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
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu_offload", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.height % 16 or args.width % 16:
        raise ValueError("--height and --width must be divisible by 16.")

    image_a = Image.open(args.image_a).convert("RGB")
    image_b = Image.open(args.image_b).convert("RGB")
    if image_a.size != image_b.size:
        print(f"Warning: input A/B sizes differ before preprocessing: {image_a.size}, {image_b.size}.")

    training_config = {}
    if args.lora is not None:
        lora_path = Path(args.lora)
        config_path = (lora_path if lora_path.is_dir() else lora_path.parent) / "training_config.json"
        if config_path.exists():
            training_config = json.loads(config_path.read_text(encoding="utf-8"))
    model = args.model or training_config.get("base_model", "black-forest-labs/FLUX.2-klein-4B")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    pipe = Flux2KleinPipeline.from_pretrained(model, torch_dtype=dtype)
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
        image=[image_a, image_b],
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images[0]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


if __name__ == "__main__":
    main()
