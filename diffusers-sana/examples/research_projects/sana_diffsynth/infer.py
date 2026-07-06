#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file

from diffusers import SanaPipeline, SanaSprintPipeline, SanaTransformer2DModel

from conditioning import ConditionedSanaTransformer, MultiImageConditionAdapter, encode_condition_batch


def load_conditioned_pipeline(model, adapter_path, pipeline_type=None, dtype=torch.bfloat16, device="cuda"):
    adapter_path = Path(adapter_path)
    config = json.loads((adapter_path / "condition_adapter.json").read_text(encoding="utf-8"))
    pipeline_type = pipeline_type or config["pipeline_type"]
    pipeline_class = SanaPipeline if pipeline_type == "sana" else SanaSprintPipeline
    if pipeline_type == "sana_sprint" and (adapter_path / "transformer").is_dir():
        transformer = SanaTransformer2DModel.from_pretrained(
            adapter_path, subfolder="transformer", torch_dtype=dtype
        )
        pipe = pipeline_class.from_pretrained(
            model or config["base_model"], transformer=transformer, torch_dtype=dtype
        )
    else:
        pipe = pipeline_class.from_pretrained(model or config["base_model"], torch_dtype=dtype)
        pipe.load_lora_weights(adapter_path)
    adapter = MultiImageConditionAdapter(
        config["latent_channels"], config["num_condition_images"], config["hidden_channels"]
    )
    adapter.load_state_dict(load_file(adapter_path / "condition_adapter.safetensors"), strict=True)
    adapter.to(device, dtype=torch.float32).eval()
    pipe.transformer = ConditionedSanaTransformer(pipe.transformer, adapter)
    pipe.to(device)
    pipe.vae.to(device=device, dtype=torch.float32)
    return pipe, config


@torch.no_grad()
def encode_images(pipe, image_paths, height, width, device):
    images = [Image.open(path).convert("RGB") for path in image_paths]
    pixels = [pipe.image_processor.preprocess(image, height=height, width=width)[0] for image in images]
    pixels = torch.stack(pixels).unsqueeze(0).to(device=device, dtype=torch.float32)
    return encode_condition_batch(pipe.vae, pixels)


@torch.no_grad()
def generate(pipe, config, image_paths, prompt, output, height, width, steps, guidance_scale, seed):
    expected = config["num_condition_images"]
    if len(image_paths) != expected:
        raise ValueError(f"This adapter requires {expected} condition images, got {len(image_paths)}.")
    condition_latents = encode_images(pipe, image_paths, height, width, pipe._execution_device)
    generator = torch.Generator(device=pipe._execution_device).manual_seed(seed)
    kwargs = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "generator": generator,
        "use_resolution_binning": False,
    }
    with pipe.transformer.use_condition(condition_latents):
        image = pipe(**kwargs).images[0]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return image


def parse_args():
    parser = argparse.ArgumentParser(description="Run a conditioned Sana or Sana Sprint adapter.")
    parser.add_argument("--model")
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--pipeline_type", choices=("sana", "sana_sprint"))
    parser.add_argument("--condition_images", nargs="+", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--num_inference_steps", type=int)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    pipe, config = load_conditioned_pipeline(
        args.model, args.adapter_path, args.pipeline_type, dtype=dtype, device=args.device
    )
    resolution = config["resolution"]
    pipeline_type = args.pipeline_type or config["pipeline_type"]
    steps = args.num_inference_steps or (2 if pipeline_type == "sana_sprint" else 20)
    generate(
        pipe,
        config,
        args.condition_images,
        args.prompt,
        args.output,
        args.height or resolution,
        args.width or resolution,
        steps,
        args.guidance_scale,
        args.seed,
    )


if __name__ == "__main__":
    main()
