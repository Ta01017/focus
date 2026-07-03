#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from diffusers import SanaSprintPipeline

from dof_utils import (
    add_metadata_args,
    add_pretrained_args,
    prepare_inference_images,
    pretrained_kwargs,
    restore_output_size,
)
from sana_dof import ConditionedSanaTransformer, create_condition_adapter, encode_condition_images


def parse_args():
    parser = argparse.ArgumentParser(description="SANA-Sprint 双图景深融合推理。")
    parser.add_argument("--model", default=None)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--image_a", required=True)
    parser.add_argument("--image_b", required=True)
    parser.add_argument("--focus_a", default=None)
    parser.add_argument("--focus_b", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="a photorealistic all-in-focus photograph")
    parser.add_argument("--height", type=int, default=None, help="Compatibility check only; must equal A height.")
    parser.add_argument("--width", type=int, default=None, help="Compatibility check only; must equal A width.")
    parser.add_argument("--max_pixels", type=int, default=None, help="Safety limit; never triggers resizing.")
    parser.add_argument("--size_divisor", type=int, default=32)
    parser.add_argument("--aspect_ratio_tolerance", type=float, default=0.01)
    parser.add_argument("--downscale_if_exceeds_max_pixels", action="store_true")
    restore_group = parser.add_mutually_exclusive_group()
    restore_group.add_argument("--restore_to_original_size", dest="restore_to_original_size", action="store_true")
    restore_group.add_argument("--no_restore_to_original_size", dest="restore_to_original_size", action="store_false")
    parser.set_defaults(restore_to_original_size=True)
    parser.add_argument("--steps", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adapter_hidden_channels", type=int, default=None)
    add_metadata_args(parser)
    add_pretrained_args(parser)
    return parser.parse_args()


def load_focus(source, height=None, width=None):
    image = source if isinstance(source, Image.Image) else Image.open(source)
    image = image.convert("L")
    if height is not None and width is not None and image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0)


def load_sana_adapter_pipeline(args, pipeline_class=SanaSprintPipeline):
    config_path = Path(args.adapter).with_name("adapter_config.json")
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    model_id = args.model or config.get(
        "base_model", "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers"
    )
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipe = pipeline_class.from_pretrained(
        model_id, torch_dtype=dtype, **pretrained_kwargs(args)
    ).to("cuda")
    pipe.vae.to(dtype=torch.float32)
    adapter_type = config.get("adapter_type", "ab")
    hidden_channels = args.adapter_hidden_channels or config.get("hidden_channels", 128)
    adapter = create_condition_adapter(adapter_type, pipe.transformer.config.in_channels, hidden_channels)
    adapter.load_state_dict(load_file(args.adapter), strict=True)
    adapter.to(device="cuda", dtype=dtype).eval()
    transformer = ConditionedSanaTransformer(pipe.transformer, adapter)
    pipe.transformer = transformer
    return pipe, transformer, config


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("该推理脚本需要 CUDA。")
    pipe, transformer, config = load_sana_adapter_pipeline(args)
    image_a = Image.open(args.image_a).convert("RGB")
    image_b = Image.open(args.image_b).convert("RGB")
    named_images = {"a": image_a, "b": image_b}
    if config.get("adapter_type", "ab") == "ab_focus":
        if args.focus_a is None or args.focus_b is None:
            raise ValueError("ab_focus adapter 推理必须提供 --focus_a 和 --focus_b。")
        named_images["focus_a"] = Image.open(args.focus_a)
        named_images["focus_b"] = Image.open(args.focus_b)
    prepared, size_info = prepare_inference_images(
        named_images,
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
    focus_a = None
    focus_b = None
    if config.get("adapter_type", "ab") == "ab_focus":
        focus_a = load_focus(prepared["focus_a"]).to("cuda")
        focus_b = load_focus(prepared["focus_b"]).to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    with transformer.use_condition(cond_a, cond_b, focus_a, focus_b):
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
