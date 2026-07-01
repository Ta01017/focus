#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from diffusers import SanaSprintPipeline

from dof_utils import add_metadata_args, add_pretrained_args, pretrained_kwargs
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
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adapter_hidden_channels", type=int, default=None)
    add_metadata_args(parser)
    add_pretrained_args(parser)
    return parser.parse_args()


def load_focus(path, height, width):
    image = Image.open(path).convert("L").resize((width, height), Image.Resampling.BILINEAR)
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
    if args.height % 32 or args.width % 32:
        raise ValueError("--height 和 --width 必须能被 32 整除。")
    pipe, transformer, config = load_sana_adapter_pipeline(args)
    image_a = Image.open(args.image_a).convert("RGB")
    image_b = Image.open(args.image_b).convert("RGB")
    cond_a, cond_b = encode_condition_images(
        pipe, image_a, image_b, args.height, args.width, torch.device("cuda")
    )
    focus_a = None
    focus_b = None
    if config.get("adapter_type", "ab") == "ab_focus":
        if args.focus_a is None or args.focus_b is None:
            raise ValueError("ab_focus adapter 推理必须提供 --focus_a 和 --focus_b。")
        focus_a = load_focus(args.focus_a, args.height, args.width).to("cuda")
        focus_b = load_focus(args.focus_b, args.height, args.width).to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    with transformer.use_condition(cond_a, cond_b, focus_a, focus_b):
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
