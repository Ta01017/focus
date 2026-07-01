#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from dof_utils import (
    add_metadata_args,
    add_pretrained_args,
    pretrained_kwargs,
    sample_output_path,
    sample_prompt,
    select_records,
)
from flux2_controlnet import focus_map_to_tokens
from infer_flux2_klein import DEFAULT_PROMPT
from infer_flux2_klein_controlnet import load_pipeline
from metadata import load_metadata, require_keys, resolve_data_path


def parse_args():
    parser = argparse.ArgumentParser(description="Batch FLUX.2 Klein focus ControlNet inference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default=None)
    add_metadata_args(parser, metadata_required=True)
    add_pretrained_args(parser)
    parser.add_argument("--control_index", type=int, choices=(2, 3), default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--conditioning_scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This reference script requires CUDA.")
    if args.height % 16 or args.width % 16:
        raise ValueError("--height and --width must be divisible by 16.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipe, transformer, config = load_pipeline(
        Path(args.checkpoint), args.model, dtype, args.conditioning_scale, pretrained_kwargs(args)
    )
    control_index = config["control_index"] if args.control_index is None else args.control_index
    records = select_records(load_metadata(args.dataset_metadata_path), args.start_index, args.max_samples)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latent_height = args.height // (pipe.vae_scale_factor * 2)
    latent_width = args.width // (pipe.vae_scale_factor * 2)
    results = []
    for index, record in records:
        require_keys(record, (args.edit_key,), index)
        destination = sample_output_path(record, index, args.edit_key, args.id_key, output_dir)
        result = dict(record)
        result[args.result_key] = str(destination)
        if args.skip_existing and destination.exists():
            results.append(result)
            continue
        try:
            edits = record[args.edit_key]
            if not isinstance(edits, list) or len(edits) <= control_index:
                raise ValueError(f"Sample {index} must contain A, B, and control index {control_index}.")
            image_a = Image.open(resolve_data_path(edits[0], args.dataset_base_path)).convert("RGB")
            image_b = Image.open(resolve_data_path(edits[1], args.dataset_base_path)).convert("RGB")
            focus = Image.open(resolve_data_path(edits[control_index], args.dataset_base_path)).convert("L")
            focus_tokens = focus_map_to_tokens(focus, latent_height, latent_width, torch.device("cuda"))
            seed = int(record.get(args.seed_key, args.seed + index))
            generator = torch.Generator(device="cuda").manual_seed(seed)
            with transformer.use_focus_condition(focus_tokens):
                image = pipe(
                    image=[image_a, image_b],
                    prompt=sample_prompt(record, args.prompt_key, args.prompt),
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                ).images[0]
            image.save(destination)
        except Exception as error:
            if not args.continue_on_error:
                raise
            result["error"] = str(error)
        results.append(result)
        print(destination)
    (output_dir / "metadata_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
