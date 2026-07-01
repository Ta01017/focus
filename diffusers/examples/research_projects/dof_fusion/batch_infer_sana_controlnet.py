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
from infer_sana_controlnet import load_pipeline
from metadata import load_metadata, require_keys, resolve_data_path
from sana_dof import encode_condition_images


def parse_args():
    parser = argparse.ArgumentParser(description="Batch ordinary SANA + focus ControlNet inference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default=None)
    add_metadata_args(parser, metadata_required=True)
    add_pretrained_args(parser)
    parser.add_argument("--control_index", type=int, choices=(2, 3), default=None)
    parser.add_argument("--prompt", default="a photorealistic all-in-focus photograph")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--conditioning_scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This reference script requires CUDA.")
    if args.batch_size < 1 or args.height % 32 or args.width % 32:
        raise ValueError("Batch size must be positive and image dimensions divisible by 32.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipe, transformer, scale, config = load_pipeline(
        Path(args.checkpoint), args.model, dtype, args.conditioning_scale, pretrained_kwargs(args)
    )
    control_index = config["control_index"] if args.control_index is None else args.control_index
    records = select_records(load_metadata(args.dataset_metadata_path), args.start_index, args.max_samples)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    pending = []
    for index, record in records:
        require_keys(record, (args.edit_key,), index)
        destination = sample_output_path(record, index, args.edit_key, args.id_key, output_dir)
        result = dict(record)
        result[args.result_key] = str(destination)
        if args.skip_existing and destination.exists():
            results.append(result)
        else:
            pending.append((index, record, destination, result))

    for offset in range(0, len(pending), args.batch_size):
        chunk = pending[offset : offset + args.batch_size]
        try:
            images_a, images_b, controls, prompts, generators = [], [], [], [], []
            for index, record, _, _ in chunk:
                edits = record[args.edit_key]
                if not isinstance(edits, list) or len(edits) <= control_index:
                    raise ValueError(f"Sample {index} must contain A, B, and control index {control_index}.")
                images_a.append(Image.open(resolve_data_path(edits[0], args.dataset_base_path)).convert("RGB"))
                images_b.append(Image.open(resolve_data_path(edits[1], args.dataset_base_path)).convert("RGB"))
                controls.append(
                    Image.open(resolve_data_path(edits[control_index], args.dataset_base_path))
                    .convert("L")
                    .convert("RGB")
                )
                prompts.append(sample_prompt(record, args.prompt_key, args.prompt))
                seed = int(record.get(args.seed_key, args.seed + index))
                generators.append(torch.Generator(device="cuda").manual_seed(seed))
            cond_a, cond_b = encode_condition_images(
                pipe, images_a, images_b, args.height, args.width, torch.device("cuda")
            )
            with transformer.use_condition(cond_a, cond_b):
                images = pipe(
                    prompt=prompts,
                    control_image=controls,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    controlnet_conditioning_scale=scale,
                    generator=generators,
                    use_resolution_binning=False,
                ).images
            for image, (_, _, destination, result) in zip(images, chunk):
                image.save(destination)
                results.append(result)
                print(destination)
        except Exception as error:
            if not args.continue_on_error:
                raise
            for _, _, _, result in chunk:
                result["error"] = str(error)
                results.append(result)
    (output_dir / "metadata_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
