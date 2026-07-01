#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from infer_sana_controlnet import load_pipeline
from metadata import load_metadata, require_keys, resolve_data_path
from sana_dof import encode_condition_images


def parse_args():
    parser = argparse.ArgumentParser(description="Batch ordinary SANA + focus ControlNet inference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset_metadata_path", required=True)
    parser.add_argument("--dataset_base_path", default=".")
    parser.add_argument("--edit_key", default="edit_image")
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
        Path(args.checkpoint), args.model, dtype, args.conditioning_scale
    )
    control_index = config["control_index"] if args.control_index is None else args.control_index
    records = load_metadata(args.dataset_metadata_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for offset in range(0, len(records), args.batch_size):
        chunk = records[offset : offset + args.batch_size]
        images_a, images_b, controls, prompts, generators = [], [], [], [], []
        for local_index, record in enumerate(chunk):
            index = offset + local_index
            require_keys(record, (args.edit_key,), index)
            edits = record[args.edit_key]
            if not isinstance(edits, list) or len(edits) <= control_index:
                raise ValueError(f"Sample {index} must contain A, B, and control index {control_index}.")
            images_a.append(Image.open(resolve_data_path(edits[0], args.dataset_base_path)).convert("RGB"))
            images_b.append(Image.open(resolve_data_path(edits[1], args.dataset_base_path)).convert("RGB"))
            controls.append(
                Image.open(resolve_data_path(edits[control_index], args.dataset_base_path)).convert("L").convert("RGB")
            )
            prompts.append(record.get("prompt") or args.prompt)
            generators.append(torch.Generator(device="cuda").manual_seed(int(record.get("seed", args.seed + index))))
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
        for local_index, (record, image) in enumerate(zip(chunk, images)):
            index = offset + local_index
            destination = output_dir / f"{index:06d}.png"
            image.save(destination)
            result = dict(record)
            result["generated_image"] = str(destination)
            results.append(result)
            print(destination)
    (output_dir / "metadata_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
