#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from infer_sana_sprint_img2img import load_pipeline
from metadata import load_metadata, require_keys, resolve_data_path
from sana_dof import encode_condition_images


def parse_args():
    parser = argparse.ArgumentParser(description="Batch SANA-Sprint img2img A/B fusion inference.")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset_metadata_path", required=True)
    parser.add_argument("--dataset_base_path", default=".")
    parser.add_argument("--edit_key", default="edit_image")
    parser.add_argument("--init_key", default=None, help="Optional metadata field for a preliminary fusion image.")
    parser.add_argument("--prompt", default="a photorealistic all-in-focus photograph")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--strength", type=float, default=0.75)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This reference script requires CUDA.")
    if args.batch_size < 1 or args.height % 32 or args.width % 32 or not 0 < args.strength <= 1:
        raise ValueError("Invalid batch size, dimensions, or strength.")
    if int(args.steps * args.strength) < 1:
        raise ValueError(
            "steps * strength must select at least one denoising step; one-step img2img needs strength=1."
        )
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipe, transformer, _ = load_pipeline(Path(args.adapter), args.model, dtype)
    records = load_metadata(args.dataset_metadata_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for offset in range(0, len(records), args.batch_size):
        chunk = records[offset : offset + args.batch_size]
        images_a, images_b, init_images, prompts, generators = [], [], [], [], []
        for local_index, record in enumerate(chunk):
            index = offset + local_index
            require_keys(record, (args.edit_key,), index)
            edits = record[args.edit_key]
            if not isinstance(edits, list) or len(edits) < 2:
                raise ValueError(f"Sample {index} must contain at least A and B.")
            image_a = Image.open(resolve_data_path(edits[0], args.dataset_base_path)).convert("RGB")
            images_a.append(image_a)
            images_b.append(Image.open(resolve_data_path(edits[1], args.dataset_base_path)).convert("RGB"))
            if args.init_key is None:
                init_images.append(image_a)
            else:
                require_keys(record, (args.init_key,), index)
                init_images.append(
                    Image.open(resolve_data_path(record[args.init_key], args.dataset_base_path)).convert("RGB")
                )
            prompts.append(record.get("prompt") or args.prompt)
            generators.append(torch.Generator(device="cuda").manual_seed(int(record.get("seed", args.seed + index))))
        cond_a, cond_b = encode_condition_images(
            pipe, images_a, images_b, args.height, args.width, torch.device("cuda")
        )
        with transformer.use_condition(cond_a, cond_b):
            images = pipe(
                prompt=prompts,
                image=init_images,
                strength=args.strength,
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                intermediate_timesteps=1.3 if args.steps == 2 else None,
                guidance_scale=args.guidance_scale,
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
