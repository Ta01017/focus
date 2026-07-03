#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from dof_utils import (
    add_metadata_args,
    add_pretrained_args,
    prepare_inference_images,
    pretrained_kwargs,
    restore_output_size,
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
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=None)
    parser.add_argument("--size_divisor", type=int, default=32)
    parser.add_argument("--aspect_ratio_tolerance", type=float, default=0.01)
    parser.add_argument("--downscale_if_exceeds_max_pixels", action="store_true")
    restore_group = parser.add_mutually_exclusive_group()
    restore_group.add_argument("--restore_to_original_size", dest="restore_to_original_size", action="store_true")
    restore_group.add_argument("--no_restore_to_original_size", dest="restore_to_original_size", action="store_false")
    parser.set_defaults(restore_to_original_size=True)
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
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive.")
    if args.height is None and args.batch_size != 1:
        raise ValueError("Dynamic-resolution batch inference requires --batch_size 1.")
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
            images_a, images_b, controls, prompts, generators, size_infos = [], [], [], [], [], []
            for index, record, _, _ in chunk:
                edits = record[args.edit_key]
                if not isinstance(edits, list) or len(edits) <= control_index:
                    raise ValueError(f"Sample {index} must contain A, B, and control index {control_index}.")
                prepared, size_info = prepare_inference_images(
                    {
                        "a": Image.open(resolve_data_path(edits[0], args.dataset_base_path)).convert("RGB"),
                        "b": Image.open(resolve_data_path(edits[1], args.dataset_base_path)).convert("RGB"),
                        "focus": Image.open(resolve_data_path(edits[control_index], args.dataset_base_path)),
                    },
                    args.height,
                    args.width,
                    args.max_pixels,
                    args.size_divisor,
                    args.aspect_ratio_tolerance,
                    args.downscale_if_exceeds_max_pixels,
                )
                images_a.append(prepared["a"])
                images_b.append(prepared["b"])
                controls.append(prepared["focus"].convert("RGB"))
                size_infos.append(size_info)
                prompts.append(sample_prompt(record, args.prompt_key, args.prompt))
                seed = int(record.get(args.seed_key, args.seed + index))
                generators.append(torch.Generator(device="cuda").manual_seed(seed))
            canvas_width, canvas_height = size_infos[0]["canvas_size"]
            cond_a, cond_b = encode_condition_images(
                pipe, images_a, images_b, canvas_height, canvas_width, torch.device("cuda")
            )
            with transformer.use_condition(cond_a, cond_b):
                images = pipe(
                    prompt=prompts,
                    control_image=controls,
                    height=canvas_height,
                    width=canvas_width,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    controlnet_conditioning_scale=scale,
                    generator=generators,
                    use_resolution_binning=False,
                ).images
            for image, size_info, (_, _, destination, result) in zip(images, size_infos, chunk):
                restore_output_size(image, size_info, args.restore_to_original_size).save(destination)
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
