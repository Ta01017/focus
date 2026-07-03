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
    restore_output_size,
    sample_output_path,
    sample_prompt,
    select_records,
)
from infer_sana_sprint import load_focus
from infer_sana_sprint_img2img import load_pipeline
from metadata import load_metadata, require_keys, resolve_data_path
from sana_dof import encode_condition_images


def parse_args():
    parser = argparse.ArgumentParser(description="批量 SANA-Sprint img2img 景深融合推理。")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--init_key", default=None)
    parser.add_argument("--prompt", default="a photorealistic all-in-focus photograph")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--height", type=int, default=None, help="Compatibility check; must match every A image.")
    parser.add_argument("--width", type=int, default=None, help="Compatibility check; must match every A image.")
    parser.add_argument("--max_pixels", type=int, default=None, help="Safety limit; never triggers resizing.")
    parser.add_argument("--size_divisor", type=int, default=32)
    parser.add_argument("--aspect_ratio_tolerance", type=float, default=0.01)
    parser.add_argument("--downscale_if_exceeds_max_pixels", action="store_true")
    restore_group = parser.add_mutually_exclusive_group()
    restore_group.add_argument("--restore_to_original_size", dest="restore_to_original_size", action="store_true")
    restore_group.add_argument("--no_restore_to_original_size", dest="restore_to_original_size", action="store_false")
    parser.set_defaults(restore_to_original_size=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--strength", type=float, default=0.75)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adapter_hidden_channels", type=int, default=None)
    add_metadata_args(parser, metadata_required=True)
    add_pretrained_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("该推理脚本需要 CUDA。")
    if args.batch_size < 1 or not 0 < args.strength <= 1:
        raise ValueError("batch size、尺寸或 strength 非法。")
    if (args.height is None) != (args.width is None):
        raise ValueError("--height and --width must be provided together or both omitted.")
    if args.height is None and args.batch_size != 1:
        raise ValueError("Dynamic-resolution batch inference requires --batch_size 1.")
    if int(args.steps * args.strength) < 1:
        raise ValueError("steps * strength 必须至少选择一个去噪步。")
    pipe, transformer, config = load_pipeline(args)
    adapter_type = config.get("adapter_type", "ab")
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
            prepared_items, size_infos, prompts, generators = [], [], [], []
            for index, record, _, _ in chunk:
                edits = record[args.edit_key]
                required = 4 if adapter_type == "ab_focus" else 2
                if not isinstance(edits, list) or len(edits) < required:
                    raise ValueError(f"Sample {index} edit_image 至少需要 {required} 项。")
                image_a = Image.open(resolve_data_path(edits[0], args.dataset_base_path)).convert("RGB")
                image_b = Image.open(resolve_data_path(edits[1], args.dataset_base_path)).convert("RGB")
                init_path = edits[0] if args.init_key is None else record[args.init_key]
                init_image = Image.open(resolve_data_path(init_path, args.dataset_base_path)).convert("RGB")
                named_images = {"a": image_a, "b": image_b, "init": init_image}
                prompts.append(sample_prompt(record, args.prompt_key, args.prompt))
                seed = int(record.get(args.seed_key, args.seed + index))
                generators.append(torch.Generator(device="cuda").manual_seed(seed))
                if adapter_type == "ab_focus":
                    named_images["focus_a"] = Image.open(resolve_data_path(edits[2], args.dataset_base_path))
                    named_images["focus_b"] = Image.open(resolve_data_path(edits[3], args.dataset_base_path))
                prepared, size_info = prepare_inference_images(
                    named_images,
                    args.height,
                    args.width,
                    args.max_pixels,
                    args.size_divisor,
                    args.aspect_ratio_tolerance,
                    args.downscale_if_exceeds_max_pixels,
                )
                prepared_items.append(prepared)
                size_infos.append(size_info)
            canvas_width, canvas_height = size_infos[0]["canvas_size"]
            cond_a, cond_b = encode_condition_images(
                pipe,
                [item["a"] for item in prepared_items],
                [item["b"] for item in prepared_items],
                canvas_height,
                canvas_width,
                torch.device("cuda"),
            )
            focus_a = None
            focus_b = None
            if adapter_type == "ab_focus":
                focus_a = torch.cat([load_focus(item["focus_a"]) for item in prepared_items]).to("cuda")
                focus_b = torch.cat([load_focus(item["focus_b"]) for item in prepared_items]).to("cuda")
            with transformer.use_condition(cond_a, cond_b, focus_a, focus_b):
                images = pipe(
                    prompt=prompts,
                    image=[item["init"] for item in prepared_items],
                    strength=args.strength,
                    height=canvas_height,
                    width=canvas_width,
                    num_inference_steps=args.steps,
                    intermediate_timesteps=1.3 if args.steps == 2 else None,
                    guidance_scale=args.guidance_scale,
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
