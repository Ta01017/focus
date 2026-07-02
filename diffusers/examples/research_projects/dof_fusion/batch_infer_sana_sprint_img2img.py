#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from dof_utils import (
    add_metadata_args,
    add_pretrained_args,
    prepare_dynamic_images,
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
    if args.batch_size != 1 or not 0 < args.strength <= 1:
        raise ValueError("batch size、尺寸或 strength 非法。")
    if (args.height is None) != (args.width is None):
        raise ValueError("--height and --width must be provided together or both omitted.")
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
            images_a, images_b, init_images, prompts, generators = [], [], [], [], []
            named_images = {}
            for index, record, _, _ in chunk:
                edits = record[args.edit_key]
                required = 4 if adapter_type == "ab_focus" else 2
                if not isinstance(edits, list) or len(edits) < required:
                    raise ValueError(f"Sample {index} edit_image 至少需要 {required} 项。")
                image_a = Image.open(resolve_data_path(edits[0], args.dataset_base_path)).convert("RGB")
                if args.height is not None and image_a.size != (args.width, args.height):
                    raise ValueError(f"Sample {index} A size must equal explicit --width/--height.")
                images_a.append(image_a)
                images_b.append(Image.open(resolve_data_path(edits[1], args.dataset_base_path)).convert("RGB"))
                init_path = edits[0] if args.init_key is None else record[args.init_key]
                init_images.append(Image.open(resolve_data_path(init_path, args.dataset_base_path)).convert("RGB"))
                named_images = {"a": images_a[-1], "b": images_b[-1], "init": init_images[-1]}
                prompts.append(sample_prompt(record, args.prompt_key, args.prompt))
                seed = int(record.get(args.seed_key, args.seed + index))
                generators.append(torch.Generator(device="cuda").manual_seed(seed))
                if adapter_type == "ab_focus":
                    named_images["focus_a"] = Image.open(resolve_data_path(edits[2], args.dataset_base_path))
                    named_images["focus_b"] = Image.open(resolve_data_path(edits[3], args.dataset_base_path))
            prepared, size_info = prepare_dynamic_images(
                named_images, args.max_pixels, args.size_divisor, args.aspect_ratio_tolerance
            )
            canvas_width, canvas_height = size_info["canvas_size"]
            cond_a, cond_b = encode_condition_images(
                pipe, prepared["a"], prepared["b"], canvas_height, canvas_width, torch.device("cuda")
            )
            focus_a = load_focus(prepared["focus_a"]).to("cuda") if "focus_a" in prepared else None
            focus_b = load_focus(prepared["focus_b"]).to("cuda") if "focus_b" in prepared else None
            with transformer.use_condition(cond_a, cond_b, focus_a, focus_b):
                images = pipe(
                    prompt=prompts,
                    image=prepared["init"],
                    strength=args.strength,
                    height=canvas_height,
                    width=canvas_width,
                    num_inference_steps=args.steps,
                    intermediate_timesteps=1.3 if args.steps == 2 else None,
                    guidance_scale=args.guidance_scale,
                    generator=generators,
                    use_resolution_binning=False,
                ).images
            for image, (_, _, destination, result) in zip(images, chunk):
                restore_output_size(image, size_info).save(destination)
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
