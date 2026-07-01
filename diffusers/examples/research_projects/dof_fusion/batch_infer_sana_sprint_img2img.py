#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from dof_utils import (
    add_metadata_args,
    add_pretrained_args,
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
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
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
    if args.batch_size < 1 or args.height % 32 or args.width % 32 or not 0 < args.strength <= 1:
        raise ValueError("batch size、尺寸或 strength 非法。")
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
            focus_a_items, focus_b_items = [], []
            for index, record, _, _ in chunk:
                edits = record[args.edit_key]
                required = 4 if adapter_type == "ab_focus" else 2
                if not isinstance(edits, list) or len(edits) < required:
                    raise ValueError(f"Sample {index} edit_image 至少需要 {required} 项。")
                image_a = Image.open(resolve_data_path(edits[0], args.dataset_base_path)).convert("RGB")
                images_a.append(image_a)
                images_b.append(Image.open(resolve_data_path(edits[1], args.dataset_base_path)).convert("RGB"))
                init_path = edits[0] if args.init_key is None else record[args.init_key]
                init_images.append(Image.open(resolve_data_path(init_path, args.dataset_base_path)).convert("RGB"))
                prompts.append(sample_prompt(record, args.prompt_key, args.prompt))
                seed = int(record.get(args.seed_key, args.seed + index))
                generators.append(torch.Generator(device="cuda").manual_seed(seed))
                if adapter_type == "ab_focus":
                    focus_a_items.append(
                        load_focus(resolve_data_path(edits[2], args.dataset_base_path), args.height, args.width)
                    )
                    focus_b_items.append(
                        load_focus(resolve_data_path(edits[3], args.dataset_base_path), args.height, args.width)
                    )
            cond_a, cond_b = encode_condition_images(
                pipe, images_a, images_b, args.height, args.width, torch.device("cuda")
            )
            focus_a = torch.cat(focus_a_items).to("cuda") if focus_a_items else None
            focus_b = torch.cat(focus_b_items).to("cuda") if focus_b_items else None
            with transformer.use_condition(cond_a, cond_b, focus_a, focus_b):
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
