#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file

from diffusers import SanaControlNetModel, SanaSprintPipeline

from dof_utils import (
    add_metadata_args,
    add_pretrained_args,
    prepare_inference_images,
    pretrained_kwargs,
    restore_output_size,
)
from infer_sana_sprint_controlnet import load_focus_map
from metadata import load_metadata, require_keys, resolve_data_path
from sana_dof import DualImageConditionAdapter, encode_condition_images
from sana_sprint_controlnet import SanaSprintFocusControlNetTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="Batch SANA-Sprint focus-map ControlNet inference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset_metadata_path", required=True)
    parser.add_argument("--dataset_base_path", default=".")
    parser.add_argument("--edit_key", default="edit_image")
    parser.add_argument("--control_index", type=int, default=None)
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--prompt", default="a photorealistic all-in-focus photograph")
    parser.add_argument("--id_key", default="id")
    parser.add_argument("--seed_key", default="seed")
    parser.add_argument("--result_key", default="generated_image")
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
    parser.add_argument("--steps", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--conditioning_scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    add_metadata_args(parser, metadata_required=True)
    add_pretrained_args(parser)
    return parser.parse_args()


def output_path(record: dict, index: int, edit_key: str, id_key: str, output_dir: str) -> Path:
    sample_id = str(record.get(id_key, Path(record[edit_key][0]).stem))
    sample_id = sample_id.replace("/", "_").replace("\\", "_")
    return Path(output_dir) / f"{index:06d}_{sample_id}.png"


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This reference script currently requires CUDA.")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be at least 1.")
    if args.height is None and args.batch_size != 1:
        raise ValueError("Dynamic-resolution batch inference requires --batch_size 1.")

    checkpoint = Path(args.checkpoint)
    config = json.loads((checkpoint / "controlnet_config.json").read_text(encoding="utf-8"))
    model_id = args.model or config["base_model"]
    control_index = config["control_index"] if args.control_index is None else args.control_index
    conditioning_scale = (
        config.get("conditioning_scale", 1.0) if args.conditioning_scale is None else args.conditioning_scale
    )
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipe = SanaSprintPipeline.from_pretrained(
        model_id, torch_dtype=dtype, **pretrained_kwargs(args)
    ).to("cuda")
    pipe.vae.to(dtype=torch.float32)
    controlnet = SanaControlNetModel.from_pretrained(
        checkpoint / "controlnet", torch_dtype=dtype, **pretrained_kwargs(args)
    ).to("cuda")
    adapter = DualImageConditionAdapter(pipe.transformer.config.in_channels, config["hidden_channels"])
    adapter.load_state_dict(load_file(checkpoint / "adapter.safetensors"), strict=True)
    adapter.to(device="cuda", dtype=dtype).eval()
    controlnet.eval()
    conditioned_transformer = SanaSprintFocusControlNetTransformer(
        pipe.transformer, controlnet, adapter, conditioning_scale=conditioning_scale
    )
    pipe.transformer = conditioned_transformer

    records = load_metadata(args.dataset_metadata_path)
    end = None if args.max_samples is None else args.start_index + args.max_samples
    selected = list(enumerate(records[args.start_index : end], start=args.start_index))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    results = []
    pending = []
    for index, record in selected:
        require_keys(record, (args.edit_key,), index)
        edit_images = record[args.edit_key]
        if not isinstance(edit_images, list) or len(edit_images) <= control_index:
            raise ValueError(
                f"Sample {index} field {args.edit_key!r} must contain A, B, and control index {control_index}."
            )
        destination = output_path(record, index, args.edit_key, args.id_key, args.output_dir)
        result = dict(record)
        result[args.result_key] = str(destination)
        if args.skip_existing and destination.exists():
            results.append(result)
        else:
            pending.append((index, record, destination, result))

    for offset in range(0, len(pending), args.batch_size):
        chunk = pending[offset : offset + args.batch_size]
        try:
            images_a, images_b, focus_maps, size_infos = [], [], [], []
            prompts = []
            generators = []
            for index, record, _, _ in chunk:
                edit_images = record[args.edit_key]
                prepared, size_info = prepare_inference_images(
                    {
                        "a": Image.open(resolve_data_path(edit_images[0], args.dataset_base_path)).convert("RGB"),
                        "b": Image.open(resolve_data_path(edit_images[1], args.dataset_base_path)).convert("RGB"),
                        "focus": Image.open(resolve_data_path(edit_images[control_index], args.dataset_base_path)),
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
                focus_maps.append(load_focus_map(prepared["focus"]))
                size_infos.append(size_info)
                prompts.append(record.get(args.prompt_key) or args.prompt)
                seed = int(record.get(args.seed_key, args.seed + index))
                generators.append(torch.Generator(device="cuda").manual_seed(seed))

            canvas_width, canvas_height = size_infos[0]["canvas_size"]
            cond_a_latents, cond_b_latents = encode_condition_images(
                pipe, images_a, images_b, canvas_height, canvas_width, torch.device("cuda")
            )
            focus_map = torch.cat(focus_maps, dim=0).to("cuda")
            with conditioned_transformer.use_conditions(cond_a_latents, cond_b_latents, focus_map):
                images = pipe(
                    prompt=prompts,
                    height=canvas_height,
                    width=canvas_width,
                    num_inference_steps=args.steps,
                    intermediate_timesteps=1.3 if args.steps == 2 else None,
                    guidance_scale=args.guidance_scale,
                    generator=generators,
                    use_resolution_binning=False,
                ).images
            for image, size_info, (index, _, destination, result) in zip(images, size_infos, chunk):
                restore_output_size(image, size_info, args.restore_to_original_size).save(destination)
                results.append(result)
                print(f"[{index + 1}] {destination}")
        except Exception as error:
            if not args.continue_on_error:
                raise
            for _, _, _, result in chunk:
                result["error"] = str(error)
                results.append(result)

    result_path = Path(args.output_dir) / "metadata_results.json"
    result_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(results)} records to {result_path}")


if __name__ == "__main__":
    main()
