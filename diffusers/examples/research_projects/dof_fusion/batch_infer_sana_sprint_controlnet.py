#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file

from diffusers import SanaControlNetModel, SanaSprintPipeline

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
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--conditioning_scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
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
    if args.height % 32 or args.width % 32:
        raise ValueError("--height and --width must be divisible by 32.")

    checkpoint = Path(args.checkpoint)
    config = json.loads((checkpoint / "controlnet_config.json").read_text(encoding="utf-8"))
    model_id = args.model or config["base_model"]
    control_index = config["control_index"] if args.control_index is None else args.control_index
    conditioning_scale = (
        config.get("conditioning_scale", 1.0) if args.conditioning_scale is None else args.conditioning_scale
    )
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipe = SanaSprintPipeline.from_pretrained(model_id, torch_dtype=dtype).to("cuda")
    pipe.vae.to(dtype=torch.float32)
    controlnet = SanaControlNetModel.from_pretrained(checkpoint / "controlnet", torch_dtype=dtype).to("cuda")
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
            images_a = []
            images_b = []
            focus_maps = []
            prompts = []
            generators = []
            for index, record, _, _ in chunk:
                edit_images = record[args.edit_key]
                images_a.append(
                    Image.open(resolve_data_path(edit_images[0], args.dataset_base_path)).convert("RGB")
                )
                images_b.append(
                    Image.open(resolve_data_path(edit_images[1], args.dataset_base_path)).convert("RGB")
                )
                focus_maps.append(
                    load_focus_map(
                        resolve_data_path(edit_images[control_index], args.dataset_base_path),
                        args.height,
                        args.width,
                    )
                )
                prompts.append(record.get(args.prompt_key) or args.prompt)
                seed = int(record.get(args.seed_key, args.seed + index))
                generators.append(torch.Generator(device="cuda").manual_seed(seed))

            cond_a_latents, cond_b_latents = encode_condition_images(
                pipe, images_a, images_b, args.height, args.width, torch.device("cuda")
            )
            focus_map = torch.cat(focus_maps, dim=0).to("cuda")
            with conditioned_transformer.use_conditions(cond_a_latents, cond_b_latents, focus_map):
                images = pipe(
                    prompt=prompts,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.steps,
                    intermediate_timesteps=1.3 if args.steps == 2 else None,
                    guidance_scale=args.guidance_scale,
                    generator=generators,
                    use_resolution_binning=False,
                ).images
            for image, (index, _, destination, result) in zip(images, chunk):
                image.save(destination)
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
