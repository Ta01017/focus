#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file

from diffusers import Flux2KleinPipeline, SanaSprintPipeline

from dof_utils import add_metadata_args, add_pretrained_args, pretrained_kwargs
from infer_flux2_klein import DEFAULT_PROMPT
from infer_sana_sprint import load_focus
from metadata import load_metadata, require_keys, resolve_data_path
from sana_dof import ConditionedSanaTransformer, create_condition_adapter, encode_condition_images


def parse_args():
    parser = argparse.ArgumentParser(description="Batch depth-of-field fusion from DiffSynth-style metadata.")
    parser.add_argument("--backend", choices=("flux2", "sana"), required=True)
    parser.add_argument("--dataset_metadata_path", required=True)
    parser.add_argument("--dataset_base_path", default=".")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--edit_key", default="edit_image")
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--id_key", default="id")
    parser.add_argument("--seed_key", default="seed")
    parser.add_argument("--result_key", default="generated_image")
    parser.add_argument("--prompt", default=None, help="Fallback prompt when a sample has no prompt field.")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1, help="True mini-batching is used by the SANA backend.")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--cpu_offload", action="store_true", help="FLUX.2 only.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--lora", default=None, help="Optional FLUX.2 Klein LoRA directory or weight file.")
    parser.add_argument("--adapter", default=None, help="Required for the SANA backend.")
    parser.add_argument("--adapter_hidden_channels", type=int, default=None)
    add_metadata_args(parser)
    add_pretrained_args(parser)
    return parser.parse_args()


def sample_output_path(record: dict, index: int, args) -> Path:
    edit_images = record[args.edit_key]
    sample_id = str(record.get(args.id_key, Path(edit_images[0]).stem))
    safe_id = sample_id.replace("/", "_").replace("\\", "_")
    return Path(args.output_dir) / f"{index:06d}_{safe_id}.png"


def sample_prompt(record: dict, args, fallback: str) -> str:
    return record.get(args.prompt_key) or args.prompt or fallback


def selected_records(records: list[dict], args) -> list[tuple[int, dict]]:
    end = None if args.max_samples is None else args.start_index + args.max_samples
    return list(enumerate(records[args.start_index : end], start=args.start_index))


def load_condition_pair(record: dict, index: int, args):
    require_keys(record, (args.edit_key,), index)
    edit_images = record[args.edit_key]
    if not isinstance(edit_images, list) or len(edit_images) < 2:
        raise ValueError(f"Sample {index} field {args.edit_key!r} must be a list containing at least A and B.")
    image_a = Image.open(resolve_data_path(edit_images[0], args.dataset_base_path)).convert("RGB")
    image_b = Image.open(resolve_data_path(edit_images[1], args.dataset_base_path)).convert("RGB")
    if image_a.size != image_b.size:
        print(f"Warning: sample {index} has different A/B sizes before resize: {image_a.size}, {image_b.size}.")
    return image_a, image_b


def run_flux2(items: list[tuple[int, dict]], args) -> list[dict]:
    training_config = {}
    if args.lora is not None:
        lora_path = Path(args.lora)
        config_path = (lora_path if lora_path.is_dir() else lora_path.parent) / "training_config.json"
        if config_path.exists():
            training_config = json.loads(config_path.read_text(encoding="utf-8"))
    model = args.model or training_config.get("base_model", "black-forest-labs/FLUX.2-klein-4B")
    steps = args.steps or 4
    guidance_scale = 1.0 if args.guidance_scale is None else args.guidance_scale
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    pipe = Flux2KleinPipeline.from_pretrained(model, torch_dtype=dtype, **pretrained_kwargs(args))
    if args.lora is not None:
        pipe.load_lora_weights(args.lora)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
        generator_device = "cpu"
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("FLUX.2 batch inference requires CUDA unless --cpu_offload is used.")
        pipe.to("cuda")
        generator_device = "cuda"

    if args.batch_size != 1:
        print("FLUX.2 Klein applies one reference list to an entire prompt batch; processing samples sequentially.")

    results = []
    for index, record in items:
        output = sample_output_path(record, index, args)
        result = dict(record)
        result[args.result_key] = str(output)
        if args.skip_existing and output.exists():
            results.append(result)
            continue
        try:
            image_a, image_b = load_condition_pair(record, index, args)
            seed = int(record.get(args.seed_key, args.seed + index))
            generator = torch.Generator(device=generator_device).manual_seed(seed)
            image = pipe(
                image=[image_a, image_b],
                prompt=sample_prompt(record, args, DEFAULT_PROMPT),
                height=args.height,
                width=args.width,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]
            output.parent.mkdir(parents=True, exist_ok=True)
            image.save(output)
        except Exception as error:
            if not args.continue_on_error:
                raise
            result["error"] = str(error)
        results.append(result)
        print(f"[{index + 1}] {output}")
    return results


def run_sana(items: list[tuple[int, dict]], args) -> list[dict]:
    if args.adapter is None:
        raise ValueError("--adapter is required when --backend sana.")
    if not torch.cuda.is_available():
        raise RuntimeError("SANA batch inference currently requires CUDA.")

    config_path = Path(args.adapter).with_name("adapter_config.json")
    adapter_config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    model = args.model or adapter_config.get(
        "base_model", "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers"
    )
    steps = args.steps or 1
    guidance_scale = 4.5 if args.guidance_scale is None else args.guidance_scale
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipe = SanaSprintPipeline.from_pretrained(model, torch_dtype=dtype, **pretrained_kwargs(args)).to("cuda")
    pipe.vae.to(dtype=torch.float32)

    hidden_channels = args.adapter_hidden_channels or adapter_config.get("hidden_channels", 128)
    adapter_type = adapter_config.get("adapter_type", "ab")
    adapter = create_condition_adapter(adapter_type, pipe.transformer.config.in_channels, hidden_channels)
    adapter.load_state_dict(load_file(args.adapter), strict=True)
    adapter.to(device="cuda", dtype=dtype).eval()
    conditioned_transformer = ConditionedSanaTransformer(pipe.transformer, adapter)
    pipe.transformer = conditioned_transformer

    results = []
    pending = []
    for index, record in items:
        output = sample_output_path(record, index, args)
        result = dict(record)
        result[args.result_key] = str(output)
        if args.skip_existing and output.exists():
            results.append(result)
        else:
            pending.append((index, record, output, result))

    for offset in range(0, len(pending), args.batch_size):
        chunk = pending[offset : offset + args.batch_size]
        try:
            pairs = [load_condition_pair(record, index, args) for index, record, _, _ in chunk]
            images_a = [pair[0] for pair in pairs]
            images_b = [pair[1] for pair in pairs]
            prompts = [
                sample_prompt(record, args, "a photorealistic all-in-focus photograph")
                for _, record, _, _ in chunk
            ]
            generators = [
                torch.Generator(device="cuda").manual_seed(int(record.get(args.seed_key, args.seed + index)))
                for index, record, _, _ in chunk
            ]
            cond_a_latents, cond_b_latents = encode_condition_images(
                pipe, images_a, images_b, args.height, args.width, torch.device("cuda")
            )
            focus_a = None
            focus_b = None
            if adapter_type == "ab_focus":
                focus_a_items = []
                focus_b_items = []
                for index, record, _, _ in chunk:
                    edits = record[args.edit_key]
                    if len(edits) < 4:
                        raise ValueError(f"Sample {index} requires edit_image[A,B,focus_a,focus_b].")
                    focus_a_items.append(
                        load_focus(
                            resolve_data_path(edits[2], args.dataset_base_path), args.height, args.width
                        )
                    )
                    focus_b_items.append(
                        load_focus(
                            resolve_data_path(edits[3], args.dataset_base_path), args.height, args.width
                        )
                    )
                focus_a = torch.cat(focus_a_items).to("cuda")
                focus_b = torch.cat(focus_b_items).to("cuda")
            with conditioned_transformer.use_condition(cond_a_latents, cond_b_latents, focus_a, focus_b):
                images = pipe(
                    prompt=prompts,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=steps,
                    intermediate_timesteps=1.3 if steps == 2 else None,
                    guidance_scale=guidance_scale,
                    generator=generators,
                    use_resolution_binning=False,
                ).images
            for image, (index, _, output, result) in zip(images, chunk):
                output.parent.mkdir(parents=True, exist_ok=True)
                image.save(output)
                results.append(result)
                print(f"[{index + 1}] {output}")
        except Exception as error:
            if not args.continue_on_error:
                raise
            for _, _, _, result in chunk:
                result["error"] = str(error)
                results.append(result)
    return results


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch_size must be at least 1.")
    if args.height % 32 or args.width % 32:
        raise ValueError("--height and --width must be divisible by 32.")

    records = load_metadata(args.dataset_metadata_path)
    items = selected_records(records, args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    results = run_flux2(items, args) if args.backend == "flux2" else run_sana(items, args)
    results.sort(key=lambda record: record[args.result_key])
    result_path = Path(args.output_dir) / "metadata_results.json"
    result_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(results)} records to {result_path}")


if __name__ == "__main__":
    main()
