#!/usr/bin/env python3

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Run a DiffSynth Studio FLUX.2 training checkpoint as teacher.")
    parser.add_argument("--config", help="JSON file containing model-loading and LoRA arguments.")
    parser.add_argument("--diffsynth_root", required=True)
    parser.add_argument("--images-json", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--embedded_guidance", type=float, default=1.0)
    parser.add_argument("--model_paths")
    parser.add_argument("--model_id_with_origin_paths")
    parser.add_argument("--tokenizer_path")
    parser.add_argument("--lora_checkpoint")
    parser.add_argument("--lora_rank", type=int)
    parser.add_argument("--lora_target_modules")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_training_module_class(diffsynth_root):
    train_path = Path(diffsynth_root) / "examples/flux2/model_training/train.py"
    if not train_path.is_file():
        raise FileNotFoundError(f"DiffSynth FLUX.2 trainer not found: {train_path}")
    spec = importlib.util.spec_from_file_location("diffsynth_flux2_training", train_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Flux2ImageTrainingModule


def main():
    args = parse_args()
    if args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        allowed = {
            "model_paths",
            "model_id_with_origin_paths",
            "tokenizer_path",
            "lora_checkpoint",
            "lora_rank",
            "lora_target_modules",
        }
        unknown = sorted(set(config) - allowed)
        if unknown:
            raise ValueError(f"Unknown teacher config fields: {unknown}.")
        for key, value in config.items():
            if getattr(args, key) is None:
                setattr(args, key, value)
    args.lora_rank = args.lora_rank or 32
    if not args.model_paths and not args.model_id_with_origin_paths:
        raise ValueError("Provide --model_paths or --model_id_with_origin_paths, directly or through --config.")
    if not args.lora_checkpoint or not args.lora_target_modules:
        raise ValueError("Provide lora_checkpoint and lora_target_modules, directly or through --config.")
    image_paths = json.loads(args.images_json)
    if not isinstance(image_paths, list) or not image_paths:
        raise ValueError("--images-json must contain a non-empty JSON list.")
    edit_images = [Image.open(path).convert("RGB") for path in image_paths]
    training_module_class = load_training_module_class(args.diffsynth_root)
    teacher = training_module_class(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        lora_base_model="dit",
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing=False,
        extra_inputs="edit_image",
        device=args.device,
        task="sft",
    )
    teacher.pipe.dit.eval()
    teacher.pipe.scheduler.training = False
    with torch.no_grad():
        image = teacher.pipe(
            prompt=args.prompt,
            edit_image=edit_images,
            height=args.height,
            width=args.width,
            seed=args.seed,
            rand_device=args.device,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
            embedded_guidance=args.embedded_guidance,
        )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)


if __name__ == "__main__":
    main()
