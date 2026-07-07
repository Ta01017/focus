#!/usr/bin/env python

import argparse
import json
import random
from pathlib import Path

import torch
from accelerate import Accelerator
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from safetensors.torch import load_file, save_file
from torch.nn import functional as F
from torch.utils.data import DataLoader

from diffusers import FlowMatchEulerDiscreteScheduler, SanaPipeline

from dof_utils import (
    add_metadata_args,
    add_pretrained_args,
    load_trainer_state,
    paired_preprocess,
    pretrained_kwargs,
    resolve_resume_checkpoint,
    save_trainer_state,
)
from focus_dataset import DiffSynthFocusDataset
from sana_dof import ConditionedSanaTransformer, DualImageConditionAdapter, encode_vae_latents

DEFAULT_SANA_LORA_TARGET_MODULES = ["to_k", "to_q", "to_v"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train ordinary SANA + A/B adapter DOF fusion.")
    parser.add_argument("--model", default="Efficient-Large-Model/Sana_600M_1024px_diffusers")
    add_metadata_args(parser, metadata_required=True)
    add_pretrained_args(parser)
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt", default="a photorealistic all-in-focus photograph")
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=None)
    parser.add_argument("--size_divisor", type=int, default=32)
    parser.add_argument("--aspect_ratio_tolerance", type=float, default=0.01)
    parser.add_argument("--downscale_if_exceeds_max_pixels", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_train_steps", type=int, default=20000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--adapter_hidden_channels", type=int, default=128)
    parser.add_argument("--train_transformer_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_modules", default=None)
    parser.add_argument("--save_transformer_lora", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="no")
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--debug_check_finite", action="store_true")
    parser.add_argument("--debug_log_adapter_stats", action="store_true")
    parser.add_argument("--overfit_fixed_noise", action="store_true")
    parser.add_argument("--overfit_timestep_index", type=int, default=None)
    parser.add_argument("--debug_dump_batch_dir", default=None)
    return parser.parse_args()


def parse_lora_target_modules(value):
    if value is None:
        return DEFAULT_SANA_LORA_TARGET_MODULES
    target_modules = [item.strip() for item in value.split(",") if item.strip()]
    if not target_modules:
        raise ValueError("--lora_target_modules must contain at least one module name when provided.")
    return target_modules


def print_lora_candidate_modules(transformer):
    keywords = ("q", "k", "v", "out", "proj")
    candidates = [
        name
        for name, module in transformer.named_modules()
        if isinstance(module, torch.nn.Linear) and any(keyword in name.lower() for keyword in keywords)
    ]
    print("SANA transformer Linear module candidates containing q/k/v/out/proj:", flush=True)
    for name in candidates:
        print(f"  {name}", flush=True)


def validate_lora_target_modules(transformer, target_modules):
    module_names = [name for name, module in transformer.named_modules() if isinstance(module, torch.nn.Linear)]
    missing = []
    for target in target_modules:
        if not any(name == target or name.endswith(f".{target}") for name in module_names):
            missing.append(target)
    if missing:
        print_lora_candidate_modules(transformer)
        raise ValueError(f"LoRA target modules were not found in SANA transformer: {missing}")


def trainable_parameters(module):
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def count_parameters(parameters):
    return sum(parameter.numel() for parameter in parameters)


def parameter_norm(parameters):
    total = 0.0
    for parameter in parameters:
        total += parameter.detach().float().square().sum().item()
    return total**0.5


def grad_norm(parameters):
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += parameter.grad.detach().float().square().sum().item()
    return total**0.5


def dump_debug_batch(batch, directory, global_step):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tensors = {key: value.detach().cpu() for key, value in batch.items() if torch.is_tensor(value)}
    tensors["prompts"] = batch["prompts"]
    torch.save(tensors, directory / f"batch_step_{global_step:06d}.pt")


def save_checkpoint(accelerator, model, optimizer, directory, args, global_step, epoch, step_in_epoch):
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    model = accelerator.unwrap_model(model)
    directory.mkdir(parents=True, exist_ok=True)
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in model.adapter.state_dict().items()},
        directory / "adapter.safetensors",
    )
    config = {
        "base_model": args.model,
        "model_type": "sana_ab_adapter_dof",
        "target_key": args.target_key,
        "edit_key": args.edit_key,
        "prompt_key": args.prompt_key,
        "cond_format": "edit_image[A,B]",
        "use_controlnet": False,
        "use_focus_maps": False,
        "latent_channels": model.transformer.config.in_channels,
        "hidden_channels": args.adapter_hidden_channels,
        "resolution": args.resolution,
        "dynamic_resolution": args.resolution is None,
        "max_pixels": args.max_pixels,
        "size_divisor": args.size_divisor,
        "aspect_ratio_tolerance": args.aspect_ratio_tolerance,
        "downscale_if_exceeds_max_pixels": args.downscale_if_exceeds_max_pixels,
        "valid_mask_loss": True,
        "train_transformer_lora": args.train_transformer_lora,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": parse_lora_target_modules(args.lora_target_modules),
        "global_step": global_step,
    }
    (directory / "adapter_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if args.train_transformer_lora and args.save_transformer_lora:
        SanaPipeline.save_lora_weights(
            save_directory=directory / "transformer_lora",
            transformer_lora_layers=get_peft_model_state_dict(model.transformer),
        )
    save_trainer_state(directory, optimizer, global_step, epoch, step_in_epoch)


def check_finite(global_step, tensors):
    for name, tensor in tensors.items():
        if tensor is None:
            continue
        finite = torch.isfinite(tensor)
        if finite.all():
            continue
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        print(
            f"Non-finite tensor at step={global_step}: {name} nan_count={nan_count} inf_count={inf_count}",
            flush=True,
        )
        raise RuntimeError(f"Non-finite values detected in {name}.")


def main():
    args = parse_args()
    if args.resolution is not None and args.resolution % args.size_divisor:
        raise ValueError("--resolution must be divisible by --size_divisor.")
    if args.resolution is None and args.batch_size != 1:
        raise ValueError("Dynamic-resolution training requires --batch_size 1; use gradient accumulation.")
    if args.log_steps < 1:
        raise ValueError("--log_steps must be at least 1.")

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    weight_dtype = torch.float32
    if accelerator.device.type == "cuda" and args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    elif accelerator.device.type == "cuda" and args.mixed_precision == "fp16":
        weight_dtype = torch.float16

    pipe = SanaPipeline.from_pretrained(args.model, torch_dtype=weight_dtype, **pretrained_kwargs(args))
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.model, subfolder="scheduler", **pretrained_kwargs(args)
    )
    pipe.transformer.requires_grad_(False).eval()
    pipe.text_encoder.requires_grad_(False).eval()
    pipe.vae.requires_grad_(False).eval()
    lora_target_modules = parse_lora_target_modules(args.lora_target_modules)
    if args.train_transformer_lora:
        validate_lora_target_modules(pipe.transformer, lora_target_modules)
        transformer_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=lora_target_modules,
        )
        pipe.transformer.add_adapter(transformer_lora_config)
    adapter = DualImageConditionAdapter(pipe.transformer.config.in_channels, args.adapter_hidden_channels)
    model = ConditionedSanaTransformer(pipe.transformer, adapter)
    if args.gradient_checkpointing:
        pipe.transformer.enable_gradient_checkpointing()
    pipe.text_encoder.to(accelerator.device, dtype=weight_dtype)
    pipe.vae.to(accelerator.device, dtype=torch.float32)
    pipe.transformer.to(accelerator.device, dtype=weight_dtype)
    adapter.to(accelerator.device, dtype=torch.float32)
    adapter_trainable_parameters = trainable_parameters(adapter)
    transformer_lora_parameters = trainable_parameters(pipe.transformer) if args.train_transformer_lora else []
    optimizer = torch.optim.AdamW(
        adapter_trainable_parameters + transformer_lora_parameters,
        lr=args.learning_rate,
        weight_decay=1e-2,
    )
    accelerator.print(f"adapter trainable params: {count_parameters(adapter_trainable_parameters):,}", flush=True)
    accelerator.print(
        f"transformer LoRA trainable params: {count_parameters(transformer_lora_parameters):,}",
        flush=True,
    )
    accelerator.print(
        f"total trainable params: {count_parameters(adapter_trainable_parameters + transformer_lora_parameters):,}",
        flush=True,
    )

    output_dir = Path(args.output_dir)
    resume_path = resolve_resume_checkpoint(output_dir, args.resume_from_checkpoint)
    resume_state = {"global_step": 0, "epoch": 0, "step_in_epoch": 0}
    if resume_path is not None:
        adapter.load_state_dict(load_file(resume_path / "adapter.safetensors"), strict=True)
        if args.train_transformer_lora:
            lora_path = resume_path / "transformer_lora"
            if not lora_path.exists():
                raise ValueError(f"Missing transformer LoRA checkpoint directory: {lora_path}")
            lora_state_dict = SanaPipeline.lora_state_dict(lora_path)
            transformer_state_dict = {
                key.replace("transformer.", ""): value
                for key, value in lora_state_dict.items()
                if key.startswith("transformer.")
            }
            set_peft_model_state_dict(pipe.transformer, transformer_state_dict, adapter_name="default")
        resume_state = load_trainer_state(resume_path, optimizer)

    dataset = DiffSynthFocusDataset(
        metadata_path=args.dataset_metadata_path,
        base_path=args.dataset_base_path,
        target_key=args.target_key,
        edit_key=args.edit_key,
        repeat=args.dataset_repeat,
        min_edit_images=2,
        use_focus_maps=False,
        default_prompt=args.prompt,
        prompt_key=args.prompt_key,
        start_index=args.start_index,
        max_samples=args.max_samples,
    )

    def collate_fn(samples):
        return paired_preprocess(
            samples,
            args.resolution,
            pipe.image_processor,
            training=True,
            max_pixels=args.max_pixels,
            size_divisor=args.size_divisor,
            aspect_ratio_tolerance=args.aspect_ratio_tolerance,
            downscale_if_exceeds_max_pixels=args.downscale_if_exceeds_max_pixels,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    scheduler_timesteps = scheduler.timesteps.to(accelerator.device)
    scheduler_sigmas = scheduler.sigmas.to(accelerator.device)
    fixed_noise_cache = {}
    global_step = int(resume_state["global_step"])
    epoch = int(resume_state["epoch"])
    resume_step = int(resume_state["step_in_epoch"])
    while global_step < args.max_train_steps:
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader):
            if epoch == int(resume_state["epoch"]) and step < resume_step:
                continue
            with accelerator.accumulate(model):
                with torch.no_grad():
                    if args.debug_dump_batch_dir and global_step == int(resume_state["global_step"]):
                        dump_debug_batch(batch, args.debug_dump_batch_dir, global_step)
                    target_latents = encode_vae_latents(
                        pipe.vae, batch["target"].to(accelerator.device, torch.float32)
                    )
                    cond_a_latents = encode_vae_latents(
                        pipe.vae, batch["cond_a"].to(accelerator.device, torch.float32)
                    )
                    cond_b_latents = encode_vae_latents(
                        pipe.vae, batch["cond_b"].to(accelerator.device, torch.float32)
                    )
                    prompt_embeds, prompt_mask, _, _ = pipe.encode_prompt(
                        batch["prompts"],
                        do_classifier_free_guidance=False,
                        device=accelerator.device,
                        clean_caption=False,
                        max_sequence_length=300,
                    )
                    prompt_embeds = prompt_embeds.to(weight_dtype)
                    if args.overfit_fixed_noise:
                        cache_key = (tuple(target_latents.shape), str(accelerator.device))
                        if cache_key not in fixed_noise_cache:
                            fixed_noise_cache[cache_key] = torch.randn_like(target_latents)
                        noise = fixed_noise_cache[cache_key]
                    else:
                        noise = torch.randn_like(target_latents)
                    if args.overfit_timestep_index is None:
                        indices = torch.randint(
                            0,
                            scheduler.config.num_train_timesteps,
                            (target_latents.shape[0],),
                            device=accelerator.device,
                        )
                    else:
                        if args.overfit_timestep_index < 0 or args.overfit_timestep_index >= scheduler.config.num_train_timesteps:
                            raise ValueError("--overfit_timestep_index is outside scheduler train timestep range.")
                        indices = torch.full(
                            (target_latents.shape[0],),
                            args.overfit_timestep_index,
                            device=accelerator.device,
                            dtype=torch.long,
                        )
                    timesteps = scheduler_timesteps[indices]
                    sigmas = scheduler_sigmas[indices].view(-1, 1, 1, 1).to(target_latents.dtype)
                    noisy_latents = (1 - sigmas) * target_latents + sigmas * noise
                    velocity_target = noise - target_latents

                prediction = model(
                    noisy_latents.to(weight_dtype),
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_mask,
                    timestep=timesteps,
                    cond_a_latents=cond_a_latents,
                    cond_b_latents=cond_b_latents,
                    return_dict=False,
                )[0]
                error = prediction.float() - velocity_target.float()
                valid_mask = F.interpolate(
                    batch["valid_mask"].to(accelerator.device), size=error.shape[-2:], mode="nearest"
                )
                denominator = (valid_mask.sum() * error.shape[1]).clamp_min(1)
                flow_mse = (error.square() * valid_mask).sum() / denominator
                flow_l1 = (error.abs() * valid_mask).sum() / denominator
                loss = flow_mse + 0.1 * flow_l1
                if args.debug_check_finite:
                    check_finite(
                        global_step,
                        {
                            "target_latents": target_latents,
                            "cond_a_latents": cond_a_latents,
                            "cond_b_latents": cond_b_latents,
                            "noisy_latents": noisy_latents,
                            "velocity_target": velocity_target,
                            "prediction": prediction,
                            "error": error,
                            "loss": loss,
                        },
                    )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if args.debug_log_adapter_stats and accelerator.is_main_process:
                        print(
                            "trainable_stats "
                            f"step={global_step} "
                            f"adapter_grad_norm={grad_norm(adapter_trainable_parameters):.6f} "
                            f"adapter_param_norm={parameter_norm(adapter_trainable_parameters):.6f} "
                            f"transformer_lora_grad_norm={grad_norm(transformer_lora_parameters):.6f} "
                            f"transformer_lora_param_norm={parameter_norm(transformer_lora_parameters):.6f}",
                            flush=True,
                        )
                    accelerator.clip_grad_norm_(adapter_trainable_parameters + transformer_lora_parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process and global_step % args.log_steps == 0:
                    print(f"step={global_step} loss={loss.detach().item():.6f}", flush=True)
                if global_step % args.save_steps == 0 or global_step == args.max_train_steps:
                    save_checkpoint(
                        accelerator,
                        model,
                        optimizer,
                        output_dir / f"checkpoint-{global_step}",
                        args,
                        global_step,
                        epoch,
                        step + 1,
                    )
                if global_step >= args.max_train_steps:
                    break
        epoch += 1
        resume_step = 0

    save_checkpoint(accelerator, model, optimizer, output_dir, args, global_step, epoch, 0)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
