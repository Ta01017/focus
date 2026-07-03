#!/usr/bin/env python

import argparse
import json
import random
from pathlib import Path

import torch
from accelerate import Accelerator
from safetensors.torch import load_file, save_file
from torch.nn import functional as F
from torch.utils.data import DataLoader

from diffusers import SanaControlNetModel, SanaSprintPipeline

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
from sana_dof import DualImageConditionAdapter, encode_vae_latents
from sana_sprint_controlnet import SanaSprintFocusControlNetTransformer, initialize_controlnet_from_transformer


def parse_args():
    parser = argparse.ArgumentParser(description="Train SANA-Sprint with a focus-map ControlNet.")
    parser.add_argument("--model", default="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers")
    add_metadata_args(parser, metadata_required=True)
    add_pretrained_args(parser)
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--control_index", type=int, default=2, help="Index of the focus map in edit_image.")
    parser.add_argument("--prompt", default="a photorealistic all-in-focus photograph")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=None)
    parser.add_argument("--size_divisor", type=int, default=32)
    parser.add_argument("--aspect_ratio_tolerance", type=float, default=0.01)
    parser.add_argument("--downscale_if_exceeds_max_pixels", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_train_steps", type=int, default=20000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--adapter_hidden_channels", type=int, default=128)
    parser.add_argument("--controlnet_layers", type=int, default=7)
    parser.add_argument("--conditioning_scale", type=float, default=1.0)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--max_timestep", type=float, default=1.57080)
    parser.add_argument("--focus_loss_weight", type=float, default=0.0)
    parser.add_argument("--focus_keep_weight", type=float, default=0.3)
    parser.add_argument("--focus_blur_weight", type=float, default=1.0)
    parser.add_argument("--focus_mask_gamma", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--resume_from_checkpoint", default=None)
    return parser.parse_args()


def validate_args(args):
    if args.resolution is not None and args.resolution % 32:
        raise ValueError("--resolution must be divisible by 32.")
    if args.resolution is None and args.batch_size != 1:
        raise ValueError("Dynamic-resolution training requires --batch_size 1.")
    if args.control_index < 2 or args.control_index > 3:
        raise ValueError("--control_index must be 2 (focus_a) or 3 (focus_b/focus_b_warp).")
    if args.focus_mask_gamma <= 0:
        raise ValueError("--focus_mask_gamma must be greater than 0.")
    if args.focus_loss_weight < 0 or args.focus_keep_weight < 0 or args.focus_blur_weight < 0:
        raise ValueError("Focus loss weights must be non-negative.")


def save_checkpoint(accelerator, model, optimizer, save_directory: Path, args, global_step, epoch, step_in_epoch):
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    unwrapped = accelerator.unwrap_model(model)
    save_directory.mkdir(parents=True, exist_ok=True)
    unwrapped.controlnet.save_pretrained(save_directory / "controlnet")
    adapter_state = {
        key: value.detach().cpu().contiguous() for key, value in unwrapped.adapter.state_dict().items()
    }
    save_file(adapter_state, save_directory / "adapter.safetensors")
    config = {
        "base_model": args.model,
        "target_key": args.target_key,
        "edit_key": args.edit_key,
        "cond_format": "edit_image[A,B,focus_a,optional_focus_b]",
        "control_index": args.control_index,
        "focus_map_range": [0.0, 1.0],
        "latent_channels": unwrapped.transformer.config.in_channels,
        "hidden_channels": args.adapter_hidden_channels,
        "controlnet_layers": args.controlnet_layers,
        "conditioning_scale": args.conditioning_scale,
        "resolution": args.resolution,
        "dynamic_resolution": args.resolution is None,
        "max_pixels": args.max_pixels,
        "size_divisor": args.size_divisor,
        "aspect_ratio_tolerance": args.aspect_ratio_tolerance,
        "downscale_if_exceeds_max_pixels": args.downscale_if_exceeds_max_pixels,
        "valid_mask_loss": True,
        "global_step": global_step,
    }
    (save_directory / "controlnet_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    save_trainer_state(save_directory, optimizer, global_step, epoch, step_in_epoch)


def main():
    args = parse_args()
    validate_args(args)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    transformer_dtype = torch.float32
    if accelerator.device.type == "cuda" and args.mixed_precision == "bf16":
        transformer_dtype = torch.bfloat16
    elif accelerator.device.type == "cuda" and args.mixed_precision == "fp16":
        transformer_dtype = torch.float16

    pipe = SanaSprintPipeline.from_pretrained(
        args.model, torch_dtype=transformer_dtype, **pretrained_kwargs(args)
    )
    pipe.transformer.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.transformer.eval()
    pipe.text_encoder.eval()
    pipe.vae.eval()

    controlnet = initialize_controlnet_from_transformer(pipe.transformer, args.controlnet_layers)
    adapter = DualImageConditionAdapter(
        latent_channels=pipe.transformer.config.in_channels,
        hidden_channels=args.adapter_hidden_channels,
    )
    model = SanaSprintFocusControlNetTransformer(
        transformer=pipe.transformer,
        controlnet=controlnet,
        adapter=adapter,
        conditioning_scale=args.conditioning_scale,
    )
    if args.gradient_checkpointing:
        pipe.transformer.enable_gradient_checkpointing()
        controlnet.enable_gradient_checkpointing()

    pipe.text_encoder.to(accelerator.device, dtype=transformer_dtype)
    pipe.vae.to(accelerator.device, dtype=torch.float32)
    pipe.transformer.to(accelerator.device, dtype=transformer_dtype)
    controlnet.to(accelerator.device, dtype=torch.float32)
    adapter.to(accelerator.device, dtype=torch.float32)
    trainable_parameters = list(controlnet.parameters()) + list(adapter.parameters())
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=1e-2)
    output_dir = Path(args.output_dir)
    resume_path = resolve_resume_checkpoint(output_dir, args.resume_from_checkpoint)
    resume_state = {"global_step": 0, "epoch": 0, "step_in_epoch": 0}
    if resume_path is not None:
        loaded_controlnet = SanaControlNetModel.from_pretrained(
            resume_path / "controlnet", **pretrained_kwargs(args)
        )
        controlnet.load_state_dict(loaded_controlnet.state_dict(), strict=True)
        adapter.load_state_dict(load_file(resume_path / "adapter.safetensors"), strict=True)
        resume_state = load_trainer_state(resume_path, optimizer)

    dataset = DiffSynthFocusDataset(
        metadata_path=args.dataset_metadata_path,
        base_path=args.dataset_base_path,
        target_key=args.target_key,
        edit_key=args.edit_key,
        repeat=args.dataset_repeat,
        min_edit_images=args.control_index + 1,
        use_focus_maps=True,
        default_prompt=args.prompt,
        prompt_key=args.prompt_key,
        start_index=args.start_index,
        max_samples=args.max_samples,
    )

    def collate_fn(samples):
        batch = paired_preprocess(
            samples,
            args.resolution,
            pipe.image_processor,
            training=True,
            max_pixels=args.max_pixels,
            size_divisor=args.size_divisor,
            aspect_ratio_tolerance=args.aspect_ratio_tolerance,
            downscale_if_exceeds_max_pixels=args.downscale_if_exceeds_max_pixels,
        )
        batch["focus_map"] = batch["focus_a"] if args.control_index == 2 else batch["focus_b"]
        return batch

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    global_step = int(resume_state["global_step"])
    epoch = int(resume_state["epoch"])
    resume_step = int(resume_state["step_in_epoch"])
    while global_step < args.max_train_steps:
        dataloader.sampler.set_epoch(epoch) if hasattr(dataloader.sampler, "set_epoch") else None
        for step, batch in enumerate(dataloader):
            if epoch == int(resume_state["epoch"]) and step < resume_step:
                continue
            with accelerator.accumulate(model):
                with torch.no_grad():
                    target = batch["target"].to(accelerator.device, dtype=torch.float32)
                    cond_a = batch["cond_a"].to(accelerator.device, dtype=torch.float32)
                    cond_b = batch["cond_b"].to(accelerator.device, dtype=torch.float32)
                    focus_map = batch["focus_map"].to(accelerator.device, dtype=torch.float32).clamp(0, 1)
                    target_latents = encode_vae_latents(pipe.vae, target)
                    cond_a_latents = encode_vae_latents(pipe.vae, cond_a)
                    cond_b_latents = encode_vae_latents(pipe.vae, cond_b)
                    prompt_embeds, prompt_attention_mask = pipe.encode_prompt(
                        batch["prompts"],
                        device=accelerator.device,
                        clean_caption=False,
                        max_sequence_length=300,
                    )
                    prompt_embeds = prompt_embeds.to(transformer_dtype)
                    prompt_attention_mask = prompt_attention_mask.to(accelerator.device)

                    sigma_data = pipe.scheduler.config.sigma_data
                    timestep = torch.full(
                        (target_latents.shape[0],),
                        args.max_timestep,
                        device=accelerator.device,
                        dtype=torch.float32,
                    )
                    target_state = target_latents * sigma_data
                    noise = torch.randn_like(target_state) * sigma_data
                    noisy_state = (
                        torch.cos(timestep).view(-1, 1, 1, 1) * target_state
                        + torch.sin(timestep).view(-1, 1, 1, 1) * noise
                    )
                    scm_timestep = torch.sin(timestep) / (torch.cos(timestep) + torch.sin(timestep))
                    scm_expanded = scm_timestep.view(-1, 1, 1, 1)
                    normalization = torch.sqrt(scm_expanded**2 + (1 - scm_expanded) ** 2)
                    model_input = noisy_state / sigma_data * normalization

                guidance = torch.full(
                    (target_latents.shape[0],),
                    args.guidance_scale * pipe.transformer.config.guidance_embeds_scale,
                    device=accelerator.device,
                    dtype=transformer_dtype,
                )
                raw_prediction = model(
                    model_input.to(transformer_dtype),
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    guidance=guidance,
                    timestep=scm_timestep,
                    cond_a_latents=cond_a_latents,
                    cond_b_latents=cond_b_latents,
                    focus_map=focus_map,
                    conditioning_scale=args.conditioning_scale,
                    return_dict=False,
                )[0]

                velocity = (
                    (1 - 2 * scm_expanded) * model_input
                    + (1 - 2 * scm_expanded + 2 * scm_expanded**2) * raw_prediction.float()
                ) / normalization
                velocity = velocity * sigma_data
                predicted_target = (
                    torch.cos(timestep).view(-1, 1, 1, 1) * noisy_state
                    - torch.sin(timestep).view(-1, 1, 1, 1) * velocity
                ) / sigma_data
                error = predicted_target.float() - target_latents.float()
                valid_mask = F.interpolate(
                    batch["valid_mask"].to(accelerator.device), size=error.shape[-2:], mode="nearest"
                )
                denominator = (valid_mask.sum() * error.shape[1]).clamp_min(1)
                loss = (error.square() * valid_mask).sum() / denominator
                loss = loss + 0.1 * (error.abs() * valid_mask).sum() / denominator

                if args.focus_loss_weight > 0:
                    focus_latent = F.interpolate(
                        focus_map,
                        size=target_latents.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    keep_mask = focus_latent.pow(args.focus_mask_gamma)
                    focus_weight = (
                        args.focus_keep_weight * keep_mask + args.focus_blur_weight * (1 - keep_mask)
                    )
                    weighted_l1 = (error.abs() * focus_weight * valid_mask).sum() / denominator
                    loss = loss + args.focus_loss_weight * weighted_l1

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process and global_step % 10 == 0:
                    print(f"step={global_step} loss={loss.detach().item():.6f}")
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
