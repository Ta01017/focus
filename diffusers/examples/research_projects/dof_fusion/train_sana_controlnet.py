#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image
from safetensors.torch import save_file
from torch.nn import functional as F
from torch.utils.data import DataLoader

from diffusers import FlowMatchEulerDiscreteScheduler, SanaPipeline

from focus_dataset import DiffSynthFocusDataset
from sana_controlnet_dof import SanaControlNetDOFModel
from sana_dof import DualImageConditionAdapter
from sana_sprint_controlnet import initialize_controlnet_from_transformer


def parse_args():
    parser = argparse.ArgumentParser(description="Train ordinary SANA + focus ControlNet for A/B fusion.")
    parser.add_argument("--model", default="Efficient-Large-Model/Sana_600M_1024px_diffusers")
    parser.add_argument("--dataset_metadata_path", required=True)
    parser.add_argument("--dataset_base_path", default=".")
    parser.add_argument("--target_key", default="image")
    parser.add_argument("--edit_key", default="edit_image")
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--control_index", type=int, choices=(2, 3), default=2)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt", default="a photorealistic all-in-focus photograph")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_train_steps", type=int, default=20000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--adapter_hidden_channels", type=int, default=128)
    parser.add_argument("--controlnet_layers", type=int, default=7)
    parser.add_argument("--conditioning_scale", type=float, default=1.0)
    parser.add_argument("--focus_loss_weight", type=float, default=0.0)
    parser.add_argument("--focus_keep_weight", type=float, default=0.3)
    parser.add_argument("--focus_blur_weight", type=float, default=1.0)
    parser.add_argument("--focus_mask_gamma", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="bf16")
    return parser.parse_args()


def save_checkpoint(accelerator, model, directory, args, global_step):
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    model = accelerator.unwrap_model(model)
    directory.mkdir(parents=True, exist_ok=True)
    model.controlnet.save_pretrained(directory / "controlnet")
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in model.adapter.state_dict().items()},
        directory / "adapter.safetensors",
    )
    config = {
        "base_model": args.model,
        "model_type": "sana_controlnet_dof",
        "target_key": args.target_key,
        "edit_key": args.edit_key,
        "cond_format": "edit_image[A,B,focus_a,optional_focus_b]",
        "control_index": args.control_index,
        "control_preprocessing": "RGB_replicated_focus_map_then_SANA_VAE",
        "latent_channels": model.transformer.config.in_channels,
        "hidden_channels": args.adapter_hidden_channels,
        "controlnet_layers": args.controlnet_layers,
        "conditioning_scale": args.conditioning_scale,
        "resolution": args.resolution,
        "global_step": global_step,
    }
    (directory / "controlnet_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main():
    args = parse_args()
    if args.resolution % 32:
        raise ValueError("--resolution must be divisible by 32.")
    if args.focus_mask_gamma <= 0:
        raise ValueError("--focus_mask_gamma must be greater than zero.")

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    torch.manual_seed(args.seed)
    weight_dtype = torch.float32
    if accelerator.device.type == "cuda" and args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    elif accelerator.device.type == "cuda" and args.mixed_precision == "fp16":
        weight_dtype = torch.float16

    pipe = SanaPipeline.from_pretrained(args.model, torch_dtype=weight_dtype)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.model, subfolder="scheduler")
    pipe.transformer.requires_grad_(False).eval()
    pipe.text_encoder.requires_grad_(False).eval()
    pipe.vae.requires_grad_(False).eval()
    controlnet = initialize_controlnet_from_transformer(pipe.transformer, args.controlnet_layers)
    adapter = DualImageConditionAdapter(pipe.transformer.config.in_channels, args.adapter_hidden_channels)
    model = SanaControlNetDOFModel(pipe.transformer, controlnet, adapter)
    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()
        pipe.transformer.enable_gradient_checkpointing()
    pipe.text_encoder.to(accelerator.device, dtype=weight_dtype)
    pipe.vae.to(accelerator.device, dtype=torch.float32)
    pipe.transformer.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=torch.float32)
    adapter.to(accelerator.device, dtype=torch.float32)
    trainable_parameters = list(controlnet.parameters()) + list(adapter.parameters())
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=1e-2)

    dataset = DiffSynthFocusDataset(
        metadata_path=args.dataset_metadata_path,
        base_path=args.dataset_base_path,
        target_key=args.target_key,
        edit_key=args.edit_key,
        repeat=args.dataset_repeat,
        min_edit_images=args.control_index + 1,
        use_focus_maps=True,
        default_prompt=args.prompt,
    )

    def collate_fn(samples):
        focus_maps = []
        focus_rgb = []
        for sample in samples:
            focus = sample["cond_images"][args.control_index].convert("L")
            focus = focus.resize((args.resolution, args.resolution), Image.Resampling.BILINEAR)
            focus_maps.append(torch.from_numpy(np.asarray(focus, dtype=np.float32) / 255.0).unsqueeze(0))
            focus_rgb.append(focus.convert("RGB"))
        preprocess = pipe.image_processor.preprocess
        return {
            "target": preprocess(
                [sample["target"] for sample in samples], height=args.resolution, width=args.resolution
            ),
            "cond_a": preprocess(
                [sample["cond_images"][0] for sample in samples], height=args.resolution, width=args.resolution
            ),
            "cond_b": preprocess(
                [sample["cond_images"][1] for sample in samples], height=args.resolution, width=args.resolution
            ),
            "control": preprocess(focus_rgb, height=args.resolution, width=args.resolution),
            "focus_map": torch.stack(focus_maps),
            "prompts": [sample["prompt"] for sample in samples],
        }

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    scheduler_timesteps = scheduler.timesteps.to(accelerator.device)
    scheduler_sigmas = scheduler.sigmas.to(accelerator.device)
    global_step = 0
    epoch = 0
    while global_step < args.max_train_steps:
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)
        for batch in dataloader:
            with accelerator.accumulate(model):
                with torch.no_grad():
                    scaling_factor = pipe.vae.config.scaling_factor
                    target_latents = pipe.vae.encode(batch["target"].to(accelerator.device, torch.float32)).latent
                    cond_a_latents = pipe.vae.encode(batch["cond_a"].to(accelerator.device, torch.float32)).latent
                    cond_b_latents = pipe.vae.encode(batch["cond_b"].to(accelerator.device, torch.float32)).latent
                    control_latents = pipe.vae.encode(batch["control"].to(accelerator.device, torch.float32)).latent
                    target_latents = target_latents * scaling_factor
                    cond_a_latents = cond_a_latents * scaling_factor
                    cond_b_latents = cond_b_latents * scaling_factor
                    control_latents = control_latents * scaling_factor
                    prompt_embeds, prompt_mask, _, _ = pipe.encode_prompt(
                        batch["prompts"],
                        do_classifier_free_guidance=False,
                        device=accelerator.device,
                        clean_caption=False,
                        max_sequence_length=300,
                    )
                    prompt_embeds = prompt_embeds.to(weight_dtype)
                    noise = torch.randn_like(target_latents)
                    indices = torch.randint(
                        0, scheduler.config.num_train_timesteps, (target_latents.shape[0],), device=accelerator.device
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
                    controlnet_cond=control_latents,
                    conditioning_scale=args.conditioning_scale,
                ).float()
                error = prediction - velocity_target.float()
                loss = error.square().mean() + 0.1 * error.abs().mean()
                if args.focus_loss_weight > 0:
                    focus = F.interpolate(
                        batch["focus_map"].to(accelerator.device),
                        size=error.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ).pow(args.focus_mask_gamma)
                    weight = args.focus_keep_weight * focus + args.focus_blur_weight * (1 - focus)
                    loss = loss + args.focus_loss_weight * (error.abs() * weight).mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process and global_step % 10 == 0:
                    print(f"step={global_step} loss={loss.detach().item():.6f}")
                if global_step % args.save_steps == 0:
                    save_checkpoint(accelerator, model, output_dir / f"checkpoint-{global_step}", args, global_step)
                if global_step >= args.max_train_steps:
                    break
        epoch += 1

    save_checkpoint(accelerator, model, output_dir, args, global_step)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
