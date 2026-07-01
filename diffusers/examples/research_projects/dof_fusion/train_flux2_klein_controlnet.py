#!/usr/bin/env python

import argparse
import copy
import json
import random
from pathlib import Path

import torch
from accelerate import Accelerator
from safetensors.torch import load_file, save_file
from torch.nn import functional as F
from torch.utils.data import DataLoader

from diffusers import Flux2KleinPipeline
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3

from dof_utils import (
    add_metadata_args,
    add_pretrained_args,
    load_trainer_state,
    paired_preprocess,
    pretrained_kwargs,
    resolve_resume_checkpoint,
    save_trainer_state,
)
from flux2_controlnet import Flux2ControlNetTransformer, Flux2FocusControlNet, parse_block_indices
from focus_dataset import DiffSynthFocusDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train a FLUX.2 Klein focus ControlNet residual branch.")
    parser.add_argument("--model", default="black-forest-labs/FLUX.2-klein-4B")
    add_metadata_args(parser, metadata_required=True)
    add_pretrained_args(parser)
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--control_index", type=int, choices=(2, 3), default=2)
    parser.add_argument("--prompt", default="a photorealistic all-in-focus photograph")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_train_steps", type=int, default=20000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--control_hidden_channels", type=int, default=256)
    parser.add_argument("--control_layers", type=int, default=3)
    parser.add_argument("--double_block_indices", default="0,1,2,3,4")
    parser.add_argument("--single_block_indices", default="0,4,8,12,16")
    parser.add_argument("--conditioning_scale", type=float, default=1.0)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--weighting_scheme", default="none")
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument("--focus_loss_weight", type=float, default=0.0)
    parser.add_argument("--focus_keep_weight", type=float, default=0.3)
    parser.add_argument("--focus_blur_weight", type=float, default=1.0)
    parser.add_argument("--focus_mask_gamma", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="bf16")
    return parser.parse_args()


def save_checkpoint(accelerator, model, optimizer, directory, args, global_step, epoch, step_in_epoch):
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    model = accelerator.unwrap_model(model)
    directory.mkdir(parents=True, exist_ok=True)
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in model.controlnet.state_dict().items()},
        directory / "controlnet.safetensors",
    )
    config = {
        "base_model": args.model,
        "model_type": "flux2_klein_focus_controlnet",
        "target_key": args.target_key,
        "edit_key": args.edit_key,
        "cond_format": "edit_image[A,B,focus_a,optional_focus_b]",
        "control_index": args.control_index,
        "focus_map_range": [0.0, 1.0],
        "focus_map_vae_encoded": False,
        "in_channels": model.controlnet.in_channels,
        "inner_dim": model.controlnet.inner_dim,
        "hidden_channels": model.controlnet.hidden_channels,
        "num_layers": model.controlnet.num_layers,
        "double_block_indices": list(model.controlnet.double_block_indices),
        "single_block_indices": list(model.controlnet.single_block_indices),
        "conditioning_scale": args.conditioning_scale,
        "resolution": args.resolution,
        "global_step": global_step,
    }
    (directory / "controlnet_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    save_trainer_state(directory, optimizer, global_step, epoch, step_in_epoch)


def main():
    args = parse_args()
    if args.resolution % 16:
        raise ValueError("--resolution must be divisible by 16.")
    if args.control_hidden_channels < 1 or args.control_layers < 1:
        raise ValueError("ControlNet dimensions must be positive.")
    if args.focus_mask_gamma <= 0:
        raise ValueError("--focus_mask_gamma must be positive.")
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

    pipe = Flux2KleinPipeline.from_pretrained(
        args.model, torch_dtype=weight_dtype, **pretrained_kwargs(args)
    )
    transformer = pipe.transformer
    transformer.requires_grad_(False).eval()
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    pipe.vae.requires_grad_(False).eval()
    pipe.text_encoder.requires_grad_(False).eval()
    double_indices = parse_block_indices(args.double_block_indices, len(transformer.transformer_blocks))
    single_indices = parse_block_indices(args.single_block_indices, len(transformer.single_transformer_blocks))
    controlnet = Flux2FocusControlNet(
        in_channels=transformer.config.in_channels,
        inner_dim=transformer.inner_dim,
        hidden_channels=args.control_hidden_channels,
        num_layers=args.control_layers,
        double_block_indices=double_indices,
        single_block_indices=single_indices,
    )
    model = Flux2ControlNetTransformer(transformer, controlnet, args.conditioning_scale)
    pipe.vae.to(accelerator.device, dtype=weight_dtype)
    pipe.text_encoder.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=args.learning_rate, weight_decay=1e-2)
    output_dir = Path(args.output_dir)
    resume_path = resolve_resume_checkpoint(output_dir, args.resume_from_checkpoint)
    resume_state = {"global_step": 0, "epoch": 0, "step_in_epoch": 0}
    if resume_path is not None:
        controlnet.load_state_dict(load_file(resume_path / "controlnet.safetensors"), strict=True)
        resume_state = load_trainer_state(resume_path, optimizer)
    scheduler = copy.deepcopy(pipe.scheduler)
    latent_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(accelerator.device, dtype=weight_dtype)
    latent_std = torch.sqrt(
        pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps
    ).to(accelerator.device, dtype=weight_dtype)

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
        batch = paired_preprocess(samples, args.resolution, pipe.image_processor, training=True)
        batch["focus_map"] = batch["focus_a"] if args.control_index == 2 else batch["focus_b"]
        return batch

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

    def encode_latents(images):
        latents = pipe.vae.encode(images.to(accelerator.device, dtype=weight_dtype)).latent_dist.mode()
        latents = Flux2KleinPipeline._patchify_latents(latents)
        return (latents - latent_mean) / latent_std

    def get_sigmas(timesteps, ndim, dtype):
        schedule_timesteps = scheduler.timesteps.to(accelerator.device)
        indices = [(schedule_timesteps == timestep).nonzero().item() for timestep in timesteps]
        sigma = scheduler.sigmas.to(accelerator.device, dtype=dtype)[indices].flatten()
        while sigma.ndim < ndim:
            sigma = sigma.unsqueeze(-1)
        return sigma

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
                    target = encode_latents(batch["target"])
                    cond_a = encode_latents(batch["cond_a"])
                    cond_b = encode_latents(batch["cond_b"])
                    prompt_embeds, text_ids = pipe.encode_prompt(
                        prompt=batch["prompts"], device=accelerator.device, max_sequence_length=512
                    )
                    prompt_embeds = prompt_embeds.to(weight_dtype)
                    density = compute_density_for_timestep_sampling(
                        weighting_scheme=args.weighting_scheme,
                        batch_size=target.shape[0],
                        logit_mean=args.logit_mean,
                        logit_std=args.logit_std,
                        mode_scale=args.mode_scale,
                    )
                    indices = (density * scheduler.config.num_train_timesteps).long()
                    timesteps = scheduler.timesteps[indices].to(accelerator.device)
                    sigmas = get_sigmas(timesteps, target.ndim, target.dtype)
                    noise = torch.randn_like(target)
                    noisy = (1 - sigmas) * target + sigmas * noise
                    latent_ids = Flux2KleinPipeline._prepare_latent_ids(target).to(accelerator.device)
                    condition_ids = Flux2KleinPipeline._prepare_image_ids([cond_a[:1], cond_b[:1]])
                    condition_ids = condition_ids.to(accelerator.device).expand(target.shape[0], -1, -1)
                    packed_noisy = Flux2KleinPipeline._pack_latents(noisy)
                    packed_a = Flux2KleinPipeline._pack_latents(cond_a)
                    packed_b = Flux2KleinPipeline._pack_latents(cond_b)
                    target_token_count = packed_noisy.shape[1]
                    transformer_input = torch.cat([packed_noisy, packed_a, packed_b], dim=1)
                    transformer_ids = torch.cat([latent_ids, condition_ids], dim=1)
                    focus = F.interpolate(
                        batch["focus_map"].to(accelerator.device),
                        size=target.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ).clamp(0, 1)
                    focus_tokens = focus.flatten(2).transpose(1, 2).contiguous()
                    guidance = None
                    if transformer.config.guidance_embeds:
                        guidance = torch.full(
                            (target.shape[0],), args.guidance_scale, device=accelerator.device, dtype=weight_dtype
                        )

                prediction = model(
                    hidden_states=transformer_input,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=transformer_ids,
                    controlnet_cond=focus_tokens,
                    conditioning_scale=args.conditioning_scale,
                    return_dict=False,
                )[0][:, :target_token_count]
                prediction = Flux2KleinPipeline._unpack_latents_with_ids(prediction, latent_ids)
                target_velocity = noise - target
                weighting = compute_loss_weighting_for_sd3(args.weighting_scheme, sigmas=sigmas)
                error = prediction.float() - target_velocity.float()
                loss = (weighting.float() * error.square()).reshape(target.shape[0], -1).mean(dim=1).mean()
                if args.focus_loss_weight > 0:
                    predicted_target = noisy - sigmas * prediction
                    mask = focus.pow(args.focus_mask_gamma)
                    focus_weight = args.focus_keep_weight * mask + args.focus_blur_weight * (1 - mask)
                    loss = loss + args.focus_loss_weight * (
                        (predicted_target.float() - target.float()).abs() * focus_weight
                    ).mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), 1.0)
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
