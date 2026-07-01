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

from diffusers import SanaSprintPipeline

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
from sana_dof import ConditionedSanaTransformer, create_condition_adapter, encode_vae_latents


def parse_args():
    parser = argparse.ArgumentParser(description="训练 SANA-Sprint 双图景深融合 Adapter。")
    parser.add_argument("--model", default="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers")
    add_metadata_args(parser, metadata_required=True)
    add_pretrained_args(parser)
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--min_edit_images", type=int, default=2)
    focus_group = parser.add_mutually_exclusive_group()
    focus_group.add_argument("--use_focus_maps", dest="use_focus_maps", action="store_true")
    focus_group.add_argument("--no_use_focus_maps", dest="use_focus_maps", action="store_false")
    parser.set_defaults(use_focus_maps=False)
    parser.add_argument("--adapter_type", choices=("ab", "ab_focus"), default="ab")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt", default="a photorealistic all-in-focus photograph")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_train_steps", type=int, default=20000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--adapter_hidden_channels", type=int, default=128)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--max_timestep", type=float, default=1.57080)
    parser.add_argument("--focus_loss_weight", type=float, default=0.0)
    parser.add_argument("--focus_keep_weight", type=float, default=0.3)
    parser.add_argument("--focus_blur_weight", type=float, default=1.0)
    parser.add_argument("--focus_mask_gamma", type=float, default=1.0)
    parser.add_argument("--resume_from_checkpoint", default=None, help="checkpoint-N 路径或 latest。")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="bf16")
    return parser.parse_args()


def adapter_config(args, latent_channels, global_step):
    return {
        "base_model": args.model,
        "model_type": "sana_sprint_dof",
        "adapter_type": args.adapter_type,
        "target_key": args.target_key,
        "edit_key": args.edit_key,
        "prompt_key": args.prompt_key,
        "cond_format": "edit_image[A,B,optional_focus_a,optional_focus_b]",
        "use_focus_maps": args.use_focus_maps,
        "latent_channels": latent_channels,
        "hidden_channels": args.adapter_hidden_channels,
        "resolution": args.resolution,
        "prompt": args.prompt,
        "global_step": global_step,
    }


def save_adapter(accelerator, model, optimizer, directory, args, global_step, epoch, step_in_epoch):
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    unwrapped = accelerator.unwrap_model(model)
    directory.mkdir(parents=True, exist_ok=True)
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in unwrapped.adapter.state_dict().items()},
        directory / "adapter.safetensors",
    )
    (directory / "adapter_config.json").write_text(
        json.dumps(adapter_config(args, unwrapped.config.in_channels, global_step), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    save_trainer_state(directory, optimizer, global_step, epoch, step_in_epoch)


def main():
    args = parse_args()
    if args.resolution % 32:
        raise ValueError("--resolution 必须能被 32 整除。")
    if args.adapter_type == "ab_focus":
        args.use_focus_maps = True
        args.min_edit_images = max(args.min_edit_images, 4)
    if args.focus_loss_weight > 0 and not args.use_focus_maps:
        raise ValueError("--focus_loss_weight > 0 需要 --use_focus_maps。")
    if args.focus_mask_gamma <= 0:
        raise ValueError("--focus_mask_gamma 必须大于 0。")

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
    pipe.transformer.requires_grad_(False).eval()
    pipe.text_encoder.requires_grad_(False).eval()
    pipe.vae.requires_grad_(False).eval()
    pipe.text_encoder.to(accelerator.device, dtype=transformer_dtype)
    pipe.vae.to(accelerator.device, dtype=torch.float32)
    pipe.transformer.to(accelerator.device, dtype=transformer_dtype)
    adapter = create_condition_adapter(
        args.adapter_type, pipe.transformer.config.in_channels, args.adapter_hidden_channels
    ).to(accelerator.device, dtype=torch.float32)
    model = ConditionedSanaTransformer(pipe.transformer, adapter)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=1e-2)

    output_dir = Path(args.output_dir)
    resume_path = resolve_resume_checkpoint(output_dir, args.resume_from_checkpoint)
    resume_state = {"global_step": 0, "epoch": 0, "step_in_epoch": 0}
    if resume_path is not None:
        adapter.load_state_dict(load_file(resume_path / "adapter.safetensors"), strict=True)
        resume_state = load_trainer_state(resume_path, optimizer)
        print(f"从 {resume_path} 恢复，global_step={resume_state['global_step']}")

    dataset = DiffSynthFocusDataset(
        metadata_path=args.dataset_metadata_path,
        base_path=args.dataset_base_path,
        target_key=args.target_key,
        edit_key=args.edit_key,
        repeat=args.dataset_repeat,
        min_edit_images=args.min_edit_images,
        use_focus_maps=args.use_focus_maps,
        default_prompt=args.prompt,
        prompt_key=args.prompt_key,
        start_index=args.start_index,
        max_samples=args.max_samples,
    )

    def collate_fn(samples):
        return paired_preprocess(samples, args.resolution, pipe.image_processor, training=True)

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
                    target_latents = encode_vae_latents(
                        pipe.vae, batch["target"].to(accelerator.device, torch.float32)
                    )
                    cond_a_latents = encode_vae_latents(
                        pipe.vae, batch["cond_a"].to(accelerator.device, torch.float32)
                    )
                    cond_b_latents = encode_vae_latents(
                        pipe.vae, batch["cond_b"].to(accelerator.device, torch.float32)
                    )
                    focus_a = batch["focus_a"]
                    focus_b = batch["focus_b"]
                    focus_a_latent = None
                    focus_b_latent = None
                    if focus_a is not None:
                        focus_a_latent = F.interpolate(
                            focus_a.to(accelerator.device),
                            size=target_latents.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        ).clamp(0, 1)
                    if focus_b is not None:
                        focus_b_latent = F.interpolate(
                            focus_b.to(accelerator.device),
                            size=target_latents.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        ).clamp(0, 1)
                    prompt_embeds, prompt_mask = pipe.encode_prompt(
                        batch["prompts"], device=accelerator.device, clean_caption=False, max_sequence_length=300
                    )
                    prompt_embeds = prompt_embeds.to(transformer_dtype)
                    sigma_data = pipe.scheduler.config.sigma_data
                    timestep = torch.full(
                        (target_latents.shape[0],), args.max_timestep, device=accelerator.device, dtype=torch.float32
                    )
                    expanded = timestep.view(-1, 1, 1, 1)
                    target_state = target_latents * sigma_data
                    noise = torch.randn_like(target_state) * sigma_data
                    noisy_state = torch.cos(expanded) * target_state + torch.sin(expanded) * noise
                    scm_timestep = torch.sin(timestep) / (torch.cos(timestep) + torch.sin(timestep))
                    scm_expanded = scm_timestep.view(-1, 1, 1, 1)
                    normalization = torch.sqrt(scm_expanded.square() + (1 - scm_expanded).square())
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
                    encoder_attention_mask=prompt_mask,
                    guidance=guidance,
                    timestep=scm_timestep,
                    cond_a_latents=cond_a_latents,
                    cond_b_latents=cond_b_latents,
                    focus_a=focus_a_latent,
                    focus_b=focus_b_latent,
                    return_dict=False,
                )[0]
                velocity = (
                    (1 - 2 * scm_expanded) * model_input
                    + (1 - 2 * scm_expanded + 2 * scm_expanded.square()) * raw_prediction.float()
                ) / normalization
                velocity = velocity * sigma_data
                predicted_target = (torch.cos(expanded) * noisy_state - torch.sin(expanded) * velocity) / sigma_data
                error = predicted_target.float() - target_latents.float()
                loss = error.square().mean() + 0.1 * error.abs().mean()
                if args.focus_loss_weight > 0 and focus_a_latent is not None:
                    keep = focus_a_latent.pow(args.focus_mask_gamma)
                    weight = args.focus_keep_weight * keep + args.focus_blur_weight * (1 - keep)
                    per_sample = (error.abs() * weight).mean(dim=(1, 2, 3))
                    valid = batch["focus_a_valid"].to(accelerator.device)
                    loss = loss + args.focus_loss_weight * (per_sample * valid).sum() / valid.sum().clamp_min(1)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(adapter.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process:
                    print(f"step={global_step} loss={loss.detach().item():.6f}")
                if global_step % args.save_steps == 0 or global_step == args.max_train_steps:
                    save_adapter(
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

    save_adapter(accelerator, model, optimizer, output_dir, args, global_step, epoch, 0)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
