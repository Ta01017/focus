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

from diffusers import SanaSprintPipeline

from focus_dataset import DiffSynthFocusDataset
from sana_dof import ConditionedSanaTransformer, DualImageConditionAdapter


def parse_args():
    parser = argparse.ArgumentParser(description="Train an external dual-image adapter for SANA-Sprint.")
    parser.add_argument("--model", default="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers")
    parser.add_argument("--dataset_metadata_path", required=True, help="DiffSynth-style metadata.json or JSONL.")
    parser.add_argument("--dataset_base_path", default=".", help="Base directory for relative paths in metadata.")
    parser.add_argument("--target_key", default="image")
    parser.add_argument("--edit_key", default="edit_image")
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--min_edit_images", type=int, default=2)
    focus_group = parser.add_mutually_exclusive_group()
    focus_group.add_argument("--use_focus_maps", dest="use_focus_maps", action="store_true")
    focus_group.add_argument("--no_use_focus_maps", dest="use_focus_maps", action="store_false")
    parser.set_defaults(use_focus_maps=False)
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="bf16")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.resolution % 32:
        raise ValueError("--resolution must be divisible by 32.")
    if args.min_edit_images < 2:
        raise ValueError("--min_edit_images must be at least 2.")
    if args.focus_mask_gamma <= 0:
        raise ValueError("--focus_mask_gamma must be greater than 0.")
    if args.focus_loss_weight > 0 and not args.use_focus_maps:
        raise ValueError("--focus_loss_weight > 0 requires --use_focus_maps.")
    if args.focus_loss_weight < 0 or args.focus_keep_weight < 0 or args.focus_blur_weight < 0:
        raise ValueError("Focus loss weights must be non-negative.")

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    torch.manual_seed(args.seed)

    transformer_dtype = torch.float32
    if accelerator.device.type == "cuda" and args.mixed_precision == "bf16":
        transformer_dtype = torch.bfloat16
    elif accelerator.device.type == "cuda" and args.mixed_precision == "fp16":
        transformer_dtype = torch.float16

    pipe = SanaSprintPipeline.from_pretrained(args.model, torch_dtype=transformer_dtype)
    pipe.transformer.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.transformer.eval()
    pipe.text_encoder.eval()
    pipe.vae.eval()

    pipe.text_encoder.to(accelerator.device)
    pipe.vae.to(accelerator.device, dtype=torch.float32)
    pipe.transformer.to(accelerator.device, dtype=transformer_dtype)
    adapter = DualImageConditionAdapter(
        latent_channels=pipe.transformer.config.in_channels,
        hidden_channels=args.adapter_hidden_channels,
    ).to(accelerator.device, dtype=torch.float32)
    model = ConditionedSanaTransformer(pipe.transformer, adapter)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=1e-2)

    dataset = DiffSynthFocusDataset(
        metadata_path=args.dataset_metadata_path,
        base_path=args.dataset_base_path,
        target_key=args.target_key,
        edit_key=args.edit_key,
        repeat=args.dataset_repeat,
        min_edit_images=args.min_edit_images,
        use_focus_maps=args.use_focus_maps,
        default_prompt=args.prompt,
    )

    def preprocess_focus_maps(samples, condition_index):
        present = [len(sample["cond_images"]) > condition_index for sample in samples]
        if not any(present):
            return None, None
        masks = torch.zeros(len(samples), 1, args.resolution, args.resolution, dtype=torch.float32)
        valid = torch.tensor(present, dtype=torch.float32).view(-1, 1, 1, 1)
        for batch_index, sample in enumerate(samples):
            if not present[batch_index]:
                continue
            focus = sample["cond_images"][condition_index].convert("L")
            focus = focus.resize((args.resolution, args.resolution), resample=Image.Resampling.BILINEAR)
            focus = torch.from_numpy(np.asarray(focus, dtype=np.float32) / 255.0).unsqueeze(0)
            masks[batch_index] = focus
        return masks, valid

    def collate_fn(samples):
        targets = [sample["target"] for sample in samples]
        cond_a = [sample["cond_images"][0] for sample in samples]
        cond_b = [sample["cond_images"][1] for sample in samples]
        focus_a, focus_a_valid = preprocess_focus_maps(samples, 2) if args.use_focus_maps else (None, None)
        focus_b, focus_b_valid = preprocess_focus_maps(samples, 3) if args.use_focus_maps else (None, None)
        return {
            "target": pipe.image_processor.preprocess(targets, height=args.resolution, width=args.resolution),
            "cond_a": pipe.image_processor.preprocess(cond_a, height=args.resolution, width=args.resolution),
            "cond_b": pipe.image_processor.preprocess(cond_b, height=args.resolution, width=args.resolution),
            "focus_a": focus_a,
            "focus_b": focus_b,
            "focus_a_valid": focus_a_valid,
            "focus_b_valid": focus_b_valid,
            "prompts": [sample["prompt"] for sample in samples],
        }

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

    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    epoch = 0
    while global_step < args.max_train_steps:
        dataloader.sampler.set_epoch(epoch) if hasattr(dataloader.sampler, "set_epoch") else None
        for batch in dataloader:
            with accelerator.accumulate(model):
                with torch.no_grad():
                    cond_a = batch["cond_a"].to(accelerator.device, dtype=torch.float32)
                    cond_b = batch["cond_b"].to(accelerator.device, dtype=torch.float32)
                    target = batch["target"].to(accelerator.device, dtype=torch.float32)
                    scaling_factor = pipe.vae.config.scaling_factor
                    cond_a_latents = pipe.vae.encode(cond_a).latent * scaling_factor
                    cond_b_latents = pipe.vae.encode(cond_b).latent * scaling_factor
                    target_latents = pipe.vae.encode(target).latent * scaling_factor

                    focus_a = batch["focus_a"]
                    focus_b = batch["focus_b"]
                    focus_a_latents = None
                    focus_b_latents = None
                    if focus_a is not None:
                        focus_a_latents = F.interpolate(
                            focus_a.to(accelerator.device, dtype=torch.float32),
                            size=target_latents.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        ).clamp(0, 1)
                    if focus_b is not None:
                        focus_b_latents = F.interpolate(
                            focus_b.to(accelerator.device, dtype=torch.float32),
                            size=target_latents.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        ).clamp(0, 1)

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
                    focus_a=focus_a_latents,
                    focus_b=focus_b_latents,
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

                mse_loss = F.mse_loss(predicted_target.float(), target_latents.float())
                l1_loss = F.l1_loss(predicted_target.float(), target_latents.float())
                loss = mse_loss + 0.1 * l1_loss
                if args.focus_loss_weight > 0 and focus_a_latents is not None:
                    keep_mask = focus_a_latents.pow(args.focus_mask_gamma)
                    blur_mask = 1 - keep_mask
                    focus_weight = args.focus_keep_weight * keep_mask + args.focus_blur_weight * blur_mask
                    weighted_l1_per_sample = (
                        (predicted_target.float() - target_latents.float()).abs() * focus_weight
                    ).mean(dim=(1, 2, 3))
                    focus_a_valid = batch["focus_a_valid"].to(accelerator.device).flatten()
                    weighted_l1 = (weighted_l1_per_sample * focus_a_valid).sum() / focus_a_valid.sum().clamp_min(1)
                    loss = loss + args.focus_loss_weight * weighted_l1
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(adapter.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process and global_step % 10 == 0:
                    print(f"step={global_step} loss={loss.detach().item():.6f}")
                if global_step % args.save_steps == 0 or global_step == args.max_train_steps:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        unwrapped = accelerator.unwrap_model(model)
                        state = {
                            key: value.detach().cpu().contiguous()
                            for key, value in unwrapped.adapter.state_dict().items()
                        }
                        save_file(state, output_dir / "adapter.safetensors")
                        metadata = {
                            "base_model": args.model,
                            "target_key": args.target_key,
                            "edit_key": args.edit_key,
                            "cond_format": "edit_image[A,B,optional_focus_a,optional_focus_b]",
                            "use_focus_maps": args.use_focus_maps,
                            "latent_channels": pipe.transformer.config.in_channels,
                            "hidden_channels": args.adapter_hidden_channels,
                            "resolution": args.resolution,
                            "prompt": args.prompt,
                            "global_step": global_step,
                        }
                        (output_dir / "adapter_config.json").write_text(
                            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
                        )
                if global_step >= args.max_train_steps:
                    break
        epoch += 1

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
