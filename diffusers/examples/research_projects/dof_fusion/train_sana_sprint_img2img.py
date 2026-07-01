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

from diffusers import SanaSprintImg2ImgPipeline

from focus_dataset import DiffSynthFocusDataset
from sana_dof import ConditionedSanaTransformer, DualImageConditionAdapter


def parse_args():
    parser = argparse.ArgumentParser(description="Train a SANA-Sprint A-initialized img2img A/B fusion adapter.")
    parser.add_argument("--model", default="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers")
    parser.add_argument("--dataset_metadata_path", required=True)
    parser.add_argument("--dataset_base_path", default=".")
    parser.add_argument("--target_key", default="image")
    parser.add_argument("--edit_key", default="edit_image")
    parser.add_argument("--dataset_repeat", type=int, default=1)
    focus = parser.add_mutually_exclusive_group()
    focus.add_argument("--use_focus_maps", dest="use_focus_maps", action="store_true")
    focus.add_argument("--no_use_focus_maps", dest="use_focus_maps", action="store_false")
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
    parser.add_argument("--min_timestep", type=float, default=0.2)
    parser.add_argument("--max_timestep", type=float, default=1.57080)
    parser.add_argument("--max_timestep_probability", type=float, default=0.25)
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
    if not 0 <= args.min_timestep < args.max_timestep <= 1.57080:
        raise ValueError("Require 0 <= min_timestep < max_timestep <= 1.57080.")
    if not 0 <= args.max_timestep_probability <= 1:
        raise ValueError("--max_timestep_probability must be in [0, 1].")
    if args.focus_loss_weight > 0 and not args.use_focus_maps:
        raise ValueError("--focus_loss_weight > 0 requires --use_focus_maps.")

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
    pipe = SanaSprintImg2ImgPipeline.from_pretrained(args.model, torch_dtype=weight_dtype)
    pipe.transformer.requires_grad_(False).eval()
    pipe.text_encoder.requires_grad_(False).eval()
    pipe.vae.requires_grad_(False).eval()
    pipe.text_encoder.to(accelerator.device, dtype=weight_dtype)
    pipe.vae.to(accelerator.device, dtype=torch.float32)
    pipe.transformer.to(accelerator.device, dtype=weight_dtype)
    adapter = DualImageConditionAdapter(pipe.transformer.config.in_channels, args.adapter_hidden_channels)
    adapter.to(accelerator.device, dtype=torch.float32)
    model = ConditionedSanaTransformer(pipe.transformer, adapter)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=1e-2)

    dataset = DiffSynthFocusDataset(
        metadata_path=args.dataset_metadata_path,
        base_path=args.dataset_base_path,
        target_key=args.target_key,
        edit_key=args.edit_key,
        repeat=args.dataset_repeat,
        min_edit_images=2,
        use_focus_maps=args.use_focus_maps,
        default_prompt=args.prompt,
    )

    def collate_fn(samples):
        focus_maps = None
        focus_valid = None
        if args.use_focus_maps:
            present = [len(sample["cond_images"]) > 2 for sample in samples]
            focus_maps = torch.zeros(len(samples), 1, args.resolution, args.resolution)
            focus_valid = torch.tensor(present, dtype=torch.float32)
            for index, sample in enumerate(samples):
                if present[index]:
                    focus = sample["cond_images"][2].convert("L").resize(
                        (args.resolution, args.resolution), Image.Resampling.BILINEAR
                    )
                    focus_maps[index, 0] = torch.from_numpy(np.asarray(focus, dtype=np.float32) / 255.0)
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
            "focus_a": focus_maps,
            "focus_a_valid": focus_valid,
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

    global_step = 0
    epoch = 0
    while global_step < args.max_train_steps:
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)
        for batch in dataloader:
            with accelerator.accumulate(model):
                with torch.no_grad():
                    scaling = pipe.vae.config.scaling_factor
                    target = pipe.vae.encode(batch["target"].to(accelerator.device, torch.float32)).latent * scaling
                    cond_a = pipe.vae.encode(batch["cond_a"].to(accelerator.device, torch.float32)).latent * scaling
                    cond_b = pipe.vae.encode(batch["cond_b"].to(accelerator.device, torch.float32)).latent * scaling
                    prompt_embeds, prompt_mask = pipe.encode_prompt(
                        batch["prompts"],
                        device=accelerator.device,
                        clean_caption=False,
                        max_sequence_length=300,
                    )
                    prompt_embeds = prompt_embeds.to(weight_dtype)
                    timestep = torch.empty(target.shape[0], device=accelerator.device).uniform_(
                        args.min_timestep, args.max_timestep
                    )
                    use_maximum = torch.rand_like(timestep) < args.max_timestep_probability
                    timestep = torch.where(use_maximum, torch.full_like(timestep, args.max_timestep), timestep)
                    expanded = timestep.view(-1, 1, 1, 1)
                    sigma_data = pipe.scheduler.config.sigma_data
                    init_state = cond_a * sigma_data
                    noise = torch.randn_like(init_state) * sigma_data
                    noisy_state = torch.cos(expanded) * init_state + torch.sin(expanded) * noise
                    scm_timestep = torch.sin(timestep) / (torch.cos(timestep) + torch.sin(timestep))
                    scm_expanded = scm_timestep.view(-1, 1, 1, 1)
                    normalization = torch.sqrt(scm_expanded.square() + (1 - scm_expanded).square())
                    model_input = noisy_state / sigma_data * normalization

                guidance = torch.full(
                    (target.shape[0],),
                    args.guidance_scale * pipe.transformer.config.guidance_embeds_scale,
                    device=accelerator.device,
                    dtype=weight_dtype,
                )
                raw_prediction = model(
                    model_input.to(weight_dtype),
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_mask,
                    guidance=guidance,
                    timestep=scm_timestep,
                    cond_a_latents=cond_a,
                    cond_b_latents=cond_b,
                    return_dict=False,
                )[0]
                velocity = (
                    (1 - 2 * scm_expanded) * model_input
                    + (1 - 2 * scm_expanded + 2 * scm_expanded.square()) * raw_prediction.float()
                ) / normalization
                velocity = velocity * sigma_data
                predicted_target = (
                    torch.cos(expanded) * noisy_state - torch.sin(expanded) * velocity
                ) / sigma_data
                error = predicted_target.float() - target.float()
                loss = error.square().mean() + 0.1 * error.abs().mean()
                if args.focus_loss_weight > 0 and batch["focus_a"] is not None:
                    focus = F.interpolate(
                        batch["focus_a"].to(accelerator.device),
                        size=target.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ).pow(args.focus_mask_gamma)
                    weight = args.focus_keep_weight * focus + args.focus_blur_weight * (1 - focus)
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
                if accelerator.is_main_process and global_step % 10 == 0:
                    print(f"step={global_step} loss={loss.detach().item():.6f}")
                if global_step % args.save_steps == 0 or global_step == args.max_train_steps:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        unwrapped = accelerator.unwrap_model(model)
                        save_file(
                            {
                                key: value.detach().cpu().contiguous()
                                for key, value in unwrapped.adapter.state_dict().items()
                            },
                            output_dir / "adapter.safetensors",
                        )
                        config = {
                            "base_model": args.model,
                            "model_type": "sana_sprint_img2img_dof",
                            "target_key": args.target_key,
                            "edit_key": args.edit_key,
                            "cond_format": "edit_image[A,B,optional_focus_a,optional_focus_b]",
                            "init_source": "A",
                            "use_focus_maps": args.use_focus_maps,
                            "latent_channels": pipe.transformer.config.in_channels,
                            "hidden_channels": args.adapter_hidden_channels,
                            "resolution": args.resolution,
                            "min_timestep": args.min_timestep,
                            "max_timestep": args.max_timestep,
                            "global_step": global_step,
                        }
                        (output_dir / "adapter_config.json").write_text(
                            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
                        )
                if global_step >= args.max_train_steps:
                    break
        epoch += 1
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
