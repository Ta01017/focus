#!/usr/bin/env python

import argparse
import copy
import json
import random
from pathlib import Path

import torch
from accelerate import Accelerator
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from torch.nn import functional as F
from torch.utils.data import DataLoader

from diffusers import Flux2KleinPipeline
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import convert_unet_state_dict_to_peft

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


def parse_args():
    parser = argparse.ArgumentParser(description="Train FLUX.2 Klein LoRA for paired depth-of-field fusion.")
    parser.add_argument("--model", default="black-forest-labs/FLUX.2-klein-4B")
    add_metadata_args(parser, metadata_required=True)
    add_pretrained_args(parser)
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--min_edit_images", type=int, default=2)
    focus_group = parser.add_mutually_exclusive_group()
    focus_group.add_argument("--use_focus_maps", dest="use_focus_maps", action="store_true")
    focus_group.add_argument("--no_use_focus_maps", dest="use_focus_maps", action="store_false")
    parser.set_defaults(use_focus_maps=False)
    parser.add_argument("--prompt", default="a photorealistic all-in-focus photograph")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_train_steps", type=int, default=20000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--weighting_scheme", default="none")
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument("--focus_loss_weight", type=float, default=0.0)
    parser.add_argument("--focus_keep_weight", type=float, default=0.3)
    parser.add_argument("--focus_blur_weight", type=float, default=1.0)
    parser.add_argument("--focus_mask_gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--resume_from_checkpoint", default=None)
    return parser.parse_args()


def validate_args(args):
    if args.resolution % 16:
        raise ValueError("--resolution must be divisible by 16 for FLUX.2 Klein.")
    if args.min_edit_images < 2:
        raise ValueError("--min_edit_images must be at least 2.")
    if args.focus_mask_gamma <= 0:
        raise ValueError("--focus_mask_gamma must be greater than 0.")
    if args.focus_loss_weight > 0 and not args.use_focus_maps:
        raise ValueError("--focus_loss_weight > 0 requires --use_focus_maps.")
    if args.focus_loss_weight < 0 or args.focus_keep_weight < 0 or args.focus_blur_weight < 0:
        raise ValueError("Focus loss weights must be non-negative.")


def save_lora(accelerator, transformer, optimizer, save_directory, args, global_step, epoch, step_in_epoch):
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    unwrapped = accelerator.unwrap_model(transformer)
    lora_state = {
        key: value.detach().cpu().contiguous()
        for key, value in get_peft_model_state_dict(unwrapped).items()
    }
    save_directory.mkdir(parents=True, exist_ok=True)
    Flux2KleinPipeline.save_lora_weights(
        save_directory=save_directory,
        transformer_lora_layers=lora_state,
        transformer_lora_adapter_metadata=unwrapped.peft_config["default"].to_dict(),
    )
    config = {
        "base_model": args.model,
        "target_key": args.target_key,
        "edit_key": args.edit_key,
        "cond_format": "edit_image[A,B,optional_focus_a,optional_focus_b]",
        "use_focus_maps": args.use_focus_maps,
        "resolution": args.resolution,
        "rank": args.rank,
        "lora_alpha": args.lora_alpha,
        "global_step": global_step,
    }
    (save_directory / "training_config.json").write_text(
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

    weight_dtype = torch.float32
    if accelerator.device.type == "cuda" and args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    elif accelerator.device.type == "cuda" and args.mixed_precision == "fp16":
        weight_dtype = torch.float16

    pipe = Flux2KleinPipeline.from_pretrained(
        args.model, torch_dtype=weight_dtype, **pretrained_kwargs(args)
    )
    transformer = pipe.transformer
    vae = pipe.vae
    scheduler = copy.deepcopy(pipe.scheduler)
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    vae.eval()
    pipe.text_encoder.eval()

    target_modules = ["to_k", "to_q", "to_v", "to_out.0", "to_qkv_mlp_proj"] + [
        f"single_transformer_blocks.{index}.attn.to_out"
        for index in range(len(transformer.single_transformer_blocks))
    ]
    transformer.add_adapter(
        LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
    )
    cast_training_params([transformer], dtype=torch.float32)
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    vae.to(accelerator.device, dtype=weight_dtype)
    pipe.text_encoder.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device)
    latent_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(accelerator.device, dtype=weight_dtype)
    latent_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
        accelerator.device, dtype=weight_dtype
    )

    trainable_parameters = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=1e-2)
    output_dir = Path(args.output_dir)
    resume_path = resolve_resume_checkpoint(output_dir, args.resume_from_checkpoint)
    resume_state = {"global_step": 0, "epoch": 0, "step_in_epoch": 0}
    if resume_path is not None:
        lora_state = Flux2KleinPipeline.lora_state_dict(resume_path)
        transformer_state = {
            key.replace("transformer.", ""): value
            for key, value in lora_state.items()
            if key.startswith("transformer.")
        }
        transformer_state = convert_unet_state_dict_to_peft(transformer_state)
        set_peft_model_state_dict(transformer, transformer_state, adapter_name="default")
        resume_state = load_trainer_state(resume_path, optimizer)
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
        drop_last=False,
    )
    transformer, optimizer, dataloader = accelerator.prepare(transformer, optimizer, dataloader)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    def get_sigmas(timesteps, ndim, dtype):
        sigmas = scheduler.sigmas.to(accelerator.device, dtype=dtype)
        schedule_timesteps = scheduler.timesteps.to(accelerator.device)
        indices = [(schedule_timesteps == timestep).nonzero().item() for timestep in timesteps]
        sigma = sigmas[indices].flatten()
        while sigma.ndim < ndim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    global_step = int(resume_state["global_step"])
    epoch = int(resume_state["epoch"])
    resume_step = int(resume_state["step_in_epoch"])
    while global_step < args.max_train_steps:
        dataloader.sampler.set_epoch(epoch) if hasattr(dataloader.sampler, "set_epoch") else None
        for step, batch in enumerate(dataloader):
            if epoch == int(resume_state["epoch"]) and step < resume_step:
                continue
            with accelerator.accumulate(transformer):
                with torch.no_grad():
                    target = batch["target"].to(accelerator.device, dtype=weight_dtype)
                    cond_a = batch["cond_a"].to(accelerator.device, dtype=weight_dtype)
                    cond_b = batch["cond_b"].to(accelerator.device, dtype=weight_dtype)
                    target_latents = vae.encode(target).latent_dist.mode()
                    cond_a_latents = vae.encode(cond_a).latent_dist.mode()
                    cond_b_latents = vae.encode(cond_b).latent_dist.mode()

                    target_latents = Flux2KleinPipeline._patchify_latents(target_latents)
                    cond_a_latents = Flux2KleinPipeline._patchify_latents(cond_a_latents)
                    cond_b_latents = Flux2KleinPipeline._patchify_latents(cond_b_latents)
                    target_latents = (target_latents - latent_mean) / latent_std
                    cond_a_latents = (cond_a_latents - latent_mean) / latent_std
                    cond_b_latents = (cond_b_latents - latent_mean) / latent_std

                    prompt_embeds, text_ids = pipe.encode_prompt(
                        prompt=batch["prompts"],
                        device=accelerator.device,
                        max_sequence_length=512,
                    )
                    prompt_embeds = prompt_embeds.to(weight_dtype)
                    text_ids = text_ids.to(accelerator.device)

                    density = compute_density_for_timestep_sampling(
                        weighting_scheme=args.weighting_scheme,
                        batch_size=target_latents.shape[0],
                        logit_mean=args.logit_mean,
                        logit_std=args.logit_std,
                        mode_scale=args.mode_scale,
                    )
                    indices = (density * scheduler.config.num_train_timesteps).long()
                    timesteps = scheduler.timesteps[indices].to(accelerator.device)
                    sigmas = get_sigmas(timesteps, target_latents.ndim, target_latents.dtype)
                    noise = torch.randn_like(target_latents)
                    noisy_latents = (1 - sigmas) * target_latents + sigmas * noise

                    latent_ids = Flux2KleinPipeline._prepare_latent_ids(target_latents).to(accelerator.device)
                    condition_ids = Flux2KleinPipeline._prepare_image_ids(
                        [cond_a_latents[0:1], cond_b_latents[0:1]]
                    ).to(accelerator.device)
                    condition_ids = condition_ids.expand(target_latents.shape[0], -1, -1)

                    packed_noisy = Flux2KleinPipeline._pack_latents(noisy_latents)
                    packed_cond_a = Flux2KleinPipeline._pack_latents(cond_a_latents)
                    packed_cond_b = Flux2KleinPipeline._pack_latents(cond_b_latents)
                    target_token_count = packed_noisy.shape[1]
                    transformer_input = torch.cat([packed_noisy, packed_cond_a, packed_cond_b], dim=1)
                    transformer_ids = torch.cat([latent_ids, condition_ids], dim=1)

                guidance = None
                if accelerator.unwrap_model(transformer).config.guidance_embeds:
                    guidance = torch.full(
                        (target_latents.shape[0],),
                        args.guidance_scale,
                        device=accelerator.device,
                        dtype=weight_dtype,
                    )
                model_prediction = transformer(
                    hidden_states=transformer_input,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=transformer_ids,
                    return_dict=False,
                )[0]
                model_prediction = model_prediction[:, :target_token_count]
                model_prediction = Flux2KleinPipeline._unpack_latents_with_ids(model_prediction, latent_ids)

                target_velocity = noise - target_latents
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme,
                    sigmas=sigmas,
                )
                loss = torch.mean(
                    (weighting.float() * (model_prediction.float() - target_velocity.float()) ** 2).reshape(
                        target_latents.shape[0], -1
                    ),
                    dim=1,
                ).mean()

                if args.focus_loss_weight > 0 and batch["focus_a"] is not None:
                    predicted_target = noisy_latents - sigmas * model_prediction
                    focus_a = F.interpolate(
                        batch["focus_a"].to(accelerator.device, dtype=torch.float32),
                        size=target_latents.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ).clamp(0, 1)
                    keep_mask = focus_a.pow(args.focus_mask_gamma)
                    focus_weight = (
                        args.focus_keep_weight * keep_mask + args.focus_blur_weight * (1 - keep_mask)
                    )
                    weighted_l1_per_sample = (
                        (predicted_target.float() - target_latents.float()).abs() * focus_weight
                    ).mean(dim=(1, 2, 3))
                    focus_valid = batch["focus_a_valid"].to(accelerator.device)
                    weighted_l1 = (weighted_l1_per_sample * focus_valid).sum() / focus_valid.sum().clamp_min(1)
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
                    save_lora(
                        accelerator,
                        transformer,
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

    save_lora(accelerator, transformer, optimizer, output_dir, args, global_step, epoch, 0)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
