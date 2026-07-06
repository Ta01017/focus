#!/usr/bin/env python3

import argparse
import copy
import json
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from diffusers import FlowMatchEulerDiscreteScheduler, SanaPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3

from conditioning import ConditionedSanaTransformer, MultiImageConditionAdapter, encode_condition_batch, encode_latents
from metadata_dataset import DiffSynthMetadataDataset, make_collate_fn
from train_utils import (
    COMPLEX_HUMAN_INSTRUCTION,
    DEFAULT_LORA_TARGETS,
    add_lora,
    get_weight_dtype,
    load_adapter_weights,
    load_lora_weights,
    save_training_weights,
    trainable_parameters,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train an image-conditioned Sana LoRA from DiffSynth metadata.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_teacher_diffusers",
    )
    parser.add_argument("--dataset_base_path", required=True)
    parser.add_argument("--dataset_metadata_path", required=True)
    parser.add_argument("--data_file_keys", default="image,edit_image")
    parser.add_argument("--extra_inputs", default="edit_image")
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--default_prompt", default="")
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_train_steps", type=int, default=20000)
    parser.add_argument("--lr_scheduler", default="constant_with_warmup")
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_target_modules", default=DEFAULT_LORA_TARGETS)
    parser.add_argument("--adapter_hidden_channels", type=int, default=128)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--weighting_scheme",
        choices=("none", "sigma_sqrt", "logit_normal", "mode", "cosmap"),
        default="none",
    )
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume_from_checkpoint")
    return parser.parse_args()


def parse_data_keys(args):
    keys = [key.strip() for key in args.data_file_keys.split(",") if key.strip()]
    extras = [key.strip() for key in args.extra_inputs.split(",") if key.strip()]
    if len(keys) != 2 or len(extras) != 1 or extras[0] != keys[1]:
        raise ValueError(
            "This trainer expects --data_file_keys 'TARGET,CONDITION' and --extra_inputs 'CONDITION'."
        )
    return keys


def main():
    args = parse_args()
    target_key, condition_key = parse_data_keys(args)
    if args.resume_from_checkpoint == "latest":
        checkpoints = sorted(
            Path(args.output_dir).glob("checkpoint-*"),
            key=lambda path: int(path.name.rsplit("-", 1)[1]),
        )
        if not checkpoints:
            raise ValueError(f"No checkpoint-* directories found in {args.output_dir}.")
        args.resume_from_checkpoint = str(checkpoints[-1])
    set_seed(args.seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    weight_dtype = get_weight_dtype(accelerator.mixed_precision)

    dataset = DiffSynthMetadataDataset(
        args.dataset_metadata_path,
        args.dataset_base_path,
        target_key=target_key,
        condition_key=condition_key,
        prompt_key=args.prompt_key,
        default_prompt=args.default_prompt,
        dataset_repeat=args.dataset_repeat,
        max_samples=args.max_samples,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        collate_fn=make_collate_fn(args.resolution, args.center_crop, args.random_flip),
    )

    pipe = SanaPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=weight_dtype)
    pipe.vae.requires_grad_(False).to(accelerator.device, dtype=torch.float32)
    pipe.text_encoder.requires_grad_(False).to(accelerator.device, dtype=weight_dtype)
    pipe.transformer.to(accelerator.device, dtype=weight_dtype)
    add_lora(
        pipe.transformer,
        args.rank,
        args.lora_alpha,
        args.lora_target_modules,
        args.gradient_checkpointing,
    )
    adapter = MultiImageConditionAdapter(
        pipe.transformer.config.in_channels,
        dataset.num_condition_images,
        args.adapter_hidden_channels,
    ).to(accelerator.device, dtype=torch.float32)
    if args.resume_from_checkpoint:
        load_lora_weights(pipe.transformer, args.resume_from_checkpoint)
        load_adapter_weights(adapter, args.resume_from_checkpoint)
    model = ConditionedSanaTransformer(pipe.transformer, adapter)

    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=args.learning_rate, weight_decay=1e-2)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler = copy.deepcopy(noise_scheduler)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def get_sigmas(timesteps, n_dim, dtype):
        schedule = noise_scheduler.timesteps.to(accelerator.device)
        indices = [(schedule == timestep).nonzero().item() for timestep in timesteps]
        sigmas = noise_scheduler.sigmas.to(accelerator.device, dtype=dtype)[indices]
        while sigmas.ndim < n_dim:
            sigmas = sigmas.unsqueeze(-1)
        return sigmas

    config = {
        "base_model": args.pretrained_model_name_or_path,
        "pipeline_type": "sana",
        "target_key": target_key,
        "condition_key": condition_key,
        "prompt_key": args.prompt_key,
        "num_condition_images": dataset.num_condition_images,
        "latent_channels": pipe.transformer.config.in_channels,
        "hidden_channels": args.adapter_hidden_channels,
        "resolution": args.resolution,
    }
    global_step = 0
    if args.resume_from_checkpoint:
        checkpoint_config = Path(args.resume_from_checkpoint) / "condition_adapter.json"
        global_step = json.loads(checkpoint_config.read_text(encoding="utf-8")).get("global_step", 0)
    progress = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Sana SFT",
    )
    model.train()
    while global_step < args.max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(model):
                with torch.no_grad():
                    target_latents = encode_latents(
                        pipe.vae, batch["target"].to(accelerator.device, dtype=torch.float32)
                    ).to(weight_dtype)
                    condition_latents = encode_condition_batch(
                        pipe.vae, batch["conditions"].to(accelerator.device, dtype=torch.float32)
                    )
                    prompt_embeds, prompt_mask, _, _ = pipe.encode_prompt(
                        batch["prompts"],
                        do_classifier_free_guidance=False,
                        device=accelerator.device,
                        clean_caption=False,
                        max_sequence_length=300,
                        complex_human_instruction=COMPLEX_HUMAN_INSTRUCTION,
                    )
                    noise = torch.randn_like(target_latents)
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=args.weighting_scheme,
                        batch_size=target_latents.shape[0],
                        logit_mean=args.logit_mean,
                        logit_std=args.logit_std,
                        mode_scale=args.mode_scale,
                    )
                    indices = (u * noise_scheduler.config.num_train_timesteps).long()
                    timesteps = noise_scheduler.timesteps[indices].to(accelerator.device)
                    sigmas = get_sigmas(timesteps, target_latents.ndim, target_latents.dtype)
                    noisy_latents = (1 - sigmas) * target_latents + sigmas * noise
                    target = noise - target_latents

                prediction = model(
                    noisy_latents,
                    encoder_hidden_states=prompt_embeds.to(weight_dtype),
                    encoder_attention_mask=prompt_mask,
                    timestep=timesteps,
                    condition_latents=condition_latents,
                    return_dict=False,
                )[0]
                weighting = compute_loss_weighting_for_sd3(args.weighting_scheme, sigmas=sigmas)
                loss = ((prediction.float() - target.float()).square() * weighting.float()).flatten(1).mean(1).mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_parameters(model), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                progress.set_postfix(loss=f"{loss.detach().item():.4f}")
                if global_step % args.checkpointing_steps == 0:
                    save_training_weights(
                        accelerator, model, output_dir / f"checkpoint-{global_step}", config, global_step
                    )
                if global_step >= args.max_train_steps:
                    break

    save_training_weights(accelerator, model, output_dir, config, global_step)
    accelerator.end_training()


if __name__ == "__main__":
    main()
