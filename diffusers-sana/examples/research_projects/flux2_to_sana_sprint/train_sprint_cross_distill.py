#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from diffusers import SanaSprintPipeline
from diffusers.optimization import get_scheduler

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
    parser = argparse.ArgumentParser(
        description="Cross-architecture response distillation from black-box FLUX.2 outputs to Sana-Sprint."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
    )
    parser.add_argument("--dataset_base_path", required=True)
    parser.add_argument("--dataset_metadata_path", required=True)
    parser.add_argument("--data_file_keys", default="teacher_image,edit_image")
    parser.add_argument("--extra_inputs", default="edit_image")
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--default_prompt", default="")
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--teacher_description", default="DiffSynth-Studio FLUX.2")
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
    parser.add_argument("--guidance_scales", default="4.0,4.5,5.0")
    parser.add_argument("--max_timestep", type=float, default=1.57080)
    parser.add_argument("--max_timestep_probability", type=float, default=0.5)
    parser.add_argument("--logit_mean", type=float, default=0.2)
    parser.add_argument("--logit_std", type=float, default=1.6)
    parser.add_argument("--l1_loss_weight", type=float, default=0.1)
    parser.add_argument("--condition_dropout", type=float, default=0.0)
    parser.add_argument("--prompt_dropout", type=float, default=0.0)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume_from_checkpoint")
    return parser.parse_args()


def validate_args(args):
    keys = [key.strip() for key in args.data_file_keys.split(",") if key.strip()]
    extras = [key.strip() for key in args.extra_inputs.split(",") if key.strip()]
    if len(keys) != 2 or extras != [keys[1]]:
        raise ValueError("Expected --data_file_keys 'TEACHER_TARGET,CONDITION' and --extra_inputs 'CONDITION'.")
    if not 0 <= args.max_timestep_probability <= 1:
        raise ValueError("--max_timestep_probability must be in [0, 1].")
    if not 0 <= args.condition_dropout < 1 or not 0 <= args.prompt_dropout < 1:
        raise ValueError("Dropout probabilities must be in [0, 1).")
    guidance_scales = [float(value) for value in args.guidance_scales.split(",") if value.strip()]
    if not guidance_scales:
        raise ValueError("--guidance_scales must contain at least one number.")
    return keys, guidance_scales


def main():
    args = parse_args()
    (target_key, condition_key), guidance_scales = validate_args(args)
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

    pipe = SanaSprintPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=weight_dtype)
    pipe.vae.requires_grad_(False).to(accelerator.device, dtype=torch.float32)
    pipe.text_encoder.requires_grad_(False).to(accelerator.device, dtype=weight_dtype)
    pipe.transformer.to(accelerator.device, dtype=weight_dtype)
    add_lora(pipe.transformer, args.rank, args.lora_alpha, args.lora_target_modules, args.gradient_checkpointing)
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
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(model, optimizer, dataloader, lr_scheduler)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "base_model": args.pretrained_model_name_or_path,
        "pipeline_type": "sana_sprint",
        "target_key": target_key,
        "condition_key": condition_key,
        "prompt_key": args.prompt_key,
        "num_condition_images": dataset.num_condition_images,
        "latent_channels": pipe.transformer.config.in_channels,
        "hidden_channels": args.adapter_hidden_channels,
        "resolution": args.resolution,
        "distillation": "cross_architecture_multitimestep_response",
        "teacher": args.teacher_description,
    }

    global_step = 0
    if args.resume_from_checkpoint:
        checkpoint_config = Path(args.resume_from_checkpoint) / "condition_adapter.json"
        global_step = json.loads(checkpoint_config.read_text(encoding="utf-8")).get("global_step", 0)
    progress = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="FLUX.2 -> Sana-Sprint",
    )
    model.train()
    guidance_values = torch.tensor(guidance_scales, device=accelerator.device, dtype=weight_dtype)

    while global_step < args.max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(model):
                with torch.no_grad():
                    target_latents = encode_latents(
                        pipe.vae, batch["target"].to(accelerator.device, dtype=torch.float32)
                    )
                    condition_latents = encode_condition_batch(
                        pipe.vae, batch["conditions"].to(accelerator.device, dtype=torch.float32)
                    )
                    if args.condition_dropout > 0:
                        keep = torch.rand(target_latents.shape[0], device=accelerator.device) >= args.condition_dropout
                        condition_latents = condition_latents * keep[:, None, None, None, None]
                    prompts = [
                        "" if torch.rand(()).item() < args.prompt_dropout else prompt for prompt in batch["prompts"]
                    ]
                    encoded = pipe.encode_prompt(
                        prompts,
                        device=accelerator.device,
                        clean_caption=False,
                        max_sequence_length=300,
                        complex_human_instruction=COMPLEX_HUMAN_INSTRUCTION,
                    )
                    prompt_embeds, prompt_mask = encoded[:2]
                    sigma_data = pipe.scheduler.config.sigma_data
                    sigma = torch.exp(
                        torch.randn(target_latents.shape[0], device=accelerator.device) * args.logit_std
                        + args.logit_mean
                    )
                    timestep = torch.atan(sigma / sigma_data)
                    use_max = torch.rand_like(timestep) < args.max_timestep_probability
                    timestep = torch.where(use_max, torch.full_like(timestep, args.max_timestep), timestep)
                    expanded = timestep[:, None, None, None]
                    target_state = target_latents * sigma_data
                    noise = torch.randn_like(target_state) * sigma_data
                    noisy_state = torch.cos(expanded) * target_state + torch.sin(expanded) * noise
                    scm_timestep = torch.sin(timestep) / (torch.cos(timestep) + torch.sin(timestep))
                    scm_expanded = scm_timestep[:, None, None, None]
                    normalization = torch.sqrt(scm_expanded.square() + (1 - scm_expanded).square())
                    model_input = noisy_state / sigma_data * normalization
                    guidance_indices = torch.randint(
                        len(guidance_scales), (target_latents.shape[0],), device=accelerator.device
                    )
                    guidance = guidance_values[guidance_indices] * pipe.transformer.config.guidance_embeds_scale

                raw_prediction = model(
                    model_input.to(weight_dtype),
                    encoder_hidden_states=prompt_embeds.to(weight_dtype),
                    encoder_attention_mask=prompt_mask,
                    guidance=guidance,
                    timestep=scm_timestep,
                    condition_latents=condition_latents,
                    return_dict=False,
                )[0]
                velocity = (
                    (1 - 2 * scm_expanded) * model_input
                    + (1 - 2 * scm_expanded + 2 * scm_expanded.square()) * raw_prediction.float()
                ) / normalization
                velocity = velocity * sigma_data
                predicted_target = (torch.cos(expanded) * noisy_state - torch.sin(expanded) * velocity) / sigma_data
                error = predicted_target.float() - target_latents.float()
                loss = error.square().mean() + args.l1_loss_weight * error.abs().mean()
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
