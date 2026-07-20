#!/usr/bin/env python

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from diffusers import FlowMatchEulerDiscreteScheduler, SanaPipeline

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dof_utils import add_metadata_args, add_pretrained_args, prepare_dynamic_images, pretrained_kwargs  # noqa: E402
from metadata import load_metadata, resolve_data_path  # noqa: E402
from sana_focus_latent_injection import (  # noqa: E402
    DEFAULT_LORA_TARGET_MODULES,
    add_lora_to_transformer,
    create_focus_model,
    default_prompt_for_mode,
    encode_vae_latents,
    load_focus_config,
    load_focus_injector,
    num_condition_images_for_mode,
    save_focus_checkpoint,
    tensor_stats,
    validate_checkpoint_mode,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SANA focus-fusion latent injection LoRA.")
    parser.add_argument("--model", default="Efficient-Large-Model/Sana_600M_1024px_diffusers")
    add_metadata_args(parser, metadata_required=True)
    add_pretrained_args(parser)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--condition_mode", choices=("a_only", "ab"), required=True)
    parser.add_argument("--prompt_mode", choices=("mode_default", "metadata"), default="mode_default")
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--max_pixels", type=int, default=1048576)
    parser.add_argument("--size_divisor", type=int, default=32)
    parser.add_argument("--aspect_ratio_tolerance", type=float, default=0.01)
    parser.add_argument("--downscale_if_exceeds_max_pixels", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_train_steps", type=int, default=20000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--condition_scale", type=float, default=1.0)
    parser.add_argument("--injector_hidden_channels", type=int, default=128)
    parser.add_argument("--train_transformer_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_scope", choices=("attn_qkv", "attn_qkvo", "wide"), default="wide")
    parser.add_argument("--lora_target_modules", default=DEFAULT_LORA_TARGET_MODULES)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="no")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--debug_check_finite", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


class FocusLatentInjectionDataset(Dataset):
    def __init__(self, args):
        records = load_metadata(args.dataset_metadata_path)
        end = None if args.max_samples is None else args.start_index + args.max_samples
        self.records = records[args.start_index:end] * args.dataset_repeat
        if not self.records:
            raise ValueError("Dataset selection is empty.")
        self.args = args
        self.base = Path(args.dataset_base_path)
        self.default_prompt = default_prompt_for_mode(args.condition_mode)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        sample = self.records[index]
        if self.args.target_key not in sample:
            raise ValueError(f"Sample {index} missing target key {self.args.target_key!r}.")
        edits = sample.get(self.args.edit_key)
        required = 1 if self.args.condition_mode == "a_only" else 2
        if not isinstance(edits, list) or len(edits) < required:
            raise ValueError(f"Sample {index} requires at least {required} edit images for {self.args.condition_mode}.")
        from PIL import Image

        target = Image.open(resolve_data_path(sample[self.args.target_key], self.base)).convert("RGB")
        image_a = Image.open(resolve_data_path(edits[0], self.base)).convert("RGB")
        image_b = Image.open(resolve_data_path(edits[1], self.base)).convert("RGB") if len(edits) > 1 else image_a
        prompt = self.default_prompt
        if self.args.prompt_mode == "metadata":
            prompt = sample.get(self.args.prompt_key) or self.default_prompt
        return {
            "target": target,
            "a": image_a,
            "b": image_b,
            "prompt": prompt,
            "has_focus_a": len(edits) > 2,
            "has_focus_b": len(edits) > 3,
            "paths": {
                "target": sample[self.args.target_key],
                "a": edits[0],
                "b": edits[1] if len(edits) > 1 else None,
                "focus_a": edits[2] if len(edits) > 2 else None,
                "focus_b": edits[3] if len(edits) > 3 else None,
            },
        }


def grad_norm(parameters):
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().norm().cpu()) ** 2
    return total**0.5


def cosine_mean(pred, target):
    return F.cosine_similarity(pred.float().flatten(1), target.float().flatten(1), dim=1).mean()


def read_resume_step(checkpoint):
    path = Path(checkpoint) / "trainer_state.json"
    if path.exists():
        return int(json.loads(path.read_text(encoding="utf-8")).get("global_step", 0))
    name = Path(checkpoint).name
    return int(name.split("-", 1)[1]) if name.startswith("checkpoint-") else 0


def main():
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("Dynamic focus latent injection training currently requires --batch_size 1.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision=args.mixed_precision)
    dtype = torch.float32
    if accelerator.device.type == "cuda" and args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif accelerator.device.type == "cuda" and args.mixed_precision == "fp16":
        dtype = torch.float16

    if args.resume_from_checkpoint:
        config, _ = load_focus_config(args.resume_from_checkpoint)
        validate_checkpoint_mode(config, args.condition_mode)

    pipe = SanaPipeline.from_pretrained(args.model, torch_dtype=dtype, **pretrained_kwargs(args))
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.model, subfolder="scheduler", **pretrained_kwargs(args))
    pipe.vae.requires_grad_(False).eval()
    pipe.text_encoder.requires_grad_(False).eval()
    pipe.transformer.requires_grad_(False).eval()

    if args.train_transformer_lora:
        matched_lora = add_lora_to_transformer(
            pipe.transformer,
            args.lora_rank,
            args.lora_alpha,
            args.lora_target_modules,
            args.lora_dropout,
            args.lora_scope,
        )
    else:
        matched_lora = []

    focus_model, latent_channels = create_focus_model(
        pipe.transformer, args.condition_mode, injector_hidden_channels=args.injector_hidden_channels
    )
    focus_model.injector.requires_grad_(True)
    focus_model.transformer.requires_grad_(False)
    if args.train_transformer_lora:
        for name, parameter in focus_model.transformer.named_parameters():
            if "lora_" in name:
                parameter.requires_grad_(True)
    if args.gradient_checkpointing and hasattr(focus_model.transformer, "enable_gradient_checkpointing"):
        focus_model.transformer.enable_gradient_checkpointing()

    if args.resume_from_checkpoint:
        load_focus_injector(args.resume_from_checkpoint, focus_model, args.condition_mode)
        lora_dir = Path(args.resume_from_checkpoint) / "transformer_lora"
        if args.train_transformer_lora and lora_dir.exists():
            pipe.load_lora_weights(lora_dir)

    pipe.vae.to(accelerator.device, dtype=torch.float32)
    pipe.text_encoder.to(accelerator.device, dtype=dtype)
    focus_model.to(accelerator.device, dtype=dtype)

    injector_params = [p for p in focus_model.injector.parameters() if p.requires_grad]
    lora_params = [p for name, p in focus_model.transformer.named_parameters() if p.requires_grad and "lora_" in name]
    optimizer_groups = [{"params": injector_params, "lr": args.learning_rate, "name": "injector"}]
    if args.train_transformer_lora:
        if not lora_params:
            raise RuntimeError("--train_transformer_lora was set but no trainable LoRA parameters were found.")
        optimizer_groups.append({"params": lora_params, "lr": args.learning_rate, "name": "transformer_lora"})
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=1e-2)

    accelerator.print(f"[FOCUS_ROUTE1] condition_mode={args.condition_mode}", flush=True)
    accelerator.print(f"[FOCUS_ROUTE1] num_condition_images={num_condition_images_for_mode(args.condition_mode)}", flush=True)
    accelerator.print(
        f"[FOCUS_ROUTE1] condition_channels={latent_channels * num_condition_images_for_mode(args.condition_mode)}",
        flush=True,
    )
    accelerator.print("[FOCUS_ROUTE1] output geometry reference=A", flush=True)
    accelerator.print("[FOCUS_ROUTE1] focus maps used=False", flush=True)
    accelerator.print(f"[FOCUS_ROUTE1] injector trainable params={sum(p.numel() for p in injector_params):,}", flush=True)
    accelerator.print(f"[FOCUS_ROUTE1] transformer LoRA params={sum(p.numel() for p in lora_params):,}", flush=True)
    accelerator.print(f"[FOCUS_ROUTE1] matched LoRA modules={len(matched_lora)}", flush=True)

    dataset = FocusLatentInjectionDataset(args)

    def collate(samples):
        sample = samples[0]
        named = {"a": sample["a"], "target": sample["target"]}
        if args.condition_mode == "ab":
            named["b"] = sample["b"]
        prepared, size_info = prepare_dynamic_images(
            named,
            args.max_pixels,
            args.size_divisor,
            args.aspect_ratio_tolerance,
            args.downscale_if_exceeds_max_pixels,
        )
        canvas_w, canvas_h = size_info["canvas_size"]
        batch = {
            "target": pipe.image_processor.preprocess(prepared["target"], height=canvas_h, width=canvas_w),
            "a": pipe.image_processor.preprocess(prepared["a"], height=canvas_h, width=canvas_w),
            "prompts": [sample["prompt"]],
            "size_info": size_info,
            "has_focus_a": sample["has_focus_a"],
            "has_focus_b": sample["has_focus_b"],
        }
        if args.condition_mode == "ab":
            batch["b"] = pipe.image_processor.preprocess(prepared["b"], height=canvas_h, width=canvas_w)
        return batch

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate)
    focus_model, optimizer, dataloader = accelerator.prepare(focus_model, optimizer, dataloader)
    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.resume_from_checkpoint:
        optimizer_path = Path(args.resume_from_checkpoint) / "optimizer.pt"
        if optimizer_path.exists():
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=accelerator.device))
        global_step = read_resume_step(args.resume_from_checkpoint)
        accelerator.print(f"[FOCUS_ROUTE1] resumed global_step={global_step}", flush=True)
    else:
        global_step = 0

    scheduler_timesteps = scheduler.timesteps.to(accelerator.device)
    scheduler_sigmas = scheduler.sigmas.to(accelerator.device)
    while global_step < args.max_train_steps:
        for batch in dataloader:
            if global_step >= args.max_train_steps:
                break
            with accelerator.accumulate(focus_model):
                with torch.no_grad():
                    z_gt = encode_vae_latents(pipe.vae, batch["target"].to(accelerator.device, torch.float32))
                    z_a = encode_vae_latents(pipe.vae, batch["a"].to(accelerator.device, torch.float32))
                    z_b = None
                    if args.condition_mode == "ab":
                        z_b = encode_vae_latents(pipe.vae, batch["b"].to(accelerator.device, torch.float32))
                    prompt_embeds, prompt_mask, _, _ = pipe.encode_prompt(
                        batch["prompts"], False, device=accelerator.device, clean_caption=False, max_sequence_length=300
                    )
                    prompt_embeds = prompt_embeds.to(dtype)
                    noise = torch.randn_like(z_gt)
                    indices = torch.randint(0, scheduler.config.num_train_timesteps, (z_gt.shape[0],), device=accelerator.device)
                    timesteps = scheduler_timesteps[indices]
                    sigmas = scheduler_sigmas[indices].view(-1, 1, 1, 1).to(z_gt.dtype)
                    z_t = (1.0 - sigmas) * z_gt + sigmas * noise
                    target_velocity = noise - z_gt
                condition_latents = z_a if args.condition_mode == "a_only" else [z_a, z_b]
                pred = focus_model(
                    hidden_states=z_t.to(dtype),
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_mask,
                    condition_latents=condition_latents,
                    condition_mode=args.condition_mode,
                    condition_scale=args.condition_scale,
                    timestep=timesteps,
                    return_dict=False,
                )[0]
                loss = F.mse_loss(pred.float(), target_velocity.float())
                if args.debug_check_finite and not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {global_step}: {loss}")
                accelerator.backward(loss)
                proj_in_grad = grad_norm(focus_model.injector.proj_in.parameters())
                proj_mid_grad = grad_norm(focus_model.injector.proj_mid.parameters())
                proj_out_grad = grad_norm(focus_model.injector.proj_out.parameters())
                lora_grad = grad_norm(lora_params)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if accelerator.is_main_process and global_step % args.log_steps == 0:
                debug = accelerator.unwrap_model(focus_model).last_debug
                accelerator.print(
                    "[FOCUS_ROUTE1] "
                    f"step={global_step} loss={loss.detach().item():.6f} lr={optimizer.param_groups[0]['lr']:.3e} "
                    f"condition_mode={args.condition_mode} target_mse={target_velocity.float().square().mean().item():.6f} "
                    f"pred_mse={pred.float().square().mean().item():.6f} pred_target_cos={cosine_mean(pred, target_velocity).item():.6f} "
                    f"condition_scale={args.condition_scale} proj_in_grad={proj_in_grad:.3e} "
                    f"proj_mid_grad={proj_mid_grad:.3e} proj_out_grad={proj_out_grad:.3e} lora_grad={lora_grad:.3e} "
                    f"gt_latent={tensor_stats(z_gt)} noisy_target={tensor_stats(z_t)} A_latent={tensor_stats(z_a)} "
                    f"B_latent={tensor_stats(z_b) if z_b is not None else None} "
                    f"condition_tensor={tensor_stats(debug['condition_tensor'])} "
                    f"injected_tokens={tensor_stats(debug['injected_tokens'])} "
                    f"hidden_tokens_after_injection={tensor_stats(debug['hidden_tokens_after_injection'])}",
                    flush=True,
                )
            if accelerator.is_main_process and global_step % args.save_steps == 0:
                ckpt = output_dir / f"checkpoint-{global_step}"
                unwrapped = accelerator.unwrap_model(focus_model)
                save_focus_checkpoint(ckpt, unwrapped, args, global_step, latent_channels, optimizer=optimizer)
                accelerator.print(f"[FOCUS_ROUTE1] saved checkpoint={ckpt}", flush=True)

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(focus_model)
        ckpt = output_dir / f"checkpoint-{global_step}"
        save_focus_checkpoint(ckpt, unwrapped, args, global_step, latent_channels, optimizer=optimizer)
        save_focus_checkpoint(output_dir, unwrapped, args, global_step, latent_channels, optimizer=optimizer)
        accelerator.print(f"[FOCUS_ROUTE1] done checkpoint={ckpt}", flush=True)


if __name__ == "__main__":
    main()
