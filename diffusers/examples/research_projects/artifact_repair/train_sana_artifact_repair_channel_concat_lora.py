#!/usr/bin/env python

import argparse
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

from artifact_repair_utils import (  # noqa: E402
    DEFAULT_REPAIR_PROMPT,
    add_pretrained_args,
    get_artifact_repair_paths,
    load_metadata,
    load_rgb,
    preprocess_triplet,
    pretrained_kwargs,
)
from sana_artifact_repair_channel_concat import (  # noqa: E402
    DEFAULT_LORA_TARGET_MODULES,
    add_lora_to_transformer,
    encode_vae_latents,
    expand_sana_patch_embedding_for_channel_concat,
    get_sana_patch_embedding,
    patch_embedding_weight_stats,
    save_route2_checkpoint,
    tensor_stats,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SANA Artifact Repair Route 2 channel-concat I2I.")
    parser.add_argument("--model", default="Efficient-Large-Model/Sana_600M_1024px_diffusers")
    parser.add_argument("--dataset_metadata_path", required=True)
    parser.add_argument("--dataset_base_path", default=".")
    parser.add_argument("--target_key", default="image")
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt", default=DEFAULT_REPAIR_PROMPT)
    parser.add_argument("--train_mode", choices=("patch_lora", "full_transformer", "patch_only"), default="patch_lora")
    parser.add_argument("--max_pixels", type=int, default=1048576)
    parser.add_argument("--size_divisor", type=int, default=32)
    parser.add_argument("--downscale_if_exceeds_max_pixels", action="store_true")
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--patch_learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_train_steps", type=int, default=20000)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_scope", choices=("attn_qkv", "attn_qkvo", "wide"), default="wide")
    parser.add_argument("--lora_target_modules", default=DEFAULT_LORA_TARGET_MODULES)
    parser.add_argument("--image_condition_dropout", type=float, default=0.0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="no")
    parser.add_argument("--debug_check_finite", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    add_pretrained_args(parser)
    return parser.parse_args()


class Route2Dataset(Dataset):
    def __init__(self, args):
        records = load_metadata(args.dataset_metadata_path)
        end = None if args.max_samples is None else args.start_index + args.max_samples
        self.records = records[args.start_index:end] * args.dataset_repeat
        if not self.records:
            raise ValueError("Dataset selection is empty.")
        self.base_path = Path(args.dataset_base_path)
        self.args = args

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        sample = self.records[index]
        paths = get_artifact_repair_paths(sample, self.base_path, index, self.args.target_key, self.args.prompt_key)
        return {
            "gt": load_rgb(paths["gt"]),
            "src": load_rgb(paths["src"]),
            "ref": load_rgb(paths["ref"]),
            "prompt": paths["prompt"] or self.args.prompt,
        }


def grad_norm(parameters):
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().norm().cpu()) ** 2
    return total**0.5


def main():
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("Dynamic Route 2 training currently requires --batch_size 1.")
    if not 0 <= args.image_condition_dropout < 1:
        raise ValueError("--image_condition_dropout must be in [0, 1).")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision=args.mixed_precision)
    dtype = torch.float32
    if accelerator.device.type == "cuda" and args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif accelerator.device.type == "cuda" and args.mixed_precision == "fp16":
        dtype = torch.float16

    accelerator.print("[ROUTE2] image/GT is target", flush=True)
    accelerator.print("[ROUTE2] edit_image[0] is src condition", flush=True)
    accelerator.print("[ROUTE2] edit_image[1] is ignored", flush=True)
    accelerator.print("[ROUTE2] condition method = channel concat before patch embedding", flush=True)

    pipe = SanaPipeline.from_pretrained(args.model, torch_dtype=dtype, **pretrained_kwargs(args))
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.model, subfolder="scheduler", **pretrained_kwargs(args))
    pipe.vae.requires_grad_(False).eval()
    pipe.text_encoder.requires_grad_(False).eval()
    pipe.transformer.requires_grad_(False).eval()

    original_latent_channels = expand_sana_patch_embedding_for_channel_concat(pipe.transformer)
    matched_lora = []
    if args.train_mode == "patch_lora":
        matched_lora = add_lora_to_transformer(
            pipe.transformer,
            args.lora_rank,
            args.lora_alpha,
            args.lora_target_modules,
            args.lora_dropout,
            args.lora_scope,
        )
    elif args.train_mode == "full_transformer":
        pipe.transformer.requires_grad_(True)
    elif args.train_mode == "patch_only":
        pass
    else:
        raise ValueError(f"Unsupported train mode: {args.train_mode}")

    patch_embed, patch_proj = get_sana_patch_embedding(pipe.transformer)
    if args.train_mode in ("patch_lora", "patch_only", "full_transformer"):
        for parameter in patch_embed.parameters():
            parameter.requires_grad_(True)

    if args.gradient_checkpointing and hasattr(pipe.transformer, "enable_gradient_checkpointing"):
        pipe.transformer.enable_gradient_checkpointing()
    pipe.text_encoder.to(accelerator.device, dtype=dtype)
    pipe.vae.to(accelerator.device, dtype=torch.float32)
    pipe.transformer.to(accelerator.device, dtype=dtype)

    patch_embed, patch_proj = get_sana_patch_embedding(pipe.transformer)
    patch_params = [parameter for parameter in patch_embed.parameters() if parameter.requires_grad]
    lora_named_params = [
        (name, parameter)
        for name, parameter in pipe.transformer.named_parameters()
        if parameter.requires_grad and "lora_" in name
    ]
    lora_params = [parameter for _, parameter in lora_named_params]
    if args.train_mode == "full_transformer":
        patch_param_ids = {id(p) for p in patch_params}
        other_params = [p for p in pipe.transformer.parameters() if p.requires_grad and id(p) not in patch_param_ids]
        optimizer_groups = [
            {"params": patch_params, "lr": args.patch_learning_rate, "name": "patch_embedding"},
            {"params": other_params, "lr": args.learning_rate, "name": "full_transformer"},
        ]
    elif args.train_mode == "patch_lora":
        if not patch_params:
            raise RuntimeError(
                "patch_lora requires a trainable expanded patch embedding, but patch_params is empty. "
                "PEFT may have frozen it after add_adapter()."
            )
        if not lora_params:
            raise RuntimeError("patch_lora requires trainable LoRA parameters, but no LoRA parameter was found.")
        if not all(parameter.requires_grad for parameter in patch_embed.parameters()):
            raise RuntimeError("Expanded patch embedding is unexpectedly frozen in patch_lora mode.")
        patch_param_ids = {id(parameter) for parameter in patch_params}
        lora_param_ids = {id(parameter) for parameter in lora_params}
        overlap = patch_param_ids & lora_param_ids
        if overlap:
            raise RuntimeError(f"Patch and LoRA optimizer groups overlap: {len(overlap)} parameters")
        optimizer_groups = [
            {"params": patch_params, "lr": args.patch_learning_rate, "name": "patch_embedding"},
            {"params": lora_params, "lr": args.learning_rate, "name": "transformer_lora"},
        ]
    else:
        optimizer_groups = [{"params": patch_params, "lr": args.patch_learning_rate, "name": "patch_embedding"}]
    trainable_params = [p for group in optimizer_groups for p in group["params"]]
    if not trainable_params:
        raise ValueError("No trainable parameters for Route 2.")
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=1e-2)

    accelerator.print(f"[ROUTE2] patch embedding trainable params: {sum(p.numel() for p in patch_params):,}", flush=True)
    accelerator.print(f"[ROUTE2] transformer LoRA trainable params: {sum(p.numel() for p in lora_params):,}", flush=True)
    accelerator.print(f"[ROUTE2] total trainable params: {sum(p.numel() for p in trainable_params):,}", flush=True)
    accelerator.print(f"[ROUTE2] patch embedding requires_grad: {[p.requires_grad for p in patch_embed.parameters()]}", flush=True)
    accelerator.print(f"[ROUTE2] LoRA matched modules: {len(matched_lora)}", flush=True)

    dataset = Route2Dataset(args)

    def collate(samples):
        sample = samples[0]
        prepared, size_info = preprocess_triplet(
            sample["gt"], sample["src"], sample["ref"], args.max_pixels, args.size_divisor, args.downscale_if_exceeds_max_pixels
        )
        canvas_w, canvas_h = size_info["canvas_size"]
        return {
            "gt": pipe.image_processor.preprocess(prepared["gt"], height=canvas_h, width=canvas_w),
            "src": pipe.image_processor.preprocess(prepared["src"], height=canvas_h, width=canvas_w),
            "prompts": [sample["prompt"]],
        }

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate)
    pipe.transformer, optimizer, dataloader = accelerator.prepare(pipe.transformer, optimizer, dataloader)
    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    scheduler_timesteps = scheduler.timesteps.to(accelerator.device)
    scheduler_sigmas = scheduler.sigmas.to(accelerator.device)
    global_step = 0
    while global_step < args.max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(pipe.transformer):
                with torch.no_grad():
                    z_gt = encode_vae_latents(pipe.vae, batch["gt"].to(accelerator.device, torch.float32))
                    z_src = encode_vae_latents(pipe.vae, batch["src"].to(accelerator.device, torch.float32))
                    if z_gt.shape != z_src.shape:
                        raise ValueError(f"z_gt and z_src must match, got {tuple(z_gt.shape)} and {tuple(z_src.shape)}")
                    if args.image_condition_dropout > 0:
                        drop_mask = torch.rand(z_src.shape[0], device=accelerator.device) < args.image_condition_dropout
                        z_src = z_src * (~drop_mask).to(z_src.dtype).view(-1, 1, 1, 1)
                    prompt_embeds, prompt_mask, _, _ = pipe.encode_prompt(
                        batch["prompts"], False, device=accelerator.device, clean_caption=False, max_sequence_length=300
                    )
                    prompt_embeds = prompt_embeds.to(dtype)
                    noise = torch.randn_like(z_gt)
                    indices = torch.randint(0, scheduler.config.num_train_timesteps, (z_gt.shape[0],), device=accelerator.device)
                    timesteps = scheduler_timesteps[indices]
                    sigmas = scheduler_sigmas[indices].view(-1, 1, 1, 1).to(z_gt.dtype)
                    z_t = (1.0 - sigmas) * z_gt + sigmas * noise
                    flow_target = noise - z_gt
                    model_input = torch.cat([z_t, z_src], dim=1)
                pred = pipe.transformer(
                    hidden_states=model_input.to(dtype),
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_mask,
                    timestep=timesteps * pipe.transformer.config.timestep_scale,
                    return_dict=False,
                )[0]
                if pred.shape != z_gt.shape:
                    raise ValueError(f"Route 2 pred shape must equal target latent shape, got {tuple(pred.shape)} vs {tuple(z_gt.shape)}")
                loss = F.mse_loss(pred.float(), flow_target.float())
                if args.debug_check_finite and not torch.isfinite(loss):
                    raise RuntimeError("[ROUTE2] Non-finite loss.")
                accelerator.backward(loss)
                unwrapped = accelerator.unwrap_model(pipe.transformer)
                current_patch_embed, current_patch_proj = get_sana_patch_embedding(unwrapped)
                if global_step == 0 and args.train_mode == "patch_lora":
                    patch_weight_grad = current_patch_proj.weight.grad
                    if patch_weight_grad is None:
                        raise RuntimeError("Expanded patch embedding has no gradient on the first backward pass.")
                    condition_grad = patch_weight_grad[:, original_latent_channels : 2 * original_latent_channels]
                    if not torch.isfinite(condition_grad).all():
                        raise FloatingPointError("Non-finite condition-channel patch gradient.")
                    if condition_grad.abs().sum().item() == 0:
                        raise RuntimeError(
                            "Condition-channel patch gradient is exactly zero. The src condition path is not learning."
                        )
                current_patch_params = [p for p in current_patch_embed.parameters() if p.requires_grad]
                current_lora_params = [p for n, p in unwrapped.named_parameters() if p.requires_grad and "lora_" in n]
                patch_grad = grad_norm(current_patch_params)
                lora_grad = grad_norm(current_lora_params)
                accelerator.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process and (global_step <= 5 or global_step % args.log_steps == 0):
                    stats = patch_embedding_weight_stats(accelerator.unwrap_model(pipe.transformer), original_latent_channels)
                    stats["patch_embedding_gradient_norm"] = patch_grad
                    print(
                        f"step={global_step} loss={loss.detach().item():.6f} "
                        f"sigma_mean={sigmas.mean().item():.6f} sigma_min={sigmas.min().item():.6f} sigma_max={sigmas.max().item():.6f}",
                        flush=True,
                    )
                    print(
                        {
                            "z_gt": tensor_stats(z_gt),
                            "z_src": tensor_stats(z_src),
                            "z_t": tensor_stats(z_t),
                            "patch": stats,
                            "lora_gradient_norm": lora_grad,
                        },
                        flush=True,
                    )
                if accelerator.is_main_process and (global_step % args.save_steps == 0 or global_step == args.max_train_steps):
                    ckpt = output_dir / f"checkpoint-{global_step}"
                    save_route2_checkpoint(ckpt, accelerator.unwrap_model(pipe.transformer), pipe, args, global_step, original_latent_channels)
                    accelerator.save_state(ckpt / "trainer_state")
                if global_step >= args.max_train_steps:
                    break
        if global_step >= args.max_train_steps:
            break

    if accelerator.is_main_process:
        save_route2_checkpoint(output_dir, accelerator.unwrap_model(pipe.transformer), pipe, args, global_step, original_latent_channels)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
