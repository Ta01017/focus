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

from artifact_repair_utils import add_pretrained_args, load_metadata, load_rgb, preprocess_triplet, pretrained_kwargs
from sana_artifact_repair_channel_concat import DEFAULT_LORA_TARGET_MODULES, add_lora_to_transformer, get_sana_patch_embedding, tensor_stats
from sana_focus_wan_crossattn import (
    build_focus_wan_model,
    encode_image_tokens,
    encode_vae_latents,
    expand_sana_patch_embedding_for_focus_wan,
    focus_prompt,
    load_focus_wan_config,
    load_focus_wan_condition_state_dict,
    load_image_encoder_and_processor,
    num_condition_images,
    save_focus_wan_checkpoint,
    validate_focus_wan_checkpoint,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train Focus Fusion SANA Wan-style latent concat + image cross-attention.")
    p.add_argument("--model", default="Efficient-Large-Model/Sana_600M_1024px_diffusers")
    p.add_argument("--condition_mode", choices=("single", "dual"), required=True)
    p.add_argument("--dataset_metadata_path", required=True)
    p.add_argument("--dataset_base_path", default=".")
    p.add_argument("--target_key", default="image")
    p.add_argument("--edit_key", default="edit_image")
    p.add_argument("--prompt_key", default="prompt")
    p.add_argument("--prompt", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_pixels", type=int, default=1048576)
    p.add_argument("--size_divisor", type=int, default=32)
    p.add_argument("--downscale_if_exceeds_max_pixels", action="store_true")
    p.add_argument("--random_crop", action="store_true")
    p.add_argument("--center_crop", action="store_true")
    p.add_argument("--random_horizontal_flip", action="store_true")
    p.add_argument("--dataset_repeat", type=int, default=1)
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--patch_learning_rate", type=float, default=1e-4)
    p.add_argument("--image_adapter_learning_rate", type=float, default=1e-4)
    p.add_argument("--max_train_steps", type=int, default=10000)
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--train_transformer_lora", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lora_scope", choices=("attn_qkv", "attn_qkvo", "wide"), default="wide")
    p.add_argument("--lora_target_modules", default=DEFAULT_LORA_TARGET_MODULES)
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--image_encoder_model", default="openai/clip-vit-large-patch14")
    p.add_argument("--image_encoder_subfolder", default=None)
    p.add_argument("--image_encoder_revision", default=None)
    p.add_argument("--image_encoder_local_files_only", action="store_true")
    p.add_argument("--image_gate_init", type=float, default=1e-3)
    p.add_argument("--image_cross_attention_scale_a", type=float, default=1.0)
    p.add_argument("--image_cross_attention_scale_b", type=float, default=1.0)
    p.add_argument("--share_image_projector", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--resume_from_checkpoint", default=None)
    p.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="no")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--debug_check_finite", action="store_true")
    p.add_argument("--use_focus_maps", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--pixel_loss_weight", type=float, default=0.0)
    p.add_argument("--x0_loss_weight", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    add_pretrained_args(p)
    return p.parse_args()


def resolve_path_once(base, value, index, label):
    raw = Path(value)
    path = raw if raw.is_absolute() else Path(base) / raw
    print(f"[FOCUS_WAN][PATH] index={index} {label}: raw={value} resolved={path}", flush=True)
    if not path.exists():
        raise FileNotFoundError(f"record index={index} {label} missing: raw={value} resolved={path}")
    return path


class FocusWanDataset(Dataset):
    def __init__(self, args):
        records = load_metadata(args.dataset_metadata_path)
        end = None if args.max_samples is None else args.start_index + args.max_samples
        self.records = records[args.start_index:end] * args.dataset_repeat
        if not self.records:
            raise ValueError("Dataset selection is empty.")
        self.args = args

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        sample = self.records[i]
        edits = sample.get(self.args.edit_key)
        need = 1 if self.args.condition_mode == "single" else 2
        if not isinstance(edits, list) or len(edits) < need:
            raise ValueError(f"record index={i} requires {need} edit images for {self.args.condition_mode}.")
        gt = load_rgb(resolve_path_once(self.args.dataset_base_path, sample[self.args.target_key], i, "GT"))
        a = load_rgb(resolve_path_once(self.args.dataset_base_path, edits[0], i, "A"))
        b = load_rgb(resolve_path_once(self.args.dataset_base_path, edits[1], i, "B")) if len(edits) > 1 else a
        prompt = sample.get(self.args.prompt_key) or self.args.prompt or focus_prompt(self.args.condition_mode)
        return {"gt": gt, "a": a, "b": b, "prompt": prompt, "has_focus": len(edits) > 2}


def count_params(ps):
    return sum(p.numel() for p in ps)


def grad_norm(ps):
    total = 0.0
    for p in ps:
        if p.grad is not None:
            total += float(p.grad.detach().float().norm().cpu()) ** 2
    return total ** 0.5


def finite_check(name, tensor, step):
    if tensor is not None and not torch.isfinite(tensor).all():
        raise RuntimeError(f"[FOCUS_WAN] non-finite tensor step={step} name={name} stats={tensor_stats(tensor)}")


def main():
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("Dynamic Focus WAN training currently requires --batch_size 1.")
    if args.pixel_loss_weight != 0 or args.x0_loss_weight != 0:
        raise ValueError("First Focus WAN version requires --pixel_loss_weight=0 and --x0_loss_weight=0.")
    random.seed(args.seed); torch.manual_seed(args.seed)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision=args.mixed_precision)
    dtype = torch.float32
    if accelerator.device.type == "cuda" and args.mixed_precision == "bf16": dtype = torch.bfloat16
    if accelerator.device.type == "cuda" and args.mixed_precision == "fp16": dtype = torch.float16

    pipe = SanaPipeline.from_pretrained(args.model, torch_dtype=dtype, **pretrained_kwargs(args))
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.model, subfolder="scheduler", **pretrained_kwargs(args))
    pipe.vae.requires_grad_(False).eval(); pipe.text_encoder.requires_grad_(False).eval(); pipe.transformer.requires_grad_(False).eval()
    latent_channels = expand_sana_patch_embedding_for_focus_wan(pipe.transformer, args.condition_mode)
    matched_lora = []
    if args.train_transformer_lora:
        matched_lora = add_lora_to_transformer(pipe.transformer, args.lora_rank, args.lora_alpha, args.lora_target_modules, args.lora_dropout, args.lora_scope)
        if not matched_lora:
            raise RuntimeError("LoRA matched module count is 0.")

    image_encoder, image_processor = load_image_encoder_and_processor(args, dtype=dtype, device=accelerator.device)
    model = build_focus_wan_model(pipe.transformer, args.condition_mode, int(image_encoder.config.hidden_size), args.image_gate_init, args.share_image_projector)
    if args.resume_from_checkpoint:
        cfg, ckpt_dir = load_focus_wan_config(args.resume_from_checkpoint)
        validate_focus_wan_checkpoint(cfg, args.condition_mode, model.transformer)
        load_focus_wan_condition_state_dict(model, ckpt_dir / "focus_wan_condition.safetensors")
        if args.train_transformer_lora and (ckpt_dir / "transformer_lora").exists():
            pipe.load_lora_weights(ckpt_dir / "transformer_lora")

    model.transformer.requires_grad_(False)
    if args.train_transformer_lora:
        for n, p in model.transformer.named_parameters():
            if "lora_" in n: p.requires_grad_(True)
    model.image_projectors.requires_grad_(True); model.image_cross_attention_adapter.requires_grad_(True)
    patch_embed, proj = get_sana_patch_embedding(model.transformer)
    patch_embed.requires_grad_(True)
    if args.gradient_checkpointing and hasattr(model.transformer, "enable_gradient_checkpointing"):
        model.transformer.enable_gradient_checkpointing()
    pipe.vae.to(accelerator.device, dtype=torch.float32); pipe.text_encoder.to(accelerator.device, dtype=dtype); model.to(accelerator.device, dtype=dtype)

    patch_params = [p for p in patch_embed.parameters() if p.requires_grad]
    lora_params = [p for n, p in model.transformer.named_parameters() if p.requires_grad and "lora_" in n]
    image_params = [p for p in list(model.image_projectors.parameters()) + list(model.image_cross_attention_adapter.parameters()) if p.requires_grad]
    groups = [
        {"params": patch_params, "lr": args.patch_learning_rate, "name": "expanded_patch"},
        {"params": image_params, "lr": args.image_adapter_learning_rate, "name": "image_wan_condition"},
    ]
    if args.train_transformer_lora:
        groups.append({"params": lora_params, "lr": args.learning_rate, "name": "transformer_lora"})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-2)
    accelerator.print(f"[FOCUS_WAN] condition_mode={args.condition_mode}", flush=True)
    accelerator.print(f"[FOCUS_WAN] total parameters={count_params(model.parameters()):,}", flush=True)
    accelerator.print(f"[FOCUS_WAN] total trainable parameters={count_params([p for p in model.parameters() if p.requires_grad]):,}", flush=True)
    accelerator.print(f"[FOCUS_WAN] expanded patch trainable parameters={count_params(patch_params):,}", flush=True)
    accelerator.print(f"[FOCUS_WAN] image projector trainable parameters={count_params(model.image_projectors.parameters()):,}", flush=True)
    accelerator.print(f"[FOCUS_WAN] image K/V and gate trainable parameters={count_params(model.image_cross_attention_adapter.parameters()):,}", flush=True)
    accelerator.print(f"[FOCUS_WAN] transformer LoRA trainable parameters={count_params(lora_params):,}", flush=True)
    accelerator.print(f"[FOCUS_WAN] LoRA matched module count={len(matched_lora)} first={matched_lora[:8]}", flush=True)
    accelerator.print(f"[FOCUS_WAN] A is used; B is {'used' if args.condition_mode == 'dual' else 'ignored'}; focus maps are ignored by the baseline WAN model", flush=True)

    dataset = FocusWanDataset(args)
    def collate(samples):
        s = samples[0]
        prepared, size_info = preprocess_triplet(s["gt"], s["a"], s["b"], args.max_pixels, args.size_divisor, args.downscale_if_exceeds_max_pixels)
        w, h = size_info["canvas_size"]
        return {"gt": pipe.image_processor.preprocess(prepared["gt"], height=h, width=w), "a": pipe.image_processor.preprocess(prepared["src"], height=h, width=w), "b": pipe.image_processor.preprocess(prepared["ref"], height=h, width=w), "a_pil": [prepared["src"]], "b_pil": [prepared["ref"]], "prompts": [s["prompt"]]}
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    outdir = Path(args.output_dir)
    if accelerator.is_main_process: outdir.mkdir(parents=True, exist_ok=True)
    global_step = 0
    scheduler_timesteps = scheduler.timesteps.to(accelerator.device); scheduler_sigmas = scheduler.sigmas.to(accelerator.device)
    scales = [args.image_cross_attention_scale_a] if args.condition_mode == "single" else [args.image_cross_attention_scale_a, args.image_cross_attention_scale_b]
    while global_step < args.max_train_steps:
        for batch in loader:
            if global_step >= args.max_train_steps: break
            with accelerator.accumulate(model):
                with torch.no_grad():
                    z_gt = encode_vae_latents(pipe.vae, batch["gt"].to(accelerator.device, torch.float32))
                    z_a = encode_vae_latents(pipe.vae, batch["a"].to(accelerator.device, torch.float32))
                    z_b = encode_vae_latents(pipe.vae, batch["b"].to(accelerator.device, torch.float32)) if args.condition_mode == "dual" else None
                    tok_a = encode_image_tokens(image_encoder, image_processor, batch["a_pil"], accelerator.device, dtype)
                    tok_b = encode_image_tokens(image_encoder, image_processor, batch["b_pil"], accelerator.device, dtype) if args.condition_mode == "dual" else None
                    prompt_embeds, prompt_mask, _, _ = pipe.encode_prompt(batch["prompts"], False, device=accelerator.device, clean_caption=False, max_sequence_length=300)
                    prompt_embeds = prompt_embeds.to(dtype)
                    noise = torch.randn_like(z_gt)
                    idx = torch.randint(0, scheduler.config.num_train_timesteps, (z_gt.shape[0],), device=accelerator.device)
                    timesteps = scheduler_timesteps[idx]
                    sigmas = scheduler_sigmas[idx].view(-1,1,1,1).to(z_gt.dtype)
                    z_t = (1.0 - sigmas) * z_gt + sigmas * noise
                    target = noise - z_gt
                    model_input = torch.cat([z_t, z_a], dim=1) if args.condition_mode == "single" else torch.cat([z_t, z_a, z_b], dim=1)
                    image_inputs = [tok_a] if args.condition_mode == "single" else [tok_a, tok_b]
                if args.debug_check_finite:
                    for name, value in (("z_gt", z_gt), ("z_t", z_t), ("tok_a", tok_a), ("tok_b", tok_b)):
                        finite_check(name, value, global_step)
                pred = model(hidden_states=model_input.to(dtype), encoder_hidden_states=prompt_embeds, encoder_attention_mask=prompt_mask, encoder_hidden_states_images=image_inputs, image_cross_attention_scales=scales, timestep=timesteps, return_dict=False)[0].float()
                loss = F.mse_loss(pred, target.float())
                if args.debug_check_finite: finite_check("prediction", pred, global_step); finite_check("loss", loss, global_step)
                accelerator.backward(loss)
                patch_grad = grad_norm(patch_params); image_grad = grad_norm(image_params); lora_grad = grad_norm(lora_params)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if accelerator.is_main_process and global_step % args.log_steps == 0:
                unwrapped = accelerator.unwrap_model(model)
                _, current_proj = get_sana_patch_embedding(unwrapped.transformer)
                c = latent_channels
                a_norm = float(current_proj.weight[:, c:2*c].detach().float().norm().cpu())
                b_norm = float(current_proj.weight[:, 2*c:3*c].detach().float().norm().cpu()) if args.condition_mode == "dual" else 0.0
                gates = unwrapped.image_cross_attention_adapter.gate_values()[0]
                accelerator.print(f"[FOCUS_WAN] step={global_step} loss={loss.item():.6f} lr={optimizer.param_groups[0]['lr']:.3e} timestep={timesteps.detach().cpu().tolist()} sigma={sigmas.flatten().detach().cpu().tolist()} z_gt={tensor_stats(z_gt)} z_t={tensor_stats(z_t)} prediction={tensor_stats(pred)} target={tensor_stats(target)} z_a={tensor_stats(z_a)} z_b={tensor_stats(z_b) if z_b is not None else None} A_vision={tensor_stats(tok_a)} B_vision={tensor_stats(tok_b) if tok_b is not None else None} projectors={unwrapped.last_debug.get('projected_image_stats')} gates={gates} A_patch_norm={a_norm:.3e} B_patch_norm={b_norm:.3e} image_grad={image_grad:.3e} lora_grad={lora_grad:.3e} patch_grad={patch_grad:.3e}", flush=True)
            if accelerator.is_main_process and global_step % args.save_steps == 0:
                ckpt = outdir / f"checkpoint-{global_step}"
                save_focus_wan_checkpoint(ckpt, accelerator.unwrap_model(model), args, global_step, latent_channels, int(image_encoder.config.hidden_size))
                accelerator.print(f"[FOCUS_WAN] saved checkpoint={ckpt}", flush=True)
    if accelerator.is_main_process:
        ckpt = outdir / f"checkpoint-{global_step}"
        save_focus_wan_checkpoint(ckpt, accelerator.unwrap_model(model), args, global_step, latent_channels, int(image_encoder.config.hidden_size))
        save_focus_wan_checkpoint(outdir, accelerator.unwrap_model(model), args, global_step, latent_channels, int(image_encoder.config.hidden_size))
        accelerator.print(f"[FOCUS_WAN] done checkpoint={ckpt}", flush=True)


if __name__ == "__main__":
    main()
