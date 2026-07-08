#!/usr/bin/env python

import argparse
import json
import random
import warnings
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from diffusers import FlowMatchEulerDiscreteScheduler, SanaPipeline

from dof_utils import add_metadata_args, add_pretrained_args, prepare_dynamic_images, pretrained_kwargs
from metadata import load_metadata, require_keys, resolve_data_path
from sana_dof import encode_vae_latents, tensor_stats
from sana_sprint_controlnet import initialize_controlnet_from_transformer


DEFAULT_LORA_TARGET_MODULES = ["to_q", "to_k", "to_v"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train ordinary SANA ControlNet A/B/focus fusion.")
    parser.add_argument("--model", default="Efficient-Large-Model/Sana_600M_1024px_diffusers")
    add_metadata_args(parser, metadata_required=True)
    add_pretrained_args(parser)
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt", default="a photorealistic all-in-focus photograph")
    parser.add_argument("--loss_mode", choices=("legacy_a_noise_gt_velocity", "standard_sft"), default="legacy_a_noise_gt_velocity")
    parser.add_argument("--control_condition_channels", type=int, default=8)
    parser.add_argument("--use_focus_conditions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--focus_default_value", type=float, default=0.5)
    parser.add_argument("--focus_normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--zero_focus_conditions", action="store_true")
    parser.add_argument("--max_pixels", type=int, default=None)
    parser.add_argument("--size_divisor", type=int, default=32)
    parser.add_argument("--aspect_ratio_tolerance", type=float, default=0.01)
    parser.add_argument("--downscale_if_exceeds_max_pixels", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_train_steps", type=int, default=20000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--controlnet_layers", type=int, default=7)
    parser.add_argument("--conditioning_scale", type=float, default=1.0)
    parser.add_argument("--train_transformer_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--lora_target_modules", default="to_q,to_k,to_v")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="no")
    return parser.parse_args()


class ControlConditionProjection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.proj.weight[:, : min(6, in_channels)])
        if in_channels > 6:
            nn.init.zeros_(self.proj.weight[:, 6:])
        nn.init.zeros_(self.proj.bias)
        print(
            f"expanded control condition embedding from {in_channels} to {out_channels} latent channels: "
            "A/B initialized, focus zero-initialized",
            flush=True,
        )

    def forward(self, x):
        return self.proj(x)


class SanaControlNetABFusionModel(nn.Module):
    def __init__(self, transformer, controlnet, condition_projection):
        super().__init__()
        self.transformer = transformer
        self.controlnet = controlnet
        self.condition_projection = condition_projection
        self.last_residual_stats = None
        self.last_shape_stats = None

    def forward(self, hidden_states, encoder_hidden_states, encoder_attention_mask, timestep, control_condition, conditioning_scale):
        raw_shape = list(control_condition.shape)
        if control_condition.shape[-2:] != hidden_states.shape[-2:]:
            control_condition = F.interpolate(
                control_condition.float(),
                size=hidden_states.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        downsampled_shape = list(control_condition.shape)
        controlnet_cond = self.condition_projection(control_condition.float()).to(hidden_states)
        self.last_shape_stats = {
            "control_condition_raw_shape": raw_shape,
            "control_condition_downsampled_shape": downsampled_shape,
            "hidden_states_shape": list(hidden_states.shape),
            "controlnet_cond_shape": list(controlnet_cond.shape),
        }
        controlnet_dtype = self.controlnet.dtype
        control_samples = self.controlnet(
            hidden_states.to(controlnet_dtype),
            encoder_hidden_states=encoder_hidden_states.to(controlnet_dtype),
            encoder_attention_mask=encoder_attention_mask,
            timestep=timestep,
            controlnet_cond=controlnet_cond.to(controlnet_dtype),
            conditioning_scale=conditioning_scale,
            return_dict=False,
        )[0]
        residual = torch.stack([sample.detach().float().norm() for sample in control_samples])
        self.last_residual_stats = {
            "mean": float(torch.stack([sample.detach().float().mean() for sample in control_samples]).mean().cpu()),
            "std": float(torch.stack([sample.detach().float().std() for sample in control_samples]).mean().cpu()),
            "norm": float(residual.sum().cpu()),
        }
        transformer_dtype = self.transformer.dtype
        return self.transformer(
            hidden_states.to(transformer_dtype),
            encoder_hidden_states=encoder_hidden_states.to(transformer_dtype),
            encoder_attention_mask=encoder_attention_mask,
            timestep=timestep,
            controlnet_block_samples=tuple(sample.to(transformer_dtype) for sample in control_samples),
            return_dict=False,
        )[0]


class DiffSynthControlFocusDataset(Dataset):
    def __init__(self, args):
        records = load_metadata(args.dataset_metadata_path)
        end = None if args.max_samples is None else args.start_index + args.max_samples
        self.records = records[args.start_index : end] * args.dataset_repeat
        if not self.records:
            raise ValueError("Dataset selection is empty.")
        self.root = Path(args.dataset_base_path)
        self.args = args

    def __len__(self):
        return len(self.records)

    def _focus_path(self, sample, edits, index, slot):
        edit_index = 2 + slot
        if len(edits) > edit_index:
            return edits[edit_index]
        for key in (("focus_image", slot), ("focus_a" if slot == 0 else "focus_b", None), ("focus_a_path" if slot == 0 else "focus_b_path", None), ("mask_keep_a" if slot == 0 else "mask_b_ref", None)):
            name, item = key
            if name in sample:
                value = sample[name]
                if isinstance(value, list):
                    if len(value) > item:
                        return value[item]
                else:
                    return value
        warnings.warn(f"Sample {index} missing focus_{'a' if slot == 0 else 'b'}; using constant {self.args.focus_default_value}.")
        return None

    def __getitem__(self, index):
        sample = self.records[index]
        require_keys(sample, (self.args.target_key, self.args.edit_key), index)
        edits = sample[self.args.edit_key]
        if not isinstance(edits, list) or len(edits) < 2:
            raise ValueError(f"Sample {index} must contain edit_image[A,B].")
        target = Image.open(resolve_data_path(sample[self.args.target_key], self.root)).convert("RGB")
        image_a = Image.open(resolve_data_path(edits[0], self.root)).convert("RGB")
        image_b = Image.open(resolve_data_path(edits[1], self.root)).convert("RGB")
        focus_images = []
        for slot in (0, 1):
            path = self._focus_path(sample, edits, index, slot)
            focus_images.append(None if path is None else Image.open(resolve_data_path(path, self.root)))
        return {
            "target": target,
            "a": image_a,
            "b": image_b,
            "focus_a": focus_images[0],
            "focus_b": focus_images[1],
            "prompt": sample.get(self.args.prompt_key) or self.args.prompt,
        }


def focus_tensor(image, size, default_value, normalize=True):
    if image is None:
        return torch.full((1, size[1], size[0]), default_value, dtype=torch.float32)
    image = image.convert("L").resize(size, Image.Resampling.BILINEAR)
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32)).unsqueeze(0)
    if normalize:
        tensor = tensor / 255.0
    return tensor.clamp(0, 1)


def build_control_condition(batch, use_focus, zero_focus=False):
    parts = [batch["cond_a"], batch["cond_b"]]
    if use_focus:
        focus_a = torch.zeros_like(batch["focus_a"]) if zero_focus else batch["focus_a"]
        focus_b = torch.zeros_like(batch["focus_b"]) if zero_focus else batch["focus_b"]
        parts.extend([focus_a, focus_b])
    return torch.cat(parts, dim=1)


def main():
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("Dynamic SANA ControlNet AB fusion currently requires --batch_size 1.")
    expected_channels = 8 if args.use_focus_conditions else 6
    if args.control_condition_channels != expected_channels:
        raise ValueError(
            f"--control_condition_channels must be {expected_channels} when "
            f"--use_focus_conditions={args.use_focus_conditions}."
        )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision=args.mixed_precision)
    dtype = torch.float32
    if accelerator.device.type == "cuda" and args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif accelerator.device.type == "cuda" and args.mixed_precision == "fp16":
        dtype = torch.float16

    accelerator.print(f"[MODE] train", flush=True)
    accelerator.print(f"[LOSS_MODE] {args.loss_mode}", flush=True)
    accelerator.print(f"[CONTROL_CONDITION_CHANNELS] {args.control_condition_channels}", flush=True)
    accelerator.print(f"[USE_FOCUS_CONDITIONS] {args.use_focus_conditions}", flush=True)
    accelerator.print(f"[MAX_PIXELS] {args.max_pixels}", flush=True)
    accelerator.print(f"[TRAIN_MAX_SAMPLES] {args.max_samples}", flush=True)
    accelerator.print(f"[MAX_TRAIN_STEPS] {args.max_train_steps}", flush=True)
    accelerator.print(f"[OUTPUT_DIR] {args.output_dir}", flush=True)
    accelerator.print(f"[MODEL] {args.model}", flush=True)
    accelerator.print(f"[DATASET_METADATA_PATH] {args.dataset_metadata_path}", flush=True)

    pipe = SanaPipeline.from_pretrained(args.model, torch_dtype=dtype, **pretrained_kwargs(args))
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.model, subfolder="scheduler", **pretrained_kwargs(args))
    pipe.transformer.requires_grad_(False).eval()
    pipe.vae.requires_grad_(False).eval()
    pipe.text_encoder.requires_grad_(False).eval()
    if args.train_transformer_lora:
        pipe.transformer.add_adapter(
            LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.0,
                init_lora_weights="gaussian",
                target_modules=[item.strip() for item in args.lora_target_modules.split(",") if item.strip()],
            )
        )
    controlnet = initialize_controlnet_from_transformer(pipe.transformer, args.controlnet_layers)
    projection = ControlConditionProjection(args.control_condition_channels, pipe.transformer.config.in_channels)
    model = SanaControlNetABFusionModel(pipe.transformer, controlnet, projection)
    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()
    pipe.text_encoder.to(accelerator.device, dtype=dtype)
    pipe.vae.to(accelerator.device, dtype=torch.float32)
    pipe.transformer.to(accelerator.device, dtype=dtype)
    controlnet.to(accelerator.device, dtype=torch.float32)
    projection.to(accelerator.device, dtype=torch.float32)
    params = list(controlnet.parameters()) + list(projection.parameters()) + [p for p in pipe.transformer.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters.")
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=1e-2)
    accelerator.print(f"total trainable params: {sum(p.numel() for p in params):,}", flush=True)
    accelerator.print(f"ControlNet trainable params: {sum(p.numel() for p in controlnet.parameters()):,}", flush=True)
    transformer_params = [p for p in pipe.transformer.parameters() if p.requires_grad]
    accelerator.print(f"transformer trainable params: {sum(p.numel() for p in transformer_params):,}", flush=True)
    accelerator.print(f"LoRA trainable params: {sum(p.numel() for p in transformer_params):,}", flush=True)

    dataset = DiffSynthControlFocusDataset(args)

    def collate(samples):
        sample = samples[0]
        prepared, size_info = prepare_dynamic_images(
            {"a": sample["a"], "b": sample["b"], "target": sample["target"]},
            args.max_pixels,
            args.size_divisor,
            args.aspect_ratio_tolerance,
            args.downscale_if_exceeds_max_pixels,
        )
        canvas_w, canvas_h = size_info["canvas_size"]
        valid_mask = torch.zeros(1, canvas_h, canvas_w)
        valid_mask[:, : size_info["content_size"][1], : size_info["content_size"][0]] = 1
        focus_a = focus_tensor(sample["focus_a"], size_info["content_size"], args.focus_default_value, args.focus_normalize)
        focus_b = focus_tensor(sample["focus_b"], size_info["content_size"], args.focus_default_value, args.focus_normalize)
        focus_a = F.pad(focus_a, (0, canvas_w - size_info["content_size"][0], 0, canvas_h - size_info["content_size"][1]))
        focus_b = F.pad(focus_b, (0, canvas_w - size_info["content_size"][0], 0, canvas_h - size_info["content_size"][1]))
        return {
            "target": pipe.image_processor.preprocess(prepared["target"], height=canvas_h, width=canvas_w),
            "cond_a": pipe.image_processor.preprocess(prepared["a"], height=canvas_h, width=canvas_w),
            "cond_b": pipe.image_processor.preprocess(prepared["b"], height=canvas_h, width=canvas_w),
            "focus_a": focus_a.unsqueeze(0),
            "focus_b": focus_b.unsqueeze(0),
            "valid_mask": valid_mask.unsqueeze(0),
            "prompts": [sample["prompt"]],
        }

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    scheduler_timesteps = scheduler.timesteps.to(accelerator.device)
    scheduler_sigmas = scheduler.sigmas.to(accelerator.device)
    global_step = 0
    while global_step < args.max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(model):
                with torch.no_grad():
                    gt_latents = encode_vae_latents(pipe.vae, batch["target"].to(accelerator.device, torch.float32))
                    a_latents = encode_vae_latents(pipe.vae, batch["cond_a"].to(accelerator.device, torch.float32))
                    prompt_embeds, prompt_mask, _, _ = pipe.encode_prompt(batch["prompts"], False, device=accelerator.device, clean_caption=False, max_sequence_length=300)
                    prompt_embeds = prompt_embeds.to(dtype)
                    noise = torch.randn_like(a_latents)
                    indices = torch.randint(0, scheduler.config.num_train_timesteps, (a_latents.shape[0],), device=accelerator.device)
                    timesteps = scheduler_timesteps[indices]
                    sigmas = scheduler_sigmas[indices].view(-1, 1, 1, 1).to(a_latents.dtype)
                    source_latents = a_latents if args.loss_mode == "legacy_a_noise_gt_velocity" else gt_latents
                    noisy_latents = (1 - sigmas) * source_latents + sigmas * noise
                    target = noise - gt_latents
                control = build_control_condition(batch, args.use_focus_conditions, args.zero_focus_conditions).to(accelerator.device)
                pred = model(
                    noisy_latents.to(dtype),
                    prompt_embeds,
                    prompt_mask,
                    timesteps,
                    control,
                    args.conditioning_scale,
                ).float()
                error = pred - target.float()
                valid_mask = F.interpolate(batch["valid_mask"].to(accelerator.device), size=error.shape[-2:], mode="nearest")
                loss = (error.square() * valid_mask).sum() / (valid_mask.sum() * error.shape[1]).clamp_min(1)
                accelerator.backward(loss)
                grad = None
                for p in accelerator.unwrap_model(model).controlnet.parameters():
                    if p.grad is not None:
                        value = p.grad.detach().float().norm()
                        grad = value if grad is None else grad + value
                if accelerator.sync_gradients:
                    if grad is None or float(grad.detach().cpu()) == 0.0:
                        print("WARNING: ControlNet grad_norm is None or zero.", flush=True)
                    accelerator.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process and (global_step <= 5 or global_step % args.log_steps == 0):
                    print(f"step={global_step} loss={loss.detach().item():.6f}", flush=True)
                    print(json.dumps({
                        "timestep": tensor_stats(timesteps),
                        "A_latents": tensor_stats(a_latents),
                        "GT_latents": tensor_stats(gt_latents),
                        "noisy_latents": tensor_stats(noisy_latents),
                        "target": tensor_stats(target),
                        "model_pred": tensor_stats(pred),
                        "focus_a": tensor_stats(batch["focus_a"]),
                        "focus_b": tensor_stats(batch["focus_b"]),
                        "controlnet_grad_norm": None if grad is None else float(grad.detach().cpu()),
                        "control_residual": accelerator.unwrap_model(model).last_residual_stats,
                        "shape_stats": accelerator.unwrap_model(model).last_shape_stats,
                    }, ensure_ascii=False), flush=True)
                if global_step % args.save_steps == 0 or global_step == args.max_train_steps:
                    unwrapped = accelerator.unwrap_model(model)
                    ckpt = output_dir / f"checkpoint-{global_step}"
                    ckpt.mkdir(parents=True, exist_ok=True)
                    unwrapped.controlnet.save_pretrained(ckpt / "controlnet")
                    save_file({k: v.detach().cpu() for k, v in unwrapped.condition_projection.state_dict().items()}, ckpt / "condition_projection.safetensors")
                    (ckpt / "controlnet_ab_config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
                if global_step >= args.max_train_steps:
                    break
        if global_step >= args.max_train_steps:
            break

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.controlnet.save_pretrained(output_dir / "controlnet")
        save_file({k: v.detach().cpu() for k, v in unwrapped.condition_projection.state_dict().items()}, output_dir / "condition_projection.safetensors")
        if args.train_transformer_lora:
            SanaPipeline.save_lora_weights(output_dir / "transformer_lora", transformer_lora_layers=get_peft_model_state_dict(unwrapped.transformer))
        (output_dir / "controlnet_ab_config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
