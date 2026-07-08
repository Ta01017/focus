#!/usr/bin/env python

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from diffusers import FlowMatchEulerDiscreteScheduler, SanaPipeline

SCRIPT_DIR = Path(__file__).resolve().parent
DOF_DIR = SCRIPT_DIR.parent / "dof_fusion"
if str(DOF_DIR) not in sys.path:
    sys.path.insert(0, str(DOF_DIR))

from sana_dof import encode_vae_latents, tensor_stats  # noqa: E402
from sana_sprint_controlnet import initialize_controlnet_from_transformer  # noqa: E402

from artifact_repair_utils import (  # noqa: E402
    DEFAULT_REPAIR_PROMPT,
    add_pretrained_args,
    artifact_paths,
    build_control_condition,
    load_metadata,
    load_rgb,
    preprocess_pair_or_triplet,
    pretrained_kwargs,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SANA ControlNet for artifact/bottom-surface repair.")
    parser.add_argument("--model", default="Efficient-Large-Model/Sana_600M_1024px_diffusers")
    parser.add_argument("--dataset_metadata_path", required=True)
    parser.add_argument("--dataset_base_path", default=".")
    parser.add_argument("--target_key", default="image")
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt", default=DEFAULT_REPAIR_PROMPT)
    parser.add_argument(
        "--loss_mode",
        choices=("legacy_src_noise_gt_velocity", "standard_sft"),
        default="legacy_src_noise_gt_velocity",
    )
    parser.add_argument("--control_condition_channels", type=int, default=6)
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
    parser.add_argument("--max_train_steps", type=int, default=20000)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--controlnet_layers", type=int, default=7)
    parser.add_argument("--conditioning_scale", type=float, default=1.0)
    parser.add_argument("--pixel_loss_weight", type=float, default=0.0)
    parser.add_argument("--src_keep_loss_weight", type=float, default=0.0)
    parser.add_argument("--lowfreq_loss_weight", type=float, default=0.0)
    parser.add_argument("--train_transformer_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_modules", default="to_q,to_k,to_v,to_out.0")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="no")
    parser.add_argument("--seed", type=int, default=0)
    add_pretrained_args(parser)
    return parser.parse_args()


class ControlConditionProjection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
        )
        nn.init.kaiming_normal_(self.proj[0].weight)
        nn.init.zeros_(self.proj[0].bias)
        nn.init.zeros_(self.proj[2].weight)
        nn.init.zeros_(self.proj[2].bias)
        print("expanded artifact repair condition embedding to 6 channels: src/ref initialized", flush=True)

    def forward(self, x):
        return self.proj(x)


class SanaArtifactRepairControlNetModel(nn.Module):
    def __init__(self, transformer, controlnet, condition_projection):
        super().__init__()
        self.transformer = transformer
        self.controlnet = controlnet
        self.condition_projection = condition_projection
        self.last_shape_stats = None
        self.last_residual_stats = None

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        encoder_attention_mask,
        timestep,
        control_condition,
        conditioning_scale,
    ):
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
        control_samples = self.controlnet(
            hidden_states.to(self.controlnet.dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.controlnet.dtype),
            encoder_attention_mask=encoder_attention_mask,
            timestep=timestep,
            controlnet_cond=controlnet_cond.to(self.controlnet.dtype),
            conditioning_scale=conditioning_scale,
            return_dict=False,
        )[0]
        self.last_residual_stats = {
            "mean": float(torch.stack([s.detach().float().mean() for s in control_samples]).mean().cpu()),
            "std": float(torch.stack([s.detach().float().std() for s in control_samples]).mean().cpu()),
            "norm": float(torch.stack([s.detach().float().norm() for s in control_samples]).sum().cpu()),
        }
        return self.transformer(
            hidden_states.to(self.transformer.dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.transformer.dtype),
            encoder_attention_mask=encoder_attention_mask,
            timestep=timestep,
            controlnet_block_samples=tuple(s.to(self.transformer.dtype) for s in control_samples),
            return_dict=False,
        )[0]


class ArtifactRepairDataset(Dataset):
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
        paths = artifact_paths(sample, self.base_path, self.args.target_key, self.args.prompt_key)
        prompt = paths["prompt"] if paths["prompt"] else self.args.prompt
        return {
            "gt": load_rgb(paths["gt"]),
            "src": load_rgb(paths["src"]),
            "ref": load_rgb(paths["ref"]),
            "prompt": prompt,
            "path_info": {k: str(v) for k, v in paths.items() if k in ("gt", "src", "ref")},
        }


def save_checkpoint(output_dir, global_step, unwrapped, pipe, args, final=False):
    ckpt = Path(output_dir) if final else Path(output_dir) / f"checkpoint-{global_step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    unwrapped.controlnet.save_pretrained(ckpt / "controlnet")
    save_file(
        {k: v.detach().cpu() for k, v in unwrapped.condition_projection.state_dict().items()},
        ckpt / "condition_projection.safetensors",
    )
    if args.train_transformer_lora:
        SanaPipeline.save_lora_weights(
            ckpt / "transformer_lora",
            transformer_lora_layers=get_peft_model_state_dict(unwrapped.transformer),
        )
    config = vars(args).copy()
    config.update(
        {
            "task_name": "artifact_repair",
            "global_step": global_step,
            "control_condition_channels": args.control_condition_channels,
            "train_transformer_lora": args.train_transformer_lora,
            "latent_channels": pipe.transformer.config.in_channels,
            "dataset_metadata_path": args.dataset_metadata_path,
            "dataset_base_path": args.dataset_base_path,
        }
    )
    text = json.dumps(config, indent=2, ensure_ascii=False)
    (ckpt / "artifact_repair_config.json").write_text(text, encoding="utf-8")
    (ckpt / "adapter_config.json").write_text(text, encoding="utf-8")


def main():
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("Dynamic artifact repair training currently requires --batch_size 1.")
    if args.control_condition_channels != 6:
        raise ValueError("--control_condition_channels must be 6 for [src_rgb, ref_rgb].")
    if args.pixel_loss_weight or args.src_keep_loss_weight or args.lowfreq_loss_weight:
        print("WARNING: pixel/src/lowfreq auxiliary losses are declared but not enabled in this first version.", flush=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    dtype = torch.float32
    if accelerator.device.type == "cuda" and args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif accelerator.device.type == "cuda" and args.mixed_precision == "fp16":
        dtype = torch.float16

    for key, value in {
        "MODE": "train",
        "TASK_NAME": "artifact_repair",
        "LOSS_MODE": args.loss_mode,
        "CONTROL_CONDITION_CHANNELS": args.control_condition_channels,
        "MODEL": args.model,
        "DATASET_METADATA_PATH": args.dataset_metadata_path,
        "DATASET_BASE_PATH": args.dataset_base_path,
        "OUTPUT_DIR": args.output_dir,
        "MAX_PIXELS": args.max_pixels,
        "MAX_TRAIN_STEPS": args.max_train_steps,
        "TRAIN_MAX_SAMPLES": args.max_samples,
        "TRAIN_TRANSFORMER_LORA": args.train_transformer_lora,
        "LORA_RANK": args.lora_rank,
        "LORA_TARGET_MODULES": args.lora_target_modules,
    }.items():
        accelerator.print(f"[{key}] {value}", flush=True)

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
                lora_dropout=args.lora_dropout,
                init_lora_weights="gaussian",
                target_modules=[x.strip() for x in args.lora_target_modules.split(",") if x.strip()],
            )
        )
    controlnet = initialize_controlnet_from_transformer(pipe.transformer, args.controlnet_layers)
    projection = ControlConditionProjection(args.control_condition_channels, pipe.transformer.config.in_channels)
    model = SanaArtifactRepairControlNetModel(pipe.transformer, controlnet, projection)
    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    pipe.text_encoder.to(accelerator.device, dtype=dtype)
    pipe.vae.to(accelerator.device, dtype=torch.float32)
    pipe.transformer.to(accelerator.device, dtype=dtype)
    controlnet.to(accelerator.device, dtype=torch.float32)
    projection.to(accelerator.device, dtype=torch.float32)

    trainable_params = list(controlnet.parameters()) + list(projection.parameters()) + [
        p for p in pipe.transformer.parameters() if p.requires_grad
    ]
    if not trainable_params:
        raise ValueError("No trainable parameters.")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=1e-2)
    transformer_params = [p for p in pipe.transformer.parameters() if p.requires_grad]
    accelerator.print(f"total trainable params: {sum(p.numel() for p in trainable_params):,}", flush=True)
    accelerator.print(f"ControlNet trainable params: {sum(p.numel() for p in controlnet.parameters()):,}", flush=True)
    accelerator.print(f"transformer trainable params: {sum(p.numel() for p in transformer_params):,}", flush=True)
    accelerator.print(f"LoRA trainable params: {sum(p.numel() for p in transformer_params):,}", flush=True)

    dataset = ArtifactRepairDataset(args)

    def collate(samples):
        sample = samples[0]
        prepared, size_info = preprocess_pair_or_triplet(
            sample["gt"],
            sample["src"],
            sample["ref"],
            max_pixels=args.max_pixels,
            size_divisor=args.size_divisor,
            downscale_if_exceeds_max_pixels=args.downscale_if_exceeds_max_pixels,
        )
        canvas_w, canvas_h = size_info["canvas_size"]
        valid_mask = torch.zeros(1, canvas_h, canvas_w)
        valid_mask[:, : size_info["content_size"][1], : size_info["content_size"][0]] = 1
        return {
            "gt": pipe.image_processor.preprocess(prepared["gt"], height=canvas_h, width=canvas_w),
            "src": pipe.image_processor.preprocess(prepared["src"], height=canvas_h, width=canvas_w),
            "ref": pipe.image_processor.preprocess(prepared["ref"], height=canvas_h, width=canvas_w),
            "valid_mask": valid_mask.unsqueeze(0),
            "prompts": [sample["prompt"]],
            "path_info": sample["path_info"],
        }

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    scheduler_timesteps = scheduler.timesteps.to(accelerator.device)
    scheduler_sigmas = scheduler.sigmas.to(accelerator.device)
    global_step = 0
    zero_residual_steps = 0
    while global_step < args.max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(model):
                with torch.no_grad():
                    gt_latents = encode_vae_latents(pipe.vae, batch["gt"].to(accelerator.device, torch.float32))
                    src_latents = encode_vae_latents(pipe.vae, batch["src"].to(accelerator.device, torch.float32))
                    prompt_embeds, prompt_mask, _, _ = pipe.encode_prompt(
                        batch["prompts"],
                        False,
                        device=accelerator.device,
                        clean_caption=False,
                        max_sequence_length=300,
                    )
                    prompt_embeds = prompt_embeds.to(dtype)
                    noise = torch.randn_like(gt_latents)
                    indices = torch.randint(0, scheduler.config.num_train_timesteps, (gt_latents.shape[0],), device=accelerator.device)
                    timesteps = scheduler_timesteps[indices]
                    sigmas = scheduler_sigmas[indices].view(-1, 1, 1, 1).to(gt_latents.dtype)
                    source_latents = src_latents if args.loss_mode == "legacy_src_noise_gt_velocity" else gt_latents
                    noisy_latents = (1 - sigmas) * source_latents + sigmas * noise
                    target = noise - gt_latents
                control = build_control_condition(batch["src"], batch["ref"]).to(accelerator.device)
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
                    accelerator.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                unwrapped = accelerator.unwrap_model(model)
                residual_norm = None if unwrapped.last_residual_stats is None else unwrapped.last_residual_stats.get("norm")
                if residual_norm == 0:
                    zero_residual_steps += 1
                else:
                    zero_residual_steps = 0
                if zero_residual_steps >= 3:
                    print("WARNING: Control residual norm has been zero for multiple steps.", flush=True)
                if accelerator.is_main_process and (global_step <= 5 or global_step % args.log_steps == 0):
                    print(f"step={global_step} loss={loss.detach().item():.6f}", flush=True)
                    print(
                        json.dumps(
                            {
                                "timestep": tensor_stats(timesteps),
                                "src_latents": tensor_stats(src_latents),
                                "gt_latents": tensor_stats(gt_latents),
                                "noisy_latents": tensor_stats(noisy_latents),
                                "target": tensor_stats(target),
                                "model_pred": tensor_stats(pred),
                                "src_rgb": tensor_stats(batch["src"]),
                                "ref_rgb": tensor_stats(batch["ref"]),
                                "controlnet_grad_norm": None if grad is None else float(grad.detach().cpu()),
                                "control_residual": unwrapped.last_residual_stats,
                                "shape_stats": unwrapped.last_shape_stats,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                if accelerator.is_main_process and (global_step % args.save_steps == 0 or global_step == args.max_train_steps):
                    save_checkpoint(output_dir, global_step, unwrapped, pipe, args, final=False)
                if global_step >= args.max_train_steps:
                    break
        if global_step >= args.max_train_steps:
            break

    if accelerator.is_main_process:
        save_checkpoint(output_dir, global_step, accelerator.unwrap_model(model), pipe, args, final=True)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
