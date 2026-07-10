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
DOF_DIR = SCRIPT_DIR.parent / "dof_fusion"
if str(DOF_DIR) not in sys.path:
    sys.path.insert(0, str(DOF_DIR))

from sana_dof import encode_vae_latents  # noqa: E402

from artifact_repair_utils import (  # noqa: E402
    DEFAULT_REPAIR_PROMPT,
    add_pretrained_args,
    get_artifact_repair_paths,
    load_metadata,
    load_rgb,
    preprocess_triplet,
    pretrained_kwargs,
)
from sana_artifact_repair_latent_concat import (  # noqa: E402
    DEFAULT_LATENT_CONCAT_LORA_TARGET_MODULES,
    SanaArtifactRepairLatentConcatModel,
    SrcLatentConditionInjector,
    add_lora_to_transformer,
    count_trainable_params,
    route1_tensor_stats,
    save_latent_concat_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Artifact Repair Route 1: SANA src latent concat / image-input injection."
    )
    parser.add_argument("--model", default="Efficient-Large-Model/Sana_600M_1024px_diffusers")
    parser.add_argument("--dataset_metadata_path", required=True)
    parser.add_argument("--dataset_base_path", default=".")
    parser.add_argument("--target_key", default="image")
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt", default=DEFAULT_REPAIR_PROMPT)
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
    parser.add_argument("--train_transformer_lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_scope", choices=("attn_qkv", "attn_qkvo", "wide"), default="wide")
    parser.add_argument("--lora_target_modules", default=DEFAULT_LATENT_CONCAT_LORA_TARGET_MODULES)
    parser.add_argument("--injector_hidden_channels", type=int, default=128)
    parser.add_argument("--pixel_loss_weight", type=float, default=0.0)
    parser.add_argument("--x0_loss_weight", type=float, default=0.0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="no")
    parser.add_argument("--debug_check_finite", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    add_pretrained_args(parser)
    return parser.parse_args()


class Route1ArtifactRepairDataset(Dataset):
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
        gt_path = paths["gt"]
        src_path = paths["src"]
        ref_path = paths["ref"]
        missing = [str(path) for path in (gt_path, src_path, ref_path) if path is not None and not Path(path).exists()]
        if missing:
            raise FileNotFoundError(f"Route 1 sample paths do not exist: {missing}")
        return {
            "gt": load_rgb(gt_path),
            "src": load_rgb(src_path),
            "ref": load_rgb(ref_path),
            "prompt": paths["prompt"] or self.args.prompt,
            "path_info": {"gt": str(gt_path), "src": str(src_path), "ref": str(ref_path)},
        }


def main():
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("Dynamic-size Route 1 training currently requires --batch_size 1.")
    if args.pixel_loss_weight != 0.0 or args.x0_loss_weight != 0.0:
        raise ValueError("pixel/x0 auxiliary losses are reserved for later work; keep their weights at 0.0.")
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
        "ROUTE": "Artifact Repair Route 1: src latent concat / image-input injection",
        "MODEL": args.model,
        "DATASET_METADATA_PATH": args.dataset_metadata_path,
        "DATASET_BASE_PATH": args.dataset_base_path,
        "OUTPUT_DIR": args.output_dir,
        "MIXED_PRECISION": args.mixed_precision,
        "MAX_PIXELS": args.max_pixels,
        "TRAIN_TRANSFORMER_LORA": args.train_transformer_lora,
        "LORA_SCOPE": args.lora_scope,
        "LORA_RANK": args.lora_rank,
        "INJECTOR_HIDDEN_CHANNELS": args.injector_hidden_channels,
    }.items():
        accelerator.print(f"[ROUTE1] {key}={value}", flush=True)
    accelerator.print("[ROUTE1] ref is read from metadata for compatibility but ignored by model computation.", flush=True)

    pipe = SanaPipeline.from_pretrained(args.model, torch_dtype=dtype, **pretrained_kwargs(args))
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.model, subfolder="scheduler", **pretrained_kwargs(args))
    pipe.transformer.requires_grad_(False).eval()
    pipe.vae.requires_grad_(False).eval()
    pipe.text_encoder.requires_grad_(False).eval()
    matched_names = []
    if args.train_transformer_lora:
        _, matched_names = add_lora_to_transformer(
            pipe.transformer,
            args.lora_rank,
            args.lora_alpha,
            args.lora_target_modules,
            args.lora_dropout,
            args.lora_scope,
        )
    if args.gradient_checkpointing and hasattr(pipe.transformer, "enable_gradient_checkpointing"):
        pipe.transformer.enable_gradient_checkpointing()

    inner_dim = pipe.transformer.config.num_attention_heads * pipe.transformer.config.attention_head_dim
    injector = SrcLatentConditionInjector(
        pipe.transformer.config.in_channels,
        inner_dim,
        patch_size=pipe.transformer.config.patch_size,
        hidden_channels=args.injector_hidden_channels,
    )
    model = SanaArtifactRepairLatentConcatModel(pipe.transformer, injector)
    pipe.transformer = model

    pipe.text_encoder.to(accelerator.device, dtype=dtype)
    pipe.vae.to(accelerator.device, dtype=torch.float32)
    model.to(accelerator.device, dtype=dtype)
    injector.to(accelerator.device, dtype=torch.float32)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    injector_params = [p for p in injector.parameters() if p.requires_grad]
    lora_params = [p for p in model.transformer.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters. Enable injector and/or transformer LoRA.")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=1e-2)
    accelerator.print(f"[ROUTE1] total trainable params: {sum(p.numel() for p in trainable_params):,}", flush=True)
    accelerator.print(f"[ROUTE1] injector trainable params: {sum(p.numel() for p in injector_params):,}", flush=True)
    accelerator.print(f"[ROUTE1] transformer LoRA trainable params: {sum(p.numel() for p in lora_params):,}", flush=True)
    accelerator.print(f"[ROUTE1] LoRA matched modules: {len(matched_names)}", flush=True)

    dataset = Route1ArtifactRepairDataset(args)

    def collate(samples):
        sample = samples[0]
        prepared, size_info = preprocess_triplet(
            sample["gt"],
            sample["src"],
            sample["ref"],
            args.max_pixels,
            args.size_divisor,
            args.downscale_if_exceeds_max_pixels,
        )
        canvas_w, canvas_h = size_info["canvas_size"]
        return {
            "gt": pipe.image_processor.preprocess(prepared["gt"], height=canvas_h, width=canvas_w),
            "src": pipe.image_processor.preprocess(prepared["src"], height=canvas_h, width=canvas_w),
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
    while global_step < args.max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(model):
                with torch.no_grad():
                    gt_latents = encode_vae_latents(pipe.vae, batch["gt"].to(accelerator.device, torch.float32))
                    src_latents = encode_vae_latents(pipe.vae, batch["src"].to(accelerator.device, torch.float32))
                    if src_latents.shape[-2:] != gt_latents.shape[-2:]:
                        src_latents = F.interpolate(
                            src_latents.float(),
                            size=gt_latents.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        ).to(gt_latents)
                    prompt_embeds, prompt_mask, _, _ = pipe.encode_prompt(
                        batch["prompts"],
                        False,
                        device=accelerator.device,
                        clean_caption=False,
                        max_sequence_length=300,
                    )
                    prompt_embeds = prompt_embeds.to(dtype)
                    noise = torch.randn_like(gt_latents)
                    indices = torch.randint(
                        0,
                        scheduler.config.num_train_timesteps,
                        (gt_latents.shape[0],),
                        device=accelerator.device,
                    )
                    timesteps = scheduler_timesteps[indices]
                    sigmas = scheduler_sigmas[indices].view(-1, 1, 1, 1).to(gt_latents.dtype)
                    noisy_latents = (1 - sigmas) * gt_latents + sigmas * noise
                    target = noise - gt_latents
                pred = model(
                    noisy_latents.to(dtype),
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_mask,
                    timestep=timesteps * model.config.timestep_scale,
                    src_latents=src_latents.to(dtype),
                    return_dict=False,
                )[0].float()
                loss = F.mse_loss(pred, target.float())
                if args.debug_check_finite and not torch.isfinite(loss):
                    raise RuntimeError("[ROUTE1] Non-finite loss detected.")
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(trainable_params, 1.0)
                if args.debug_check_finite:
                    for name, parameter in accelerator.unwrap_model(model).named_parameters():
                        if parameter.requires_grad and parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                            raise RuntimeError(f"[ROUTE1] Non-finite gradient detected: {name}")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process and (global_step <= 5 or global_step % args.log_steps == 0):
                    unwrapped = accelerator.unwrap_model(model)
                    print(
                        f"step={global_step} loss={loss.detach().item():.6f} lr={optimizer.param_groups[0]['lr']:.3e}",
                        flush=True,
                    )
                    print(
                        {
                            "target_latents": route1_tensor_stats(gt_latents),
                            "src_latents": route1_tensor_stats(src_latents),
                            "injection": unwrapped.last_injection_stats,
                            "trainable_params": count_trainable_params(unwrapped),
                        },
                        flush=True,
                    )
                if accelerator.is_main_process and (global_step % args.save_steps == 0 or global_step == args.max_train_steps):
                    save_latent_concat_checkpoint(
                        output_dir / f"checkpoint-{global_step}",
                        accelerator.unwrap_model(model),
                        pipe,
                        args,
                        global_step,
                    )
                if global_step >= args.max_train_steps:
                    break
        if global_step >= args.max_train_steps:
            break

    if accelerator.is_main_process:
        save_latent_concat_checkpoint(output_dir, accelerator.unwrap_model(model), pipe, args, global_step)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
