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

from sana_dof import encode_vae_latents, tensor_stats  # noqa: E402

from artifact_repair_utils import (  # noqa: E402
    DEFAULT_REPAIR_PROMPT,
    add_pretrained_args,
    get_artifact_repair_paths,
    load_metadata,
    load_rgb,
    preprocess_triplet,
    pretrained_kwargs,
)
from sana_native_edit_utils import (  # noqa: E402
    DEFAULT_NATIVE_EDIT_LORA_TARGET_MODULES,
    SanaNativeEditCrossAttentionWrapper,
    add_lora_to_transformer,
    count_trainable_params,
    save_native_edit_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SANA native edit LoRA for artifact repair.")
    parser.add_argument("--model", default="Efficient-Large-Model/Sana_600M_1024px_diffusers")
    parser.add_argument("--dataset_metadata_path", required=True)
    parser.add_argument("--dataset_base_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=20000)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="no")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_pixels", type=int, default=1048576)
    parser.add_argument("--size_divisor", type=int, default=32)
    parser.add_argument("--downscale_if_exceeds_max_pixels", action="store_true")
    parser.add_argument("--train_transformer_lora", action="store_true", default=True)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_modules", default=DEFAULT_NATIVE_EDIT_LORA_TARGET_MODULES)
    parser.add_argument("--lora_scope", choices=("cross_attention", "all_attention", "wide"), default="wide")
    parser.add_argument("--enable_native_edit_tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--native_edit_impl", choices=("cross_attention_v2",), default="cross_attention_v2")
    parser.add_argument("--num_edit_images", type=int, default=2)
    parser.add_argument("--edit_role_embedding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--edit_condition_scale", type=float, default=1.0)
    parser.add_argument("--use_edit_token_norm", dest="use_edit_token_norm", action="store_true", default=True)
    parser.add_argument("--no_edit_token_norm", dest="use_edit_token_norm", action="store_false")
    parser.add_argument("--debug_nan", action="store_true")
    parser.add_argument("--drop_src_prob", type=float, default=0.0)
    parser.add_argument("--drop_ref_prob", type=float, default=0.0)
    parser.add_argument("--swap_src_ref_prob", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    add_pretrained_args(parser)
    return parser.parse_args()


class NativeEditArtifactDataset(Dataset):
    def __init__(self, args):
        records = load_metadata(args.dataset_metadata_path)
        end = None if args.max_samples is None else args.start_index + args.max_samples
        self.records = records[args.start_index:end] * args.dataset_repeat
        if not self.records:
            raise ValueError("Dataset selection is empty.")
        self.base = Path(args.dataset_base_path)
        self.args = args

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        sample = self.records[index]
        paths = get_artifact_repair_paths(sample, self.base, index)
        return {
            "gt": load_rgb(paths["gt"]),
            "src": load_rgb(paths["src"]),
            "ref": load_rgb(paths["ref"]),
            "prompt": paths["prompt"] or DEFAULT_REPAIR_PROMPT,
        }


def main():
    args = parse_args()
    if args.num_edit_images != 2:
        raise ValueError("This first native edit route expects exactly two edit images: src and ref.")
    if not args.enable_native_edit_tokens:
        raise ValueError("--enable_native_edit_tokens must stay enabled for this route.")
    if args.batch_size != 1:
        raise ValueError("Dynamic native edit training currently requires --batch_size 1.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision=args.mixed_precision)
    dtype = torch.float32
    if accelerator.device.type == "cuda" and args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif accelerator.device.type == "cuda" and args.mixed_precision == "fp16":
        dtype = torch.float16

    accelerator.print("[NATIVE_EDIT_V2] implementation=cross_attention", flush=True)
    accelerator.print("[NATIVE_EDIT_V2] target stream=H×W target-only", flush=True)
    accelerator.print("[NATIVE_EDIT_V2] src/ref stream=cross-attention conditions", flush=True)
    accelerator.print("[NATIVE_EDIT_V2] legacy H×3W disabled", flush=True)
    accelerator.print("[NATIVE_EDIT_V2] edit_image[0]=src", flush=True)
    accelerator.print("[NATIVE_EDIT_V2] edit_image[1]=ref", flush=True)
    accelerator.print("[NATIVE_EDIT_V2] GT is clean flow target", flush=True)
    accelerator.print("[NATIVE_EDIT_V2] no ControlNet", flush=True)
    accelerator.print("[NATIVE_EDIT_V2] no Adapter", flush=True)
    if args.drop_src_prob or args.drop_ref_prob or args.swap_src_ref_prob:
        accelerator.print("[NATIVE_EDIT] drop/swap augmentation parameters are accepted but not enabled in this first version.", flush=True)

    pipe = SanaPipeline.from_pretrained(args.model, torch_dtype=dtype, **pretrained_kwargs(args))
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.model, subfolder="scheduler", **pretrained_kwargs(args))
    pipe.vae.requires_grad_(False).eval()
    pipe.text_encoder.requires_grad_(False).eval()
    pipe.transformer.requires_grad_(False).eval()
    if args.gradient_checkpointing:
        pipe.transformer.enable_gradient_checkpointing()
    matched_names = []
    matched_counts = {"attn1": 0, "attn2": 0, "ff_proj": 0}
    if args.train_transformer_lora:
        _, matched_names, matched_counts = add_lora_to_transformer(
            pipe.transformer,
            args.lora_rank,
            args.lora_alpha,
            args.lora_target_modules,
            args.lora_dropout,
            args.lora_scope,
        )
    native_transformer = SanaNativeEditCrossAttentionWrapper(
        pipe.transformer,
        num_edit_images=args.num_edit_images,
        use_edit_role_embedding=args.edit_role_embedding,
        edit_condition_scale=args.edit_condition_scale,
        use_edit_token_norm=args.use_edit_token_norm,
    )
    pipe.transformer = native_transformer

    pipe.text_encoder.to(accelerator.device, dtype=dtype)
    pipe.vae.to(accelerator.device, dtype=torch.float32)
    native_transformer.to(accelerator.device, dtype=dtype)
    if native_transformer.edit_role_embedding is not None:
        native_transformer.edit_role_embedding.requires_grad_(True)
    params = [p for p in native_transformer.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters. Enable transformer LoRA or role embedding.")
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=1e-2)
    accelerator.print(f"[NATIVE_EDIT] trainable params: {sum(p.numel() for p in params):,}", flush=True)
    accelerator.print(f"[NATIVE_EDIT] LoRA module count: {len(matched_names)}", flush=True)
    accelerator.print(f"[NATIVE_EDIT] attn2 LoRA module count: {matched_counts['attn2']}", flush=True)
    accelerator.print(
        f"[NATIVE_EDIT] edit_role_embedding params: "
        f"{0 if native_transformer.edit_role_embedding is None else native_transformer.edit_role_embedding.numel():,}",
        flush=True,
    )
    accelerator.print(
        f"[NATIVE_EDIT] edit_token_norm params: "
        f"{sum(p.numel() for p in native_transformer.edit_token_norm.parameters() if p.requires_grad):,}",
        flush=True,
    )
    accelerator.print(
        f"[NATIVE_EDIT] edit_condition_projection params: "
        f"{sum(p.numel() for p in native_transformer.edit_condition_projection.parameters() if p.requires_grad):,}",
        flush=True,
    )

    dataset = NativeEditArtifactDataset(args)

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
            "ref": pipe.image_processor.preprocess(prepared["ref"], height=canvas_h, width=canvas_w),
            "prompts": [sample["prompt"]],
        }

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate)
    native_transformer, optimizer, dataloader = accelerator.prepare(native_transformer, optimizer, dataloader)
    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    scheduler_timesteps = scheduler.timesteps.to(accelerator.device)
    scheduler_sigmas = scheduler.sigmas.to(accelerator.device)
    global_step = 0
    while global_step < args.max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(native_transformer):
                with torch.no_grad():
                    gt_latents = encode_vae_latents(pipe.vae, batch["gt"].to(accelerator.device, torch.float32))
                    src_latents = encode_vae_latents(pipe.vae, batch["src"].to(accelerator.device, torch.float32))
                    ref_latents = encode_vae_latents(pipe.vae, batch["ref"].to(accelerator.device, torch.float32))
                    if src_latents.shape[-2:] != gt_latents.shape[-2:]:
                        src_latents = F.interpolate(src_latents.float(), size=gt_latents.shape[-2:], mode="bilinear", align_corners=False).to(gt_latents)
                    if ref_latents.shape[-2:] != gt_latents.shape[-2:]:
                        ref_latents = F.interpolate(ref_latents.float(), size=gt_latents.shape[-2:], mode="bilinear", align_corners=False).to(gt_latents)
                    prompt_embeds, prompt_mask, _, _ = pipe.encode_prompt(
                        batch["prompts"], False, device=accelerator.device, clean_caption=False, max_sequence_length=300
                    )
                    prompt_embeds = prompt_embeds.to(dtype)
                    noise = torch.randn_like(gt_latents)
                    indices = torch.randint(0, scheduler.config.num_train_timesteps, (gt_latents.shape[0],), device=accelerator.device)
                    timesteps = scheduler_timesteps[indices]
                    sigmas = scheduler_sigmas[indices].view(-1, 1, 1, 1).to(gt_latents.dtype)
                    noisy_latents = (1 - sigmas) * gt_latents + sigmas * noise
                    target = noise - gt_latents
                pred = native_transformer(
                    noisy_latents.to(dtype),
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_mask,
                    timestep=timesteps * native_transformer.config.timestep_scale,
                    edit_hidden_states=[src_latents.to(dtype), ref_latents.to(dtype)],
                    debug_nan=args.debug_nan,
                    return_dict=False,
                )[0].float()
                loss = F.mse_loss(pred, target.float())
                if args.debug_nan and not torch.isfinite(loss):
                    raise RuntimeError("[NATIVE_EDIT_V2] Non-finite loss detected.")
                accelerator.backward(loss)
                grad_norm = accelerator.clip_grad_norm_(params, 1.0)
                if args.debug_nan:
                    for name, parameter in accelerator.unwrap_model(native_transformer).named_parameters():
                        if parameter.requires_grad and parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                            raise RuntimeError(f"[NATIVE_EDIT_V2] Non-finite gradient detected: {name}")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process and (global_step <= 5 or global_step % args.log_steps == 0):
                    stats = accelerator.unwrap_model(native_transformer).last_token_stats
                    print(
                        f"step={global_step} loss={loss.detach().item():.6f} lr={optimizer.param_groups[0]['lr']:.3e} "
                        f"target_tokens={stats.get('target_token_length')} "
                        f"encoder_condition_tokens={stats.get('total_encoder_condition_length')}",
                        flush=True,
                    )
                    print(
                        {
                            "timestep": tensor_stats(timesteps),
                            "gt_latents": tensor_stats(gt_latents),
                            "src_latents": tensor_stats(src_latents),
                            "ref_latents": tensor_stats(ref_latents),
                            "target_grid_h": stats.get("target_grid_h"),
                            "target_grid_w": stats.get("target_grid_w"),
                            "target_token_len": stats.get("target_token_length"),
                            "text_token_len": stats.get("text_token_length"),
                            "src_condition_token_len": stats.get("src_condition_token_length"),
                            "ref_condition_token_len": stats.get("ref_condition_token_length"),
                            "total_encoder_condition_len": stats.get("total_encoder_condition_length"),
                            "edit_condition_scale": stats.get("edit_condition_scale"),
                            "dtype": str(dtype),
                            "grad_norm": None if grad_norm is None else float(grad_norm),
                            "trainable_params": count_trainable_params(accelerator.unwrap_model(native_transformer)),
                            "lora_module_count": len(matched_names),
                            "attn2_lora_module_count": matched_counts["attn2"],
                        },
                        flush=True,
                    )
                if accelerator.is_main_process and (global_step % args.save_steps == 0 or global_step == args.max_train_steps):
                    save_native_edit_checkpoint(
                        output_dir / f"checkpoint-{global_step}",
                        accelerator.unwrap_model(native_transformer),
                        pipe,
                        args,
                        global_step,
                    )
                if global_step >= args.max_train_steps:
                    break
        if global_step >= args.max_train_steps:
            break

    if accelerator.is_main_process:
        save_native_edit_checkpoint(output_dir, accelerator.unwrap_model(native_transformer), pipe, args, global_step)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
