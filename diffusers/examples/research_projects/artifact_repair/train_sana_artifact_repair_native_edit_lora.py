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
    NativeEditSanaTransformer,
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
    parser.add_argument("--enable_native_edit_tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_edit_images", type=int, default=2)
    parser.add_argument("--edit_role_embedding", action=argparse.BooleanOptionalAction, default=True)
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

    accelerator.print("[NATIVE_EDIT] enabled", flush=True)
    accelerator.print("[NATIVE_EDIT] model = SANA", flush=True)
    accelerator.print("[NATIVE_EDIT] using GT as clean diffusion target", flush=True)
    accelerator.print("[NATIVE_EDIT] edit_image[0] = src", flush=True)
    accelerator.print("[NATIVE_EDIT] edit_image[1] = ref", flush=True)
    accelerator.print("[NATIVE_EDIT] token concat enabled", flush=True)
    accelerator.print(f"[NATIVE_EDIT] role embedding {'enabled' if args.edit_role_embedding else 'disabled'}", flush=True)
    accelerator.print("[NATIVE_EDIT] no ControlNet", flush=True)
    accelerator.print("[NATIVE_EDIT] no Adapter", flush=True)
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
    if args.train_transformer_lora:
        _, matched_names = add_lora_to_transformer(
            pipe.transformer, args.lora_rank, args.lora_alpha, args.lora_target_modules, args.lora_dropout
        )
    native_transformer = NativeEditSanaTransformer(pipe.transformer, use_role_embedding=args.edit_role_embedding)
    pipe.transformer = native_transformer

    pipe.text_encoder.to(accelerator.device, dtype=dtype)
    pipe.vae.to(accelerator.device, dtype=torch.float32)
    native_transformer.to(accelerator.device, dtype=dtype)
    if native_transformer.role_embed is not None:
        native_transformer.role_embed.requires_grad_(True)
    params = [p for p in native_transformer.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters. Enable transformer LoRA or role embedding.")
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=1e-2)
    accelerator.print(f"[NATIVE_EDIT] trainable params: {sum(p.numel() for p in params):,}", flush=True)
    accelerator.print(f"[NATIVE_EDIT] LoRA module count: {len(matched_names)}", flush=True)

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
                    edit_role_ids=[1, 2],
                    enable_native_edit_tokens=True,
                    return_dict=False,
                )[0].float()
                loss = F.mse_loss(pred, target.float())
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process and (global_step <= 5 or global_step % args.log_steps == 0):
                    stats = accelerator.unwrap_model(native_transformer).last_token_stats
                    print(
                        f"step={global_step} loss={loss.detach().item():.6f} lr={optimizer.param_groups[0]['lr']:.3e} "
                        f"target_tokens={stats.get('target_token_length')} total_tokens={stats.get('total_token_length')}",
                        flush=True,
                    )
                    print(
                        {
                            "timestep": tensor_stats(timesteps),
                            "gt_latents": tensor_stats(gt_latents),
                            "src_latents": tensor_stats(src_latents),
                            "ref_latents": tensor_stats(ref_latents),
                            "target_token_length": stats.get("target_token_length"),
                            "src_token_length": stats.get("edit_token_lengths", [None, None])[0],
                            "ref_token_length": stats.get("edit_token_lengths", [None, None])[1],
                            "total_token_length": stats.get("total_token_length"),
                            "trainable_params": count_trainable_params(accelerator.unwrap_model(native_transformer)),
                            "lora_module_count": len(matched_names),
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
