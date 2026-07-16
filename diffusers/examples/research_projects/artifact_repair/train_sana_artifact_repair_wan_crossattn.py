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

from diffusers import FlowMatchEulerDiscreteScheduler, SanaPipeline, SanaTransformer2DModel

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
    load_route2_patch_embedding,
    patch_embedding_weight_stats,
    route2_full_transformer_path,
    route2_lora_path,
    tensor_stats,
)
from sana_artifact_repair_wan_crossattn import (  # noqa: E402
    IMPLEMENTATION,
    build_route3_model,
    encode_image_tokens,
    load_image_encoder_and_processor,
    load_route3_config,
    save_route3_checkpoint,
    validate_route2_init,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SANA Artifact Repair Route 3 latent-concat + Wan image xattn.")
    parser.add_argument("--model", default="Efficient-Large-Model/Sana_600M_1024px_diffusers")
    parser.add_argument("--dataset_metadata_path", required=True)
    parser.add_argument("--dataset_base_path", default=".")
    parser.add_argument("--target_key", default="image")
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt", default=DEFAULT_REPAIR_PROMPT)
    parser.add_argument("--train_mode", choices=("patch_lora", "full_transformer"), default="patch_lora")
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
    parser.add_argument("--image_adapter_learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_train_steps", type=int, default=20000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_scope", choices=("attn_qkv", "attn_qkvo", "wide"), default="wide")
    parser.add_argument("--lora_target_modules", default=DEFAULT_LORA_TARGET_MODULES)
    parser.add_argument("--image_condition_dropout", type=float, default=0.0)
    parser.add_argument("--image_encoder_model", required=True)
    parser.add_argument("--image_encoder_subfolder", default=None)
    parser.add_argument("--image_encoder_revision", default=None)
    parser.add_argument("--image_encoder_local_files_only", action="store_true")
    parser.add_argument("--image_gate_init", type=float, default=1e-3)
    parser.add_argument("--image_cross_attention_scale", type=float, default=1.0)
    parser.add_argument("--init_from_route2_checkpoint", default=None)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="no")
    parser.add_argument("--debug_check_finite", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    add_pretrained_args(parser)
    return parser.parse_args()


class Route3Dataset(Dataset):
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


def count_params(params):
    return sum(p.numel() for p in params)


def grad_norm(params):
    total = 0.0
    for parameter in params:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().norm().cpu()) ** 2
    return total**0.5


def unique_optimizer_groups(groups):
    seen = {}
    for group in groups:
        for parameter in group["params"]:
            if id(parameter) in seen:
                raise RuntimeError(f"Optimizer parameter overlap: {group['name']} and {seen[id(parameter)]}")
            seen[id(parameter)] = group["name"]


def read_resume_step(checkpoint):
    path = Path(checkpoint) / "trainer_state.json"
    if path.exists():
        return int(json.loads(path.read_text(encoding="utf-8")).get("global_step", 0))
    name = Path(checkpoint).name
    if name.startswith("checkpoint-"):
        return int(name.split("-", 1)[1])
    return 0


def main():
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("Dynamic Route3 training currently requires --batch_size 1.")
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

    accelerator.print(f"[ROUTE3] implementation={IMPLEMENTATION}", flush=True)
    accelerator.print("[ROUTE3] condition = Route2 latent concat + Wan-style image cross-attention", flush=True)
    accelerator.print("[ROUTE3] edit_image[0] is SRC; edit_image[1] is ignored", flush=True)

    if args.resume_from_checkpoint:
        resume_config, _ = load_route3_config(args.resume_from_checkpoint)
        if resume_config.get("train_mode") != args.train_mode:
            raise ValueError(f"Resume train mode mismatch: ckpt={resume_config.get('train_mode')} args={args.train_mode}")
        accelerator.print(f"[ROUTE3] resume checkpoint={args.resume_from_checkpoint}", flush=True)

    if args.init_from_route2_checkpoint:
        validate_route2_init(args.init_from_route2_checkpoint, args.train_mode)
        accelerator.print(f"[ROUTE3] initializing from Route2 checkpoint={args.init_from_route2_checkpoint}", flush=True)

    pipe_kwargs = pretrained_kwargs(args)
    if args.init_from_route2_checkpoint and args.train_mode == "full_transformer":
        full_path = route2_full_transformer_path(args.init_from_route2_checkpoint)
        if full_path is None:
            raise FileNotFoundError("Route2 full transformer checkpoint missing transformer/ subdir.")
        transformer = SanaTransformer2DModel.from_pretrained(full_path, torch_dtype=dtype)
        pipe = SanaPipeline.from_pretrained(args.model, transformer=transformer, torch_dtype=dtype, **pipe_kwargs)
    else:
        pipe = SanaPipeline.from_pretrained(args.model, torch_dtype=dtype, **pipe_kwargs)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.model, subfolder="scheduler", **pipe_kwargs)
    pipe.vae.requires_grad_(False).eval()
    pipe.text_encoder.requires_grad_(False).eval()
    pipe.transformer.requires_grad_(False).eval()

    original_latent_channels = expand_sana_patch_embedding_for_channel_concat(pipe.transformer)
    if args.init_from_route2_checkpoint:
        load_route2_patch_embedding(args.init_from_route2_checkpoint, pipe.transformer)

    matched_lora = []
    excluded_image_modules = []
    if args.train_mode == "patch_lora":
        if args.init_from_route2_checkpoint:
            lora_path = route2_lora_path(args.init_from_route2_checkpoint)
            if lora_path is None:
                raise FileNotFoundError("Route2 patch_lora checkpoint missing transformer_lora/ subdir.")
            pipe.load_lora_weights(lora_path)
            matched_lora = ["loaded_from_route2"]
        else:
            matched_lora = add_lora_to_transformer(
                pipe.transformer,
                args.lora_rank,
                args.lora_alpha,
                args.lora_target_modules,
                args.lora_dropout,
                args.lora_scope,
            )
        excluded_image_modules = ["to_k_img", "to_v_img"]
    elif args.train_mode == "full_transformer":
        pipe.transformer.requires_grad_(True)

    image_encoder, image_processor = load_image_encoder_and_processor(args, dtype=dtype, device=accelerator.device)
    image_hidden_size = int(image_encoder.config.hidden_size)
    route3_model = build_route3_model(pipe.transformer, image_hidden_size, image_gate_init=args.image_gate_init)

    pipe.text_encoder.to(accelerator.device, dtype=dtype)
    pipe.vae.to(accelerator.device, dtype=torch.float32)
    route3_model.to(accelerator.device, dtype=dtype)

    if args.train_mode == "patch_lora":
        route3_model.transformer.requires_grad_(False)
        for name, parameter in route3_model.transformer.named_parameters():
            if "lora_" in name:
                parameter.requires_grad_(True)
    elif args.train_mode == "full_transformer":
        route3_model.transformer.requires_grad_(True)
    route3_model.image_projector.requires_grad_(True)
    route3_model.image_cross_attention_adapter.requires_grad_(True)

    patch_embed, _ = get_sana_patch_embedding(route3_model.transformer)
    for parameter in patch_embed.parameters():
        parameter.requires_grad_(True)

    if args.gradient_checkpointing and hasattr(route3_model.transformer, "enable_gradient_checkpointing"):
        route3_model.transformer.enable_gradient_checkpointing()

    patch_params = [p for p in patch_embed.parameters() if p.requires_grad]
    lora_named_params = [(n, p) for n, p in route3_model.transformer.named_parameters() if p.requires_grad and "lora_" in n]
    lora_params = [p for _, p in lora_named_params]
    image_projector_params = [p for p in route3_model.image_projector.parameters() if p.requires_grad]
    image_adapter_params = [p for p in route3_model.image_cross_attention_adapter.parameters() if p.requires_grad]
    image_gate_params = [layer.image_gate for layer in route3_model.image_cross_attention_adapter.layers]

    patch_ids = {id(p) for p in patch_params}
    image_ids = {id(p) for p in image_projector_params + image_adapter_params}
    if args.train_mode == "full_transformer":
        full_params = [p for p in route3_model.transformer.parameters() if p.requires_grad and id(p) not in patch_ids]
        groups = [
            {"params": patch_params, "lr": args.patch_learning_rate, "name": "patch_embedding"},
            {"params": full_params, "lr": args.learning_rate, "name": "full_transformer"},
            {"params": image_projector_params + image_adapter_params, "lr": args.image_adapter_learning_rate, "name": "image_adapter"},
        ]
    else:
        if not lora_params:
            raise RuntimeError("patch_lora mode requires trainable transformer LoRA parameters.")
        groups = [
            {"params": patch_params, "lr": args.patch_learning_rate, "name": "patch_embedding"},
            {"params": lora_params, "lr": args.learning_rate, "name": "transformer_lora"},
            {"params": image_projector_params + image_adapter_params, "lr": args.image_adapter_learning_rate, "name": "image_adapter"},
        ]
    unique_optimizer_groups(groups)
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-2)

    kv_params = []
    for layer in route3_model.image_cross_attention_adapter.layers:
        kv_params.extend(list(layer.to_k_img.parameters()))
        kv_params.extend(list(layer.to_v_img.parameters()))
        kv_params.extend([p for p in layer.norm_k_img.parameters() if p.requires_grad])
    trainable = [p for group in groups for p in group["params"]]
    accelerator.print(f"[ROUTE3] patch embedding trainable params: {count_params(patch_params):,}", flush=True)
    accelerator.print(f"[ROUTE3] transformer LoRA trainable params: {count_params(lora_params):,}", flush=True)
    accelerator.print(f"[ROUTE3] full transformer trainable params: {count_params(groups[1]['params']):,}", flush=True)
    accelerator.print(f"[ROUTE3] image projector trainable params: {count_params(image_projector_params):,}", flush=True)
    accelerator.print(f"[ROUTE3] image K/V trainable params: {count_params(kv_params):,}", flush=True)
    accelerator.print(f"[ROUTE3] image gate trainable params: {count_params(image_gate_params):,}", flush=True)
    accelerator.print(f"[ROUTE3] total trainable params: {count_params(trainable):,}", flush=True)
    accelerator.print(f"[ROUTE3] LoRA matched modules: {len(matched_lora)}", flush=True)
    accelerator.print(f"[ROUTE3] LoRA excluded image modules: {excluded_image_modules}", flush=True)

    dataset = Route3Dataset(args)

    def collate(samples):
        sample = samples[0]
        prepared, _ = preprocess_triplet(
            sample["gt"], sample["src"], sample["ref"], args.max_pixels, args.size_divisor, args.downscale_if_exceeds_max_pixels
        )
        canvas_w, canvas_h = prepared["src"].size
        return {
            "gt": pipe.image_processor.preprocess(prepared["gt"], height=canvas_h, width=canvas_w),
            "src": pipe.image_processor.preprocess(prepared["src"], height=canvas_h, width=canvas_w),
            "src_pil": [prepared["src"]],
            "prompts": [sample["prompt"]],
        }

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate)
    route3_model, optimizer, dataloader = accelerator.prepare(route3_model, optimizer, dataloader)

    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.resume_from_checkpoint:
        accelerator.load_state(str(Path(args.resume_from_checkpoint) / "trainer_state"))
        global_step = read_resume_step(args.resume_from_checkpoint)
        accelerator.print("[ROUTE3] trainer state loaded", flush=True)
        accelerator.print(f"[ROUTE3] resumed global_step={global_step}", flush=True)
        accelerator.print(f"[ROUTE3] target global_step={args.max_train_steps}", flush=True)
    else:
        global_step = 0

    scheduler_timesteps = scheduler.timesteps.to(accelerator.device)
    scheduler_sigmas = scheduler.sigmas.to(accelerator.device)
    while global_step < args.max_train_steps:
        for batch in dataloader:
            if global_step >= args.max_train_steps:
                break
            with accelerator.accumulate(route3_model):
                with torch.no_grad():
                    z_gt = encode_vae_latents(pipe.vae, batch["gt"].to(accelerator.device, torch.float32))
                    z_src = encode_vae_latents(pipe.vae, batch["src"].to(accelerator.device, torch.float32))
                    image_tokens = encode_image_tokens(
                        image_encoder,
                        image_processor,
                        batch["src_pil"],
                        device=accelerator.device,
                        dtype=dtype,
                    )
                    if args.image_condition_dropout > 0:
                        drop_mask = torch.rand(z_src.shape[0], device=accelerator.device) < args.image_condition_dropout
                        keep = (~drop_mask).to(z_src.dtype).view(-1, 1, 1, 1)
                        z_src = z_src * keep
                        image_tokens = image_tokens * keep.view(-1, 1, 1).to(image_tokens.dtype)
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
                pred = route3_model(
                    hidden_states=model_input.to(dtype),
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_mask,
                    encoder_hidden_states_image=image_tokens,
                    image_cross_attention_scale=args.image_cross_attention_scale,
                    timestep=timesteps,
                    return_dict=False,
                )[0].float()
                loss = F.mse_loss(pred, flow_target.float())
                if args.debug_check_finite and not torch.isfinite(loss):
                    accelerator.print("[ROUTE3] Non-finite loss", tensor_stats(pred), tensor_stats(flow_target), flush=True)
                    raise RuntimeError("Non-finite Route3 loss.")
                accelerator.backward(loss)
                last_patch_grad = grad_norm(patch_params)
                last_image_grad = grad_norm(image_projector_params + image_adapter_params)
                last_lora_grad = grad_norm(lora_params)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if accelerator.is_main_process and global_step % args.log_steps == 0:
                accelerator.print(
                    f"[ROUTE3] step={global_step} loss={loss.detach().item():.6f} "
                    f"patch_grad={last_patch_grad:.3e} lora_grad={last_lora_grad:.3e} image_grad={last_image_grad:.3e} "
                    f"patch_stats={patch_embedding_weight_stats(route3_model.transformer, original_latent_channels)}",
                    flush=True,
                )
            if accelerator.is_main_process and global_step % args.save_steps == 0:
                ckpt = output_dir / f"checkpoint-{global_step}"
                unwrapped = accelerator.unwrap_model(route3_model)
                save_route3_checkpoint(ckpt, unwrapped, pipe, args, global_step, original_latent_channels, image_hidden_size)
                accelerator.save_state(str(ckpt / "trainer_state"))
                (ckpt / "trainer_state.json").write_text(json.dumps({"global_step": global_step}, indent=2), encoding="utf-8")
                accelerator.print(f"[ROUTE3] saved checkpoint={ckpt}", flush=True)

    if accelerator.is_main_process:
        ckpt = output_dir / f"checkpoint-{global_step}"
        unwrapped = accelerator.unwrap_model(route3_model)
        save_route3_checkpoint(ckpt, unwrapped, pipe, args, global_step, original_latent_channels, image_hidden_size)
        accelerator.save_state(str(ckpt / "trainer_state"))
        (ckpt / "trainer_state.json").write_text(json.dumps({"global_step": global_step}, indent=2), encoding="utf-8")
        accelerator.print(f"[ROUTE3] done checkpoint={ckpt}", flush=True)


if __name__ == "__main__":
    main()
