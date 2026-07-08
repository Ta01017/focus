#!/usr/bin/env python

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file
from torch.nn import functional as F

from diffusers import SanaPipeline

SCRIPT_DIR = Path(__file__).resolve().parent
DOF_DIR = SCRIPT_DIR.parent / "dof_fusion"
if str(DOF_DIR) not in sys.path:
    sys.path.insert(0, str(DOF_DIR))

from sana_dof import decode_vae_latents, encode_vae_latents  # noqa: E402

from artifact_repair_utils import (  # noqa: E402
    DEFAULT_REPAIR_PROMPT,
    add_pretrained_args,
    build_control_condition,
    load_rgb,
    preprocess_pair,
    pretrained_kwargs,
    restore_output_size,
    tensor_stats,
    write_json,
)
from train_sana_artifact_repair_froma import SrcRefAdapter  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Infer SANA artifact repair from-src adapter without ControlNet.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--src_image", required=True)
    parser.add_argument("--ref_image", required=True)
    parser.add_argument("--prompt", default=DEFAULT_REPAIR_PROMPT)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--strength", type=float, default=0.15)
    parser.add_argument("--img2img_schedule_mode", choices=("pipeline_full", "sliced"), default="sliced")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--adapter_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_pixels", type=int, default=1048576)
    parser.add_argument("--size_divisor", type=int, default=32)
    parser.add_argument("--downscale_if_exceeds_max_pixels", action="store_true")
    restore = parser.add_mutually_exclusive_group()
    restore.add_argument("--restore_to_original_size", dest="restore_to_original_size", action="store_true")
    restore.add_argument("--no_restore_to_original_size", dest="restore_to_original_size", action="store_false")
    parser.set_defaults(restore_to_original_size=True)
    parser.add_argument("--debug_latent_dir", default=None)
    parser.add_argument("--zero_ref_condition", action="store_true")
    parser.add_argument("--zero_src_condition", action="store_true")
    add_pretrained_args(parser)
    return parser.parse_args()


@torch.no_grad()
def encode_image_latents(pipe, image, height, width, device):
    pixel_values = pipe.image_processor.preprocess(image, height=height, width=width).to(device, torch.float32)
    return encode_vae_latents(pipe.vae, pixel_values)


@torch.no_grad()
def decode_latents_to_pil(pipe, latents):
    image = decode_vae_latents(pipe.vae, latents)
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


def load_config(checkpoint):
    checkpoint = Path(checkpoint)
    for name in ("artifact_repair_config.json", "adapter_config.json"):
        path = checkpoint / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"Missing artifact_repair_config.json or adapter_config.json in {checkpoint}")


def load_pipeline(checkpoint, model, dtype, options):
    checkpoint = Path(checkpoint)
    config = load_config(checkpoint)
    model_id = model or config.get("model") or config.get("base_model")
    pipe = SanaPipeline.from_pretrained(model_id, torch_dtype=dtype, **options).to("cuda")
    pipe.vae.to(dtype=torch.float32)
    lora_path = checkpoint / "transformer_lora"
    if config.get("train_transformer_lora", False) or lora_path.exists():
        if not lora_path.exists():
            raise FileNotFoundError(f"Config requires transformer LoRA, but {lora_path} does not exist.")
        pipe.load_lora_weights(lora_path)
    adapter_path = checkpoint / "adapter.safetensors"
    if not adapter_path.exists():
        raise FileNotFoundError(f"Missing adapter weights: {adapter_path}")
    adapter = SrcRefAdapter(int(config.get("adapter_condition_channels", config.get("condition_channels", 6))), pipe.transformer.config.in_channels)
    adapter.load_state_dict(load_file(adapter_path), strict=True)
    adapter.to("cuda", dtype=torch.float32).eval()
    return pipe, adapter, config


def make_condition(pipe, prepared, size_info, zero_src=False, zero_ref=False):
    canvas_w, canvas_h = size_info["canvas_size"]
    src = Image.new("RGB", prepared["src"].size, (0, 0, 0)) if zero_src else prepared["src"]
    ref = Image.new("RGB", prepared["ref"].size, (0, 0, 0)) if zero_ref else prepared["ref"]
    src_tensor = pipe.image_processor.preprocess(src, height=canvas_h, width=canvas_w)
    ref_tensor = pipe.image_processor.preprocess(ref, height=canvas_h, width=canvas_w)
    return build_control_condition(src_tensor, ref_tensor)


@torch.no_grad()
def generate(
    pipe,
    adapter,
    prompt,
    negative_prompt,
    src_image,
    src_ref_condition,
    height,
    width,
    steps,
    guidance_scale,
    strength,
    mode,
    generator,
    adapter_scale,
):
    if not 0 <= strength <= 1:
        raise ValueError("--strength must be in [0, 1].")
    device = torch.device("cuda")
    do_cfg = guidance_scale > 1.0
    prompt_embeds, prompt_mask, neg_embeds, neg_mask = pipe.encode_prompt(
        prompt,
        do_cfg,
        negative_prompt=negative_prompt,
        num_images_per_prompt=1,
        device=device,
        clean_caption=False,
        max_sequence_length=300,
    )
    if do_cfg:
        prompt_embeds = torch.cat([neg_embeds, prompt_embeds], dim=0)
        prompt_mask = torch.cat([neg_mask, prompt_mask], dim=0)

    pipe.scheduler.set_timesteps(steps, device=device)
    timesteps = pipe.scheduler.timesteps
    src_latents = encode_image_latents(pipe, src_image, height, width, device)
    noise = torch.randn(src_latents.shape, generator=generator, device=device, dtype=src_latents.dtype)
    scheduler_order = getattr(pipe.scheduler, "order", 1)
    begin_index = None
    if mode == "sliced":
        init_timestep = max(min(int(steps * strength), steps), 1)
        t_start = max(steps - init_timestep, 0)
        sliced_timesteps = timesteps[t_start:]
        begin_index = t_start * scheduler_order
        if hasattr(pipe.scheduler, "set_begin_index"):
            pipe.scheduler.set_begin_index(begin_index)
        sigma = pipe.scheduler.sigmas[t_start].to(device=device, dtype=torch.float32).reshape(1, 1, 1, 1)
        init_latents = (1 - sigma.to(src_latents)) * src_latents + sigma.to(src_latents) * noise
    else:
        t_start = 0
        sliced_timesteps = timesteps
        sigma = torch.tensor(strength, device=device, dtype=torch.float32).reshape(1, 1, 1, 1)
        init_latents = (1 - sigma.to(src_latents)) * src_latents + sigma.to(src_latents) * noise
    latents = init_latents

    src_ref_condition = src_ref_condition.to(device).float()
    raw_shape = list(src_ref_condition.shape)
    if src_ref_condition.shape[-2:] != latents.shape[-2:]:
        src_ref_condition = F.interpolate(src_ref_condition, size=latents.shape[-2:], mode="bilinear", align_corners=False)
    downsampled_shape = list(src_ref_condition.shape)
    adapter_residual = adapter(src_ref_condition).to(latents)
    adapter_residual_shape = list(adapter_residual.shape)
    residual_stats = {
        "mean": float(adapter_residual.detach().float().mean().cpu()),
        "std": float(adapter_residual.detach().float().std().cpu()),
        "norm": float(adapter_residual.detach().float().norm().cpu()),
    }

    for timestep in sliced_timesteps:
        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
        timestep_input = timestep.expand(latent_model_input.shape[0]) * pipe.transformer.config.timestep_scale
        residual = adapter_residual
        if residual.shape[0] != latent_model_input.shape[0]:
            residual = residual.repeat(latent_model_input.shape[0], 1, 1, 1)
        conditioned_input = latent_model_input + adapter_scale * residual.to(latent_model_input)
        noise_pred = pipe.transformer(
            conditioned_input.to(pipe.transformer.dtype),
            encoder_hidden_states=prompt_embeds.to(pipe.transformer.dtype),
            encoder_attention_mask=prompt_mask,
            timestep=timestep_input,
            return_dict=False,
        )[0].float()
        if do_cfg:
            uncond, text = noise_pred.chunk(2)
            noise_pred = uncond + guidance_scale * (text - uncond)
        latents = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

    image = decode_latents_to_pil(pipe, latents)
    stats = {
        "img2img_schedule_mode": mode,
        "strength": strength,
        "adapter_scale": adapter_scale,
        "t_start": t_start,
        "selected_sigma": float(sigma.flatten()[0].detach().cpu()),
        "actual_num_denoise_steps": len(sliced_timesteps),
        "scheduler_has_set_begin_index": hasattr(pipe.scheduler, "set_begin_index"),
        "scheduler_order": scheduler_order,
        "begin_index_value": begin_index,
        "init_latents": tensor_stats(init_latents),
        "src_latents": tensor_stats(src_latents),
        "final_latents": tensor_stats(latents),
        "src_ref_condition_raw_shape": raw_shape,
        "src_ref_condition_downsampled_shape": downsampled_shape,
        "latents_shape": list(latents.shape),
        "adapter_residual_shape": adapter_residual_shape,
        "adapter_residual": residual_stats,
    }
    return image, latents, src_latents, init_latents, stats


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipe, adapter, _ = load_pipeline(Path(args.checkpoint), args.model, dtype, pretrained_kwargs(args))
    src = load_rgb(args.src_image)
    ref = load_rgb(args.ref_image)
    prepared, size_info = preprocess_pair(src, ref, args.max_pixels, args.size_divisor, args.downscale_if_exceeds_max_pixels)
    canvas_w, canvas_h = size_info["canvas_size"]
    condition = make_condition(pipe, prepared, size_info, args.zero_src_condition, args.zero_ref_condition)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    image, _, src_latents, init_latents, stats = generate(
        pipe,
        adapter,
        args.prompt,
        args.negative_prompt,
        prepared["src"],
        condition,
        canvas_h,
        canvas_w,
        args.steps,
        args.guidance_scale,
        args.strength,
        args.img2img_schedule_mode,
        generator,
        args.adapter_scale,
    )
    if args.debug_latent_dir:
        debug = Path(args.debug_latent_dir)
        debug.mkdir(parents=True, exist_ok=True)
        prepared["src"].save(debug / "raw_src.png")
        prepared["ref"].save(debug / "raw_ref.png")
        decode_latents_to_pil(pipe, src_latents).save(debug / "vae_roundtrip_src.png")
        ref_latents = encode_image_latents(pipe, prepared["ref"], canvas_h, canvas_w, torch.device("cuda"))
        decode_latents_to_pil(pipe, ref_latents).save(debug / "vae_roundtrip_ref.png")
        decode_latents_to_pil(pipe, init_latents).save(debug / "decoded_init_latents.png")
        image.save(debug / "final_output.png")
        write_json(debug / "latent_stats.json", stats)
        write_json(debug / "condition_stats.json", tensor_stats(condition))

    image = restore_output_size(image, size_info, args.restore_to_original_size)
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


if __name__ == "__main__":
    main()
