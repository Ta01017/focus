#!/usr/bin/env python

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

from diffusers import SanaPipeline

SCRIPT_DIR = Path(__file__).resolve().parent
DOF_DIR = SCRIPT_DIR.parent / "dof_fusion"
if str(DOF_DIR) not in sys.path:
    sys.path.insert(0, str(DOF_DIR))

from sana_dof import decode_vae_latents, encode_vae_latents  # noqa: E402

from artifact_repair_utils import (  # noqa: E402
    DEFAULT_REPAIR_PROMPT,
    add_pretrained_args,
    load_rgb,
    preprocess_pair,
    pretrained_kwargs,
    restore_output_size,
)
from sana_native_edit_utils import NativeEditSanaTransformer, load_native_edit_assets  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Infer SANA native edit LoRA for artifact repair.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--src_image", required=True)
    parser.add_argument("--ref_image", required=True)
    parser.add_argument("--prompt", default=DEFAULT_REPAIR_PROMPT)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init_mode", choices=("noise", "src"), default="src")
    parser.add_argument("--strength", type=float, default=0.15)
    parser.add_argument("--max_pixels", type=int, default=1048576)
    parser.add_argument("--size_divisor", type=int, default=32)
    parser.add_argument("--downscale_if_exceeds_max_pixels", action="store_true")
    parser.add_argument("--restore_to_original_size", action=argparse.BooleanOptionalAction, default=True)
    add_pretrained_args(parser)
    return parser.parse_args()


@torch.no_grad()
def encode_image_latents(pipe, image, height, width, device):
    pixels = pipe.image_processor.preprocess(image, height=height, width=width).to(device, torch.float32)
    return encode_vae_latents(pipe.vae, pixels)


@torch.no_grad()
def decode_latents_to_pil(pipe, latents):
    image = decode_vae_latents(pipe.vae, latents)
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


def load_pipeline(args, dtype):
    checkpoint = Path(args.checkpoint)
    config_model = None
    edit_role_embedding = True
    config_path = checkpoint / "native_edit_config.json"
    if not config_path.exists() and (checkpoint.parent / "native_edit_config.json").exists():
        config_path = checkpoint.parent / "native_edit_config.json"
    if config_path.exists():
        import json

        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
        config_model = raw_config.get("model")
        edit_role_embedding = bool(raw_config.get("edit_role_embedding", True))
    model_id = args.model or config_model or "Efficient-Large-Model/Sana_600M_1024px_diffusers"
    print(f"[NATIVE_EDIT] checkpoint = {checkpoint}", flush=True)
    print(f"[NATIVE_EDIT] model = {model_id}", flush=True)
    pipe = SanaPipeline.from_pretrained(model_id, torch_dtype=dtype, **pretrained_kwargs(args)).to("cuda")
    pipe.vae.to(dtype=torch.float32)
    native_transformer = NativeEditSanaTransformer(pipe.transformer, use_role_embedding=edit_role_embedding).to("cuda", dtype=dtype)
    config, lora_path = load_native_edit_assets(checkpoint, native_transformer)
    pipe.load_lora_weights(lora_path)
    pipe.transformer = native_transformer
    return pipe, native_transformer, config


@torch.no_grad()
def generate(pipe, native_transformer, prompt, negative_prompt, src_image, ref_image, height, width, steps, guidance_scale, init_mode, strength, generator):
    device = torch.device("cuda")
    do_cfg = guidance_scale > 1.0
    prompt_embeds, prompt_mask, neg_embeds, neg_mask = pipe.encode_prompt(
        prompt, do_cfg, negative_prompt=negative_prompt, num_images_per_prompt=1, device=device, clean_caption=False, max_sequence_length=300
    )
    if do_cfg:
        prompt_embeds = torch.cat([neg_embeds, prompt_embeds], dim=0)
        prompt_mask = torch.cat([neg_mask, prompt_mask], dim=0)
    src_latents = encode_image_latents(pipe, src_image, height, width, device)
    ref_latents = encode_image_latents(pipe, ref_image, height, width, device)
    pipe.scheduler.set_timesteps(steps, device=device)
    timesteps = pipe.scheduler.timesteps
    if init_mode == "src":
        init_timestep = max(min(int(steps * strength), steps), 1)
        t_start = max(steps - init_timestep, 0)
        sliced = timesteps[t_start:]
        if hasattr(pipe.scheduler, "set_begin_index"):
            pipe.scheduler.set_begin_index(t_start * pipe.scheduler.order)
        sigma = pipe.scheduler.sigmas[t_start].to(device=device, dtype=torch.float32).reshape(1, 1, 1, 1)
        noise = torch.randn(src_latents.shape, generator=generator, device=device, dtype=src_latents.dtype)
        latents = (1 - sigma.to(src_latents)) * src_latents + sigma.to(src_latents) * noise
    else:
        t_start = 0
        sliced = timesteps
        latents = torch.randn(src_latents.shape, generator=generator, device=device, dtype=src_latents.dtype)
    for t in sliced:
        latent_input = torch.cat([latents] * 2) if do_cfg else latents
        timestep = t.expand(latent_input.shape[0]) * pipe.transformer.config.timestep_scale
        noise_pred = native_transformer(
            latent_input.to(pipe.transformer.dtype),
            encoder_hidden_states=prompt_embeds.to(pipe.transformer.dtype),
            encoder_attention_mask=prompt_mask,
            timestep=timestep,
            edit_hidden_states=[src_latents.to(pipe.transformer.dtype), ref_latents.to(pipe.transformer.dtype)],
            edit_role_ids=[1, 2],
            enable_native_edit_tokens=True,
            return_dict=False,
        )[0].float()
        if do_cfg:
            uncond, text = noise_pred.chunk(2)
            noise_pred = uncond + guidance_scale * (text - uncond)
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    print(f"[NATIVE_EDIT] token lengths: {native_transformer.last_token_stats}", flush=True)
    return decode_latents_to_pil(pipe, latents)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipe, native_transformer, _ = load_pipeline(args, dtype)
    src = load_rgb(args.src_image)
    ref = load_rgb(args.ref_image)
    prepared, size_info = preprocess_pair(src, ref, args.max_pixels, args.size_divisor, args.downscale_if_exceeds_max_pixels)
    canvas_w, canvas_h = size_info["canvas_size"]
    print(f"[NATIVE_EDIT] init_mode = {args.init_mode}", flush=True)
    print(f"[NATIVE_EDIT] strength = {args.strength}", flush=True)
    print(f"[NATIVE_EDIT] src path = {args.src_image}", flush=True)
    print(f"[NATIVE_EDIT] ref path = {args.ref_image}", flush=True)
    print(f"[NATIVE_EDIT] prompt = {args.prompt}", flush=True)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    image = generate(
        pipe,
        native_transformer,
        args.prompt,
        args.negative_prompt,
        prepared["src"],
        prepared["ref"],
        canvas_h,
        canvas_w,
        args.steps,
        args.guidance_scale,
        args.init_mode,
        args.strength,
        generator,
    )
    image = restore_output_size(image, size_info, args.restore_to_original_size)
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(f"[NATIVE_EDIT] output path = {output}", flush=True)


if __name__ == "__main__":
    main()
