#!/usr/bin/env python

import argparse
import json
import sys
from pathlib import Path

import torch

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
    write_json,
)
from sana_artifact_repair_latent_concat import (  # noqa: E402
    SanaArtifactRepairLatentConcatModel,
    SrcLatentConditionInjector,
    load_latent_concat_assets,
    route1_tensor_stats,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Infer Artifact Repair Route 1: SANA src latent concat / image-input injection."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--image_src", "--src_image", dest="image_src", required=True)
    parser.add_argument("--image_ref", "--ref_image", dest="image_ref", required=True)
    parser.add_argument("--output", "--output_path", dest="output", required=True)
    parser.add_argument("--prompt", default=DEFAULT_REPAIR_PROMPT)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--strength", type=float, default=0.15)
    parser.add_argument("--img2img_schedule_mode", choices=("pipeline_full", "sliced"), default="sliced")
    parser.add_argument("--use_src_latent_init", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16", "auto"), default="auto")
    parser.add_argument("--max_pixels", type=int, default=1048576)
    parser.add_argument("--size_divisor", type=int, default=32)
    parser.add_argument("--downscale_if_exceeds_max_pixels", action="store_true")
    parser.add_argument("--restore_to_original_size", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug_latent_dir", default=None)
    add_pretrained_args(parser)
    return parser.parse_args()


def select_dtype(value):
    if value == "fp32":
        return torch.float32
    if value == "bf16":
        return torch.bfloat16
    if value == "fp16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


@torch.no_grad()
def encode_image_latents(pipe, image, height, width, device):
    pixels = pipe.image_processor.preprocess(image, height=height, width=width).to(device, torch.float32)
    return encode_vae_latents(pipe.vae, pixels)


@torch.no_grad()
def decode_latents_to_pil(pipe, latents):
    image = decode_vae_latents(pipe.vae, latents)
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


def read_checkpoint_config(checkpoint):
    checkpoint = Path(checkpoint)
    for path in (checkpoint / "route1_config.json", checkpoint / "artifact_repair_config.json"):
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    for path in (checkpoint.parent / "route1_config.json", checkpoint.parent / "artifact_repair_config.json"):
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def load_pipeline(args, dtype):
    checkpoint = Path(args.checkpoint)
    config = read_checkpoint_config(checkpoint)
    model_id = args.model or config.get("model") or config.get("base_model") or "Efficient-Large-Model/Sana_600M_1024px_diffusers"
    print(f"[ROUTE1] checkpoint={checkpoint}", flush=True)
    print(f"[ROUTE1] model={model_id}", flush=True)
    pipe = SanaPipeline.from_pretrained(model_id, torch_dtype=dtype, **pretrained_kwargs(args)).to("cuda")
    pipe.vae.to(dtype=torch.float32)
    inner_dim = pipe.transformer.config.num_attention_heads * pipe.transformer.config.attention_head_dim
    injector = SrcLatentConditionInjector(
        pipe.transformer.config.in_channels,
        inner_dim,
        patch_size=pipe.transformer.config.patch_size,
        hidden_channels=int(config.get("src_injector_hidden_channels", config.get("injector_hidden_channels", 128))),
    )
    model = SanaArtifactRepairLatentConcatModel(pipe.transformer, injector).to("cuda", dtype=dtype)
    model.src_condition_injector.to("cuda", dtype=torch.float32)
    config, lora_path = load_latent_concat_assets(checkpoint, model)
    if config.get("train_transformer_lora", False) or lora_path is not None:
        if lora_path is None:
            raise FileNotFoundError("Route 1 checkpoint config expects transformer LoRA, but transformer_lora is missing.")
        pipe.load_lora_weights(lora_path)
    pipe.transformer = model
    model.eval()
    return pipe, model, config


@torch.no_grad()
def generate(
    pipe,
    model,
    prompt,
    negative_prompt,
    src_image,
    height,
    width,
    steps,
    guidance_scale,
    strength,
    img2img_schedule_mode,
    use_src_latent_init,
    generator,
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

    src_latents = encode_image_latents(pipe, src_image, height, width, device)
    noise = torch.randn(src_latents.shape, generator=generator, device=device, dtype=src_latents.dtype)
    pipe.scheduler.set_timesteps(steps, device=device)
    timesteps = pipe.scheduler.timesteps
    scheduler_order = getattr(pipe.scheduler, "order", 1)
    begin_index = None
    if use_src_latent_init:
        if img2img_schedule_mode == "sliced":
            init_timestep = max(min(int(steps * strength), steps), 1)
            t_start = max(steps - init_timestep, 0)
            sliced_timesteps = timesteps[t_start:]
            begin_index = t_start * scheduler_order
            if hasattr(pipe.scheduler, "set_begin_index"):
                pipe.scheduler.set_begin_index(begin_index)
            sigma = pipe.scheduler.sigmas[t_start].to(device=device, dtype=torch.float32).reshape(1, 1, 1, 1)
            latents = (1 - sigma.to(src_latents)) * src_latents + sigma.to(src_latents) * noise
        else:
            t_start = 0
            sliced_timesteps = timesteps
            sigma = torch.tensor(strength, device=device, dtype=torch.float32).reshape(1, 1, 1, 1)
            latents = (1 - sigma.to(src_latents)) * src_latents + sigma.to(src_latents) * noise
    else:
        t_start = 0
        sliced_timesteps = timesteps
        sigma = torch.tensor(1.0, device=device, dtype=torch.float32).reshape(1, 1, 1, 1)
        latents = noise

    init_or_noisy_latents = latents.clone()
    for timestep in sliced_timesteps:
        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
        timestep_input = timestep.expand(latent_model_input.shape[0]) * pipe.transformer.config.timestep_scale
        noise_pred = model(
            latent_model_input.to(pipe.transformer.dtype),
            encoder_hidden_states=prompt_embeds.to(pipe.transformer.dtype),
            encoder_attention_mask=prompt_mask,
            timestep=timestep_input,
            src_latents=src_latents.to(pipe.transformer.dtype),
            return_dict=False,
        )[0].float()
        if do_cfg:
            uncond, text = noise_pred.chunk(2)
            noise_pred = uncond + guidance_scale * (text - uncond)
        latents = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

    image = decode_latents_to_pil(pipe, latents)
    stats = {
        "strength": strength,
        "img2img_schedule_mode": img2img_schedule_mode,
        "use_src_latent_init": use_src_latent_init,
        "t_start": t_start,
        "selected_sigma": float(sigma.flatten()[0].detach().cpu()),
        "actual_num_denoise_steps": len(sliced_timesteps),
        "scheduler_has_set_begin_index": hasattr(pipe.scheduler, "set_begin_index"),
        "scheduler_order": scheduler_order,
        "begin_index_value": begin_index,
        "src_latents": route1_tensor_stats(src_latents),
        "target_or_noisy_latents": route1_tensor_stats(init_or_noisy_latents),
        "final_latents": route1_tensor_stats(latents),
        "injected_condition": model.last_injection_stats,
    }
    return image, stats, src_latents, init_or_noisy_latents


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    dtype = select_dtype(args.dtype)
    pipe, model, _ = load_pipeline(args, dtype)
    src = load_rgb(args.image_src)
    ref = load_rgb(args.image_ref)
    prepared, size_info = preprocess_pair(
        src,
        ref,
        args.max_pixels,
        args.size_divisor,
        args.downscale_if_exceeds_max_pixels,
    )
    canvas_w, canvas_h = size_info["canvas_size"]
    print("[ROUTE1] image_ref is read for metadata/size compatibility but ignored in model computation.", flush=True)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    image, stats, src_latents, init_latents = generate(
        pipe,
        model,
        args.prompt,
        args.negative_prompt,
        prepared["src"],
        canvas_h,
        canvas_w,
        args.steps,
        args.guidance_scale,
        args.strength,
        args.img2img_schedule_mode,
        args.use_src_latent_init,
        generator,
    )
    if args.debug_latent_dir:
        debug = Path(args.debug_latent_dir)
        debug.mkdir(parents=True, exist_ok=True)
        src.save(debug / "raw_src.png")
        prepared["src"].save(debug / "resized_src.png")
        ref.save(debug / "raw_ref_ignored.png")
        image.save(debug / "final_output_before_restore.png")
        decode_latents_to_pil(pipe, src_latents).save(debug / "vae_roundtrip_src.png")
        decode_latents_to_pil(pipe, init_latents).save(debug / "decoded_initial_latents.png")
        write_json(debug / "latent_stats.json", stats)
    image = restore_output_size(image, size_info, args.restore_to_original_size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    write_json(output.with_suffix(output.suffix + ".stats.json"), stats)
    print(f"[ROUTE1] output={output}", flush=True)


if __name__ == "__main__":
    main()
