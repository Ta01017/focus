#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file

from diffusers import SanaPipeline

from dof_utils import add_pretrained_args, prepare_inference_images, pretrained_kwargs, restore_output_size
from sana_dof import (
    ConditionedSanaTransformer,
    DualImageConditionAdapter,
    decode_vae_latents,
    encode_condition_images,
    encode_vae_latents,
    tensor_stats,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Ordinary SANA + A/B adapter-only DOF fusion inference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--image_a", required=True)
    parser.add_argument("--image_b", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="a photorealistic all-in-focus photograph")
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=None)
    parser.add_argument("--size_divisor", type=int, default=32)
    parser.add_argument("--aspect_ratio_tolerance", type=float, default=0.01)
    parser.add_argument("--downscale_if_exceeds_max_pixels", action="store_true")
    restore_group = parser.add_mutually_exclusive_group()
    restore_group.add_argument("--restore_to_original_size", dest="restore_to_original_size", action="store_true")
    restore_group.add_argument("--no_restore_to_original_size", dest="restore_to_original_size", action="store_false")
    parser.set_defaults(restore_to_original_size=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_a_latent_init", action="store_true")
    parser.add_argument("--strength", type=float, default=0.6)
    parser.add_argument("--zero_condition_images", action="store_true")
    parser.add_argument("--img2img_schedule_mode", choices=("pipeline_full", "sliced"), default="pipeline_full")
    parser.add_argument("--debug_latent_dir", default=None)
    add_pretrained_args(parser)
    return parser.parse_args()


def load_pipeline(checkpoint, model_id, dtype, pretrained_options):
    config = json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
    model_id = model_id or config["base_model"]
    pipe = SanaPipeline.from_pretrained(model_id, torch_dtype=dtype, **pretrained_options).to("cuda")
    pipe.vae.to(dtype=torch.float32)
    lora_path = checkpoint / "transformer_lora"
    if config.get("train_transformer_lora", False) or lora_path.exists():
        if not lora_path.exists():
            raise ValueError(f"adapter_config.json requests transformer LoRA, but {lora_path} does not exist.")
        pipe.load_lora_weights(lora_path)
    adapter = DualImageConditionAdapter(pipe.transformer.config.in_channels, config["hidden_channels"])
    adapter.load_state_dict(load_file(checkpoint / "adapter.safetensors"), strict=True)
    adapter.to(device="cuda", dtype=dtype).eval()
    transformer = ConditionedSanaTransformer(pipe.transformer, adapter)
    pipe.transformer = transformer
    return pipe, transformer, config


@torch.no_grad()
def prepare_a_latent_init(pipe, image_a, height, width, strength, generator, device):
    if strength < 0 or strength > 1:
        raise ValueError("--strength must be in [0, 1].")
    pixel_values = pipe.image_processor.preprocess(image_a, height=height, width=width).to(device, torch.float32)
    a_latents = encode_vae_latents(pipe.vae, pixel_values)
    noise = torch.randn(a_latents.shape, generator=generator, device=device, dtype=a_latents.dtype)
    return (1 - strength) * a_latents + strength * noise


@torch.no_grad()
def encode_image_latents(pipe, image, height, width, device):
    pixel_values = pipe.image_processor.preprocess(image, height=height, width=width).to(device, torch.float32)
    return encode_vae_latents(pipe.vae, pixel_values)


@torch.no_grad()
def decode_latents_to_pil(pipe, latents):
    image = decode_vae_latents(pipe.vae, latents)
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


@torch.no_grad()
def vae_roundtrip(pipe, image, height, width, device):
    latents = encode_image_latents(pipe, image, height, width, device)
    return decode_latents_to_pil(pipe, latents), latents


@torch.no_grad()
def generate_sana_img2img_sliced(
    pipe,
    transformer,
    prompt,
    negative_prompt,
    image_a,
    cond_a,
    cond_b,
    height,
    width,
    num_inference_steps,
    guidance_scale,
    strength,
    generator,
):
    if strength < 0 or strength > 1:
        raise ValueError("--strength must be in [0, 1].")
    if not hasattr(pipe.scheduler, "sigmas"):
        raise NotImplementedError("Sliced SANA img2img requires scheduler.sigmas.")

    device = torch.device("cuda")
    pipe._guidance_scale = guidance_scale
    pipe._attention_kwargs = None
    pipe._interrupt = False
    do_classifier_free_guidance = guidance_scale > 1.0
    prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = pipe.encode_prompt(
        prompt,
        do_classifier_free_guidance,
        negative_prompt=negative_prompt,
        num_images_per_prompt=1,
        device=device,
        clean_caption=False,
        max_sequence_length=300,
    )
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    init_timestep = max(init_timestep, 1)
    t_start = max(num_inference_steps - init_timestep, 0)
    sliced_timesteps = timesteps[t_start:]
    if not hasattr(pipe.scheduler, "sigmas"):
        raise NotImplementedError("Sliced SANA img2img requires scheduler.sigmas.")
    sigma = pipe.scheduler.sigmas[t_start].to(device=device, dtype=torch.float32).reshape(1, 1, 1, 1)

    a_latents = encode_image_latents(pipe, image_a, height, width, device)
    noise = torch.randn(a_latents.shape, generator=generator, device=device, dtype=a_latents.dtype)
    init_latents = (1 - sigma.to(a_latents)) * a_latents + sigma.to(a_latents) * noise
    latents = init_latents
    transformer_dtype = pipe.transformer.dtype
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, 0.0)
    with pipe.progress_bar(total=len(sliced_timesteps)) as progress_bar:
        for t in sliced_timesteps:
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            timestep = t.expand(latent_model_input.shape[0])
            timestep = timestep * pipe.transformer.config.timestep_scale
            with transformer.use_condition(cond_a, cond_b):
                noise_pred = pipe.transformer(
                    hidden_states=latent_model_input.to(dtype=transformer_dtype),
                    encoder_hidden_states=prompt_embeds.to(dtype=transformer_dtype),
                    encoder_attention_mask=prompt_attention_mask,
                    timestep=timestep,
                    return_dict=False,
                    attention_kwargs=None,
                )[0]
            noise_pred = noise_pred.float()
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            if pipe.transformer.config.out_channels // 2 == pipe.transformer.config.in_channels:
                noise_pred = noise_pred.chunk(2, dim=1)[0]
            latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
            progress_bar.update()

    image = decode_latents_to_pil(pipe, latents)
    stats = {
        "num_inference_steps": num_inference_steps,
        "init_timestep": init_timestep,
        "t_start": t_start,
        "actual_num_denoise_steps": len(sliced_timesteps),
        "selected_sigma": float(sigma.flatten()[0].detach().cpu()),
        "timesteps_first": float(timesteps[0].detach().cpu()),
        "timesteps_last": float(timesteps[-1].detach().cpu()),
        "sliced_timesteps_first": float(sliced_timesteps[0].detach().cpu()),
        "sliced_timesteps_last": float(sliced_timesteps[-1].detach().cpu()),
        "a_latents": tensor_stats(a_latents),
        "noise": tensor_stats(noise),
        "init_latents": tensor_stats(init_latents),
        "final_latents": tensor_stats(latents),
    }
    return image, latents, stats, noise, init_latents


def save_latent_debug(
    debug_dir,
    *,
    args,
    model_id,
    pipe,
    prepared,
    size_info,
    canvas_width,
    canvas_height,
    cond_a,
    cond_b,
    noise,
    init_latents,
    final_latents,
    image,
    schedule_stats,
):
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    prepared["a"].save(debug_dir / "raw_A.png")
    prepared["b"].save(debug_dir / "raw_B.png")
    roundtrip_a, a_latents = vae_roundtrip(pipe, prepared["a"], canvas_height, canvas_width, torch.device("cuda"))
    roundtrip_b, b_latents = vae_roundtrip(pipe, prepared["b"], canvas_height, canvas_width, torch.device("cuda"))
    roundtrip_a.save(debug_dir / "vae_roundtrip_A.png")
    roundtrip_b.save(debug_dir / "vae_roundtrip_B.png")
    if init_latents is not None:
        decode_latents_to_pil(pipe, init_latents).save(debug_dir / "decoded_init_latents.png")
    image.save(debug_dir / "final_output.png")
    stats = {
        "checkpoint": args.checkpoint,
        "model": model_id,
        "use_a_latent_init": args.use_a_latent_init,
        "img2img_schedule_mode": args.img2img_schedule_mode,
        "strength": args.strength,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "canvas_size": list(size_info["canvas_size"]),
        "original_size": list(size_info["original_size"]),
        "content_size": list(size_info["content_size"]),
        "vae.config.scaling_factor": getattr(pipe.vae.config, "scaling_factor", None),
        "vae.config.shift_factor": getattr(pipe.vae.config, "shift_factor", None),
        "scheduler class name": pipe.scheduler.__class__.__name__,
        "a_latents": tensor_stats(a_latents),
        "b_latents": tensor_stats(b_latents),
        "cond_a_latents": tensor_stats(cond_a),
        "cond_b_latents": tensor_stats(cond_b),
    }
    if noise is not None:
        stats["noise"] = tensor_stats(noise)
    if init_latents is not None:
        stats["init_latents"] = tensor_stats(init_latents)
    if final_latents is not None:
        stats["final_latents"] = tensor_stats(final_latents)
    stats.update(schedule_stats)
    (debug_dir / "latent_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This reference script requires CUDA.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipe, transformer, config = load_pipeline(Path(args.checkpoint), args.model, dtype, pretrained_kwargs(args))
    model_id = args.model or config["base_model"]
    image_a = Image.open(args.image_a).convert("RGB")
    image_b = Image.open(args.image_b).convert("RGB")
    prepared, size_info = prepare_inference_images(
        {"a": image_a, "b": image_b},
        args.height,
        args.width,
        args.max_pixels,
        args.size_divisor,
        args.aspect_ratio_tolerance,
        args.downscale_if_exceeds_max_pixels,
    )
    canvas_width, canvas_height = size_info["canvas_size"]
    condition_a = Image.new("RGB", prepared["a"].size, (0, 0, 0)) if args.zero_condition_images else prepared["a"]
    condition_b = Image.new("RGB", prepared["b"].size, (0, 0, 0)) if args.zero_condition_images else prepared["b"]
    cond_a, cond_b = encode_condition_images(
        pipe, condition_a, condition_b, canvas_height, canvas_width, torch.device("cuda")
    )
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    print(f"[USE_A_LATENT_INIT] {int(args.use_a_latent_init)}", flush=True)
    print(f"[STRENGTH] {args.strength}", flush=True)
    print(f"[ZERO_CONDITION_IMAGES] {int(args.zero_condition_images)}", flush=True)
    print(f"[IMG2IMG_SCHEDULE_MODE] {args.img2img_schedule_mode}", flush=True)
    print(f"[DEBUG_LATENT_DIR] {args.debug_latent_dir}", flush=True)
    latents = None
    noise = None
    final_latents = None
    schedule_stats = {}
    if args.use_a_latent_init and args.img2img_schedule_mode == "pipeline_full":
        a_latents = encode_image_latents(pipe, prepared["a"], canvas_height, canvas_width, torch.device("cuda"))
        noise = torch.randn(a_latents.shape, generator=generator, device="cuda", dtype=a_latents.dtype)
        latents = (1 - args.strength) * a_latents + args.strength * noise
        schedule_stats = {"init_timestep": None, "t_start": None, "actual_num_denoise_steps": args.steps}
    if args.use_a_latent_init and args.img2img_schedule_mode == "sliced":
        image, final_latents, schedule_stats, noise, latents = generate_sana_img2img_sliced(
            pipe,
            transformer,
            args.prompt,
            args.negative_prompt,
            prepared["a"],
            cond_a,
            cond_b,
            canvas_height,
            canvas_width,
            args.steps,
            args.guidance_scale,
            args.strength,
            generator,
        )
    else:
        with transformer.use_condition(cond_a, cond_b):
            output = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                height=canvas_height,
                width=canvas_width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                latents=latents,
                output_type="latent",
                use_resolution_binning=False,
            )
            final_latents = output.images
            image = decode_latents_to_pil(pipe, final_latents)
        pipe.scheduler.set_timesteps(args.steps, device=torch.device("cuda"))
        timesteps = pipe.scheduler.timesteps
        schedule_stats = {
            "num_inference_steps": args.steps,
            "init_timestep": None,
            "t_start": None,
            "actual_num_denoise_steps": len(timesteps),
            "selected_sigma": None,
            "timesteps_first": float(timesteps[0].detach().cpu()),
            "timesteps_last": float(timesteps[-1].detach().cpu()),
            "sliced_timesteps_first": None,
            "sliced_timesteps_last": None,
        }
    if args.debug_latent_dir is not None:
        save_latent_debug(
            args.debug_latent_dir,
            args=args,
            model_id=model_id,
            pipe=pipe,
            prepared=prepared,
            size_info=size_info,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            cond_a=cond_a,
            cond_b=cond_b,
            noise=noise,
            init_latents=latents,
            final_latents=final_latents,
            image=image,
            schedule_stats=schedule_stats,
        )
    image = restore_output_size(image, size_info, args.restore_to_original_size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


if __name__ == "__main__":
    main()
