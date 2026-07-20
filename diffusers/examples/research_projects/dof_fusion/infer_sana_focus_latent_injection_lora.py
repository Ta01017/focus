#!/usr/bin/env python

import argparse
import sys
from pathlib import Path

import torch

from diffusers import SanaPipeline

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dof_utils import add_pretrained_args, prepare_inference_images, pretrained_kwargs, restore_output_size  # noqa: E402
from sana_focus_latent_injection import (  # noqa: E402
    IMPLEMENTATION,
    create_focus_model,
    decode_vae_latents,
    default_prompt_for_mode,
    encode_vae_latents,
    load_focus_config,
    load_focus_injector,
    tensor_stats,
    validate_checkpoint_mode,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Infer SANA focus latent injection LoRA.")
    parser.add_argument("--model", default="Efficient-Large-Model/Sana_600M_1024px_diffusers")
    parser.add_argument("--condition_mode", choices=("a_only", "ab"), required=True)
    parser.add_argument("--image_a", required=True)
    parser.add_argument("--image_b", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--allow_untrained_injector", action="store_true")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--use_a_latent_init", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strength", type=float, default=0.3)
    parser.add_argument("--img2img_schedule_mode", choices=("sliced", "pipeline_full"), default="sliced")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16", "auto"), default="auto")
    parser.add_argument("--condition_scale", type=float, default=1.0)
    parser.add_argument("--zero_a_condition", action="store_true")
    parser.add_argument("--zero_b_condition", action="store_true")
    parser.add_argument("--swap_ab", action="store_true")
    parser.add_argument("--disable_condition_injection", action="store_true")
    parser.add_argument("--max_pixels", type=int, default=1048576)
    parser.add_argument("--size_divisor", type=int, default=32)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--aspect_ratio_tolerance", type=float, default=0.01)
    parser.add_argument("--downscale_if_exceeds_max_pixels", action="store_true")
    parser.add_argument("--restore_to_original_size", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
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
    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


def write_json(path, data):
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


@torch.no_grad()
def decode_latents_to_pil(pipe, latents):
    image = decode_vae_latents(pipe.vae, latents)
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


@torch.no_grad()
def encode_prepared_image(pipe, image, height, width, device):
    pixels = pipe.image_processor.preprocess(image, height=height, width=width).to(device, torch.float32)
    return encode_vae_latents(pipe.vae, pixels)


def load_pipeline_and_model(args, dtype):
    if args.condition_mode == "a_only":
        if args.zero_b_condition or args.swap_ab:
            raise ValueError("A-only mode forbids --zero_b_condition and --swap_ab.")
    if args.condition_mode == "ab" and args.image_b is None:
        raise ValueError("AB mode requires --image_b.")
    if args.checkpoint is None and not args.allow_untrained_injector:
        raise ValueError("Provide --checkpoint or explicitly pass --allow_untrained_injector for zero-init sanity check.")

    model_id = args.model
    config = None
    checkpoint_dir = None
    if args.checkpoint:
        config, checkpoint_dir = load_focus_config(args.checkpoint)
        validate_checkpoint_mode(config, args.condition_mode)
        model_id = config.get("base_model", args.model)
    pipe = SanaPipeline.from_pretrained(model_id, torch_dtype=dtype, **pretrained_kwargs(args)).to("cuda")
    pipe.vae.to(dtype=torch.float32)
    pipe.vae.requires_grad_(False).eval()
    pipe.text_encoder.requires_grad_(False).eval()
    pipe.transformer.requires_grad_(False).eval()
    if args.checkpoint and (checkpoint_dir / "transformer_lora").exists():
        pipe.load_lora_weights(checkpoint_dir / "transformer_lora")
    focus_model, latent_channels = create_focus_model(
        pipe.transformer,
        args.condition_mode,
        injector_hidden_channels=(config or {}).get("injector_hidden_channels", 128),
    )
    if args.checkpoint:
        load_focus_injector(checkpoint_dir, focus_model, args.condition_mode)
    else:
        print("[FOCUS_ROUTE1] no trained checkpoint supplied", flush=True)
        print("[FOCUS_ROUTE1] using zero-init injector", flush=True)
        print("[FOCUS_ROUTE1] sanity check only", flush=True)
        print("[FOCUS_ROUTE1] no repair/fusion improvement is expected before training", flush=True)
        config = {
            "implementation": IMPLEMENTATION,
            "condition_mode": args.condition_mode,
            "base_model": model_id,
            "latent_channels": latent_channels,
            "injector_hidden_channels": 128,
            "global_step": 0,
        }
    focus_model.to("cuda", dtype=dtype).eval()
    return pipe, focus_model, config, checkpoint_dir


def compute_timesteps(pipe, steps, strength, use_a_latent_init, device):
    pipe.scheduler.set_timesteps(steps, device=device)
    timesteps = pipe.scheduler.timesteps
    if not use_a_latent_init:
        return timesteps, 0, steps
    effective_steps = max(1, int(steps * strength))
    effective_steps = min(effective_steps, steps)
    start_index = max(steps - effective_steps, 0)
    sliced = timesteps[start_index:]
    if hasattr(pipe.scheduler, "set_begin_index"):
        pipe.scheduler.set_begin_index(start_index)
    return sliced, start_index, effective_steps


def expand_for_cfg(tensor, do_cfg):
    return torch.cat([tensor, tensor], dim=0) if do_cfg else tensor


@torch.no_grad()
def zero_init_noop_check(focus_model, prompt_embeds, prompt_mask, z_a, z_b, z_t, timestep, condition_mode):
    condition = z_a if condition_mode == "a_only" else [z_a, z_b]
    timestep_input = timestep.expand(z_t.shape[0]) * focus_model.config.timestep_scale
    pred_base = focus_model(
        hidden_states=z_t.to(focus_model.dtype),
        encoder_hidden_states=prompt_embeds.to(focus_model.dtype),
        encoder_attention_mask=prompt_mask,
        timestep=timestep_input,
        condition_latents=condition,
        condition_mode=condition_mode,
        disable_condition_injection=True,
        return_dict=False,
    )[0].float()
    pred_zero = focus_model(
        hidden_states=z_t.to(focus_model.dtype),
        encoder_hidden_states=prompt_embeds.to(focus_model.dtype),
        encoder_attention_mask=prompt_mask,
        timestep=timestep_input,
        condition_latents=condition,
        condition_mode=condition_mode,
        condition_scale=1.0,
        return_dict=False,
    )[0].float()
    return {
        "max_abs_delta": float((pred_base - pred_zero).abs().max().cpu()),
        "mean_abs_delta": float((pred_base - pred_zero).abs().mean().cpu()),
    }


@torch.no_grad()
def generate(pipe, focus_model, config, args, image_a, image_b=None):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    prompt = args.prompt or default_prompt_for_mode(args.condition_mode)
    do_cfg = args.guidance_scale > 1.0
    prompt_embeds, prompt_mask, neg_embeds, neg_mask = pipe.encode_prompt(
        prompt,
        do_cfg,
        negative_prompt=args.negative_prompt,
        num_images_per_prompt=1,
        device=device,
        clean_caption=False,
        max_sequence_length=300,
    )
    if do_cfg:
        prompt_embeds = torch.cat([neg_embeds, prompt_embeds], dim=0)
        prompt_mask = torch.cat([neg_mask, prompt_mask], dim=0)

    named = {"a": image_a}
    if args.condition_mode == "ab":
        named["b"] = image_b
    prepared, size_info = prepare_inference_images(
        named,
        height=args.height,
        width=args.width,
        max_pixels=args.max_pixels,
        size_divisor=args.size_divisor,
        aspect_ratio_tolerance=args.aspect_ratio_tolerance,
        downscale_if_exceeds_max_pixels=args.downscale_if_exceeds_max_pixels,
    )
    canvas_w, canvas_h = size_info["canvas_size"]
    z_a = encode_prepared_image(pipe, prepared["a"], canvas_h, canvas_w, device)
    z_b = None
    if args.condition_mode == "ab":
        z_b = encode_prepared_image(pipe, prepared["b"], canvas_h, canvas_w, device)
    cond_a = torch.zeros_like(z_a) if args.zero_a_condition else z_a
    if args.condition_mode == "ab":
        cond_b = torch.zeros_like(z_b) if args.zero_b_condition else z_b
        if args.swap_ab:
            cond_a, cond_b = cond_b, cond_a
        condition = [cond_a, cond_b]
    else:
        condition = cond_a

    timesteps, start_index, effective_steps = compute_timesteps(
        pipe, args.steps, args.strength, args.use_a_latent_init, device
    )
    noise = torch.randn(z_a.shape, generator=generator, device=device, dtype=z_a.dtype)
    if args.use_a_latent_init:
        latent_timestep = pipe.scheduler.timesteps[start_index].expand(z_a.shape[0])
        latents = pipe.scheduler.add_noise(original_samples=z_a, noise=noise, timesteps=latent_timestep)
    else:
        latents = noise
    initial_latents = latents.clone()

    sanity = None
    if args.checkpoint is None and args.allow_untrained_injector:
        sanity = zero_init_noop_check(
            focus_model, prompt_embeds[-1:], prompt_mask[-1:], z_a, z_b, latents, timesteps[0], args.condition_mode
        )
        print(f"[FOCUS_ROUTE1] zero-init no-op max_abs_delta={sanity['max_abs_delta']:.6e}", flush=True)
        print(f"[FOCUS_ROUTE1] zero-init no-op mean_abs_delta={sanity['mean_abs_delta']:.6e}", flush=True)

    for timestep in timesteps:
        latent_input = expand_for_cfg(latents, do_cfg)
        if args.condition_mode == "ab":
            condition_input = [expand_for_cfg(condition[0], do_cfg), expand_for_cfg(condition[1], do_cfg)]
        else:
            condition_input = expand_for_cfg(condition, do_cfg)
        timestep_input = timestep.expand(latent_input.shape[0]) * focus_model.config.timestep_scale
        pred = focus_model(
            hidden_states=latent_input.to(focus_model.dtype),
            encoder_hidden_states=prompt_embeds.to(focus_model.dtype),
            encoder_attention_mask=prompt_mask,
            timestep=timestep_input,
            condition_latents=condition_input,
            condition_mode=args.condition_mode,
            condition_scale=args.condition_scale,
            disable_condition_injection=args.disable_condition_injection,
            return_dict=False,
        )[0].float()
        if do_cfg:
            uncond, text = pred.chunk(2)
            pred = uncond + args.guidance_scale * (text - uncond)
        latents = pipe.scheduler.step(pred, timestep, latents, return_dict=False)[0]

    image = decode_latents_to_pil(pipe, latents)
    stats = {
        "implementation": IMPLEMENTATION,
        "condition_mode": args.condition_mode,
        "condition_scale": args.condition_scale,
        "use_a_latent_init": args.use_a_latent_init,
        "strength": args.strength,
        "requested_steps": args.steps,
        "effective_steps": effective_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "zero_a_condition": args.zero_a_condition,
        "zero_b_condition": args.zero_b_condition,
        "swap_ab": args.swap_ab,
        "disable_condition_injection": args.disable_condition_injection,
        "checkpoint": args.checkpoint,
        "global_step": config.get("global_step"),
        "original_size": list(size_info["original_size"]),
        "canvas_size": list(size_info["canvas_size"]),
        "initial_latents": tensor_stats(initial_latents),
        "final_latents": tensor_stats(latents),
        "sanity": sanity,
        "status": "success",
    }
    return image, stats, prepared, size_info


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    dtype = select_dtype(args.dtype)
    pipe, focus_model, config, _ = load_pipeline_and_model(args, dtype)
    from PIL import Image

    image_a = Image.open(args.image_a).convert("RGB")
    image_b = Image.open(args.image_b).convert("RGB") if args.image_b else None
    image, stats, prepared, size_info = generate(pipe, focus_model, config, args, image_a, image_b)
    if args.debug_latent_dir:
        debug = Path(args.debug_latent_dir)
        debug.mkdir(parents=True, exist_ok=True)
        prepared["a"].save(debug / "prepared_a.png")
        if "b" in prepared:
            prepared["b"].save(debug / "prepared_b.png")
        image.save(debug / "before_restore.png")
        write_json(debug / "latent_stats.json", stats)
    image = restore_output_size(image, size_info, args.restore_to_original_size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    stats["output_size"] = list(image.size)
    write_json(output.with_suffix(output.suffix + ".stats.json"), stats)
    print(f"[FOCUS_ROUTE1] output={output}", flush=True)


if __name__ == "__main__":
    main()
