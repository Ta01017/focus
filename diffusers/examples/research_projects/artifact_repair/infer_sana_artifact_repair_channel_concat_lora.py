#!/usr/bin/env python

import argparse
import sys
from pathlib import Path

import torch

from diffusers import SanaPipeline, SanaTransformer2DModel

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from artifact_repair_utils import (  # noqa: E402
    DEFAULT_REPAIR_PROMPT,
    add_pretrained_args,
    load_rgb,
    preprocess_pair,
    pretrained_kwargs,
    restore_output_size,
    write_json,
)
from sana_artifact_repair_channel_concat import (  # noqa: E402
    CONCAT_ORDER,
    decode_vae_latents,
    encode_vae_latents,
    expand_sana_patch_embedding_for_channel_concat,
    get_sana_output_channels,
    get_sana_patch_embedding,
    load_route2_config,
    load_route2_patch_embedding,
    patch_embedding_weight_stats,
    route2_full_transformer_path,
    route2_lora_path,
    tensor_stats,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Infer SANA Artifact Repair Route 2 channel-concat I2I.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--image_src", "--src_image", dest="image_src", required=True)
    parser.add_argument("--image_ref", "--ref_image", dest="image_ref", required=True)
    parser.add_argument("--output", "--output_path", dest="output", required=True)
    parser.add_argument("--prompt", default=DEFAULT_REPAIR_PROMPT)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16", "auto"), default="auto")
    parser.add_argument("--max_pixels", type=int, default=1048576)
    parser.add_argument("--size_divisor", type=int, default=32)
    parser.add_argument("--downscale_if_exceeds_max_pixels", action="store_true")
    parser.add_argument("--restore_to_original_size", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug_latent_dir", default=None)
    parser.add_argument("--save_condition_sensitivity_debug", action="store_true")
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


@torch.no_grad()
def encode_image_latents(pipe, image, height, width, device):
    pixels = pipe.image_processor.preprocess(image, height=height, width=width).to(device, torch.float32)
    return encode_vae_latents(pipe.vae, pixels)


@torch.no_grad()
def decode_latents_to_pil(pipe, latents):
    image = decode_vae_latents(pipe.vae, latents)
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


def _validate_loaded_transformer(transformer, config):
    expected_original_c = int(config["original_latent_channels"])
    expected_expanded_c = int(config["expanded_input_channels"])
    expected_output_c = int(config["output_channels"])
    actual_in_channels = int(transformer.config.in_channels)
    actual_out_channels = get_sana_output_channels(transformer)
    _, proj = get_sana_patch_embedding(transformer)
    if actual_in_channels != expected_expanded_c:
        raise ValueError(f"Route 2 transformer config.in_channels mismatch: expected {expected_expanded_c}, actual {actual_in_channels}")
    if actual_out_channels != expected_output_c:
        raise ValueError(f"Route 2 transformer output channels mismatch: expected {expected_output_c}, actual {actual_out_channels}")
    if int(proj.in_channels) != expected_expanded_c:
        raise ValueError(f"Route 2 patch projection in_channels mismatch: expected {expected_expanded_c}, actual {proj.in_channels}")
    if int(proj.out_channels) <= 0:
        raise ValueError("Route 2 patch projection has invalid out_channels.")
    return expected_original_c


def load_pipeline(args, dtype):
    checkpoint = Path(args.checkpoint)
    config = load_route2_config(checkpoint)
    train_mode = config.get("train_mode")
    model_id = args.model or config["base_model"]
    print(f"[ROUTE2] checkpoint={checkpoint}", flush=True)
    print(f"[ROUTE2] base_model={model_id}", flush=True)
    print(f"[ROUTE2] train_mode={train_mode}", flush=True)
    patch_path = None
    lora_path = None
    if train_mode == "full_transformer":
        full_transformer_dir = route2_full_transformer_path(checkpoint)
        if full_transformer_dir is None:
            raise FileNotFoundError(
                "Route 2 checkpoint was trained with full_transformer, but the complete transformer directory is missing. "
                "Using only i2i_patch_embedding would silently discard trained transformer weights."
            )
        transformer = SanaTransformer2DModel.from_pretrained(
            full_transformer_dir,
            torch_dtype=dtype,
            local_files_only=args.local_files_only,
            cache_dir=args.cache_dir,
            revision=args.revision,
        )
        pipe = SanaPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=dtype, **pretrained_kwargs(args)).to("cuda")
        original_latent_channels = _validate_loaded_transformer(pipe.transformer, config)
        print("[ROUTE2] full transformer loaded=true", flush=True)
        print(f"[ROUTE2] full transformer path={full_transformer_dir}", flush=True)
        print("[ROUTE2] patch embedding loaded from full transformer=true", flush=True)
        print("[ROUTE2] LoRA loaded=false", flush=True)
    elif train_mode in ("patch_lora", "patch_only"):
        pipe = SanaPipeline.from_pretrained(model_id, torch_dtype=dtype, **pretrained_kwargs(args)).to("cuda")
        original_latent_channels = expand_sana_patch_embedding_for_channel_concat(pipe.transformer)
        patch_path = load_route2_patch_embedding(checkpoint, pipe.transformer)
        _validate_loaded_transformer(pipe.transformer, config)
        if train_mode == "patch_lora":
            lora_path = route2_lora_path(checkpoint)
            if lora_path is None:
                raise FileNotFoundError("Route 2 checkpoint train_mode=patch_lora but transformer_lora is missing.")
            pipe.load_lora_weights(lora_path)
        print("[ROUTE2] full transformer loaded=false", flush=True)
        print(f"[ROUTE2] patch embedding loaded=true path={patch_path}", flush=True)
        print(f"[ROUTE2] LoRA loaded={lora_path is not None}", flush=True)
    else:
        raise ValueError(f"Unsupported Route 2 train_mode in checkpoint: {train_mode!r}")
    pipe.vae.to(dtype=torch.float32)
    pipe.transformer.eval()
    print(f"[ROUTE2] original_latent_channels={original_latent_channels}", flush=True)
    print(f"[ROUTE2] expanded_input_channels={config['expanded_input_channels']}", flush=True)
    print("[ROUTE2] inference init=pure_noise", flush=True)
    print("[ROUTE2] condition=clean src latent channel concat", flush=True)
    return pipe, config, original_latent_channels


@torch.no_grad()
def condition_sensitivity(pipe, latents, z_src, prompt_embeds, prompt_mask, timestep, original_latent_channels):
    timestep_input = timestep.expand(latents.shape[0]) * pipe.transformer.config.timestep_scale
    model_input = torch.cat([latents, z_src], dim=1)
    pred_real = pipe.transformer(
        hidden_states=model_input.to(pipe.transformer.dtype),
        encoder_hidden_states=prompt_embeds.to(pipe.transformer.dtype),
        encoder_attention_mask=prompt_mask,
        timestep=timestep_input,
        return_dict=False,
    )[0].float()
    zeros = torch.zeros_like(z_src)
    pred_zero = pipe.transformer(
        hidden_states=torch.cat([latents, zeros], dim=1).to(pipe.transformer.dtype),
        encoder_hidden_states=prompt_embeds.to(pipe.transformer.dtype),
        encoder_attention_mask=prompt_mask,
        timestep=timestep_input,
        return_dict=False,
    )[0].float()
    shuffled = z_src[torch.randperm(z_src.shape[0], device=z_src.device)] if z_src.shape[0] > 1 else zeros
    pred_shuffle = pipe.transformer(
        hidden_states=torch.cat([latents, shuffled], dim=1).to(pipe.transformer.dtype),
        encoder_hidden_states=prompt_embeds.to(pipe.transformer.dtype),
        encoder_attention_mask=prompt_mask,
        timestep=timestep_input,
        return_dict=False,
    )[0].float()
    real_abs = pred_real.abs().mean().clamp_min(1e-8)
    patch_stats = patch_embedding_weight_stats(pipe.transformer, original_latent_channels)
    stats = {
        "pred_real_vs_zero_mae": float((pred_real - pred_zero).abs().mean().cpu()),
        "pred_real_vs_zero_rel": float(((pred_real - pred_zero).abs().mean() / real_abs).cpu()),
        "pred_real_vs_shuffle_mae": float((pred_real - pred_shuffle).abs().mean().cpu()),
        "condition_latent_abs_mean": float(z_src.detach().float().abs().mean().cpu()),
        "condition_weight_norm": patch_stats["condition_weight_norm"],
    }
    if stats["pred_real_vs_zero_mae"] == 0.0:
        print("[WARNING] Model output is insensitive to src condition. Check patch embedding training/loading and concat path.", flush=True)
    return stats


@torch.no_grad()
def generate(pipe, prompt, negative_prompt, src_image, height, width, steps, guidance_scale, seed, original_latent_channels, save_condition_debug=False):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    do_cfg = guidance_scale > 1.0
    prompt_embeds, prompt_mask, neg_embeds, neg_mask = pipe.encode_prompt(
        prompt, do_cfg, negative_prompt=negative_prompt, num_images_per_prompt=1, device=device, clean_caption=False, max_sequence_length=300
    )
    if do_cfg:
        prompt_embeds = torch.cat([neg_embeds, prompt_embeds], dim=0)
        prompt_mask = torch.cat([neg_mask, prompt_mask], dim=0)
    z_src = encode_image_latents(pipe, src_image, height, width, device)
    latents = torch.randn(z_src.shape, generator=generator, device=device, dtype=z_src.dtype)
    initial_latents = latents.clone()
    pipe.scheduler.set_timesteps(steps, device=device)
    sensitivity = None
    for step_index, timestep in enumerate(pipe.scheduler.timesteps):
        latent_input = torch.cat([latents] * 2) if do_cfg else latents
        src_input = torch.cat([z_src, z_src], dim=0) if do_cfg else z_src
        timestep_input = timestep.expand(latent_input.shape[0]) * pipe.transformer.config.timestep_scale
        if save_condition_debug and step_index == 0:
            sensitivity = condition_sensitivity(pipe, latents, z_src, prompt_embeds[-latents.shape[0] :], prompt_mask[-latents.shape[0] :], timestep, original_latent_channels)
        model_input = torch.cat([latent_input, src_input], dim=1)
        pred = pipe.transformer(
            hidden_states=model_input.to(pipe.transformer.dtype),
            encoder_hidden_states=prompt_embeds.to(pipe.transformer.dtype),
            encoder_attention_mask=prompt_mask,
            timestep=timestep_input,
            return_dict=False,
        )[0].float()
        if do_cfg:
            uncond, text = pred.chunk(2)
            pred = uncond + guidance_scale * (text - uncond)
        latents = pipe.scheduler.step(pred, timestep, latents, return_dict=False)[0]
    image = decode_latents_to_pil(pipe, latents)
    stats = {
        "route": "channel_concat_v1",
        "steps": steps,
        "guidance_scale": guidance_scale,
        "initialization": "pure_noise",
        "condition_type": "clean_src_vae_latent",
        "concat_order": CONCAT_ORDER,
        "src_latents": tensor_stats(z_src),
        "initial_latents": tensor_stats(initial_latents),
        "final_latents": tensor_stats(latents),
        "condition_sensitivity": sensitivity,
    }
    return image, stats, z_src, initial_latents


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    dtype = select_dtype(args.dtype)
    pipe, _, original_latent_channels = load_pipeline(args, dtype)
    src = load_rgb(args.image_src)
    ref = load_rgb(args.image_ref)
    prepared, size_info = preprocess_pair(src, ref, args.max_pixels, args.size_divisor, args.downscale_if_exceeds_max_pixels)
    canvas_w, canvas_h = size_info["canvas_size"]
    image, stats, z_src, initial_latents = generate(
        pipe, args.prompt, args.negative_prompt, prepared["src"], canvas_h, canvas_w, args.steps, args.guidance_scale, args.seed, original_latent_channels, args.save_condition_sensitivity_debug
    )
    stats.update({"src": str(args.image_src), "ref": str(args.image_ref), "prompt": args.prompt, "original_size": list(size_info["original_size"]), "inference_size": list(size_info["canvas_size"])})
    if args.debug_latent_dir:
        debug = Path(args.debug_latent_dir)
        debug.mkdir(parents=True, exist_ok=True)
        src.save(debug / "raw_src.png")
        prepared["src"].save(debug / "resized_src.png")
        image.save(debug / "final_output_before_restore.png")
        decode_latents_to_pil(pipe, z_src).save(debug / "vae_roundtrip_src.png")
        decode_latents_to_pil(pipe, initial_latents).save(debug / "decoded_initial_noise.png")
        write_json(debug / "latent_stats.json", stats)
    image = restore_output_size(image, size_info, args.restore_to_original_size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    stats["restored_size"] = list(image.size)
    write_json(output.with_suffix(output.suffix + ".stats.json"), stats)
    print(f"[ROUTE2] output={output}", flush=True)


if __name__ == "__main__":
    main()
