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
    decode_vae_latents,
    encode_vae_latents,
    expand_sana_patch_embedding_for_channel_concat,
    get_sana_output_channels,
    get_sana_patch_embedding,
    load_route2_patch_embedding,
    route2_lora_path,
    tensor_stats,
)
from sana_artifact_repair_wan_crossattn import (  # noqa: E402
    IMPLEMENTATION,
    build_route3_model,
    encode_image_tokens,
    load_image_encoder_and_processor,
    load_route3_components,
    load_route3_config,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Infer SANA Artifact Repair Route 3 Wan-style image cross-attention.")
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
    parser.add_argument("--init_mode", choices=("pure_noise", "src_latent"), default="pure_noise")
    parser.add_argument("--src_init_strength", type=float, default=0.3)
    parser.add_argument("--disable_image_cross_attention", action="store_true")
    parser.add_argument("--zero_image_tokens", action="store_true")
    parser.add_argument("--swap_image_tokens_path", default=None)
    parser.add_argument("--image_cross_attention_scale", type=float, default=None)
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


@torch.no_grad()
def encode_image_latents(pipe, image, height, width, device):
    pixels = pipe.image_processor.preprocess(image, height=height, width=width).to(device, torch.float32)
    return encode_vae_latents(pipe.vae, pixels)


@torch.no_grad()
def decode_latents_to_pil(pipe, latents):
    image = decode_vae_latents(pipe.vae, latents)
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


def _validate_loaded_transformer(transformer, config):
    expected_expanded_c = int(config["expanded_input_channels"])
    expected_output_c = int(config["output_channels"])
    _, proj = get_sana_patch_embedding(transformer)
    if int(transformer.config.in_channels) != expected_expanded_c:
        raise ValueError(f"Route3 transformer config.in_channels mismatch: expected {expected_expanded_c}")
    if int(proj.in_channels) != expected_expanded_c:
        raise ValueError(f"Route3 patch projection in_channels mismatch: expected {expected_expanded_c}")
    if get_sana_output_channels(transformer) != expected_output_c:
        raise ValueError(f"Route3 output channels mismatch: expected {expected_output_c}")


def load_pipeline(args, dtype):
    checkpoint = Path(args.checkpoint)
    config, ckpt_dir = load_route3_config(checkpoint)
    model_id = args.model or config["base_model"]
    train_mode = config["train_mode"]
    print(f"[ROUTE3] implementation={IMPLEMENTATION}", flush=True)
    print(f"[ROUTE3] checkpoint={ckpt_dir}", flush=True)
    print(f"[ROUTE3] base_model={model_id}", flush=True)
    print(f"[ROUTE3] train_mode={train_mode}", flush=True)

    if train_mode == "full_transformer":
        transformer_dir = ckpt_dir / "transformer"
        if not transformer_dir.exists():
            raise FileNotFoundError(f"Missing Route3 full transformer dir: {transformer_dir}")
        transformer = SanaTransformer2DModel.from_pretrained(transformer_dir, torch_dtype=dtype, **pretrained_kwargs(args))
        pipe = SanaPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=dtype, **pretrained_kwargs(args)).to("cuda")
    elif train_mode == "patch_lora":
        pipe = SanaPipeline.from_pretrained(model_id, torch_dtype=dtype, **pretrained_kwargs(args)).to("cuda")
        expand_sana_patch_embedding_for_channel_concat(pipe.transformer)
        load_route2_patch_embedding(ckpt_dir, pipe.transformer)
        lora_path = route2_lora_path(ckpt_dir)
        if lora_path is None:
            raise FileNotFoundError(f"Missing Route3 transformer_lora in {ckpt_dir}")
        pipe.load_lora_weights(lora_path)
    else:
        raise ValueError(f"Unsupported Route3 train mode: {train_mode}")

    _validate_loaded_transformer(pipe.transformer, config)
    image_args = argparse.Namespace(
        image_encoder_model=config["image_encoder_model"],
        image_encoder_subfolder=config.get("image_encoder_subfolder"),
        image_encoder_revision=config.get("image_encoder_revision"),
        image_encoder_local_files_only=args.local_files_only,
    )
    image_encoder, image_processor = load_image_encoder_and_processor(image_args, dtype=dtype, device="cuda")
    route3_model = build_route3_model(pipe.transformer, int(config["image_encoder_hidden_size"]), image_gate_init=config.get("image_gate_init", 1e-3))
    route3_model.to("cuda", dtype=dtype).eval()
    load_route3_components(ckpt_dir, route3_model)
    pipe.transformer = route3_model
    pipe.vae.to(dtype=torch.float32)
    print(f"[ROUTE3] image_encoder_model={config['image_encoder_model']}", flush=True)
    return pipe, image_encoder, image_processor, config


def compute_timesteps_for_init(pipe, steps, init_mode, src_init_strength, device):
    pipe.scheduler.set_timesteps(steps, device=device)
    timesteps = pipe.scheduler.timesteps
    if init_mode == "pure_noise":
        return timesteps, 0, steps
    effective_steps = max(1, int(steps * src_init_strength))
    effective_steps = min(effective_steps, steps)
    start_index = max(steps - effective_steps, 0)
    sliced_timesteps = timesteps[start_index:]
    if hasattr(pipe.scheduler, "set_begin_index"):
        pipe.scheduler.set_begin_index(start_index)
    return sliced_timesteps, start_index, effective_steps


@torch.no_grad()
def generate(
    pipe,
    image_encoder,
    image_processor,
    config,
    prompt,
    negative_prompt,
    src_image,
    image_token_image,
    height,
    width,
    steps,
    guidance_scale,
    seed,
    init_mode,
    src_init_strength,
    disable_image_cross_attention=False,
    zero_image_tokens=False,
    image_cross_attention_scale=None,
):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
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
    z_src = encode_image_latents(pipe, src_image, height, width, device)
    image_tokens = encode_image_tokens(image_encoder, image_processor, [image_token_image], device, pipe.transformer.dtype)
    if zero_image_tokens:
        image_tokens = torch.zeros_like(image_tokens)

    timesteps, start_index, effective_steps = compute_timesteps_for_init(
        pipe, steps, init_mode, src_init_strength, device
    )
    noise = torch.randn(z_src.shape, generator=generator, device=device, dtype=z_src.dtype)
    if init_mode == "src_latent":
        latent_timestep = pipe.scheduler.timesteps[start_index].expand(z_src.shape[0])
        latents = pipe.scheduler.add_noise(original_samples=z_src, noise=noise, timesteps=latent_timestep)
    else:
        latents = noise
    initial_latents = latents.clone()
    scale = config.get("image_cross_attention_scale", 1.0) if image_cross_attention_scale is None else image_cross_attention_scale

    print(f"[ROUTE3] init_mode={init_mode}", flush=True)
    print(f"[ROUTE3] src_init_strength={src_init_strength}", flush=True)
    print(f"[ROUTE3] requested_steps={steps}", flush=True)
    print(f"[ROUTE3] effective_steps={effective_steps}", flush=True)
    print(f"[ROUTE3] scheduler={pipe.scheduler.__class__.__name__}", flush=True)
    print(f"[ROUTE3] image_context_length={image_tokens.shape[1]}", flush=True)

    for timestep in timesteps:
        latent_input = torch.cat([latents] * 2) if do_cfg else latents
        src_input = torch.cat([z_src, z_src], dim=0) if do_cfg else z_src
        prompt_input = prompt_embeds.to(pipe.transformer.dtype)
        mask_input = prompt_mask
        image_input = torch.cat([image_tokens, image_tokens], dim=0) if do_cfg else image_tokens
        timestep_input = timestep.expand(latent_input.shape[0]) * pipe.transformer.config.timestep_scale
        pred = pipe.transformer(
            hidden_states=torch.cat([latent_input, src_input], dim=1).to(pipe.transformer.dtype),
            encoder_hidden_states=prompt_input,
            encoder_attention_mask=mask_input,
            encoder_hidden_states_image=image_input,
            image_cross_attention_scale=scale,
            disable_image_cross_attention=disable_image_cross_attention,
            timestep=timestep_input,
            return_dict=False,
        )[0].float()
        if do_cfg:
            uncond, text = pred.chunk(2)
            pred = uncond + guidance_scale * (text - uncond)
        latents = pipe.scheduler.step(pred, timestep, latents, return_dict=False)[0]

    image = decode_latents_to_pil(pipe, latents)
    stats = {
        "route": IMPLEMENTATION,
        "initialization": init_mode,
        "src_init_strength": src_init_strength,
        "requested_steps": steps,
        "effective_steps": effective_steps,
        "guidance_scale": guidance_scale,
        "image_encoder_model": config["image_encoder_model"],
        "image_context_length": int(image_tokens.shape[1]),
        "image_cross_attention_scale": float(scale),
        "checkpoint": str(config.get("_checkpoint", "")),
        "train_mode": config["train_mode"],
        "seed": seed,
        "status": "success",
        "disable_image_cross_attention": disable_image_cross_attention,
        "zero_image_tokens": zero_image_tokens,
        "src_latents": tensor_stats(z_src),
        "initial_latents": tensor_stats(initial_latents),
        "final_latents": tensor_stats(latents),
    }
    return image, stats


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    dtype = select_dtype(args.dtype)
    pipe, image_encoder, image_processor, config = load_pipeline(args, dtype)
    config["_checkpoint"] = str(args.checkpoint)
    src = load_rgb(args.image_src)
    ref = load_rgb(args.image_ref)
    token_img = load_rgb(args.swap_image_tokens_path) if args.swap_image_tokens_path else src
    prepared, size_info = preprocess_pair(src, ref, args.max_pixels, args.size_divisor, args.downscale_if_exceeds_max_pixels)
    canvas_w, canvas_h = size_info["canvas_size"]
    token_img = token_img.resize(prepared["src"].size)
    image, stats = generate(
        pipe,
        image_encoder,
        image_processor,
        config,
        args.prompt,
        args.negative_prompt,
        prepared["src"],
        token_img,
        canvas_h,
        canvas_w,
        args.steps,
        args.guidance_scale,
        args.seed,
        args.init_mode,
        args.src_init_strength,
        args.disable_image_cross_attention,
        args.zero_image_tokens,
        args.image_cross_attention_scale,
    )
    stats.update({"src": str(args.image_src), "ref": str(args.image_ref), "prompt": args.prompt, "original_size": list(size_info["original_size"]), "inference_size": list(size_info["canvas_size"])})
    if args.debug_latent_dir:
        debug = Path(args.debug_latent_dir)
        debug.mkdir(parents=True, exist_ok=True)
        prepared["src"].save(debug / "resized_src.png")
        image.save(debug / "final_output_before_restore.png")
        write_json(debug / "latent_stats.json", stats)
    image = restore_output_size(image, size_info, args.restore_to_original_size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    stats["restored_size"] = list(image.size)
    write_json(output.with_suffix(output.suffix + ".stats.json"), stats)
    print(f"[ROUTE3] output={output}", flush=True)


if __name__ == "__main__":
    main()
