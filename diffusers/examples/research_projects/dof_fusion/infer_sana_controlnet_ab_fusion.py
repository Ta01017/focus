#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from diffusers import SanaControlNetModel, SanaPipeline

from dof_utils import add_pretrained_args, prepare_inference_images, pretrained_kwargs, restore_output_size
from infer_sana_ab_adapter import decode_latents_to_pil, encode_image_latents
from sana_dof import tensor_stats
from train_sana_controlnet_ab_fusion import ControlConditionProjection


def parse_args():
    parser = argparse.ArgumentParser(description="Infer SANA ControlNet A/B/focus fusion.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--image_a", required=True)
    parser.add_argument("--image_b", required=True)
    parser.add_argument("--focus_a", default=None)
    parser.add_argument("--focus_b", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="a photorealistic all-in-focus photograph")
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--use_a_latent_init", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strength", type=float, default=0.2)
    parser.add_argument("--img2img_schedule_mode", choices=("pipeline_full", "sliced"), default="sliced")
    parser.add_argument("--debug_latent_dir", default=None)
    parser.add_argument("--zero_condition_images", action="store_true")
    parser.add_argument("--zero_focus_conditions", action="store_true")
    parser.add_argument("--focus_default_value", type=float, default=0.5)
    parser.add_argument("--focus_normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=None)
    parser.add_argument("--size_divisor", type=int, default=32)
    parser.add_argument("--aspect_ratio_tolerance", type=float, default=0.01)
    parser.add_argument("--downscale_if_exceeds_max_pixels", action="store_true")
    restore = parser.add_mutually_exclusive_group()
    restore.add_argument("--restore_to_original_size", dest="restore_to_original_size", action="store_true")
    restore.add_argument("--no_restore_to_original_size", dest="restore_to_original_size", action="store_false")
    parser.set_defaults(restore_to_original_size=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=10.0)
    parser.add_argument("--conditioning_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    add_pretrained_args(parser)
    return parser.parse_args()


def focus_tensor(image, size, default_value, normalize=True):
    if image is None:
        return torch.full((1, size[1], size[0]), default_value, dtype=torch.float32)
    image = image.convert("L").resize(size, Image.Resampling.BILINEAR)
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32)).unsqueeze(0)
    if normalize:
        tensor = tensor / 255.0
    return tensor.clamp(0, 1)


def build_condition(pipe, prepared, focus_a, focus_b, size_info, args):
    canvas_w, canvas_h = size_info["canvas_size"]
    cond_a = Image.new("RGB", prepared["a"].size, (0, 0, 0)) if args.zero_condition_images else prepared["a"]
    cond_b = Image.new("RGB", prepared["b"].size, (0, 0, 0)) if args.zero_condition_images else prepared["b"]
    a = pipe.image_processor.preprocess(cond_a, height=canvas_h, width=canvas_w)[0]
    b = pipe.image_processor.preprocess(cond_b, height=canvas_h, width=canvas_w)[0]
    fa = focus_tensor(None if args.zero_focus_conditions else focus_a, size_info["content_size"], args.focus_default_value, args.focus_normalize)
    fb = focus_tensor(None if args.zero_focus_conditions else focus_b, size_info["content_size"], args.focus_default_value, args.focus_normalize)
    fa = torch.nn.functional.pad(fa, (0, canvas_w - size_info["content_size"][0], 0, canvas_h - size_info["content_size"][1]))
    fb = torch.nn.functional.pad(fb, (0, canvas_w - size_info["content_size"][0], 0, canvas_h - size_info["content_size"][1]))
    return torch.cat([a, b, fa, fb], dim=0).unsqueeze(0), fa, fb


def load_pipeline(checkpoint, model, dtype, options):
    config = json.loads((checkpoint / "controlnet_ab_config.json").read_text(encoding="utf-8"))
    model_id = model or config["model"]
    pipe = SanaPipeline.from_pretrained(model_id, torch_dtype=dtype, **options).to("cuda")
    pipe.vae.to(dtype=torch.float32)
    controlnet = SanaControlNetModel.from_pretrained(checkpoint / "controlnet", torch_dtype=dtype, **options).to("cuda")
    projection = ControlConditionProjection(config["control_condition_channels"], pipe.transformer.config.in_channels)
    projection.load_state_dict(load_file(checkpoint / "condition_projection.safetensors"))
    projection.to("cuda", dtype=torch.float32).eval()
    if (checkpoint / "transformer_lora").exists():
        pipe.load_lora_weights(checkpoint / "transformer_lora")
    return pipe, controlnet, projection, config


@torch.no_grad()
def generate(pipe, controlnet, projection, prompt, negative_prompt, image_a, control_condition, height, width, steps, guidance_scale, strength, mode, generator, conditioning_scale):
    device = torch.device("cuda")
    pipe._guidance_scale = guidance_scale
    do_cfg = guidance_scale > 1.0
    prompt_embeds, prompt_mask, neg_embeds, neg_mask = pipe.encode_prompt(
        prompt, do_cfg, negative_prompt=negative_prompt, num_images_per_prompt=1, device=device, clean_caption=False, max_sequence_length=300
    )
    if do_cfg:
        prompt_embeds = torch.cat([neg_embeds, prompt_embeds], dim=0)
        prompt_mask = torch.cat([neg_mask, prompt_mask], dim=0)
    pipe.scheduler.set_timesteps(steps, device=device)
    timesteps = pipe.scheduler.timesteps
    if mode == "sliced":
        init_timestep = max(min(int(steps * strength), steps), 1)
        t_start = max(steps - init_timestep, 0)
        sliced = timesteps[t_start:]
        begin_index = t_start * pipe.scheduler.order
        if hasattr(pipe.scheduler, "set_begin_index"):
            pipe.scheduler.set_begin_index(begin_index)
        sigma = pipe.scheduler.sigmas[t_start].to(device=device, dtype=torch.float32).reshape(1, 1, 1, 1)
        a_latents = encode_image_latents(pipe, image_a, height, width, device)
        noise = torch.randn(a_latents.shape, generator=generator, device=device, dtype=a_latents.dtype)
        latents = (1 - sigma.to(a_latents)) * a_latents + sigma.to(a_latents) * noise
    else:
        sliced = timesteps
        t_start = 0
        begin_index = 0
        a_latents = encode_image_latents(pipe, image_a, height, width, device)
        noise = torch.randn(a_latents.shape, generator=generator, device=device, dtype=a_latents.dtype)
        latents = (1 - strength) * a_latents + strength * noise if strength > 0 else noise
        sigma = None
    controlnet_cond = projection(control_condition.to(device).float()).to(latents)
    final_residual_stats = None
    for t in sliced:
        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
        timestep = t.expand(latent_model_input.shape[0]) * pipe.transformer.config.timestep_scale
        expanded_control = controlnet_cond.repeat(latent_model_input.shape[0], 1, 1, 1) if latent_model_input.shape[0] != controlnet_cond.shape[0] else controlnet_cond
        samples = controlnet(
            latent_model_input.to(controlnet.dtype),
            encoder_hidden_states=prompt_embeds.to(controlnet.dtype),
            encoder_attention_mask=prompt_mask,
            timestep=timestep,
            controlnet_cond=expanded_control.to(controlnet.dtype),
            conditioning_scale=conditioning_scale,
            return_dict=False,
        )[0]
        final_residual_stats = {
            "mean": float(torch.stack([s.detach().float().mean() for s in samples]).mean().cpu()),
            "std": float(torch.stack([s.detach().float().std() for s in samples]).mean().cpu()),
            "norm": float(torch.stack([s.detach().float().norm() for s in samples]).sum().cpu()),
        }
        noise_pred = pipe.transformer(
            latent_model_input.to(pipe.transformer.dtype),
            encoder_hidden_states=prompt_embeds.to(pipe.transformer.dtype),
            encoder_attention_mask=prompt_mask,
            timestep=timestep,
            controlnet_block_samples=tuple(s.to(pipe.transformer.dtype) for s in samples),
            return_dict=False,
        )[0].float()
        if do_cfg:
            uncond, text = noise_pred.chunk(2)
            noise_pred = uncond + guidance_scale * (text - uncond)
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    image = decode_latents_to_pil(pipe, latents)
    return image, latents, {
        "img2img_schedule_mode": mode,
        "strength": strength,
        "t_start": t_start,
        "selected_sigma": None if sigma is None else float(sigma.flatten()[0].cpu()),
        "actual_num_denoise_steps": len(sliced),
        "scheduler_order": pipe.scheduler.order,
        "begin_index_value": begin_index,
        "init_latents": tensor_stats(a_latents),
        "final_latents": tensor_stats(latents),
        "control_residual": final_residual_stats,
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipe, controlnet, projection, _ = load_pipeline(Path(args.checkpoint), args.model, dtype, pretrained_kwargs(args))
    image_a = Image.open(args.image_a).convert("RGB")
    image_b = Image.open(args.image_b).convert("RGB")
    focus_a = Image.open(args.focus_a) if args.focus_a else None
    focus_b = Image.open(args.focus_b) if args.focus_b else None
    prepared, size_info = prepare_inference_images(
        {"a": image_a, "b": image_b},
        args.height,
        args.width,
        args.max_pixels,
        args.size_divisor,
        args.aspect_ratio_tolerance,
        args.downscale_if_exceeds_max_pixels,
    )
    canvas_w, canvas_h = size_info["canvas_size"]
    control_condition, focus_a_tensor, focus_b_tensor = build_condition(pipe, prepared, focus_a, focus_b, size_info, args)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    image, final_latents, stats = generate(
        pipe, controlnet, projection, args.prompt, args.negative_prompt, prepared["a"], control_condition, canvas_h, canvas_w,
        args.steps, args.guidance_scale, args.strength if args.use_a_latent_init else 1.0, args.img2img_schedule_mode, generator, args.conditioning_scale
    )
    if args.debug_latent_dir:
        debug = Path(args.debug_latent_dir)
        debug.mkdir(parents=True, exist_ok=True)
        prepared["a"].save(debug / "raw_A.png")
        prepared["b"].save(debug / "raw_B.png")
        if focus_a:
            focus_a.save(debug / "raw_focus_a.png")
        if focus_b:
            focus_b.save(debug / "raw_focus_b.png")
        Image.fromarray((focus_a_tensor.squeeze().numpy() * 255).astype(np.uint8)).save(debug / "resized_focus_a.png")
        Image.fromarray((focus_b_tensor.squeeze().numpy() * 255).astype(np.uint8)).save(debug / "resized_focus_b.png")
        image.save(debug / "final_output.png")
        (debug / "latent_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
        (debug / "control_condition_stats.json").write_text(json.dumps(tensor_stats(control_condition), indent=2), encoding="utf-8")
    image = restore_output_size(image, size_info, args.restore_to_original_size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


if __name__ == "__main__":
    main()
