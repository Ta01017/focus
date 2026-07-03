#!/usr/bin/env python

import argparse
from pathlib import Path

import torch
from PIL import Image

from diffusers import SanaSprintImg2ImgPipeline
from diffusers.utils.torch_utils import randn_tensor

from dof_utils import add_metadata_args, add_pretrained_args, prepare_inference_images, restore_output_size
from infer_sana_sprint import load_focus, load_sana_adapter_pipeline
from sana_dof import encode_condition_images, encode_vae_latents


class DofSanaSprintImg2ImgPipeline(SanaSprintImg2ImgPipeline):
    """SANA-Sprint img2img pipeline using the shared VAE encode compatibility helper."""

    def prepare_latents(
        self, image, timestep, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if image.shape[1] != num_channels_latents:
            image_latents = encode_vae_latents(self.vae, image) * self.scheduler.config.sigma_data
        else:
            image_latents = image
        if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
            image_latents = torch.cat([image_latents] * (batch_size // image_latents.shape[0]), dim=0)
        elif batch_size > image_latents.shape[0]:
            raise ValueError(
                f"Cannot duplicate image batch {image_latents.shape[0]} to requested batch {batch_size}."
            )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f"Generator list length {len(generator)} does not match batch size {batch_size}.")
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        noise = noise * self.scheduler.config.sigma_data
        return torch.cos(timestep) * image_latents + torch.sin(timestep) * noise


def parse_args():
    parser = argparse.ArgumentParser(description="SANA-Sprint img2img A/B fusion inference.")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--image_a", required=True)
    parser.add_argument("--image_b", required=True)
    parser.add_argument("--focus_a", default=None)
    parser.add_argument("--focus_b", default=None)
    parser.add_argument("--init_image", default=None, help="Defaults to A; may be a preliminary fusion image.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="a photorealistic all-in-focus photograph")
    parser.add_argument("--height", type=int, default=None, help="Compatibility check only; must equal A height.")
    parser.add_argument("--width", type=int, default=None, help="Compatibility check only; must equal A width.")
    parser.add_argument("--max_pixels", type=int, default=None, help="Safety limit; never triggers resizing.")
    parser.add_argument("--size_divisor", type=int, default=32)
    parser.add_argument("--aspect_ratio_tolerance", type=float, default=0.01)
    parser.add_argument("--downscale_if_exceeds_max_pixels", action="store_true")
    restore_group = parser.add_mutually_exclusive_group()
    restore_group.add_argument("--restore_to_original_size", dest="restore_to_original_size", action="store_true")
    restore_group.add_argument("--no_restore_to_original_size", dest="restore_to_original_size", action="store_false")
    parser.set_defaults(restore_to_original_size=True)
    parser.add_argument("--steps", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--strength", type=float, default=0.75)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adapter_hidden_channels", type=int, default=None)
    add_metadata_args(parser)
    add_pretrained_args(parser)
    return parser.parse_args()


def load_pipeline(args):
    return load_sana_adapter_pipeline(args, pipeline_class=DofSanaSprintImg2ImgPipeline)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This reference script requires CUDA.")
    if not 0 < args.strength <= 1:
        raise ValueError("strength must be in (0, 1].")
    if int(args.steps * args.strength) < 1:
        raise ValueError(
            "steps * strength must select at least one denoising step; one-step img2img needs strength=1."
        )
    pipe, transformer, config = load_pipeline(args)
    image_a = Image.open(args.image_a).convert("RGB")
    image_b = Image.open(args.image_b).convert("RGB")
    init_image = image_a if args.init_image is None else Image.open(args.init_image).convert("RGB")
    named_images = {"a": image_a, "b": image_b, "init": init_image}
    if config.get("adapter_type", "ab") == "ab_focus":
        if args.focus_a is None or args.focus_b is None:
            raise ValueError("ab_focus adapter 推理必须提供 --focus_a 和 --focus_b。")
        named_images["focus_a"] = Image.open(args.focus_a)
        named_images["focus_b"] = Image.open(args.focus_b)
    prepared, size_info = prepare_inference_images(
        named_images,
        args.height,
        args.width,
        args.max_pixels,
        args.size_divisor,
        args.aspect_ratio_tolerance,
        args.downscale_if_exceeds_max_pixels,
    )
    canvas_width, canvas_height = size_info["canvas_size"]
    cond_a, cond_b = encode_condition_images(
        pipe, prepared["a"], prepared["b"], canvas_height, canvas_width, torch.device("cuda")
    )
    focus_a = None
    focus_b = None
    if config.get("adapter_type", "ab") == "ab_focus":
        focus_a = load_focus(prepared["focus_a"]).to("cuda")
        focus_b = load_focus(prepared["focus_b"]).to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    with transformer.use_condition(cond_a, cond_b, focus_a, focus_b):
        image = pipe(
            prompt=args.prompt,
            image=prepared["init"],
            strength=args.strength,
            height=canvas_height,
            width=canvas_width,
            num_inference_steps=args.steps,
            intermediate_timesteps=1.3 if args.steps == 2 else None,
            guidance_scale=args.guidance_scale,
            generator=generator,
            use_resolution_binning=False,
        ).images[0]
    image = restore_output_size(image, size_info, args.restore_to_original_size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


if __name__ == "__main__":
    main()
