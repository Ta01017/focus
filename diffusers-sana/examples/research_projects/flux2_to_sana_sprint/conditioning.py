from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F


class MultiImageConditionAdapter(nn.Module):
    """Maps a fixed number of aligned image latents to an additive Sana latent condition."""

    def __init__(self, latent_channels, num_condition_images, hidden_channels=128):
        super().__init__()
        if num_condition_images < 1:
            raise ValueError("num_condition_images must be positive.")
        self.latent_channels = latent_channels
        self.num_condition_images = num_condition_images
        self.hidden_channels = hidden_channels
        self.proj_in = nn.Conv2d(latent_channels * num_condition_images, hidden_channels, 3, padding=1)
        self.proj_mid = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.proj_out = nn.Conv2d(hidden_channels, latent_channels, 3, padding=1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, condition_latents):
        if condition_latents.ndim != 5:
            raise ValueError("condition_latents must have shape [batch, images, channels, height, width].")
        batch, images, channels, height, width = condition_latents.shape
        if images != self.num_condition_images or channels != self.latent_channels:
            raise ValueError(
                f"Expected {self.num_condition_images} images with {self.latent_channels} channels, "
                f"got {images} images with {channels} channels."
            )
        hidden_states = condition_latents.reshape(batch, images * channels, height, width)
        hidden_states = F.silu(self.proj_in(hidden_states))
        hidden_states = hidden_states + F.silu(self.proj_mid(hidden_states))
        return self.proj_out(hidden_states)


class ConditionedSanaTransformer(nn.Module):
    def __init__(self, transformer, adapter):
        super().__init__()
        self.transformer = transformer
        self.adapter = adapter
        self._condition_latents = None

    @property
    def config(self):
        return self.transformer.config

    @property
    def dtype(self):
        return self.transformer.dtype

    @contextmanager
    def use_condition(self, condition_latents):
        if self._condition_latents is not None:
            raise RuntimeError("An image condition is already active.")
        self._condition_latents = condition_latents
        try:
            yield
        finally:
            self._condition_latents = None

    def forward(self, hidden_states, *args, condition_latents=None, **kwargs):
        condition_latents = self._condition_latents if condition_latents is None else condition_latents
        if condition_latents is None:
            raise ValueError("condition_latents are required for the conditioned Sana transformer.")
        parameter = next(self.adapter.parameters())
        condition_latents = condition_latents.to(device=parameter.device, dtype=parameter.dtype)
        batch, images, channels, height, width = condition_latents.shape
        if (height, width) != hidden_states.shape[-2:]:
            condition_latents = F.interpolate(
                condition_latents.reshape(batch * images, channels, height, width),
                size=hidden_states.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).reshape(batch, images, channels, *hidden_states.shape[-2:])
        if batch != hidden_states.shape[0]:
            if hidden_states.shape[0] % batch:
                raise ValueError("Condition batch cannot be expanded to the denoiser batch.")
            repeats = hidden_states.shape[0] // batch
            condition_latents = condition_latents.repeat(repeats, 1, 1, 1, 1)
        hidden_states = hidden_states + self.adapter(condition_latents).to(hidden_states)
        return self.transformer(hidden_states, *args, **kwargs)


def encode_latents(vae, pixel_values):
    encoded = vae.encode(pixel_values)
    if hasattr(encoded, "latent"):
        latents = encoded.latent
    elif hasattr(encoded, "latents"):
        latents = encoded.latents
    elif hasattr(encoded, "latent_dist"):
        latents = encoded.latent_dist.sample()
    else:
        raise TypeError(f"Unsupported VAE output {type(encoded).__name__}.")
    return latents * getattr(vae.config, "scaling_factor", 1.0)


def encode_condition_batch(vae, pixel_values):
    batch, images, channels, height, width = pixel_values.shape
    latents = encode_latents(vae, pixel_values.reshape(batch * images, channels, height, width))
    return latents.reshape(batch, images, *latents.shape[1:])
