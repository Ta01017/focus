from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F


class DualImageConditionAdapter(nn.Module):
    """Projects two aligned SANA image latents to an additive denoiser condition."""

    def __init__(self, latent_channels: int, hidden_channels: int = 128):
        super().__init__()
        self.latent_channels = latent_channels
        self.proj_in = nn.Conv2d(2 * latent_channels, hidden_channels, kernel_size=3, padding=1)
        self.proj_mid = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.proj_out = nn.Conv2d(hidden_channels, latent_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, cond_a_latents: torch.Tensor, cond_b_latents: torch.Tensor) -> torch.Tensor:
        if cond_a_latents.shape != cond_b_latents.shape:
            raise ValueError(
                f"Condition A and B latent shapes must match, got {cond_a_latents.shape} and {cond_b_latents.shape}."
            )
        hidden_states = torch.cat([cond_a_latents, cond_b_latents], dim=1)
        hidden_states = F.silu(self.proj_in(hidden_states))
        hidden_states = hidden_states + F.silu(self.proj_mid(hidden_states))
        return self.proj_out(hidden_states)


class DualImageFocusConditionAdapter(nn.Module):
    """Projects A/B latents and two [0, 1] focus maps to an additive SANA condition."""

    def __init__(self, latent_channels: int, hidden_channels: int = 128):
        super().__init__()
        self.latent_channels = latent_channels
        self.proj_in = nn.Conv2d(2 * latent_channels + 2, hidden_channels, kernel_size=3, padding=1)
        self.proj_mid = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.proj_out = nn.Conv2d(hidden_channels, latent_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, cond_a_latents, cond_b_latents, focus_a, focus_b):
        if cond_a_latents.shape != cond_b_latents.shape:
            raise ValueError("Condition A and B latent shapes must match.")
        size = cond_a_latents.shape[-2:]
        focus_a = F.interpolate(focus_a.float(), size=size, mode="bilinear", align_corners=False).clamp(0, 1)
        focus_b = F.interpolate(focus_b.float(), size=size, mode="bilinear", align_corners=False).clamp(0, 1)
        hidden_states = torch.cat(
            [cond_a_latents, cond_b_latents, focus_a.to(cond_a_latents), focus_b.to(cond_a_latents)], dim=1
        )
        hidden_states = F.silu(self.proj_in(hidden_states))
        hidden_states = hidden_states + F.silu(self.proj_mid(hidden_states))
        return self.proj_out(hidden_states)


def encode_vae_latents(vae, pixel_values, scaling_factor=None, sample=True, generator=None):
    encoded = vae.encode(pixel_values)
    if hasattr(encoded, "latent"):
        latents = encoded.latent
    elif hasattr(encoded, "latents"):
        latents = encoded.latents
    elif hasattr(encoded, "latent_dist"):
        if sample:
            latents = encoded.latent_dist.sample(generator=generator)
        elif hasattr(encoded.latent_dist, "mode"):
            latents = encoded.latent_dist.mode()
        else:
            latents = encoded.latent_dist.sample(generator=generator)
    else:
        raise TypeError(f"Unsupported VAE encode output type: {type(encoded).__name__}.")
    if scaling_factor is None:
        scaling_factor = getattr(vae.config, "scaling_factor", 1.0)
    return latents * scaling_factor


def decode_vae_latents(vae, latents):
    scaling_factor = getattr(vae.config, "scaling_factor", 1.0)
    return vae.decode((latents / scaling_factor).to(vae.dtype), return_dict=False)[0]


def tensor_stats(x):
    x = x.detach().float().cpu()
    return {
        "shape": list(x.shape),
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "dtype": str(x.dtype),
    }


def create_condition_adapter(adapter_type, latent_channels, hidden_channels):
    if adapter_type == "ab":
        return DualImageConditionAdapter(latent_channels, hidden_channels)
    if adapter_type == "ab_focus":
        return DualImageFocusConditionAdapter(latent_channels, hidden_channels)
    raise ValueError(f"Unsupported adapter_type {adapter_type!r}; expected 'ab' or 'ab_focus'.")


class ConditionedSanaTransformer(nn.Module):
    """External wrapper that injects paired image conditions without editing Diffusers source files."""

    def __init__(self, transformer: nn.Module, adapter: DualImageConditionAdapter):
        super().__init__()
        self.transformer = transformer
        self.adapter = adapter
        self._condition = None

    @property
    def config(self):
        return self.transformer.config

    @property
    def dtype(self):
        return self.transformer.dtype

    @contextmanager
    def use_condition(
        self,
        cond_a_latents: torch.Tensor,
        cond_b_latents: torch.Tensor,
        focus_a: torch.Tensor | None = None,
        focus_b: torch.Tensor | None = None,
    ):
        if self._condition is not None:
            raise RuntimeError("A SANA depth-of-field condition is already active.")
        self._condition = (cond_a_latents, cond_b_latents, focus_a, focus_b)
        try:
            yield
        finally:
            self._condition = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        cond_a_latents: torch.Tensor | None = None,
        cond_b_latents: torch.Tensor | None = None,
        focus_a: torch.Tensor | None = None,
        focus_b: torch.Tensor | None = None,
        **kwargs,
    ):
        if (cond_a_latents is None) != (cond_b_latents is None):
            raise ValueError("cond_a_latents and cond_b_latents must be provided together.")
        if cond_a_latents is None and self._condition is not None:
            cond_a_latents, cond_b_latents, focus_a, focus_b = self._condition
        if cond_a_latents is None:
            return self.transformer(hidden_states, *args, **kwargs)

        adapter_parameter = next(self.adapter.parameters())
        cond_a_latents = cond_a_latents.to(device=adapter_parameter.device, dtype=adapter_parameter.dtype)
        cond_b_latents = cond_b_latents.to(device=adapter_parameter.device, dtype=adapter_parameter.dtype)
        if cond_a_latents.shape[-2:] != hidden_states.shape[-2:]:
            size = hidden_states.shape[-2:]
            cond_a_latents = F.interpolate(cond_a_latents, size=size, mode="bilinear", align_corners=False)
            cond_b_latents = F.interpolate(cond_b_latents, size=size, mode="bilinear", align_corners=False)
        if cond_a_latents.shape[0] != hidden_states.shape[0]:
            if hidden_states.shape[0] % cond_a_latents.shape[0]:
                raise ValueError("Condition batch size cannot be expanded to match the denoiser batch size.")
            repeats = hidden_states.shape[0] // cond_a_latents.shape[0]
            cond_a_latents = cond_a_latents.repeat_interleave(repeats, dim=0)
            cond_b_latents = cond_b_latents.repeat_interleave(repeats, dim=0)

        if isinstance(self.adapter, DualImageFocusConditionAdapter):
            if focus_a is None or focus_b is None:
                raise ValueError("ab_focus adapter requires both focus_a and focus_b.")
            focus_a = focus_a.to(device=adapter_parameter.device, dtype=torch.float32)
            focus_b = focus_b.to(device=adapter_parameter.device, dtype=torch.float32)
            if focus_a.shape[0] != hidden_states.shape[0]:
                repeats = hidden_states.shape[0] // focus_a.shape[0]
                focus_a = focus_a.repeat_interleave(repeats, dim=0)
                focus_b = focus_b.repeat_interleave(repeats, dim=0)
            condition = self.adapter(cond_a_latents, cond_b_latents, focus_a, focus_b)
        else:
            condition = self.adapter(cond_a_latents, cond_b_latents)
        condition = condition.to(device=hidden_states.device, dtype=hidden_states.dtype)
        hidden_states = hidden_states + condition
        return self.transformer(hidden_states, *args, **kwargs)


@torch.no_grad()
def encode_condition_images(pipe, image_a, image_b, height: int, width: int, device: torch.device):
    cond_a = pipe.image_processor.preprocess(image_a, height=height, width=width).to(device, pipe.vae.dtype)
    cond_b = pipe.image_processor.preprocess(image_b, height=height, width=width).to(device, pipe.vae.dtype)
    cond_a_latents = encode_vae_latents(pipe.vae, cond_a)
    cond_b_latents = encode_vae_latents(pipe.vae, cond_b)
    return cond_a_latents, cond_b_latents
