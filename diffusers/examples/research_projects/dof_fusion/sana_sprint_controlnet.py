from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F

from diffusers import SanaControlNetModel

from sana_dof import DualImageConditionAdapter


def initialize_controlnet_from_transformer(transformer, num_layers: int) -> SanaControlNetModel:
    if num_layers < 1 or num_layers > len(transformer.transformer_blocks):
        raise ValueError(
            f"num_layers must be between 1 and {len(transformer.transformer_blocks)}, got {num_layers}."
        )
    config = transformer.config
    controlnet = SanaControlNetModel(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        num_attention_heads=config.num_attention_heads,
        attention_head_dim=config.attention_head_dim,
        num_layers=num_layers,
        num_cross_attention_heads=config.num_cross_attention_heads,
        cross_attention_head_dim=config.cross_attention_head_dim,
        cross_attention_dim=config.cross_attention_dim,
        caption_channels=config.caption_channels,
        mlp_ratio=config.mlp_ratio,
        dropout=config.dropout,
        attention_bias=config.attention_bias,
        sample_size=config.sample_size,
        patch_size=config.patch_size,
        norm_elementwise_affine=config.norm_elementwise_affine,
        norm_eps=config.norm_eps,
        interpolation_scale=config.interpolation_scale,
    )
    controlnet.patch_embed.load_state_dict(transformer.patch_embed.state_dict())
    controlnet.time_embed.load_state_dict(transformer.time_embed.state_dict(), strict=False)
    controlnet.caption_projection.load_state_dict(transformer.caption_projection.state_dict())
    controlnet.caption_norm.load_state_dict(transformer.caption_norm.state_dict())
    for control_block, transformer_block in zip(controlnet.transformer_blocks, transformer.transformer_blocks):
        control_block.load_state_dict(transformer_block.state_dict(), strict=False)
    return controlnet


class SanaSprintFocusControlNetTransformer(nn.Module):
    """Injects A/B latent conditions and a focus-map ControlNet into a frozen SANA-Sprint transformer."""

    def __init__(
        self,
        transformer: nn.Module,
        controlnet: SanaControlNetModel,
        adapter: DualImageConditionAdapter,
        conditioning_scale: float = 1.0,
    ):
        super().__init__()
        self.transformer = transformer
        self.controlnet = controlnet
        self.adapter = adapter
        self.conditioning_scale = conditioning_scale
        self._conditions = None

    @property
    def config(self):
        return self.transformer.config

    @property
    def dtype(self):
        return self.transformer.dtype

    @contextmanager
    def use_conditions(
        self,
        cond_a_latents: torch.Tensor,
        cond_b_latents: torch.Tensor,
        focus_map: torch.Tensor,
    ):
        if self._conditions is not None:
            raise RuntimeError("SANA-Sprint ControlNet conditions are already active.")
        self._conditions = (cond_a_latents, cond_b_latents, focus_map)
        try:
            yield
        finally:
            self._conditions = None

    @staticmethod
    def _expand_batch(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        if tensor.shape[0] == batch_size:
            return tensor
        if batch_size % tensor.shape[0]:
            raise ValueError(f"Cannot expand condition batch {tensor.shape[0]} to denoiser batch {batch_size}.")
        return tensor.repeat_interleave(batch_size // tensor.shape[0], dim=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        attention_kwargs: dict | None = None,
        cond_a_latents: torch.Tensor | None = None,
        cond_b_latents: torch.Tensor | None = None,
        focus_map: torch.Tensor | None = None,
        conditioning_scale: float | None = None,
        return_dict: bool = True,
    ):
        explicit_conditions = (cond_a_latents, cond_b_latents, focus_map)
        if all(condition is None for condition in explicit_conditions) and self._conditions is not None:
            cond_a_latents, cond_b_latents, focus_map = self._conditions
        elif any(condition is None for condition in explicit_conditions):
            raise ValueError("cond_a_latents, cond_b_latents, and focus_map must be provided together.")

        if cond_a_latents is None:
            return self.transformer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                guidance=guidance,
                encoder_attention_mask=encoder_attention_mask,
                attention_mask=attention_mask,
                attention_kwargs=attention_kwargs,
                return_dict=return_dict,
            )

        batch_size = hidden_states.shape[0]
        adapter_parameter = next(self.adapter.parameters())
        cond_a_latents = self._expand_batch(cond_a_latents, batch_size).to(
            adapter_parameter.device, adapter_parameter.dtype
        )
        cond_b_latents = self._expand_batch(cond_b_latents, batch_size).to(
            adapter_parameter.device, adapter_parameter.dtype
        )
        if cond_a_latents.shape[-2:] != hidden_states.shape[-2:]:
            cond_a_latents = F.interpolate(
                cond_a_latents, size=hidden_states.shape[-2:], mode="bilinear", align_corners=False
            )
            cond_b_latents = F.interpolate(
                cond_b_latents, size=hidden_states.shape[-2:], mode="bilinear", align_corners=False
            )
        image_condition = self.adapter(cond_a_latents, cond_b_latents).to(
            hidden_states.device, hidden_states.dtype
        )
        conditioned_hidden_states = hidden_states + image_condition

        focus_map = self._expand_batch(focus_map, batch_size).to(hidden_states.device, torch.float32)
        if focus_map.shape[1] != 1:
            focus_map = focus_map.mean(dim=1, keepdim=True)
        focus_map = F.interpolate(
            focus_map, size=hidden_states.shape[-2:], mode="bilinear", align_corners=False
        ).clamp(0, 1)
        controlnet_condition = focus_map.repeat(1, self.controlnet.config.in_channels, 1, 1)
        scale = self.conditioning_scale if conditioning_scale is None else conditioning_scale
        controlnet_dtype = self.controlnet.dtype
        controlnet_samples = self.controlnet(
            conditioned_hidden_states.to(controlnet_dtype),
            encoder_hidden_states=encoder_hidden_states.to(controlnet_dtype),
            timestep=timestep,
            controlnet_cond=controlnet_condition.to(controlnet_dtype),
            conditioning_scale=scale,
            encoder_attention_mask=encoder_attention_mask,
            attention_mask=attention_mask,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]
        controlnet_samples = tuple(sample.to(hidden_states.dtype) for sample in controlnet_samples)
        return self.transformer(
            conditioned_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            guidance=guidance,
            encoder_attention_mask=encoder_attention_mask,
            attention_mask=attention_mask,
            attention_kwargs=attention_kwargs,
            controlnet_block_samples=controlnet_samples,
            return_dict=return_dict,
        )
