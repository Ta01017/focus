from contextlib import contextmanager

import torch
from torch import nn

from sana_dof import DualImageConditionAdapter


class SanaControlNetDOFModel(nn.Module):
    """Training wrapper for a frozen SANA transformer, a focus ControlNet, and an A/B adapter."""

    def __init__(self, transformer, controlnet, adapter: DualImageConditionAdapter):
        super().__init__()
        self.transformer = transformer
        self.controlnet = controlnet
        self.adapter = adapter

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        encoder_attention_mask,
        timestep,
        cond_a_latents,
        cond_b_latents,
        controlnet_cond,
        conditioning_scale=1.0,
    ):
        controlnet_dtype = self.controlnet.dtype
        controlnet_samples = self.controlnet(
            hidden_states.to(controlnet_dtype),
            encoder_hidden_states=encoder_hidden_states.to(controlnet_dtype),
            encoder_attention_mask=encoder_attention_mask,
            timestep=timestep,
            controlnet_cond=controlnet_cond.to(controlnet_dtype),
            conditioning_scale=conditioning_scale,
            return_dict=False,
        )[0]
        adapter_parameter = next(self.adapter.parameters())
        image_condition = self.adapter(
            cond_a_latents.to(adapter_parameter.device, adapter_parameter.dtype),
            cond_b_latents.to(adapter_parameter.device, adapter_parameter.dtype),
        ).to(hidden_states.device, hidden_states.dtype)
        transformer_dtype = self.transformer.dtype
        return self.transformer(
            (hidden_states + image_condition).to(transformer_dtype),
            encoder_hidden_states=encoder_hidden_states.to(transformer_dtype),
            encoder_attention_mask=encoder_attention_mask,
            timestep=timestep,
            controlnet_block_samples=tuple(sample.to(transformer_dtype) for sample in controlnet_samples),
            return_dict=False,
        )[0]


class SanaControlNetInferenceTransformer(nn.Module):
    """Adds A/B latents to the ordinary SANA denoiser while its pipeline runs ControlNet normally."""

    def __init__(self, transformer, adapter: DualImageConditionAdapter):
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
    def use_condition(self, cond_a_latents, cond_b_latents):
        if self._condition is not None:
            raise RuntimeError("A SANA ControlNet A/B condition is already active.")
        self._condition = (cond_a_latents, cond_b_latents)
        try:
            yield
        finally:
            self._condition = None

    def forward(self, hidden_states, *args, **kwargs):
        if self._condition is None:
            return self.transformer(hidden_states, *args, **kwargs)
        cond_a_latents, cond_b_latents = self._condition
        if hidden_states.shape[0] % cond_a_latents.shape[0]:
            raise ValueError("The denoiser batch is not divisible by the A/B condition batch.")
        repeats = hidden_states.shape[0] // cond_a_latents.shape[0]
        if repeats > 1:
            cond_a_latents = cond_a_latents.repeat(repeats, 1, 1, 1)
            cond_b_latents = cond_b_latents.repeat(repeats, 1, 1, 1)
        adapter_parameter = next(self.adapter.parameters())
        condition = self.adapter(
            cond_a_latents.to(adapter_parameter.device, adapter_parameter.dtype),
            cond_b_latents.to(adapter_parameter.device, adapter_parameter.dtype),
        ).to(hidden_states.device, hidden_states.dtype)
        return self.transformer(hidden_states + condition, *args, **kwargs)
