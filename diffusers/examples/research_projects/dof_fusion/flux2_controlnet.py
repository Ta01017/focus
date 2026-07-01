from contextlib import contextmanager

import torch
from torch import nn

from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModelOutput


def parse_block_indices(value: str, upper_bound: int) -> tuple[int, ...]:
    indices = tuple(int(index.strip()) for index in value.split(",") if index.strip())
    if not indices:
        raise ValueError("At least one block index is required.")
    if len(set(indices)) != len(indices) or any(index < 0 or index >= upper_bound for index in indices):
        raise ValueError(f"Block indices must be unique and in [0, {upper_bound - 1}], got {indices}.")
    return indices


class Flux2FocusControlNet(nn.Module):
    """Lightweight ControlNet-style branch producing zero-initialized FLUX.2 block residuals."""

    def __init__(
        self,
        in_channels: int,
        inner_dim: int,
        hidden_channels: int,
        num_layers: int,
        double_block_indices: tuple[int, ...],
        single_block_indices: tuple[int, ...],
    ):
        super().__init__()
        self.in_channels = in_channels
        self.inner_dim = inner_dim
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.double_block_indices = double_block_indices
        self.single_block_indices = single_block_indices
        self.input_projection = nn.Linear(in_channels + 1, hidden_channels)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_channels),
                    nn.Linear(hidden_channels, hidden_channels * 2),
                    nn.SiLU(),
                    nn.Linear(hidden_channels * 2, hidden_channels),
                )
                for _ in range(num_layers)
            ]
        )
        self.double_residuals = nn.ModuleList(
            [nn.Linear(hidden_channels, inner_dim) for _ in double_block_indices]
        )
        self.single_residuals = nn.ModuleList(
            [nn.Linear(hidden_channels, inner_dim) for _ in single_block_indices]
        )
        for projection in [*self.double_residuals, *self.single_residuals]:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, hidden_states, focus_tokens, target_token_count, conditioning_scale=1.0):
        if focus_tokens.ndim != 3 or focus_tokens.shape[-1] != 1:
            raise ValueError("focus_tokens must have shape [batch, target_tokens, 1].")
        if focus_tokens.shape[1] != target_token_count:
            raise ValueError("focus token count does not match target_token_count.")
        if focus_tokens.shape[0] != hidden_states.shape[0]:
            if hidden_states.shape[0] % focus_tokens.shape[0]:
                raise ValueError("Focus-map batch cannot be expanded to the FLUX.2 batch.")
            focus_tokens = focus_tokens.repeat(hidden_states.shape[0] // focus_tokens.shape[0], 1, 1)
        focus_tokens = focus_tokens.to(hidden_states.device, torch.float32).clamp(0, 1)
        if hidden_states.shape[1] < target_token_count:
            raise ValueError("FLUX.2 hidden-state sequence is shorter than the focus-map sequence.")
        reference_tokens = hidden_states.shape[1] - target_token_count
        if reference_tokens:
            focus_tokens = torch.cat(
                [focus_tokens, focus_tokens.new_zeros(focus_tokens.shape[0], reference_tokens, 1)], dim=1
            )
        parameter = self.input_projection.weight
        control_input = torch.cat(
            [hidden_states.to(parameter.device, parameter.dtype), focus_tokens.to(parameter.device, parameter.dtype)],
            dim=-1,
        )
        control_states = self.input_projection(control_input)
        for block in self.blocks:
            control_states = control_states + block(control_states)
        token_mask = control_states.new_zeros(control_states.shape[0], control_states.shape[1], 1)
        token_mask[:, :target_token_count] = 1
        double_residuals = {
            index: projection(control_states) * token_mask * conditioning_scale
            for index, projection in zip(self.double_block_indices, self.double_residuals)
        }
        single_residuals = {
            index: projection(control_states) * token_mask * conditioning_scale
            for index, projection in zip(self.single_block_indices, self.single_residuals)
        }
        return double_residuals, single_residuals


class Flux2ControlNetTransformer(nn.Module):
    """External FLUX.2 wrapper injecting focus ControlNet residuals without editing Diffusers source."""

    def __init__(self, transformer, controlnet: Flux2FocusControlNet, conditioning_scale=1.0):
        super().__init__()
        self.transformer = transformer
        self.controlnet = controlnet
        self.conditioning_scale = conditioning_scale
        self._focus_condition = None

    @property
    def config(self):
        return self.transformer.config

    @property
    def dtype(self):
        return self.transformer.dtype

    def cache_context(self, *args, **kwargs):
        return self.transformer.cache_context(*args, **kwargs)

    @contextmanager
    def use_focus_condition(self, focus_tokens):
        if self._focus_condition is not None:
            raise RuntimeError("A FLUX.2 focus condition is already active.")
        self._focus_condition = focus_tokens
        try:
            yield
        finally:
            self._focus_condition = None

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        img_ids=None,
        txt_ids=None,
        guidance=None,
        joint_attention_kwargs=None,
        return_dict=True,
        kv_cache=None,
        kv_cache_mode=None,
        num_ref_tokens=0,
        ref_fixed_timestep=0.0,
        controlnet_cond=None,
        conditioning_scale=None,
    ):
        del kv_cache, num_ref_tokens, ref_fixed_timestep
        if kv_cache_mode is not None:
            raise ValueError("The external FLUX.2 focus ControlNet does not support KV-cache mode.")
        focus_tokens = self._focus_condition if controlnet_cond is None else controlnet_cond
        if focus_tokens is None:
            return self.transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
                joint_attention_kwargs=joint_attention_kwargs,
                return_dict=return_dict,
            )

        target_token_count = focus_tokens.shape[1]
        scale = self.conditioning_scale if conditioning_scale is None else conditioning_scale
        double_residuals, single_residuals = self.controlnet(
            hidden_states, focus_tokens, target_token_count, conditioning_scale=scale
        )
        num_txt_tokens = encoder_hidden_states.shape[1]
        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        temb = self.transformer.time_guidance_embed(timestep, guidance)
        double_mod_img = self.transformer.double_stream_modulation_img(temb)
        double_mod_txt = self.transformer.double_stream_modulation_txt(temb)
        single_mod = self.transformer.single_stream_modulation(temb)
        hidden_states = self.transformer.x_embedder(hidden_states)
        encoder_hidden_states = self.transformer.context_embedder(encoder_hidden_states)
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        image_rotary_emb = self.transformer.pos_embed(img_ids)
        text_rotary_emb = self.transformer.pos_embed(txt_ids)
        rotary_emb = (
            torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
            torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
        )
        for index, block in enumerate(self.transformer.transformer_blocks):
            if index in double_residuals:
                hidden_states = hidden_states + double_residuals[index].to(hidden_states.dtype)
            if torch.is_grad_enabled() and self.transformer.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self.transformer._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    double_mod_img,
                    double_mod_txt,
                    rotary_emb,
                    joint_attention_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb_mod_img=double_mod_img,
                    temb_mod_txt=double_mod_txt,
                    image_rotary_emb=rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        for index, block in enumerate(self.transformer.single_transformer_blocks):
            if index in single_residuals:
                residual = single_residuals[index].to(hidden_states.dtype)
                residual = torch.cat(
                    [residual.new_zeros(residual.shape[0], num_txt_tokens, residual.shape[2]), residual], dim=1
                )
                hidden_states = hidden_states + residual
            if torch.is_grad_enabled() and self.transformer.gradient_checkpointing:
                hidden_states = self.transformer._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    None,
                    single_mod,
                    rotary_emb,
                    joint_attention_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=None,
                    temb_mod=single_mod,
                    image_rotary_emb=rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )
        hidden_states = hidden_states[:, num_txt_tokens:, ...]
        hidden_states = self.transformer.norm_out(hidden_states, temb)
        output = self.transformer.proj_out(hidden_states)
        if not return_dict:
            return (output,)
        return Flux2Transformer2DModelOutput(sample=output)


def focus_map_to_tokens(focus_map, height, width, device):
    import numpy as np
    from PIL import Image

    if isinstance(focus_map, (str, bytes)):
        focus_map = Image.open(focus_map)
    if isinstance(focus_map, Image.Image):
        focus_map = focus_map.convert("L").resize((width, height), Image.Resampling.BILINEAR)
        focus_map = torch.from_numpy(np.asarray(focus_map, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    if focus_map.ndim != 4:
        raise ValueError("focus_map must resolve to a [batch, 1, height, width] tensor.")
    focus_map = torch.nn.functional.interpolate(
        focus_map.to(device=device, dtype=torch.float32), size=(height, width), mode="bilinear", align_corners=False
    )
    return focus_map.flatten(2).transpose(1, 2).contiguous()
