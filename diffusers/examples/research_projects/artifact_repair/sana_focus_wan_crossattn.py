import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from peft.utils import get_peft_model_state_dict
from safetensors.torch import load_file, save_file
from torch import nn

from diffusers import SanaPipeline
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from sana_artifact_repair_channel_concat import DEFAULT_LORA_TARGET_MODULES, get_sana_output_channels, get_sana_patch_embedding, tensor_stats


# This implementation is derived from the existing artifact-repair Wan-style
# latent-concat and image cross-attention route.
ROUTE = "focus_wan_crossattn"
IMPLEMENTATION = "sana_focus_wan_crossattn_v1"
CONDITION_MODES = ("single", "dual")


def num_condition_images(condition_mode):
    if condition_mode == "single":
        return 1
    if condition_mode == "dual":
        return 2
    raise ValueError(f"Unsupported condition_mode={condition_mode!r}.")


def focus_prompt(condition_mode):
    if condition_mode == "single":
        return "Restore an all-in-focus photograph from Image A while preserving geometry, viewpoint, color, and valid sharp details."
    return "Fuse Image A and Image B into one all-in-focus photograph aligned to Image A. Preserve Image A geometry and use Image B only for complementary sharp details."


def encode_vae_latents(vae, pixel_values, sample=True, generator=None):
    encoded = vae.encode(pixel_values)
    if hasattr(encoded, "latent"):
        latents = encoded.latent
    elif hasattr(encoded, "latents"):
        latents = encoded.latents
    elif hasattr(encoded, "latent_dist"):
        latents = encoded.latent_dist.sample(generator=generator) if sample else encoded.latent_dist.mode()
    else:
        raise TypeError(f"Unsupported VAE encode output type: {type(encoded).__name__}.")
    return latents * getattr(vae.config, "scaling_factor", 1.0)


def decode_vae_latents(vae, latents):
    decoded = vae.decode(latents / getattr(vae.config, "scaling_factor", 1.0))
    if hasattr(decoded, "sample"):
        return decoded.sample
    if isinstance(decoded, (tuple, list)):
        return decoded[0]
    return decoded


def expand_sana_patch_embedding_for_focus_wan(transformer, condition_mode):
    patch_embed, old_proj = get_sana_patch_embedding(transformer)
    input_channels = int(old_proj.in_channels)
    ncond = num_condition_images(condition_mode)
    if hasattr(transformer, "focus_wan_original_latent_channels"):
        original_c = int(transformer.focus_wan_original_latent_channels)
        expected = (1 + ncond) * original_c
        if input_channels != expected:
            raise ValueError(f"Existing focus WAN patch input channels={input_channels}, expected {expected} for {condition_mode}.")
        return original_c
    config_c = int(getattr(transformer.config, "in_channels", input_channels))
    if input_channels == (1 + ncond) * config_c:
        transformer.focus_wan_original_latent_channels = config_c
        return config_c
    if input_channels != config_c:
        raise ValueError(f"Cannot infer original SANA latent channels from patch input={input_channels}, config={config_c}.")
    new_proj = nn.Conv2d(
        in_channels=(1 + ncond) * input_channels,
        out_channels=old_proj.out_channels,
        kernel_size=old_proj.kernel_size,
        stride=old_proj.stride,
        padding=old_proj.padding,
        dilation=old_proj.dilation,
        groups=old_proj.groups,
        bias=old_proj.bias is not None,
        padding_mode=old_proj.padding_mode,
        device=old_proj.weight.device,
        dtype=old_proj.weight.dtype,
    )
    with torch.no_grad():
        new_proj.weight.zero_()
        new_proj.weight[:, :input_channels].copy_(old_proj.weight)
        if old_proj.bias is not None:
            new_proj.bias.copy_(old_proj.bias)
        copied_diff = float((new_proj.weight[:, :input_channels] - old_proj.weight).detach().float().abs().max().cpu())
        original_norm = float(old_proj.weight.detach().float().norm().cpu())
        a_norm = float(new_proj.weight[:, input_channels:2 * input_channels].detach().float().norm().cpu())
        b_norm = 0.0
        if ncond == 2:
            b_norm = float(new_proj.weight[:, 2 * input_channels:3 * input_channels].detach().float().norm().cpu())
    patch_embed.proj = new_proj
    if hasattr(transformer, "register_to_config"):
        transformer.register_to_config(in_channels=(1 + ncond) * input_channels)
    else:
        transformer.config.in_channels = (1 + ncond) * input_channels
    transformer.focus_wan_original_latent_channels = input_channels
    print(f"[FOCUS_WAN] original patch weight norm={original_norm:.6e}", flush=True)
    print(f"[FOCUS_WAN] target-channel copied weight difference={copied_diff:.6e}", flush=True)
    print(f"[FOCUS_WAN] A extra-channel weight norm={a_norm:.6e}", flush=True)
    print(f"[FOCUS_WAN] B extra-channel weight norm={b_norm:.6e}", flush=True)
    print(f"[FOCUS_WAN] expanded patch embedding input channels={(1+ncond)*input_channels}", flush=True)
    return input_channels


class FocusWanImageProjector(nn.Module):
    def __init__(self, image_hidden_size, cross_attention_dim, hidden_size=None):
        super().__init__()
        hidden_size = int(hidden_size or cross_attention_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(image_hidden_size),
            nn.Linear(image_hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, cross_attention_dim),
            nn.LayerNorm(cross_attention_dim),
        )

    def forward(self, image_hidden_states):
        return self.net(image_hidden_states)


def _clone_norm(norm):
    return nn.Identity() if norm is None else copy.deepcopy(norm)


class FocusWanImageCrossAttentionBranch(nn.Module):
    def __init__(self, attn2, image_gate_init=1e-3):
        super().__init__()
        self.to_k_img = nn.Linear(attn2.to_k.in_features, attn2.to_k.out_features, bias=attn2.to_k.bias is not None)
        self.to_v_img = nn.Linear(attn2.to_v.in_features, attn2.to_v.out_features, bias=attn2.to_v.bias is not None)
        if tuple(self.to_k_img.weight.shape) != tuple(attn2.to_k.weight.shape):
            raise ValueError("to_k_img and text to_k shape mismatch.")
        if tuple(self.to_v_img.weight.shape) != tuple(attn2.to_v.weight.shape):
            raise ValueError("to_v_img and text to_v shape mismatch.")
        with torch.no_grad():
            self.to_k_img.weight.copy_(attn2.to_k.weight)
            self.to_v_img.weight.copy_(attn2.to_v.weight)
            if self.to_k_img.bias is not None:
                self.to_k_img.bias.copy_(attn2.to_k.bias)
            if self.to_v_img.bias is not None:
                self.to_v_img.bias.copy_(attn2.to_v.bias)
        self.norm_k_img = _clone_norm(getattr(attn2, "norm_k", None))
        self.image_gate = nn.Parameter(torch.tensor(float(image_gate_init)))
        self.last_image_output = None


class FocusWanCrossAttnProcessor:
    def __init__(self, branches):
        self.branches = branches
        self.image_context_lengths = []
        self.image_cross_attention_scales = []
        self.disable_image_cross_attention = False

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        image_lengths = [int(x) for x in self.image_context_lengths]
        total_image = sum(image_lengths)
        if encoder_hidden_states is None or total_image <= 0:
            text_states = encoder_hidden_states
            image_states = []
        else:
            if encoder_hidden_states.shape[1] < total_image:
                raise ValueError(f"Context length {encoder_hidden_states.shape[1]} smaller than image token length {total_image}.")
            text_len = encoder_hidden_states.shape[1] - total_image
            text_states = encoder_hidden_states[:, :text_len]
            image_states = []
            offset = text_len
            for length in image_lengths:
                image_states.append(encoder_hidden_states[:, offset:offset + length])
                offset += length

        batch_size = hidden_states.shape[0]
        text_len = hidden_states.shape[1] if text_states is None else text_states.shape[1]
        text_mask = attention_mask
        if attention_mask is not None and total_image > 0 and attention_mask.shape[-1] == encoder_hidden_states.shape[1]:
            text_mask = attention_mask[..., :text_len]
        if text_mask is not None:
            text_mask = attn.prepare_attention_mask(text_mask, text_len, batch_size)
            text_mask = text_mask.view(batch_size, attn.heads, -1, text_mask.shape[-1])

        query = attn.to_q(hidden_states)
        if text_states is None:
            text_states = hidden_states
        key_text = attn.to_k(text_states)
        value_text = attn.to_v(text_states)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key_text = attn.norm_k(key_text)
        inner_dim = key_text.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key_text = key_text.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value_text = value_text.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        hidden = F.scaled_dot_product_attention(query, key_text, value_text, attn_mask=text_mask, dropout_p=0.0, is_causal=False)

        if image_states and not self.disable_image_cross_attention:
            for branch_index, (branch, states) in enumerate(zip(self.branches, image_states)):
                key_img = branch.norm_k_img(branch.to_k_img(states))
                value_img = branch.to_v_img(states)
                key_img = key_img.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                value_img = value_img.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                out_img = F.scaled_dot_product_attention(query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False)
                branch.last_image_output = out_img.detach()
                scale = float(self.image_cross_attention_scales[branch_index]) if branch_index < len(self.image_cross_attention_scales) else 1.0
                hidden = hidden + scale * branch.image_gate.to(out_img.dtype) * out_img

        hidden = hidden.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden = hidden.to(query.dtype)
        hidden = attn.to_out[0](hidden)
        hidden = attn.to_out[1](hidden)
        hidden = hidden / attn.rescale_output_factor
        return hidden


class FocusWanImageCrossAttentionAdapter(nn.Module):
    def __init__(self, transformer, number_of_image_branches, image_gate_init=1e-3):
        super().__init__()
        self.number_of_image_branches = int(number_of_image_branches)
        self.layers = nn.ModuleList()
        self.processors = []
        for index, block in enumerate(transformer.transformer_blocks):
            attn2 = getattr(block, "attn2", None)
            if attn2 is None:
                raise ValueError(f"SANA block {index} has no attn2 cross-attention.")
            branches = nn.ModuleList([FocusWanImageCrossAttentionBranch(attn2, image_gate_init) for _ in range(self.number_of_image_branches)])
            processor = FocusWanCrossAttnProcessor(branches)
            attn2.set_processor(processor)
            self.layers.append(branches)
            self.processors.append(processor)

    def set_runtime(self, image_context_lengths, image_cross_attention_scales=None, disable_image_cross_attention=False):
        if len(image_context_lengths) != self.number_of_image_branches:
            raise ValueError(f"Expected {self.number_of_image_branches} image lengths, got {len(image_context_lengths)}.")
        if image_cross_attention_scales is None:
            image_cross_attention_scales = [1.0] * self.number_of_image_branches
        for processor in self.processors:
            processor.image_context_lengths = [int(x) for x in image_context_lengths]
            processor.image_cross_attention_scales = [float(x) for x in image_cross_attention_scales]
            processor.disable_image_cross_attention = bool(disable_image_cross_attention)

    def gate_values(self):
        return [[float(branch.image_gate.detach().float().cpu()) for branch in branches] for branches in self.layers]


class FocusWanSanaModel(nn.Module):
    def __init__(self, transformer, image_projectors, image_cross_attention_adapter, condition_mode):
        super().__init__()
        self.transformer = transformer
        self.image_projectors = nn.ModuleList(image_projectors)
        self.image_cross_attention_adapter = image_cross_attention_adapter
        self.condition_mode = condition_mode
        self.last_debug = {}

    @property
    def config(self):
        return self.transformer.config

    @property
    def dtype(self):
        return next(self.transformer.parameters()).dtype

    def forward(self, hidden_states, encoder_hidden_states, timestep, guidance=None, encoder_attention_mask=None, attention_mask=None,
                encoder_hidden_states_images=None, image_cross_attention_scales=None, disable_image_cross_attention=False,
                attention_kwargs=None, controlnet_block_samples=None, return_dict=True):
        transformer = self.transformer
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
        batch_size, _, height, width = hidden_states.shape
        p = transformer.config.patch_size
        post_patch_height, post_patch_width = height // p, width // p
        hidden_states = transformer.patch_embed(hidden_states)
        if guidance is not None:
            timestep, embedded_timestep = transformer.time_embed(timestep, guidance=guidance, hidden_dtype=hidden_states.dtype)
        else:
            timestep, embedded_timestep = transformer.time_embed(timestep, batch_size=batch_size, hidden_dtype=hidden_states.dtype)
        text_states = transformer.caption_projection(encoder_hidden_states)
        text_states = text_states.view(batch_size, -1, hidden_states.shape[-1])
        text_states = transformer.caption_norm(text_states)
        image_lengths = []
        projected = []
        if encoder_hidden_states_images is not None:
            if len(encoder_hidden_states_images) != len(self.image_projectors):
                raise ValueError(f"Expected {len(self.image_projectors)} image token tensors, got {len(encoder_hidden_states_images)}.")
            for projector, image_tokens in zip(self.image_projectors, encoder_hidden_states_images):
                image_states = projector(image_tokens.to(dtype=hidden_states.dtype))
                projected.append(image_states)
                image_lengths.append(image_states.shape[1])
            encoder_hidden_states = torch.cat([text_states, *projected], dim=1)
            if encoder_attention_mask is not None:
                zeros = torch.zeros(encoder_attention_mask.shape[0], encoder_attention_mask.shape[1], sum(image_lengths), device=encoder_attention_mask.device, dtype=encoder_attention_mask.dtype)
                encoder_attention_mask = torch.cat([encoder_attention_mask, zeros], dim=-1)
        else:
            encoder_hidden_states = text_states
        self.image_cross_attention_adapter.set_runtime(image_lengths, image_cross_attention_scales, disable_image_cross_attention)
        self.last_debug = {"projected_image_stats": [tensor_stats(x) for x in projected], "image_lengths": image_lengths}
        if torch.is_grad_enabled() and transformer.gradient_checkpointing:
            for index_block, block in enumerate(transformer.transformer_blocks):
                hidden_states = transformer._gradient_checkpointing_func(block, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, timestep, post_patch_height, post_patch_width)
                if controlnet_block_samples is not None and 0 < index_block <= len(controlnet_block_samples):
                    hidden_states = hidden_states + controlnet_block_samples[index_block - 1]
        else:
            for index_block, block in enumerate(transformer.transformer_blocks):
                hidden_states = block(hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, timestep, post_patch_height, post_patch_width)
                if controlnet_block_samples is not None and 0 < index_block <= len(controlnet_block_samples):
                    hidden_states = hidden_states + controlnet_block_samples[index_block - 1]
        hidden_states = transformer.norm_out(hidden_states, embedded_timestep, transformer.scale_shift_table)
        hidden_states = transformer.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch_size, post_patch_height, post_patch_width, transformer.config.patch_size, transformer.config.patch_size, -1)
        hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
        output = hidden_states.reshape(batch_size, -1, post_patch_height * p, post_patch_width * p)
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


def load_image_encoder_and_processor(args, dtype=None, device=None):
    from transformers import AutoImageProcessor, CLIPVisionModel
    kwargs = {"revision": getattr(args, "image_encoder_revision", None), "local_files_only": getattr(args, "image_encoder_local_files_only", False)}
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    subfolder = getattr(args, "image_encoder_subfolder", None) or None
    processor = AutoImageProcessor.from_pretrained(args.image_encoder_model, subfolder=subfolder, **kwargs)
    image_encoder = CLIPVisionModel.from_pretrained(args.image_encoder_model, subfolder=subfolder, torch_dtype=dtype, **kwargs)
    image_encoder.requires_grad_(False).eval()
    if device is not None:
        image_encoder.to(device, dtype=dtype)
    return image_encoder, processor


@torch.no_grad()
def encode_image_tokens(image_encoder, image_processor, images, device, dtype):
    inputs = image_processor(images=images, return_tensors="pt")
    inputs = {key: value.to(device=device) for key, value in inputs.items()}
    return image_encoder(**inputs).last_hidden_state.to(dtype=dtype)


def build_focus_wan_model(transformer, condition_mode, image_encoder_hidden_size, image_gate_init=1e-3, share_image_projector=False):
    branches = num_condition_images(condition_mode)
    cross_dim = int(getattr(transformer.config, "cross_attention_dim"))
    if share_image_projector:
        shared = FocusWanImageProjector(image_encoder_hidden_size, cross_dim)
        projectors = [shared] if branches == 1 else [shared, shared]
    else:
        projectors = [FocusWanImageProjector(image_encoder_hidden_size, cross_dim) for _ in range(branches)]
    adapter = FocusWanImageCrossAttentionAdapter(transformer, branches, image_gate_init)
    return FocusWanSanaModel(transformer, projectors, adapter, condition_mode)


def focus_wan_condition_state_dict(model):
    patch_embed, _ = get_sana_patch_embedding(model.transformer)
    state = {f"patch_embed.{k}": v.detach().cpu() for k, v in patch_embed.state_dict().items()}
    for i, projector in enumerate(model.image_projectors):
        for k, v in projector.state_dict().items():
            state[f"image_projectors.{i}.{k}"] = v.detach().cpu()
    for k, v in model.image_cross_attention_adapter.state_dict().items():
        state[f"image_cross_attention_adapter.{k}"] = v.detach().cpu()
    return state


def load_focus_wan_condition_state_dict(model, path):
    state = load_file(path)
    patch_state = {k.removeprefix("patch_embed."): v for k, v in state.items() if k.startswith("patch_embed.")}
    missing, unexpected = get_sana_patch_embedding(model.transformer)[0].load_state_dict(patch_state, strict=True)
    if missing or unexpected:
        raise ValueError(f"Patch embedding strict load failed: missing={missing}, unexpected={unexpected}")
    for i, projector in enumerate(model.image_projectors):
        prefix = f"image_projectors.{i}."
        p_state = {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}
        missing, unexpected = projector.load_state_dict(p_state, strict=True)
        if missing or unexpected:
            raise ValueError(f"Projector {i} strict load failed: missing={missing}, unexpected={unexpected}")
    prefix = "image_cross_attention_adapter."
    a_state = {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}
    missing, unexpected = model.image_cross_attention_adapter.load_state_dict(a_state, strict=True)
    if missing or unexpected:
        raise ValueError(f"Image cross-attention strict load failed: missing={missing}, unexpected={unexpected}")


def save_focus_wan_checkpoint(directory, model, args, global_step, latent_channels, image_encoder_hidden_size):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_file(focus_wan_condition_state_dict(model), directory / "focus_wan_condition.safetensors")
    if getattr(args, "train_transformer_lora", True):
        SanaPipeline.save_lora_weights(directory / "transformer_lora", transformer_lora_layers=get_peft_model_state_dict(model.transformer))
    patch_embed, proj = get_sana_patch_embedding(model.transformer)
    config = {
        "route": ROUTE,
        "implementation": IMPLEMENTATION,
        "condition_mode": args.condition_mode,
        "latent_channels": int(latent_channels),
        "patch_embed_in_channels": int(proj.in_channels),
        "number_of_image_branches": num_condition_images(args.condition_mode),
        "share_image_projector": bool(args.share_image_projector),
        "image_encoder": args.image_encoder_model,
        "image_encoder_hidden_size": int(image_encoder_hidden_size),
        "image_gate_init": float(args.image_gate_init),
        "image_cross_attention_scale_a": float(args.image_cross_attention_scale_a),
        "image_cross_attention_scale_b": float(getattr(args, "image_cross_attention_scale_b", 1.0)),
        "lora_scope": args.lora_scope,
        "lora_rank": int(args.lora_rank),
        "lora_alpha": int(args.lora_alpha),
        "base_model": args.model,
        "max_pixels": int(args.max_pixels),
        "size_divisor": int(args.size_divisor),
        "global_step": int(global_step),
    }
    (directory / "focus_wan_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (directory / "trainer_state.json").write_text(json.dumps({"global_step": int(global_step)}, indent=2), encoding="utf-8")
    return config


def load_focus_wan_config(checkpoint):
    checkpoint = Path(checkpoint)
    for path in (checkpoint / "focus_wan_config.json", checkpoint.parent / "focus_wan_config.json"):
        if path.exists():
            config = json.loads(path.read_text(encoding="utf-8"))
            if config.get("route") != ROUTE:
                raise ValueError(f"Not a Focus WAN checkpoint: {path}")
            return config, path.parent
    raise FileNotFoundError(f"focus_wan_config.json not found in {checkpoint} or parent.")


def validate_focus_wan_checkpoint(config, condition_mode, transformer):
    expected_branches = num_condition_images(condition_mode)
    if config.get("condition_mode") != condition_mode:
        raise ValueError(f"checkpoint condition_mode={config.get('condition_mode')} is incompatible with requested condition_mode={condition_mode}")
    if int(config.get("number_of_image_branches")) != expected_branches:
        raise ValueError("number_of_image_branches mismatch.")
    _, proj = get_sana_patch_embedding(transformer)
    if int(config.get("patch_embed_in_channels")) != int(proj.in_channels):
        raise ValueError(f"patch_embed_in_channels mismatch: ckpt={config.get('patch_embed_in_channels')} model={proj.in_channels}")
    if int(config.get("latent_channels")) * (1 + expected_branches) != int(proj.in_channels):
        raise ValueError("latent_channels and patch_embed_in_channels are inconsistent.")
