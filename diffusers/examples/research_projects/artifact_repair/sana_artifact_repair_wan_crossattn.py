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

from sana_artifact_repair_channel_concat import (
    CONCAT_ORDER,
    DEFAULT_LORA_TARGET_MODULES,
    encode_vae_latents,
    expand_sana_patch_embedding_for_channel_concat,
    get_sana_output_channels,
    get_sana_patch_embedding,
    load_route2_config,
    load_route2_patch_embedding,
    route2_full_transformer_path,
    route2_lora_path,
    tensor_stats,
)


IMPLEMENTATION = "sana_artifact_repair_wan_crossattn_v1"


class Route3ImageProjector(nn.Module):
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


def _clone_norm_k(norm_k):
    if norm_k is None:
        return nn.Identity()
    return copy.deepcopy(norm_k)


class Route3ImageCrossAttentionLayer(nn.Module):
    def __init__(self, attn2, image_gate_init=1e-3):
        super().__init__()
        self.to_k_img = nn.Linear(
            attn2.to_k.in_features,
            attn2.to_k.out_features,
            bias=attn2.to_k.bias is not None,
        )
        self.to_v_img = nn.Linear(
            attn2.to_v.in_features,
            attn2.to_v.out_features,
            bias=attn2.to_v.bias is not None,
        )
        if tuple(self.to_k_img.weight.shape) != tuple(attn2.to_k.weight.shape):
            raise ValueError("to_k_img and attn2.to_k shape mismatch; refusing unsafe initialization.")
        if tuple(self.to_v_img.weight.shape) != tuple(attn2.to_v.weight.shape):
            raise ValueError("to_v_img and attn2.to_v shape mismatch; refusing unsafe initialization.")
        with torch.no_grad():
            self.to_k_img.weight.copy_(attn2.to_k.weight)
            self.to_v_img.weight.copy_(attn2.to_v.weight)
            if self.to_k_img.bias is not None:
                self.to_k_img.bias.copy_(attn2.to_k.bias)
            if self.to_v_img.bias is not None:
                self.to_v_img.bias.copy_(attn2.to_v.bias)
        self.norm_k_img = _clone_norm_k(getattr(attn2, "norm_k", None))
        self.image_gate = nn.Parameter(torch.tensor(float(image_gate_init)))


class Route3WanCrossAttnProcessor:
    def __init__(self, image_layer):
        self.image_layer = image_layer
        self.image_context_length = 0
        self.image_cross_attention_scale = 1.0
        self.disable_image_cross_attention = False

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        if encoder_hidden_states is None or self.image_context_length <= 0:
            text_states = encoder_hidden_states
            image_states = None
        else:
            image_len = int(self.image_context_length)
            if encoder_hidden_states.shape[1] < image_len:
                raise ValueError(
                    f"encoder_hidden_states length {encoder_hidden_states.shape[1]} is smaller than "
                    f"image_context_length {image_len}."
                )
            text_states = encoder_hidden_states[:, :-image_len]
            image_states = encoder_hidden_states[:, -image_len:]

        batch_size, text_len, _ = hidden_states.shape if text_states is None else text_states.shape
        query_batch = hidden_states.shape[0]
        if batch_size != query_batch:
            raise ValueError(f"Batch mismatch in Route3 attention: query={query_batch}, context={batch_size}")

        text_mask = attention_mask
        if attention_mask is not None and image_states is not None and attention_mask.shape[-1] == encoder_hidden_states.shape[1]:
            text_mask = attention_mask[..., : text_states.shape[1]]
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

        out_text = F.scaled_dot_product_attention(
            query, key_text, value_text, attn_mask=text_mask, dropout_p=0.0, is_causal=False
        )
        hidden = out_text

        if image_states is not None and not self.disable_image_cross_attention:
            key_img = self.image_layer.to_k_img(image_states)
            value_img = self.image_layer.to_v_img(image_states)
            key_img = self.image_layer.norm_k_img(key_img)
            key_img = key_img.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value_img = value_img.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            out_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            gate = self.image_layer.image_gate.to(dtype=out_img.dtype)
            hidden = hidden + float(self.image_cross_attention_scale) * gate * out_img

        hidden = hidden.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden = hidden.to(query.dtype)
        hidden = attn.to_out[0](hidden)
        hidden = attn.to_out[1](hidden)
        hidden = hidden / attn.rescale_output_factor
        return hidden


class Route3ImageCrossAttentionAdapter(nn.Module):
    def __init__(self, transformer, image_gate_init=1e-3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.processors = []
        for index, block in enumerate(transformer.transformer_blocks):
            attn2 = getattr(block, "attn2", None)
            if attn2 is None:
                raise ValueError(f"SANA block {index} has no attn2 cross-attention.")
            layer = Route3ImageCrossAttentionLayer(attn2, image_gate_init=image_gate_init)
            processor = Route3WanCrossAttnProcessor(layer)
            attn2.set_processor(processor)
            self.layers.append(layer)
            self.processors.append(processor)

    def set_runtime(self, image_context_length, image_cross_attention_scale=1.0, disable_image_cross_attention=False):
        for processor in self.processors:
            processor.image_context_length = int(image_context_length or 0)
            processor.image_cross_attention_scale = float(image_cross_attention_scale)
            processor.disable_image_cross_attention = bool(disable_image_cross_attention)

    @property
    def image_context_length(self):
        return self.processors[0].image_context_length if self.processors else 0


class Route3SanaModel(nn.Module):
    def __init__(self, transformer, image_projector, image_cross_attention_adapter):
        super().__init__()
        self.transformer = transformer
        self.image_projector = image_projector
        self.image_cross_attention_adapter = image_cross_attention_adapter

    @property
    def config(self):
        return self.transformer.config

    @property
    def dtype(self):
        return next(self.transformer.parameters()).dtype

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        guidance=None,
        encoder_attention_mask=None,
        attention_mask=None,
        encoder_hidden_states_image=None,
        image_attention_mask=None,
        image_cross_attention_scale=1.0,
        disable_image_cross_attention=False,
        attention_kwargs=None,
        controlnet_block_samples=None,
        return_dict=True,
    ):
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
            timestep, embedded_timestep = transformer.time_embed(
                timestep, guidance=guidance, hidden_dtype=hidden_states.dtype
            )
        else:
            timestep, embedded_timestep = transformer.time_embed(
                timestep, batch_size=batch_size, hidden_dtype=hidden_states.dtype
            )

        text_states = transformer.caption_projection(encoder_hidden_states)
        text_states = text_states.view(batch_size, -1, hidden_states.shape[-1])
        text_states = transformer.caption_norm(text_states)

        image_context_length = 0
        if encoder_hidden_states_image is not None:
            image_states = self.image_projector(encoder_hidden_states_image.to(dtype=hidden_states.dtype))
            image_context_length = image_states.shape[1]
            encoder_hidden_states = torch.cat([text_states, image_states], dim=1)
            if encoder_attention_mask is not None:
                zeros = torch.zeros(
                    encoder_attention_mask.shape[0],
                    encoder_attention_mask.shape[1],
                    image_context_length,
                    device=encoder_attention_mask.device,
                    dtype=encoder_attention_mask.dtype,
                )
                encoder_attention_mask = torch.cat([encoder_attention_mask, zeros], dim=-1)
        else:
            encoder_hidden_states = text_states

        self.image_cross_attention_adapter.set_runtime(
            image_context_length,
            image_cross_attention_scale=image_cross_attention_scale,
            disable_image_cross_attention=disable_image_cross_attention,
        )

        if torch.is_grad_enabled() and transformer.gradient_checkpointing:
            for index_block, block in enumerate(transformer.transformer_blocks):
                hidden_states = transformer._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    post_patch_height,
                    post_patch_width,
                )
                if controlnet_block_samples is not None and 0 < index_block <= len(controlnet_block_samples):
                    hidden_states = hidden_states + controlnet_block_samples[index_block - 1]
        else:
            for index_block, block in enumerate(transformer.transformer_blocks):
                hidden_states = block(
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    post_patch_height,
                    post_patch_width,
                )
                if controlnet_block_samples is not None and 0 < index_block <= len(controlnet_block_samples):
                    hidden_states = hidden_states + controlnet_block_samples[index_block - 1]

        hidden_states = transformer.norm_out(hidden_states, embedded_timestep, transformer.scale_shift_table)
        hidden_states = transformer.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(
            batch_size, post_patch_height, post_patch_width, transformer.config.patch_size, transformer.config.patch_size, -1
        )
        hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
        output = hidden_states.reshape(batch_size, -1, post_patch_height * p, post_patch_width * p)
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


def load_image_encoder_and_processor(args, dtype=None, device=None):
    from transformers import AutoImageProcessor, CLIPVisionModel

    kwargs = {
        "revision": getattr(args, "image_encoder_revision", None),
        "local_files_only": getattr(args, "image_encoder_local_files_only", False),
    }
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
    hidden = image_encoder(**inputs).last_hidden_state
    return hidden.to(dtype=dtype)


def build_route3_model(transformer, image_encoder_hidden_size, image_gate_init=1e-3):
    cross_dim = int(getattr(transformer.config, "cross_attention_dim"))
    image_projector = Route3ImageProjector(image_encoder_hidden_size, cross_dim)
    image_adapter = Route3ImageCrossAttentionAdapter(transformer, image_gate_init=image_gate_init)
    return Route3SanaModel(transformer, image_projector, image_adapter)


def _strict_load_module(module, path, name):
    state = load_file(path)
    missing, unexpected = module.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise ValueError(f"{name} strict load failed: missing={missing}, unexpected={unexpected}")


def save_route3_checkpoint(directory, route3_model, pipe, args, global_step, original_latent_channels, image_encoder_hidden_size):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    patch_embed, proj = get_sana_patch_embedding(route3_model.transformer)
    save_file({k: v.detach().cpu() for k, v in patch_embed.state_dict().items()}, directory / "i2i_patch_embedding.safetensors")
    save_file(
        {k: v.detach().cpu() for k, v in route3_model.image_projector.state_dict().items()},
        directory / "image_projector.safetensors",
    )
    save_file(
        {k: v.detach().cpu() for k, v in route3_model.image_cross_attention_adapter.state_dict().items()},
        directory / "image_cross_attention.safetensors",
    )
    if args.train_mode == "patch_lora":
        SanaPipeline.save_lora_weights(
            directory / "transformer_lora",
            transformer_lora_layers=get_peft_model_state_dict(route3_model.transformer),
        )
    elif args.train_mode == "full_transformer":
        route3_model.transformer.save_pretrained(directory / "transformer", safe_serialization=True)
    else:
        raise ValueError(f"Unsupported Route3 train_mode={args.train_mode}")

    config = {
        "implementation": IMPLEMENTATION,
        "base_model": args.model,
        "model": args.model,
        "train_mode": args.train_mode,
        "global_step": int(global_step),
        "condition_type": "clean_src_vae_latent_plus_src_image_cross_attention",
        "concat_order": CONCAT_ORDER,
        "original_latent_channels": int(original_latent_channels),
        "expanded_input_channels": int(2 * original_latent_channels),
        "output_channels": int(get_sana_output_channels(route3_model.transformer)),
        "image_encoder_model": args.image_encoder_model,
        "image_encoder_subfolder": getattr(args, "image_encoder_subfolder", None),
        "image_encoder_revision": getattr(args, "image_encoder_revision", None),
        "image_encoder_hidden_size": int(image_encoder_hidden_size),
        "image_context_dynamic": True,
        "image_projector_type": "LayerNorm-Linear-GELU-Linear-LayerNorm",
        "image_cross_attention_type": "wan_style_independent_image_kv",
        "image_gate_init": float(args.image_gate_init),
        "image_cross_attention_scale": float(args.image_cross_attention_scale),
        "uses_text_cross_attention": True,
        "uses_image_cross_attention": True,
        "uses_src_latent_concat": True,
        "uses_ref": False,
        "supports_pure_noise": True,
        "supports_src_latent_init": True,
        "learning_rate": float(args.learning_rate),
        "patch_learning_rate": float(args.patch_learning_rate),
        "image_adapter_learning_rate": float(args.image_adapter_learning_rate),
        "lora_rank": int(getattr(args, "lora_rank", 0)),
        "lora_alpha": int(getattr(args, "lora_alpha", 0)),
        "lora_scope": getattr(args, "lora_scope", None),
        "max_pixels": int(args.max_pixels),
        "size_divisor": int(args.size_divisor),
        "mixed_precision": args.mixed_precision,
        "checkpoint_format": "route3_wan_crossattn_v1",
    }
    (directory / "route3_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return config


def load_route3_config(checkpoint):
    checkpoint = Path(checkpoint)
    candidates = [checkpoint / "route3_config.json", checkpoint.parent / "route3_config.json"]
    for path in candidates:
        if path.exists():
            config = json.loads(path.read_text(encoding="utf-8"))
            if config.get("implementation") != IMPLEMENTATION:
                raise ValueError(f"Not a Route3 checkpoint: {path}")
            return config, path.parent
    raise FileNotFoundError(f"route3_config.json not found in {checkpoint} or parent.")


def load_route3_components(checkpoint, route3_model):
    config, ckpt_dir = load_route3_config(checkpoint)
    _strict_load_module(route3_model.image_projector, ckpt_dir / "image_projector.safetensors", "image_projector")
    _strict_load_module(
        route3_model.image_cross_attention_adapter,
        ckpt_dir / "image_cross_attention.safetensors",
        "image_cross_attention",
    )
    return config, ckpt_dir


def validate_route2_init(route2_checkpoint, train_mode):
    config = load_route2_config(route2_checkpoint)
    if config.get("train_mode") != train_mode:
        raise ValueError(
            f"--init_from_route2_checkpoint train mode mismatch: route2={config.get('train_mode')} route3={train_mode}"
        )
    return config
