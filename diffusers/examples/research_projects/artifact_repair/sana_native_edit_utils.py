import json
import warnings
from pathlib import Path

import torch
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F

from diffusers import SanaPipeline


DEFAULT_NATIVE_EDIT_LORA_TARGET_MODULES = "to_q,to_k,to_v,to_out.0,proj,linear,fc1,fc2"


class NativeEditSanaTransformer(nn.Module):
    """SANA wrapper that concatenates target/src/ref latent tokens only when explicitly enabled."""

    def __init__(self, transformer, num_roles=3, use_role_embedding=True):
        super().__init__()
        self.transformer = transformer
        self.use_role_embedding = use_role_embedding
        inner_dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
        self.role_embed = nn.Parameter(torch.zeros(num_roles, inner_dim)) if use_role_embedding else None
        self.last_token_stats = {}

    @property
    def config(self):
        return self.transformer.config

    @property
    def dtype(self):
        return self.transformer.dtype

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        guidance=None,
        encoder_attention_mask=None,
        attention_mask=None,
        attention_kwargs=None,
        controlnet_block_samples=None,
        return_dict=True,
        edit_hidden_states=None,
        edit_role_ids=None,
        enable_native_edit_tokens=False,
        target_token_length=None,
    ):
        if not enable_native_edit_tokens or edit_hidden_states is None:
            return self.transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                guidance=guidance,
                encoder_attention_mask=encoder_attention_mask,
                attention_mask=attention_mask,
                attention_kwargs=attention_kwargs,
                controlnet_block_samples=controlnet_block_samples,
                return_dict=return_dict,
            )
        if controlnet_block_samples is not None:
            raise ValueError("Native edit token route does not support ControlNet residuals.")
        if not isinstance(edit_hidden_states, (list, tuple)) or len(edit_hidden_states) < 1:
            raise ValueError("edit_hidden_states must be a list/tuple of edit latents.")

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

        target_tokens = transformer.patch_embed(hidden_states)
        if target_token_length is None:
            target_token_length = target_tokens.shape[1]
        tokens = [target_tokens]
        role_ids = edit_role_ids or list(range(1, len(edit_hidden_states) + 1))
        edit_lengths = []
        for edit_latents, role_id in zip(edit_hidden_states, role_ids):
            if edit_latents.shape[0] != hidden_states.shape[0]:
                if hidden_states.shape[0] % edit_latents.shape[0]:
                    raise ValueError("Edit latent batch size cannot be expanded to target batch size.")
                edit_latents = edit_latents.repeat_interleave(hidden_states.shape[0] // edit_latents.shape[0], dim=0)
            if edit_latents.shape[-2:] != hidden_states.shape[-2:]:
                edit_latents = F.interpolate(edit_latents.float(), size=hidden_states.shape[-2:], mode="bilinear", align_corners=False).to(hidden_states)
            edit_tokens = transformer.patch_embed(edit_latents)
            if self.role_embed is not None:
                edit_tokens = edit_tokens + self.role_embed[int(role_id)].to(edit_tokens)
            tokens.append(edit_tokens)
            edit_lengths.append(edit_tokens.shape[1])
        if self.role_embed is not None:
            tokens[0] = tokens[0] + self.role_embed[0].to(tokens[0])
        hidden_tokens = torch.cat(tokens, dim=1)

        if guidance is not None:
            timestep, embedded_timestep = transformer.time_embed(timestep, guidance=guidance, hidden_dtype=hidden_tokens.dtype)
        else:
            timestep, embedded_timestep = transformer.time_embed(timestep, batch_size=batch_size, hidden_dtype=hidden_tokens.dtype)

        encoder_hidden_states = transformer.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_tokens.shape[-1])
        encoder_hidden_states = transformer.caption_norm(encoder_hidden_states)

        if torch.is_grad_enabled() and transformer.gradient_checkpointing:
            for block in transformer.transformer_blocks:
                hidden_tokens = transformer._gradient_checkpointing_func(
                    block,
                    hidden_tokens,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    post_patch_height,
                    post_patch_width,
                )
        else:
            for block in transformer.transformer_blocks:
                hidden_tokens = block(
                    hidden_tokens,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    post_patch_height,
                    post_patch_width,
                )

        hidden_tokens = hidden_tokens[:, :target_token_length]
        hidden_tokens = transformer.norm_out(hidden_tokens, embedded_timestep, transformer.scale_shift_table)
        hidden_tokens = transformer.proj_out(hidden_tokens)
        hidden_tokens = hidden_tokens.reshape(
            batch_size,
            post_patch_height,
            post_patch_width,
            transformer.config.patch_size,
            transformer.config.patch_size,
            -1,
        )
        hidden_tokens = hidden_tokens.permute(0, 5, 1, 3, 2, 4)
        output = hidden_tokens.reshape(batch_size, -1, post_patch_height * p, post_patch_width * p)
        self.last_token_stats = {
            "target_token_length": int(target_token_length),
            "edit_token_lengths": [int(x) for x in edit_lengths],
            "total_token_length": int(target_token_length + sum(edit_lengths)),
        }
        if not return_dict:
            return (output,)
        from diffusers.models.modeling_outputs import Transformer2DModelOutput

        return Transformer2DModelOutput(sample=output)


def find_lora_target_modules(transformer, requested):
    requested = [item.strip() for item in requested.split(",") if item.strip()]
    linear_names = [name for name, module in transformer.named_modules() if isinstance(module, torch.nn.Linear)]
    matched_patterns = []
    matched_names = []
    for target in requested:
        names = [name for name in linear_names if name == target or name.endswith(f".{target}") or target in name]
        if names:
            matched_patterns.append(target)
            matched_names.extend(names)
        else:
            warnings.warn(f"LoRA target module pattern did not match any Linear module: {target}")
    matched_names = sorted(set(matched_names))
    if not matched_patterns:
        raise ValueError("No LoRA target modules matched the SANA transformer.")
    return matched_patterns, matched_names


def add_lora_to_transformer(transformer, rank, alpha, target_modules, dropout=0.0):
    matched_patterns, matched_names = find_lora_target_modules(transformer, target_modules)
    print(f"[NATIVE_EDIT] matched LoRA module patterns: {matched_patterns}", flush=True)
    print(f"[NATIVE_EDIT] matched LoRA modules count: {len(matched_names)}", flush=True)
    for name in matched_names[:40]:
        print(f"[NATIVE_EDIT] LoRA module: {name}", flush=True)
    transformer.add_adapter(
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            init_lora_weights="gaussian",
            target_modules=matched_names,
        )
    )
    return matched_patterns, matched_names


def save_native_edit_checkpoint(directory, native_transformer, pipe, args, global_step=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    SanaPipeline.save_lora_weights(
        directory / "transformer_lora",
        transformer_lora_layers=get_peft_model_state_dict(native_transformer.transformer),
    )
    config = {
        "enable_native_edit_tokens": True,
        "num_edit_images": args.num_edit_images,
        "edit_role_embedding": args.edit_role_embedding,
        "hidden_dim": native_transformer.transformer.config.num_attention_heads * native_transformer.transformer.config.attention_head_dim,
        "implementation": "sana_native_edit_token_concat",
        "model": args.model,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_target_modules": args.lora_target_modules,
        "max_pixels": args.max_pixels,
        "size_divisor": args.size_divisor,
        "global_step": global_step,
    }
    (directory / "native_edit_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    if native_transformer.role_embed is not None:
        save_file({"role_embed": native_transformer.role_embed.detach().cpu()}, directory / "edit_role_embedding.safetensors")


def load_native_edit_assets(checkpoint, native_transformer):
    checkpoint = Path(checkpoint)
    config_path = checkpoint / "native_edit_config.json"
    if not config_path.exists() and (checkpoint.parent / "native_edit_config.json").exists():
        config_path = checkpoint.parent / "native_edit_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"native_edit_config.json not found in {checkpoint} or its parent.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    print(f"[LOAD] native_edit_config: {config_path}", flush=True)
    role_path = checkpoint / "edit_role_embedding.safetensors"
    if not role_path.exists() and (checkpoint.parent / "edit_role_embedding.safetensors").exists():
        role_path = checkpoint.parent / "edit_role_embedding.safetensors"
    if native_transformer.role_embed is not None:
        if not role_path.exists():
            raise FileNotFoundError(f"edit_role_embedding.safetensors not found for role embedding route: {checkpoint}")
        state = load_file(role_path)
        native_transformer.role_embed.data.copy_(state["role_embed"].to(native_transformer.role_embed))
        print(f"[LOAD] edit_role_embedding: {role_path}", flush=True)
    lora_path = checkpoint / "transformer_lora"
    if not lora_path.exists() and (checkpoint.parent / "transformer_lora").exists():
        lora_path = checkpoint.parent / "transformer_lora"
    if not lora_path.exists():
        raise FileNotFoundError(f"transformer_lora not found in {checkpoint} or its parent.")
    print(f"[LOAD] transformer LoRA: {lora_path}", flush=True)
    return config, lora_path


def count_trainable_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
