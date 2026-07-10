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
from diffusers.models.normalization import RMSNorm


DEFAULT_NATIVE_EDIT_LORA_TARGET_MODULES = "to_q,to_k,to_v,to_out.0,proj,linear,fc1,fc2"


class SanaNativeEditCrossAttentionWrapper(nn.Module):
    """SANA wrapper that feeds src/ref as cross-attention encoder conditions.

    The target noisy latent is the only image hidden/query stream and keeps its real HxW grid.
    Edit latents are patch-embedded with the shared SANA patch_embed and appended to the text
    encoder condition sequence, so SANA attn2 can read them as K/V.
    """

    def __init__(
        self,
        transformer,
        num_edit_images=2,
        use_edit_role_embedding=True,
        edit_condition_scale=1.0,
        use_edit_token_norm=True,
    ):
        super().__init__()
        self.transformer = transformer
        self.num_edit_images = num_edit_images
        self.use_edit_role_embedding = use_edit_role_embedding
        self.edit_condition_scale = edit_condition_scale
        self.use_edit_token_norm = use_edit_token_norm
        inner_dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
        self.edit_role_embedding = (
            nn.Parameter(torch.zeros(num_edit_images, inner_dim)) if use_edit_role_embedding else None
        )
        self.edit_token_norm = RMSNorm(inner_dim, eps=1e-5, elementwise_affine=True) if use_edit_token_norm else nn.Identity()
        cross_attention_dim = self._get_cross_attention_dim()
        self.edit_condition_projection = (
            nn.Identity() if inner_dim == cross_attention_dim else nn.Linear(inner_dim, cross_attention_dim, bias=False)
        )
        self.inner_dim = inner_dim
        self.cross_attention_dim = cross_attention_dim
        self.last_token_stats = {}
        print(f"[NATIVE_EDIT_V2] inner_dim={inner_dim}", flush=True)
        print(f"[NATIVE_EDIT_V2] cross_attention_dim={cross_attention_dim}", flush=True)
        print(
            f"[NATIVE_EDIT_V2] edit projection={self.edit_condition_projection.__class__.__name__}",
            flush=True,
        )

    def _get_cross_attention_dim(self):
        for block in self.transformer.transformer_blocks:
            attn2 = getattr(block, "attn2", None)
            to_k = getattr(attn2, "to_k", None)
            if to_k is not None and hasattr(to_k, "in_features"):
                return to_k.in_features
        return self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim

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
        debug_nan=False,
        **kwargs,
    ):
        if edit_hidden_states is None:
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
        if not isinstance(edit_hidden_states, (list, tuple)) or len(edit_hidden_states) != self.num_edit_images:
            raise ValueError(f"edit_hidden_states must contain exactly {self.num_edit_images} edit latents.")

        transformer = self.transformer
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        batch_size, _, height, width = hidden_states.shape
        p = transformer.config.patch_size
        post_patch_height, post_patch_width = height // p, width // p

        target_tokens = transformer.patch_embed(hidden_states)
        expected_target_len = post_patch_height * post_patch_width
        if target_tokens.shape[1] != expected_target_len:
            raise ValueError(
                f"target token length={target_tokens.shape[1]} must equal target latent grid "
                f"{post_patch_height}x{post_patch_width}={expected_target_len}."
            )

        if guidance is not None:
            timestep, embedded_timestep = transformer.time_embed(
                timestep, guidance=guidance, hidden_dtype=target_tokens.dtype
            )
        else:
            timestep, embedded_timestep = transformer.time_embed(
                timestep, batch_size=batch_size, hidden_dtype=target_tokens.dtype
            )

        text_tokens = transformer.caption_projection(encoder_hidden_states)
        text_tokens = text_tokens.view(batch_size, -1, target_tokens.shape[-1])
        text_tokens = transformer.caption_norm(text_tokens)

        edit_tokens_list = []
        for role_id, edit_latents in enumerate(edit_hidden_states):
            if edit_latents.shape[0] != hidden_states.shape[0]:
                if hidden_states.shape[0] % edit_latents.shape[0]:
                    raise ValueError("Edit latent batch size cannot be expanded to target batch size.")
                edit_latents = edit_latents.repeat_interleave(hidden_states.shape[0] // edit_latents.shape[0], dim=0)
            if edit_latents.shape[-2:] != hidden_states.shape[-2:]:
                edit_latents = F.interpolate(
                    edit_latents.float(),
                    size=hidden_states.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).to(hidden_states)
            edit_tokens = transformer.patch_embed(edit_latents)
            if self.edit_role_embedding is not None:
                edit_tokens = edit_tokens + self.edit_role_embedding[role_id].to(edit_tokens)
            edit_tokens = self.edit_token_norm(edit_tokens)
            edit_tokens = self.edit_condition_projection(edit_tokens)
            edit_tokens_list.append(edit_tokens)

        encoder_condition_tokens = torch.cat(
            [text_tokens] + [self.edit_condition_scale * tokens for tokens in edit_tokens_list],
            dim=1,
        )
        combined_encoder_attention_mask = self._build_encoder_attention_mask(
            encoder_attention_mask,
            batch_size,
            text_tokens.shape[1],
            [tokens.shape[1] for tokens in edit_tokens_list],
            encoder_condition_tokens.dtype,
            encoder_condition_tokens.device,
        )
        if combined_encoder_attention_mask.shape[-1] != encoder_condition_tokens.shape[1]:
            raise ValueError(
                "combined encoder attention mask length mismatch: "
                f"mask={combined_encoder_attention_mask.shape[-1]} condition={encoder_condition_tokens.shape[1]} "
                f"text={text_tokens.shape[1]} edit={[tokens.shape[1] for tokens in edit_tokens_list]}"
            )

        self._check_finite("target_tokens", target_tokens, debug_nan)
        self._check_finite("text_tokens", text_tokens, debug_nan)
        for index, tokens in enumerate(edit_tokens_list):
            self._check_finite(f"edit_tokens_{index}", tokens, debug_nan)
        self._check_finite("encoder_condition_tokens", encoder_condition_tokens, debug_nan)

        if torch.is_grad_enabled() and transformer.gradient_checkpointing:
            for index, block in enumerate(transformer.transformer_blocks):
                target_tokens = transformer._gradient_checkpointing_func(
                    block,
                    target_tokens,
                    attention_mask,
                    encoder_condition_tokens,
                    combined_encoder_attention_mask,
                    timestep,
                    post_patch_height,
                    post_patch_width,
                )
                self._check_finite(f"block_{index}_output", target_tokens, debug_nan)
        else:
            for index, block in enumerate(transformer.transformer_blocks):
                target_tokens = block(
                    target_tokens,
                    attention_mask,
                    encoder_condition_tokens,
                    combined_encoder_attention_mask,
                    timestep,
                    post_patch_height,
                    post_patch_width,
                )
                self._check_finite(f"block_{index}_output", target_tokens, debug_nan)

        target_tokens = transformer.norm_out(target_tokens, embedded_timestep, transformer.scale_shift_table)
        target_tokens = transformer.proj_out(target_tokens)
        target_tokens = target_tokens.reshape(
            batch_size,
            post_patch_height,
            post_patch_width,
            transformer.config.patch_size,
            transformer.config.patch_size,
            -1,
        )
        target_tokens = target_tokens.permute(0, 5, 1, 3, 2, 4)
        output = target_tokens.reshape(batch_size, -1, post_patch_height * p, post_patch_width * p)
        self.last_token_stats = {
            "target_token_length": int(expected_target_len),
            "target_hidden_tokens": int(expected_target_len),
            "text_token_length": int(text_tokens.shape[1]),
            "src_condition_token_length": int(edit_tokens_list[0].shape[1]),
            "ref_condition_token_length": int(edit_tokens_list[1].shape[1]) if len(edit_tokens_list) > 1 else None,
            "total_encoder_condition_length": int(encoder_condition_tokens.shape[1]),
            "target_grid_h": int(post_patch_height),
            "target_grid_w": int(post_patch_width),
            "edit_condition_scale": float(self.edit_condition_scale),
            "output_stream": "target_only",
        }
        self._check_finite("pred", output, debug_nan)
        if not return_dict:
            return (output,)
        from diffusers.models.modeling_outputs import Transformer2DModelOutput

        return Transformer2DModelOutput(sample=output)

    def _build_encoder_attention_mask(self, text_mask, batch_size, text_len, edit_lengths, dtype, device):
        edit_len = sum(edit_lengths)
        if text_mask is None:
            mask = torch.ones(batch_size, text_len + edit_len, device=device, dtype=dtype)
            return (1 - mask)[:, None, :] * -10000.0
        if text_mask.ndim == 2:
            text_mask = text_mask.to(device=device)
            edit_mask = torch.ones(batch_size, edit_len, device=device, dtype=text_mask.dtype)
            mask = torch.cat([text_mask, edit_mask], dim=1)
            return (1 - mask.to(dtype))[:, None, :] * -10000.0
        if text_mask.ndim == 3:
            text_mask = text_mask.to(device=device, dtype=dtype)
            edit_bias = torch.zeros(batch_size, 1, edit_len, device=device, dtype=dtype)
            return torch.cat([text_mask, edit_bias], dim=-1)
        raise ValueError(f"Unsupported encoder_attention_mask ndim={text_mask.ndim}.")

    def _check_finite(self, name, tensor, enabled):
        if enabled and not torch.isfinite(tensor).all():
            raise RuntimeError(f"[NATIVE_EDIT_V2] Non-finite tensor detected: {name}")


def _scope_targets(scope, requested):
    if scope == "cross_attention":
        return ["attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0"]
    if scope == "all_attention":
        return ["attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0", "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0"]
    return [item.strip() for item in requested.split(",") if item.strip()]


def _lora_match_counts(names):
    return {
        "attn1": sum(".attn1." in name for name in names),
        "attn2": sum(".attn2." in name for name in names),
        "ff_proj": sum((".ff." in name) or ("proj" in name) or ("linear" in name) or ("fc" in name) for name in names),
    }


def find_lora_target_modules(transformer, requested, scope="wide"):
    requested = _scope_targets(scope, requested)
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
    counts = _lora_match_counts(matched_names)
    if counts["attn2"] == 0:
        raise ValueError("LoRA matched zero attn2 modules. Native-edit v2 requires attn2 LoRA coverage.")
    return matched_patterns, matched_names


def add_lora_to_transformer(transformer, rank, alpha, target_modules, dropout=0.0, scope="wide"):
    matched_patterns, matched_names = find_lora_target_modules(transformer, target_modules, scope)
    counts = _lora_match_counts(matched_names)
    print(f"[LORA] scope={scope}", flush=True)
    print(f"[LORA] matched patterns={matched_patterns}", flush=True)
    print(f"[LORA] matched total={len(matched_names)}", flush=True)
    print(f"[LORA] matched attn1={counts['attn1']}", flush=True)
    print(f"[LORA] matched attn2={counts['attn2']}", flush=True)
    print(f"[LORA] matched ff/proj={counts['ff_proj']}", flush=True)
    for name in matched_names[:40]:
        print(f"[LORA] module: {name}", flush=True)
    transformer.add_adapter(
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            init_lora_weights="gaussian",
            target_modules=matched_names,
        )
    )
    return matched_patterns, matched_names, counts


def save_native_edit_checkpoint(directory, native_transformer, pipe, args, global_step=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    SanaPipeline.save_lora_weights(
        directory / "transformer_lora",
        transformer_lora_layers=get_peft_model_state_dict(native_transformer.transformer),
    )
    config = {
        "implementation": "sana_cross_attention_edit_condition_v2",
        "version": 2,
        "num_edit_images": args.num_edit_images,
        "roles": ["src", "ref"],
        "use_edit_role_embedding": args.edit_role_embedding,
        "use_edit_token_norm": args.use_edit_token_norm,
        "edit_condition_scale": args.edit_condition_scale,
        "target_stream": "target_only",
        "edit_stream": "cross_attention_encoder_condition",
        "spatial_layout": "target_HxW_only",
        "hidden_dim": native_transformer.inner_dim,
        "cross_attention_dim": native_transformer.cross_attention_dim,
        "model": args.model,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_target_modules": args.lora_target_modules,
        "lora_scope": args.lora_scope,
        "native_edit_impl": args.native_edit_impl,
        "max_pixels": args.max_pixels,
        "size_divisor": args.size_divisor,
        "global_step": global_step,
    }
    (directory / "native_edit_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    if native_transformer.edit_role_embedding is not None:
        save_file({"edit_role_embedding": native_transformer.edit_role_embedding.detach().cpu()}, directory / "edit_role_embedding.safetensors")
    if not isinstance(native_transformer.edit_token_norm, nn.Identity):
        save_file({k: v.detach().cpu() for k, v in native_transformer.edit_token_norm.state_dict().items()}, directory / "edit_token_norm.safetensors")
    if not isinstance(native_transformer.edit_condition_projection, nn.Identity):
        save_file({k: v.detach().cpu() for k, v in native_transformer.edit_condition_projection.state_dict().items()}, directory / "edit_condition_projection.safetensors")


def load_native_edit_assets(checkpoint, native_transformer):
    checkpoint = Path(checkpoint)
    config_path = checkpoint / "native_edit_config.json"
    if not config_path.exists() and (checkpoint.parent / "native_edit_config.json").exists():
        config_path = checkpoint.parent / "native_edit_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"native_edit_config.json not found in {checkpoint} or its parent.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("implementation") != "sana_cross_attention_edit_condition_v2":
        raise ValueError(
            "[ERROR] Legacy Hx3W native-edit checkpoint is incompatible with cross-attention v2. "
            "Retraining is required."
        )
    print(f"[LOAD] native_edit_config: {config_path}", flush=True)
    role_path = checkpoint / "edit_role_embedding.safetensors"
    if not role_path.exists() and (checkpoint.parent / "edit_role_embedding.safetensors").exists():
        role_path = checkpoint.parent / "edit_role_embedding.safetensors"
    if native_transformer.edit_role_embedding is not None:
        if not role_path.exists():
            raise FileNotFoundError(f"edit_role_embedding.safetensors not found for role embedding route: {checkpoint}")
        state = load_file(role_path)
        role = state.get("edit_role_embedding")
        if role is None:
            raise ValueError(
                "[ERROR] Legacy Hx3W role embedding checkpoint is incompatible with cross-attention v2. "
                "Retraining is required."
            )
        if role.shape != native_transformer.edit_role_embedding.shape:
            raise ValueError(
                f"[ERROR] edit role embedding shape mismatch: checkpoint={tuple(role.shape)} "
                f"model={tuple(native_transformer.edit_role_embedding.shape)}. Retraining is required."
            )
        native_transformer.edit_role_embedding.data.copy_(role.to(native_transformer.edit_role_embedding))
        print(f"[LOAD] edit_role_embedding: {role_path}", flush=True)
    norm_path = checkpoint / "edit_token_norm.safetensors"
    if not norm_path.exists() and (checkpoint.parent / "edit_token_norm.safetensors").exists():
        norm_path = checkpoint.parent / "edit_token_norm.safetensors"
    if not isinstance(native_transformer.edit_token_norm, nn.Identity):
        if not norm_path.exists():
            raise FileNotFoundError(f"edit_token_norm.safetensors not found: {checkpoint}")
        native_transformer.edit_token_norm.load_state_dict(load_file(norm_path))
        print(f"[LOAD] edit_token_norm: {norm_path}", flush=True)
    proj_path = checkpoint / "edit_condition_projection.safetensors"
    if not proj_path.exists() and (checkpoint.parent / "edit_condition_projection.safetensors").exists():
        proj_path = checkpoint.parent / "edit_condition_projection.safetensors"
    if not isinstance(native_transformer.edit_condition_projection, nn.Identity):
        if not proj_path.exists():
            raise FileNotFoundError(f"edit_condition_projection.safetensors not found: {checkpoint}")
        native_transformer.edit_condition_projection.load_state_dict(load_file(proj_path))
        print(f"[LOAD] edit_condition_projection: {proj_path}", flush=True)
    lora_path = checkpoint / "transformer_lora"
    if not lora_path.exists() and (checkpoint.parent / "transformer_lora").exists():
        lora_path = checkpoint.parent / "transformer_lora"
    if not lora_path.exists():
        raise FileNotFoundError(f"transformer_lora not found in {checkpoint} or its parent.")
    print(f"[LOAD] transformer LoRA: {lora_path}", flush=True)
    return config, lora_path


def count_trainable_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
