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
from diffusers.models.modeling_outputs import Transformer2DModelOutput


DEFAULT_LATENT_CONCAT_LORA_TARGET_MODULES = (
    "to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out,"
    "linear_in,linear_out,proj,linear,fc1,fc2"
)


def route1_tensor_stats(x):
    x = x.detach().float().cpu()
    return {
        "shape": list(x.shape),
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "dtype": str(x.dtype),
    }


class SrcLatentConditionInjector(nn.Module):
    """Project a source VAE latent into additive SANA image-token residuals.

    Expected inputs:
    - src_latents: `(B, C, H, W)` VAE latents for the source image.
    - target_hidden_states: `(B, C, H, W)` noisy target latents before patch embedding.

    The final projection is zero-initialized, so the untrained route exactly starts as a
    no-op injection path.
    """

    def __init__(self, latent_channels, inner_dim, patch_size=1, hidden_channels=128):
        super().__init__()
        if patch_size < 1:
            raise ValueError("patch_size must be positive.")
        self.latent_channels = latent_channels
        self.inner_dim = inner_dim
        self.patch_size = patch_size
        self.hidden_channels = hidden_channels
        self.proj_in = nn.Conv2d(latent_channels, hidden_channels, kernel_size=3, padding=1)
        self.proj_mid = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.proj_out = nn.Conv2d(hidden_channels, inner_dim, kernel_size=patch_size, stride=patch_size)
        self.act = nn.SiLU()
        nn.init.kaiming_normal_(self.proj_in.weight)
        nn.init.zeros_(self.proj_in.bias)
        nn.init.kaiming_normal_(self.proj_mid.weight)
        nn.init.zeros_(self.proj_mid.bias)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, src_latents, target_hidden_states):
        if src_latents.shape[0] != target_hidden_states.shape[0]:
            if target_hidden_states.shape[0] % src_latents.shape[0]:
                raise ValueError(
                    f"src latent batch={src_latents.shape[0]} cannot be expanded to target batch="
                    f"{target_hidden_states.shape[0]}."
                )
            src_latents = src_latents.repeat_interleave(target_hidden_states.shape[0] // src_latents.shape[0], dim=0)
        if src_latents.shape[-2:] != target_hidden_states.shape[-2:]:
            src_latents = F.interpolate(
                src_latents.float(),
                size=target_hidden_states.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).to(target_hidden_states)
        x = self.act(self.proj_in(src_latents.float()))
        x = self.act(self.proj_mid(x))
        x = self.proj_out(x).to(target_hidden_states)
        return x.flatten(2, 3).transpose(1, 2)


class SanaArtifactRepairLatentConcatModel(nn.Module):
    """SANA wrapper for Route 1: src latent injection into the image input stream."""

    def __init__(self, transformer, src_condition_injector):
        super().__init__()
        self.transformer = transformer
        self.src_condition_injector = src_condition_injector
        self.last_injection_stats = {}

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
        src_latents=None,
    ):
        if src_latents is None:
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
            raise ValueError("Route 1 src latent injection does not use ControlNet residuals.")
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
        injected_tokens = self.src_condition_injector(src_latents, hidden_states)
        if injected_tokens.shape != target_tokens.shape:
            raise ValueError(
                f"src injection token shape {tuple(injected_tokens.shape)} must match target token shape "
                f"{tuple(target_tokens.shape)}."
            )
        hidden_tokens = target_tokens + injected_tokens

        if guidance is not None:
            timestep, embedded_timestep = transformer.time_embed(
                timestep, guidance=guidance, hidden_dtype=hidden_tokens.dtype
            )
        else:
            timestep, embedded_timestep = transformer.time_embed(
                timestep, batch_size=batch_size, hidden_dtype=hidden_tokens.dtype
            )

        encoder_hidden_states = transformer.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_tokens.shape[-1])
        encoder_hidden_states = transformer.caption_norm(encoder_hidden_states)

        if torch.is_grad_enabled() and transformer.gradient_checkpointing:
            for index_block, block in enumerate(transformer.transformer_blocks):
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
        self.last_injection_stats = {
            "target_tokens": route1_tensor_stats(target_tokens),
            "src_injected_tokens": route1_tensor_stats(injected_tokens),
            "hidden_tokens_after_injection": route1_tensor_stats(target_tokens + injected_tokens),
            "target_grid_h": int(post_patch_height),
            "target_grid_w": int(post_patch_width),
        }
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


def lora_targets_for_scope(scope, requested):
    if scope == "attn_qkv":
        return ["to_q", "to_k", "to_v"]
    if scope == "attn_qkvo":
        return ["to_q", "to_k", "to_v", "to_out.0"]
    return [item.strip() for item in requested.split(",") if item.strip()]


def find_lora_target_modules(transformer, requested, scope="wide"):
    patterns = lora_targets_for_scope(scope, requested)
    linear_names = [name for name, module in transformer.named_modules() if isinstance(module, nn.Linear)]
    matched_patterns = []
    matched_names = []
    for pattern in patterns:
        names = [name for name in linear_names if name == pattern or name.endswith(f".{pattern}") or pattern in name]
        if names:
            matched_patterns.append(pattern)
            matched_names.extend(names)
        else:
            warnings.warn(f"LoRA target module pattern did not match any Linear module: {pattern}")
    matched_names = sorted(set(matched_names))
    if not matched_names:
        raise ValueError("No LoRA target modules matched the SANA transformer.")
    return matched_patterns, matched_names


def add_lora_to_transformer(transformer, rank, alpha, target_modules, dropout=0.0, scope="wide"):
    matched_patterns, matched_names = find_lora_target_modules(transformer, target_modules, scope)
    print(f"[ROUTE1_LORA] scope={scope}", flush=True)
    print(f"[ROUTE1_LORA] matched patterns={matched_patterns}", flush=True)
    print(f"[ROUTE1_LORA] matched module count={len(matched_names)}", flush=True)
    for name in matched_names[:60]:
        print(f"[ROUTE1_LORA] module: {name}", flush=True)
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


def count_trainable_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def save_latent_concat_checkpoint(directory, model, pipe, args, global_step):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.detach().cpu() for k, v in model.src_condition_injector.state_dict().items()},
        directory / "src_condition_injector.safetensors",
    )
    if args.train_transformer_lora:
        SanaPipeline.save_lora_weights(
            directory / "transformer_lora",
            transformer_lora_layers=get_peft_model_state_dict(model.transformer),
        )
    config = vars(args).copy()
    config.update(
        {
            "task_name": "artifact_repair",
            "route": "latent_concat_src_injection",
            "route_name": "Artifact Repair – Route 1: src latent concat / image-input injection",
            "implementation": "sana_artifact_repair_src_latent_injection_v1",
            "version": 1,
            "base_model": args.model,
            "model": args.model,
            "latent_channels": pipe.transformer.config.in_channels,
            "inner_dim": pipe.transformer.config.num_attention_heads * pipe.transformer.config.attention_head_dim,
            "patch_size": pipe.transformer.config.patch_size,
            "src_injector_hidden_channels": args.injector_hidden_channels,
            "target_stream": "noisy_target_latent_image_stream",
            "condition_stream": "src_latent_additive_image_token_residual",
            "ref_usage": "metadata_read_only_ignored_in_route1",
            "global_step": global_step,
        }
    )
    (directory / "route1_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (directory / "artifact_repair_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def load_latent_concat_assets(checkpoint, model):
    checkpoint = Path(checkpoint)
    config_path = checkpoint / "route1_config.json"
    if not config_path.exists():
        config_path = checkpoint / "artifact_repair_config.json"
    if not config_path.exists() and (checkpoint.parent / "route1_config.json").exists():
        config_path = checkpoint.parent / "route1_config.json"
    if not config_path.exists() and (checkpoint.parent / "artifact_repair_config.json").exists():
        config_path = checkpoint.parent / "artifact_repair_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing route1_config.json or artifact_repair_config.json in {checkpoint}.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("implementation") != "sana_artifact_repair_src_latent_injection_v1":
        raise ValueError(f"Checkpoint is not Route 1 src latent injection: {config_path}")
    injector_path = checkpoint / "src_condition_injector.safetensors"
    if not injector_path.exists() and (checkpoint.parent / "src_condition_injector.safetensors").exists():
        injector_path = checkpoint.parent / "src_condition_injector.safetensors"
    if not injector_path.exists():
        raise FileNotFoundError(f"Missing src_condition_injector.safetensors in {checkpoint} or its parent.")
    model.src_condition_injector.load_state_dict(load_file(injector_path), strict=True)
    lora_path = checkpoint / "transformer_lora"
    if not lora_path.exists() and (checkpoint.parent / "transformer_lora").exists():
        lora_path = checkpoint.parent / "transformer_lora"
    return config, lora_path if lora_path.exists() else None
