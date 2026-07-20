import json
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from safetensors.torch import load_file, save_file
from torch import nn

from diffusers import SanaPipeline
from diffusers.models.modeling_outputs import Transformer2DModelOutput


IMPLEMENTATION = "sana_focus_latent_injection"
VERSION = 1
DEFAULT_LORA_TARGET_MODULES = "to_q,to_k,to_v,to_out.0,proj,linear,fc1,fc2"
A_ONLY_PROMPT = (
    "Restore an all-in-focus photograph from Image A while preserving its original structure, viewpoint, geometry, "
    "color, and valid sharp details. Remove defocus blur without introducing duplicated edges, artifacts, geometric "
    "changes, or color shifts."
)
AB_PROMPT = (
    "Fuse Image A and Image B into one all-in-focus photograph. Preserve Image A's structure, viewpoint, geometry, "
    "color, and valid sharp regions. Use Image B only to restore regions that are blurred in Image A. Avoid ghosting, "
    "duplicated edges, artifacts, geometric changes, and color shifts."
)


def default_prompt_for_mode(condition_mode):
    if condition_mode == "a_only":
        return A_ONLY_PROMPT
    if condition_mode == "ab":
        return AB_PROMPT
    raise ValueError(f"Unsupported condition_mode={condition_mode!r}")


def num_condition_images_for_mode(condition_mode):
    if condition_mode == "a_only":
        return 1
    if condition_mode == "ab":
        return 2
    raise ValueError(f"Unsupported condition_mode={condition_mode!r}")


def tensor_stats(x):
    x = x.detach().float().cpu()
    return {
        "shape": list(x.shape),
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std()) if x.numel() > 1 else 0.0,
        "abs_mean": float(x.abs().mean()),
        "dtype": str(x.dtype),
    }


def encode_vae_latents(vae, pixel_values, scaling_factor=None, sample=True, generator=None):
    encoded = vae.encode(pixel_values)
    if hasattr(encoded, "latent"):
        latents = encoded.latent
    elif hasattr(encoded, "latents"):
        latents = encoded.latents
    elif hasattr(encoded, "latent_dist"):
        latents = encoded.latent_dist.sample(generator=generator) if sample else encoded.latent_dist.mode()
    else:
        raise TypeError(f"Unsupported VAE encode output type: {type(encoded).__name__}.")
    if scaling_factor is None:
        scaling_factor = getattr(vae.config, "scaling_factor", 1.0)
    return latents * scaling_factor


def decode_vae_latents(vae, latents):
    latents = latents / getattr(vae.config, "scaling_factor", 1.0)
    decoded = vae.decode(latents)
    if hasattr(decoded, "sample"):
        return decoded.sample
    if isinstance(decoded, (tuple, list)):
        return decoded[0]
    return decoded


def get_sana_patch_embedding(transformer):
    patch_embed = getattr(transformer, "patch_embed", None)
    if patch_embed is None:
        raise AttributeError("SANA transformer has no patch_embed module.")
    proj = getattr(patch_embed, "proj", None)
    if not isinstance(proj, nn.Conv2d):
        raise TypeError(f"Expected transformer.patch_embed.proj to be nn.Conv2d, got {type(proj).__name__}.")
    return patch_embed, proj


def get_sana_latent_channels(transformer):
    _, proj = get_sana_patch_embedding(transformer)
    return int(proj.in_channels)


def get_sana_output_channels(transformer):
    patch_size = int(getattr(transformer.config, "patch_size", 1))
    proj_out = getattr(transformer, "proj_out", None)
    if isinstance(proj_out, nn.Linear):
        return int(proj_out.out_features // (patch_size * patch_size))
    return int(getattr(transformer.config, "out_channels", getattr(transformer.config, "in_channels")))


def lora_targets_for_scope(scope, requested):
    if scope == "attn_qkv":
        return ["to_q", "to_k", "to_v"]
    if scope == "attn_qkvo":
        return ["to_q", "to_k", "to_v", "to_out.0"]
    return [item.strip() for item in requested.split(",") if item.strip()]


def add_lora_to_transformer(transformer, rank, alpha, target_modules, dropout=0.0, scope="wide"):
    patterns = lora_targets_for_scope(scope, target_modules)
    linear_names = [name for name, module in transformer.named_modules() if isinstance(module, nn.Linear)]
    matched = []
    for pattern in patterns:
        names = [name for name in linear_names if name == pattern or name.endswith(f".{pattern}") or pattern in name]
        if names:
            matched.extend(names)
        else:
            warnings.warn(f"LoRA target pattern did not match any Linear module: {pattern}")
    matched = sorted(set(matched))
    if not matched:
        raise ValueError("No LoRA target modules matched SANA transformer.")
    print(f"[FOCUS_ROUTE1] matched LoRA modules={len(matched)}", flush=True)
    for name in matched[:50]:
        print(f"[FOCUS_ROUTE1] LoRA module: {name}", flush=True)
    transformer.add_adapter(
        LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, init_lora_weights="gaussian", target_modules=matched)
    )
    return matched


class FocusLatentConditionInjector(nn.Module):
    def __init__(self, latent_channels, num_condition_images, inner_dim, patch_size, hidden_channels=128):
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.num_condition_images = int(num_condition_images)
        self.condition_channels = self.latent_channels * self.num_condition_images
        self.inner_dim = int(inner_dim)
        self.patch_size = int(patch_size)
        self.hidden_channels = int(hidden_channels)
        self.proj_in = nn.Conv2d(self.condition_channels, self.hidden_channels, kernel_size=3, padding=1)
        self.proj_mid = nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1)
        self.proj_out = nn.Conv2d(
            self.hidden_channels, self.inner_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        nn.init.kaiming_normal_(self.proj_in.weight, nonlinearity="relu")
        nn.init.zeros_(self.proj_in.bias)
        nn.init.kaiming_normal_(self.proj_mid.weight, nonlinearity="relu")
        nn.init.zeros_(self.proj_mid.bias)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, condition_latents, target_latents=None):
        hidden = F.silu(self.proj_in(condition_latents))
        hidden = F.silu(self.proj_mid(hidden))
        injected = self.proj_out(hidden)
        return injected.flatten(2, 3).transpose(1, 2)


class SanaFocusLatentInjectionModel(nn.Module):
    def __init__(self, transformer, injector, condition_mode):
        super().__init__()
        self.transformer = transformer
        self.injector = injector
        self.condition_mode = condition_mode
        self.last_debug = {}

    @property
    def config(self):
        return self.transformer.config

    @property
    def dtype(self):
        return next(self.transformer.parameters()).dtype

    def _condition_tensor(self, condition_latents, condition_mode):
        mode = condition_mode or self.condition_mode
        if condition_latents is None:
            return None
        if mode == "a_only":
            if isinstance(condition_latents, (list, tuple)):
                if len(condition_latents) != 1:
                    raise ValueError("a_only condition_latents must be z_a or [z_a].")
                condition_latents = condition_latents[0]
            return condition_latents
        if mode == "ab":
            if not isinstance(condition_latents, (list, tuple)) or len(condition_latents) != 2:
                raise ValueError("ab condition_latents must be [z_a, z_b].")
            return torch.cat([condition_latents[0], condition_latents[1]], dim=1)
        raise ValueError(f"Unsupported condition_mode={mode!r}")

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        condition_latents=None,
        condition_mode=None,
        condition_scale=1.0,
        guidance=None,
        encoder_attention_mask=None,
        attention_mask=None,
        attention_kwargs=None,
        controlnet_block_samples=None,
        disable_condition_injection=False,
        return_dict=True,
    ):
        if condition_latents is None or disable_condition_injection:
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
        condition_tensor = self._condition_tensor(condition_latents, condition_mode)
        condition_tensor = condition_tensor.to(device=hidden_states.device, dtype=target_tokens.dtype)
        injected_tokens = self.injector(condition_tensor, hidden_states)
        if injected_tokens.shape != target_tokens.shape:
            raise ValueError(
                f"Injected token shape {tuple(injected_tokens.shape)} must match target tokens "
                f"{tuple(target_tokens.shape)}."
            )
        hidden_states = target_tokens + float(condition_scale) * injected_tokens
        self.last_debug = {
            "condition_tensor": condition_tensor.detach(),
            "target_tokens": target_tokens.detach(),
            "injected_tokens": injected_tokens.detach(),
            "hidden_tokens_after_injection": hidden_states.detach(),
            "post_patch_height": post_patch_height,
            "post_patch_width": post_patch_width,
        }

        if guidance is not None:
            timestep, embedded_timestep = transformer.time_embed(
                timestep, guidance=guidance, hidden_dtype=hidden_states.dtype
            )
        else:
            timestep, embedded_timestep = transformer.time_embed(
                timestep, batch_size=batch_size, hidden_dtype=hidden_states.dtype
            )
        encoder_hidden_states = transformer.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])
        encoder_hidden_states = transformer.caption_norm(encoder_hidden_states)

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


def create_focus_model(transformer, condition_mode, injector_hidden_channels=128):
    patch_embed, proj = get_sana_patch_embedding(transformer)
    latent_channels = int(proj.in_channels)
    inner_dim = int(proj.out_channels)
    patch_size = int(getattr(transformer.config, "patch_size", proj.kernel_size[0]))
    num_condition_images = num_condition_images_for_mode(condition_mode)
    injector = FocusLatentConditionInjector(
        latent_channels=latent_channels,
        num_condition_images=num_condition_images,
        inner_dim=inner_dim,
        patch_size=patch_size,
        hidden_channels=injector_hidden_channels,
    )
    return SanaFocusLatentInjectionModel(transformer, injector, condition_mode), latent_channels


def validate_checkpoint_mode(config, requested_condition_mode):
    ckpt_mode = config.get("condition_mode")
    if ckpt_mode != requested_condition_mode:
        raise ValueError(
            f"[ERROR] checkpoint condition_mode={ckpt_mode} is incompatible with requested "
            f"condition_mode={requested_condition_mode}"
        )


def focus_config(args, model, global_step, latent_channels):
    num_condition_images = num_condition_images_for_mode(args.condition_mode)
    roles = ["A"] if args.condition_mode == "a_only" else ["A", "B"]
    config = {
        "task": "focus_fusion",
        "implementation": IMPLEMENTATION,
        "version": VERSION,
        "condition_mode": args.condition_mode,
        "num_condition_images": num_condition_images,
        "condition_roles": roles,
        "latent_fusion": "channel_concat_before_injector",
        "image_stream_injection": "additive_patch_token_residual",
        "output_reference": "A",
        "focus_maps_used": False,
        "base_model": args.model,
        "global_step": int(global_step),
        "latent_channels": int(latent_channels),
        "condition_channels": int(latent_channels * num_condition_images),
        "injector_hidden_channels": int(args.injector_hidden_channels),
        "condition_scale": float(args.condition_scale),
        "lora_rank": int(args.lora_rank),
        "lora_alpha": int(args.lora_alpha),
        "lora_scope": args.lora_scope,
        "learning_rate": float(args.learning_rate),
        "max_pixels": int(args.max_pixels),
        "size_divisor": int(args.size_divisor),
        "mixed_precision": args.mixed_precision,
    }
    if args.condition_mode == "ab":
        config["condition_channel_order"] = ["A", "B"]
    return config


def save_focus_checkpoint(directory, focus_model, args, global_step, latent_channels, optimizer=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_file(
        {key: value.detach().cpu() for key, value in focus_model.injector.state_dict().items()},
        directory / "focus_condition_injector.safetensors",
    )
    SanaPipeline.save_lora_weights(
        directory / "transformer_lora", transformer_lora_layers=get_peft_model_state_dict(focus_model.transformer)
    )
    config = focus_config(args, focus_model, global_step, latent_channels)
    (directory / "focus_latent_injection_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (directory / "trainer_state.json").write_text(
        json.dumps({"global_step": int(global_step)}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if optimizer is not None:
        torch.save(optimizer.state_dict(), directory / "optimizer.pt")
    return config


def load_focus_config(checkpoint):
    checkpoint = Path(checkpoint)
    candidates = [checkpoint / "focus_latent_injection_config.json", checkpoint.parent / "focus_latent_injection_config.json"]
    for path in candidates:
        if path.exists():
            config = json.loads(path.read_text(encoding="utf-8"))
            if config.get("implementation") != IMPLEMENTATION:
                raise ValueError(f"Not a focus latent injection checkpoint: {path}")
            return config, path.parent
    raise FileNotFoundError(f"focus_latent_injection_config.json not found in {checkpoint} or parent.")


def load_focus_injector(checkpoint, focus_model, condition_mode):
    config, checkpoint_dir = load_focus_config(checkpoint)
    validate_checkpoint_mode(config, condition_mode)
    state = load_file(checkpoint_dir / "focus_condition_injector.safetensors")
    missing, unexpected = focus_model.injector.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise ValueError(f"Focus injector strict load failed: missing={missing}, unexpected={unexpected}")
    return config, checkpoint_dir
