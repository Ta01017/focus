import json
import warnings
from pathlib import Path

import torch
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from safetensors.torch import load_file, save_file
from torch import nn

from diffusers import SanaPipeline


IMPLEMENTATION = "sana_artifact_repair_channel_concat_v1"
DEFAULT_LORA_TARGET_MODULES = "to_q,to_k,to_v,to_out.0,proj,linear,fc1,fc2"
CONCAT_ORDER = ["current_latent", "src_latent"]


def tensor_stats(x):
    x = x.detach().float().cpu()
    return {
        "shape": list(x.shape),
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "abs_mean": float(x.abs().mean()),
        "dtype": str(x.dtype),
    }


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


def get_sana_output_channels(transformer):
    patch_size = int(getattr(transformer.config, "patch_size", 1))
    proj_out = getattr(transformer, "proj_out", None)
    if isinstance(proj_out, nn.Linear):
        return int(proj_out.out_features // (patch_size * patch_size))
    config_out = getattr(transformer.config, "out_channels", None)
    if config_out is None:
        return int(getattr(transformer, "original_latent_channels", getattr(transformer.config, "in_channels")))
    return int(config_out)


def expand_sana_patch_embedding_for_channel_concat(transformer):
    patch_embed, old_proj = get_sana_patch_embedding(transformer)
    input_channels = int(old_proj.in_channels)
    if hasattr(transformer, "original_latent_channels"):
        original_channels = int(transformer.original_latent_channels)
        if input_channels != 2 * original_channels:
            raise ValueError(
                f"Transformer declares original_latent_channels={original_channels}, but patch input channels "
                f"are {input_channels}; expected {2 * original_channels}."
            )
        print(
            f"[ROUTE2] patch embedding already expanded: {input_channels} input channels, original C={original_channels}",
            flush=True,
        )
        return original_channels
    config_in_channels = int(getattr(transformer.config, "in_channels", input_channels))
    if input_channels == 2 * config_in_channels:
        original_channels = config_in_channels
        print(
            f"[ROUTE2] patch embedding already expanded: {input_channels} input channels, original C={original_channels}",
            flush=True,
        )
        return original_channels
    if input_channels != config_in_channels:
        raise ValueError(
            f"Patch embedding input channels ({input_channels}) do not match transformer.config.in_channels "
            f"({config_in_channels}); refusing ambiguous expansion."
        )
    new_proj = nn.Conv2d(
        in_channels=2 * input_channels,
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
    patch_embed.proj = new_proj
    if hasattr(transformer, "register_to_config"):
        transformer.register_to_config(in_channels=2 * input_channels)
    else:
        transformer.config.in_channels = 2 * input_channels
    transformer.original_latent_channels = input_channels
    print(
        f"[ROUTE2] expanded patch embedding {old_proj.__class__.__name__}: {input_channels} -> {2 * input_channels} input channels; "
        f"out={old_proj.out_channels}, kernel={old_proj.kernel_size}, stride={old_proj.stride}, padding={old_proj.padding}",
        flush=True,
    )
    return input_channels


def patch_embedding_weight_stats(transformer, original_latent_channels):
    _, proj = get_sana_patch_embedding(transformer)
    weight = proj.weight
    target_weight = weight[:, :original_latent_channels]
    condition_weight = weight[:, original_latent_channels : 2 * original_latent_channels]
    target_norm = target_weight.detach().float().norm().clamp_min(1e-12)
    condition_norm = condition_weight.detach().float().norm()
    full_weight_grad = proj.weight.grad
    if full_weight_grad is None:
        condition_gradient_norm = 0.0
    else:
        condition_gradient_norm = float(
            full_weight_grad[:, original_latent_channels : 2 * original_latent_channels]
            .detach()
            .float()
            .norm()
            .cpu()
        )
    return {
        "target_weight_abs_mean": float(target_weight.detach().float().abs().mean().cpu()),
        "condition_weight_abs_mean": float(condition_weight.detach().float().abs().mean().cpu()),
        "target_weight_norm": float(target_norm.cpu()),
        "condition_weight_norm": float(condition_norm.cpu()),
        "condition_target_norm_ratio": float((condition_norm / target_norm).cpu()),
        "condition_gradient_norm": condition_gradient_norm,
    }


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
    print(f"[ROUTE2] LoRA scope={scope} matched_modules={len(matched)}", flush=True)
    for name in matched[:50]:
        print(f"[ROUTE2] LoRA module: {name}", flush=True)
    transformer.add_adapter(
        LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, init_lora_weights="gaussian", target_modules=matched)
    )
    return matched


def save_route2_checkpoint(directory, transformer, pipe, args, global_step, original_latent_channels):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    patch_embed, proj = get_sana_patch_embedding(transformer)
    save_file({k: v.detach().cpu() for k, v in patch_embed.state_dict().items()}, directory / "i2i_patch_embedding.safetensors")
    if args.train_mode == "patch_lora":
        SanaPipeline.save_lora_weights(directory / "transformer_lora", transformer_lora_layers=get_peft_model_state_dict(transformer))
    if args.train_mode == "full_transformer":
        transformer.save_pretrained(directory / "transformer", safe_serialization=True)
    full_transformer_saved = args.train_mode == "full_transformer"
    lora_saved = args.train_mode == "patch_lora"
    config = {
        "implementation": IMPLEMENTATION,
        "condition_type": "clean_src_vae_latent",
        "model_input": "concat_noisy_target_and_clean_src",
        "concat_order": CONCAT_ORDER,
        "original_latent_channels": original_latent_channels,
        "expanded_input_channels": 2 * original_latent_channels,
        "output_channels": get_sana_output_channels(transformer),
        "inference_initialization": "pure_noise",
        "uses_src_latent_init": False,
        "uses_ref": False,
        "base_model": args.model,
        "model": args.model,
        "global_step": global_step,
        "patch_embedding_class": patch_embed.__class__.__name__,
        "patch_projection_class": proj.__class__.__name__,
        "train_mode": args.train_mode,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_scope": args.lora_scope,
        "patch_learning_rate": args.patch_learning_rate,
        "learning_rate": args.learning_rate,
        "image_condition_dropout": args.image_condition_dropout,
        "mixed_precision": args.mixed_precision,
        "max_pixels": args.max_pixels,
        "size_divisor": args.size_divisor,
        "full_transformer_saved": full_transformer_saved,
        "full_transformer_subdir": "transformer" if full_transformer_saved else None,
        "patch_embedding_saved": True,
        "lora_saved": lora_saved,
    }
    (directory / "route2_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return config


def load_route2_config(checkpoint):
    checkpoint = Path(checkpoint)
    for path in (checkpoint / "route2_config.json", checkpoint.parent / "route2_config.json"):
        if path.exists():
            config = json.loads(path.read_text(encoding="utf-8"))
            if config.get("implementation") != IMPLEMENTATION:
                raise ValueError(f"Not a Route 2 checkpoint: {path}")
            return config
    raise FileNotFoundError(f"route2_config.json not found in {checkpoint} or its parent.")


def load_route2_patch_embedding(checkpoint, transformer):
    checkpoint = Path(checkpoint)
    path = checkpoint / "i2i_patch_embedding.safetensors"
    if not path.exists() and (checkpoint.parent / "i2i_patch_embedding.safetensors").exists():
        path = checkpoint.parent / "i2i_patch_embedding.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"Missing i2i_patch_embedding.safetensors in {checkpoint} or parent.")
    patch_embed, _ = get_sana_patch_embedding(transformer)
    state = load_file(path)
    current = patch_embed.state_dict()
    for key, value in current.items():
        if key not in state:
            raise ValueError(f"missing keys when loading patch embedding: {[key]}")
        if tuple(state[key].shape) != tuple(value.shape):
            raise ValueError(
                f"Patch embedding shape mismatch for {key}: expected {tuple(value.shape)}, actual {tuple(state[key].shape)}"
            )
    unexpected = sorted(set(state) - set(current))
    if unexpected:
        raise ValueError(f"unexpected keys when loading patch embedding: {unexpected}")
    missing, unexpected = patch_embed.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise ValueError(f"Patch embedding strict load failed, missing={missing}, unexpected={unexpected}")
    return path


def route2_full_transformer_path(checkpoint):
    checkpoint = Path(checkpoint)
    candidates = [checkpoint / "transformer", checkpoint.parent / "transformer"]
    for path in candidates:
        if path.exists():
            return path
    return None


def route2_lora_path(checkpoint):
    checkpoint = Path(checkpoint)
    path = checkpoint / "transformer_lora"
    if not path.exists() and (checkpoint.parent / "transformer_lora").exists():
        path = checkpoint.parent / "transformer_lora"
    return path if path.exists() else None
