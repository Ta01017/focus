#!/usr/bin/env python

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from diffusers import SanaTransformer2DModel

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sana_artifact_repair_channel_concat import (  # noqa: E402
    expand_sana_patch_embedding_for_channel_concat,
    get_sana_patch_embedding,
    save_route2_checkpoint,
)


def make_tiny_real_sana():
    return SanaTransformer2DModel(
        in_channels=4,
        out_channels=4,
        num_attention_heads=2,
        attention_head_dim=4,
        num_layers=1,
        num_cross_attention_heads=2,
        cross_attention_head_dim=4,
        cross_attention_dim=8,
        caption_channels=8,
        mlp_ratio=2.0,
        sample_size=4,
        patch_size=1,
        guidance_embeds=False,
    )


def forward(model, hidden_states, encoder_hidden_states, encoder_attention_mask, timestep):
    return model(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        timestep=timestep,
        return_dict=True,
    ).sample


def main():
    torch.manual_seed(123)
    model = make_tiny_real_sana()
    c = expand_sana_patch_embedding_for_channel_concat(model)
    z_t = torch.randn(1, c, 4, 4)
    z_src = torch.randn_like(z_t)
    encoder_hidden_states = torch.randn(1, 3, 8)
    encoder_attention_mask = torch.ones(1, 3)
    timestep = torch.tensor([500.0])
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    pred = forward(model, torch.cat([z_t, z_src], dim=1), encoder_hidden_states, encoder_attention_mask, timestep)
    pred.square().mean().backward()
    optimizer.step()
    with torch.no_grad():
        before_save = forward(model, torch.cat([z_t, z_src], dim=1), encoder_hidden_states, encoder_attention_mask, timestep)
    args = SimpleNamespace(
        train_mode="full_transformer",
        model="tiny-real-sana",
        lora_rank=2,
        lora_alpha=2,
        lora_scope="attn_qkv",
        patch_learning_rate=1e-4,
        learning_rate=2e-5,
        image_condition_dropout=0.0,
        mixed_precision="no",
        max_pixels=1024,
        size_divisor=32,
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        save_route2_checkpoint(tmp, model, SimpleNamespace(transformer=model), args, 1, c)
        loaded = SanaTransformer2DModel.from_pretrained(tmp / "transformer")
        with torch.no_grad():
            after_load = forward(loaded, torch.cat([z_t, z_src], dim=1), encoder_hidden_states, encoder_attention_mask, timestep)
        _, proj = get_sana_patch_embedding(loaded)
        max_abs_error = (before_save - after_load).abs().max().item()
        print(f"full_transformer_round_trip_max_abs_error={max_abs_error:.10f}", flush=True)
        print(f"loaded_config_in_channels={loaded.config.in_channels}", flush=True)
        print(f"loaded_patch_projection_in_channels={proj.in_channels}", flush=True)
        if loaded.config.in_channels != 8 or proj.in_channels != 8:
            raise AssertionError("Loaded full transformer is not the expanded Route 2 transformer.")
        if max_abs_error >= 1e-6:
            raise AssertionError(f"Full transformer round-trip mismatch: {max_abs_error}")


if __name__ == "__main__":
    main()
