import gc
import os

import pytest
import torch

from diffusers import SanaTransformer2DModel

from sana_artifact_repair_channel_concat import expand_sana_patch_embedding_for_channel_concat, get_sana_patch_embedding


pytestmark = [pytest.mark.slow, pytest.mark.integration]


def load_pretrained_transformer():
    if os.environ.get("RUN_SANA_PRETRAINED_TESTS") != "1":
        pytest.skip("Set RUN_SANA_PRETRAINED_TESTS=1 to run local SANA pretrained integration tests.")
    model_id = os.environ.get("SANA_TEST_MODEL", "Efficient-Large-Model/Sana_600M_1024px_diffusers")
    device = os.environ.get("SANA_TEST_DEVICE", "cpu")
    model = SanaTransformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        local_files_only=True,
        torch_dtype=torch.float32,
    ).to(device)
    return model, torch.device(device)


def cleanup(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_pretrained_actual_structure_and_zero_init_compatibility():
    model, device = load_pretrained_transformer()
    try:
        patch_embed, proj = get_sana_patch_embedding(model)
        original_c = int(model.config.in_channels)
        out_c = int(getattr(model.config, "out_channels", original_c) or original_c)
        print("transformer class", model.__class__.__name__)
        print("patch embedding class", patch_embed.__class__.__name__)
        print("projection class", proj.__class__.__name__)
        print("original in_channels", original_c)
        print("out_channels", out_c)
        print("patch size", model.config.patch_size)
        z_t = torch.randn(1, original_c, 4, 4, device=device)
        z_src = torch.randn_like(z_t)
        encoder_hidden_states = torch.randn(1, 3, model.config.caption_channels, device=device)
        encoder_attention_mask = torch.ones(1, 3, device=device)
        timestep = torch.tensor([500.0], device=device)
        with torch.no_grad():
            out_before = model(
                hidden_states=z_t,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                return_dict=True,
            ).sample
            c = expand_sana_patch_embedding_for_channel_concat(model)
            out_after = model(
                hidden_states=torch.cat([z_t, z_src], dim=1),
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                return_dict=True,
            ).sample
        assert c == original_c
        assert out_after.shape == out_before.shape
        assert (out_before - out_after).abs().max().item() < 1e-5
    finally:
        cleanup(model)


def test_pretrained_condition_channel_gradient():
    model, device = load_pretrained_transformer()
    try:
        model.requires_grad_(False)
        c = expand_sana_patch_embedding_for_channel_concat(model)
        patch_embed, proj = get_sana_patch_embedding(model)
        patch_embed.requires_grad_(True)
        z_t = torch.randn(1, c, 4, 4, device=device)
        z_src = torch.randn_like(z_t)
        encoder_hidden_states = torch.randn(1, 3, model.config.caption_channels, device=device)
        encoder_attention_mask = torch.ones(1, 3, device=device)
        timestep = torch.tensor([500.0], device=device)
        out = model(
            hidden_states=torch.cat([z_t, z_src], dim=1),
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            timestep=timestep,
            return_dict=True,
        ).sample
        out.square().mean().backward()
        grad = proj.weight.grad[:, c : 2 * c]
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert grad.abs().sum().item() > 0
    finally:
        cleanup(model)
