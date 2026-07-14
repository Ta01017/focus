import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from diffusers import SanaTransformer2DModel

from sana_artifact_repair_channel_concat import (
    CONCAT_ORDER,
    add_lora_to_transformer,
    expand_sana_patch_embedding_for_channel_concat,
    get_sana_output_channels,
    get_sana_patch_embedding,
    load_route2_patch_embedding,
    patch_embedding_weight_stats,
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


def make_inputs(batch_size=1, height=4, width=4):
    return {
        "z_t": torch.randn(batch_size, 4, height, width),
        "z_src": torch.randn(batch_size, 4, height, width),
        "encoder_hidden_states": torch.randn(batch_size, 3, 8),
        "encoder_attention_mask": torch.ones(batch_size, 3),
        "timestep": torch.tensor([500.0] * batch_size),
    }


def forward_real_sana(model, hidden_states, inputs):
    return model(
        hidden_states=hidden_states,
        encoder_hidden_states=inputs["encoder_hidden_states"],
        encoder_attention_mask=inputs["encoder_attention_mask"],
        timestep=inputs["timestep"],
        return_dict=True,
    ).sample


def route2_args(train_mode):
    return SimpleNamespace(
        train_mode=train_mode,
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


class DummyPipe:
    def __init__(self, transformer):
        self.transformer = transformer


def test_real_sana_zero_init_compatibility():
    torch.manual_seed(0)
    inputs = make_inputs()
    original = make_tiny_real_sana().eval()
    expanded = copy.deepcopy(original).eval()
    with torch.no_grad():
        out_before = forward_real_sana(original, inputs["z_t"], inputs)
        c = expand_sana_patch_embedding_for_channel_concat(expanded)
        out_after = forward_real_sana(expanded, torch.cat([inputs["z_t"], inputs["z_src"]], dim=1), inputs)
    assert c == 4
    assert expanded.config.in_channels == 8
    assert out_before.shape == out_after.shape
    assert out_after.shape == inputs["z_t"].shape
    assert torch.allclose(out_before, out_after, atol=1e-5, rtol=1e-5)


def test_real_sana_condition_channel_gradient():
    torch.manual_seed(1)
    model = make_tiny_real_sana()
    model.requires_grad_(False)
    c = expand_sana_patch_embedding_for_channel_concat(model)
    patch_embed, proj = get_sana_patch_embedding(model)
    patch_embed.requires_grad_(True)
    inputs = make_inputs()
    out = forward_real_sana(model, torch.cat([inputs["z_t"], inputs["z_src"]], dim=1), inputs)
    out.square().mean().backward()
    condition_grad = proj.weight.grad[:, c : 2 * c]
    target_grad = proj.weight.grad[:, :c]
    assert condition_grad is not None
    assert target_grad is not None
    assert torch.isfinite(condition_grad).all()
    assert torch.isfinite(target_grad).all()
    assert condition_grad.abs().sum() > 0
    assert target_grad.abs().sum() > 0


def test_real_sana_condition_effect_after_one_patch_update():
    torch.manual_seed(2)
    model = make_tiny_real_sana()
    model.requires_grad_(False)
    c = expand_sana_patch_embedding_for_channel_concat(model)
    patch_embed, _ = get_sana_patch_embedding(model)
    patch_embed.requires_grad_(True)
    optimizer = torch.optim.SGD(patch_embed.parameters(), lr=1e-1)
    inputs = make_inputs()
    out = forward_real_sana(model, torch.cat([inputs["z_t"], inputs["z_src"]], dim=1), inputs)
    out.square().mean().backward()
    optimizer.step()
    with torch.no_grad():
        pred_real = forward_real_sana(model, torch.cat([inputs["z_t"], inputs["z_src"]], dim=1), inputs)
        pred_zero = forward_real_sana(model, torch.cat([inputs["z_t"], torch.zeros_like(inputs["z_src"])], dim=1), inputs)
    delta = (pred_real - pred_zero).abs().mean()
    assert torch.isfinite(delta)
    assert delta.item() > 0


def test_real_sana_patch_save_load_consistency(tmp_path):
    torch.manual_seed(3)
    inputs = make_inputs()
    base = make_tiny_real_sana()
    model_a = copy.deepcopy(base)
    model_a.requires_grad_(False)
    c = expand_sana_patch_embedding_for_channel_concat(model_a)
    patch_embed, _ = get_sana_patch_embedding(model_a)
    patch_embed.requires_grad_(True)
    optimizer = torch.optim.SGD(patch_embed.parameters(), lr=1e-1)
    out = forward_real_sana(model_a, torch.cat([inputs["z_t"], inputs["z_src"]], dim=1), inputs)
    out.square().mean().backward()
    optimizer.step()
    with torch.no_grad():
        expected = forward_real_sana(model_a, torch.cat([inputs["z_t"], inputs["z_src"]], dim=1), inputs)
    save_route2_checkpoint(tmp_path, model_a, DummyPipe(model_a), route2_args("patch_only"), 1, c)
    model_b = copy.deepcopy(base)
    expand_sana_patch_embedding_for_channel_concat(model_b)
    load_route2_patch_embedding(tmp_path, model_b)
    with torch.no_grad():
        actual = forward_real_sana(model_b, torch.cat([inputs["z_t"], inputs["z_src"]], dim=1), inputs)
    assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-6)


def test_real_sana_full_transformer_save_load_round_trip(tmp_path):
    torch.manual_seed(4)
    inputs = make_inputs()
    model = make_tiny_real_sana()
    c = expand_sana_patch_embedding_for_channel_concat(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    out = forward_real_sana(model, torch.cat([inputs["z_t"], inputs["z_src"]], dim=1), inputs)
    out.square().mean().backward()
    optimizer.step()
    with torch.no_grad():
        before_save = forward_real_sana(model, torch.cat([inputs["z_t"], inputs["z_src"]], dim=1), inputs)
    save_route2_checkpoint(tmp_path, model, DummyPipe(model), route2_args("full_transformer"), 1, c)
    loaded = SanaTransformer2DModel.from_pretrained(tmp_path / "transformer")
    with torch.no_grad():
        after_load = forward_real_sana(loaded, torch.cat([inputs["z_t"], inputs["z_src"]], dim=1), inputs)
    _, proj = get_sana_patch_embedding(loaded)
    assert loaded.config.in_channels == 8
    assert get_sana_output_channels(loaded) == 4
    assert proj.in_channels == 8
    assert torch.allclose(before_save, after_load, atol=1e-6, rtol=1e-6)


def test_real_sana_config_projection_consistency_and_repeat_expand():
    model = make_tiny_real_sana()
    c1 = expand_sana_patch_embedding_for_channel_concat(model)
    c2 = expand_sana_patch_embedding_for_channel_concat(model)
    _, proj = get_sana_patch_embedding(model)
    assert c1 == 4
    assert c2 == 4
    assert model.config.in_channels == proj.in_channels
    assert get_sana_output_channels(model) == 4
    assert proj.in_channels == 8


def test_checkpoint_mode_directory_structure(tmp_path):
    torch.manual_seed(5)
    c = 4
    for mode in ("patch_only", "patch_lora", "full_transformer"):
        model = make_tiny_real_sana()
        c = expand_sana_patch_embedding_for_channel_concat(model)
        if mode == "patch_lora":
            add_lora_to_transformer(model, 2, 2, "to_q,to_k,to_v", 0.0, "attn_qkv")
        out_dir = tmp_path / mode
        save_route2_checkpoint(out_dir, model, DummyPipe(model), route2_args(mode), 1, c)
        config = json.loads((out_dir / "route2_config.json").read_text())
        assert (out_dir / "i2i_patch_embedding.safetensors").exists()
        assert config["patch_embedding_saved"] is True
        if mode == "patch_only":
            assert not (out_dir / "transformer_lora").exists()
            assert not (out_dir / "transformer").exists()
            assert config["lora_saved"] is False
            assert config["full_transformer_saved"] is False
        elif mode == "patch_lora":
            assert (out_dir / "transformer_lora").exists()
            assert not (out_dir / "transformer").exists()
            assert config["lora_saved"] is True
            assert config["full_transformer_saved"] is False
        else:
            assert (out_dir / "transformer").exists()
            assert not (out_dir / "transformer_lora").exists()
            assert config["lora_saved"] is False
            assert config["full_transformer_saved"] is True
            assert config["full_transformer_subdir"] == "transformer"


def test_concat_order_real_sana():
    assert CONCAT_ORDER == ["current_latent", "src_latent"]
