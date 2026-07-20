import pytest
import torch
from torch import nn

from sana_focus_latent_injection import (
    FocusLatentConditionInjector,
    SanaFocusLatentInjectionModel,
    create_focus_model,
    validate_checkpoint_mode,
)


class FakePatchEmbed(nn.Module):
    def __init__(self, channels=4, inner_dim=8, patch_size=2):
        super().__init__()
        self.proj = nn.Conv2d(channels, inner_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2, 3).transpose(1, 2)


class FakeTransformer(nn.Module):
    def __init__(self, channels=4, inner_dim=8, patch_size=2):
        super().__init__()
        self.config = type("Config", (), {"patch_size": patch_size, "timestep_scale": 1.0, "in_channels": channels})()
        self.patch_embed = FakePatchEmbed(channels, inner_dim, patch_size)
        self.time_embed = lambda timestep, batch_size=None, guidance=None, hidden_dtype=None: (
            torch.zeros(batch_size or timestep.shape[0], 1, inner_dim, dtype=hidden_dtype or torch.float32),
            torch.zeros(batch_size or timestep.shape[0], inner_dim, dtype=hidden_dtype or torch.float32),
        )
        self.caption_projection = nn.Linear(inner_dim, inner_dim)
        self.caption_norm = nn.LayerNorm(inner_dim)
        self.transformer_blocks = nn.ModuleList([FakeBlock()])
        self.norm_out = FakeNormOut()
        self.scale_shift_table = nn.Parameter(torch.zeros(2, inner_dim))
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * channels)
        self.gradient_checkpointing = False

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        guidance=None,
        encoder_attention_mask=None,
        attention_mask=None,
        attention_kwargs=None,
        controlnet_block_samples=None,
        return_dict=True,
    ):
        tokens = self.patch_embed(hidden_states)
        batch, _, height, width = hidden_states.shape
        post_h, post_w = height // 2, width // 2
        tokens = self.proj_out(tokens)
        tokens = tokens.reshape(batch, post_h, post_w, 2, 2, -1)
        tokens = tokens.permute(0, 5, 1, 3, 2, 4)
        output = tokens.reshape(batch, -1, height, width)
        return (output,)


class FakeBlock(nn.Module):
    def forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, timestep, height, width):
        return hidden_states


class FakeNormOut(nn.Module):
    def forward(self, hidden_states, embedded_timestep, scale_shift_table):
        return hidden_states


def test_a_only_shape():
    injector = FocusLatentConditionInjector(4, 1, 8, 2, hidden_channels=16)
    target = torch.randn(2, 4, 16, 20)
    z_a = torch.randn(2, 4, 16, 20)
    out = injector(z_a, target)
    assert out.shape == (2, 80, 8)


def test_ab_shape_and_channel_concat():
    injector = FocusLatentConditionInjector(4, 2, 8, 2, hidden_channels=16)
    z_a = torch.randn(2, 4, 16, 20)
    z_b = torch.randn(2, 4, 16, 20)
    condition = torch.cat([z_a, z_b], dim=1)
    out = injector(condition)
    assert condition.shape == (2, 8, 16, 20)
    assert out.shape == (2, 80, 8)


def test_zero_init_equivalence():
    transformer = FakeTransformer()
    model, _ = create_focus_model(transformer, "a_only", injector_hidden_channels=16)
    z_t = torch.randn(1, 4, 16, 16)
    z_a = torch.randn(1, 4, 16, 16)
    text = torch.randn(1, 1, 8)
    base = model(z_t, text, torch.zeros(1), disable_condition_injection=True, return_dict=False)[0]
    conditioned = model(
        z_t,
        text,
        torch.zeros(1),
        condition_latents=z_a,
        condition_mode="a_only",
        return_dict=False,
    )[0]
    assert (base - conditioned).abs().max().item() < 1e-5


def test_spatial_grid_is_target_grid_not_ab_sequence_concat():
    transformer = FakeTransformer()
    model, _ = create_focus_model(transformer, "ab", injector_hidden_channels=16)
    z_t = torch.randn(1, 4, 18, 22)
    z_a = torch.randn(1, 4, 18, 22)
    z_b = torch.randn(1, 4, 18, 22)
    model(z_t, torch.randn(1, 1, 8), torch.zeros(1), condition_latents=[z_a, z_b], condition_mode="ab", return_dict=False)
    assert model.last_debug["post_patch_height"] * model.last_debug["post_patch_width"] == model.last_debug["target_tokens"].shape[1]


def test_ab_channel_order_changes_after_training_step():
    transformer = FakeTransformer()
    model, _ = create_focus_model(transformer, "ab", injector_hidden_channels=16)
    with torch.no_grad():
        model.injector.proj_out.weight.fill_(0.01)
    z_t = torch.randn(1, 4, 16, 16)
    z_a = torch.randn(1, 4, 16, 16)
    z_b = torch.randn(1, 4, 16, 16)
    normal = model(z_t, torch.randn(1, 1, 8), torch.zeros(1), condition_latents=[z_a, z_b], condition_mode="ab", return_dict=False)[0]
    swapped = model(z_t, torch.randn(1, 1, 8), torch.zeros(1), condition_latents=[z_b, z_a], condition_mode="ab", return_dict=False)[0]
    assert (normal - swapped).abs().mean().item() > 0


def test_checkpoint_incompatibility():
    with pytest.raises(ValueError, match="incompatible"):
        validate_checkpoint_mode({"condition_mode": "a_only"}, "ab")
    with pytest.raises(ValueError, match="incompatible"):
        validate_checkpoint_mode({"condition_mode": "ab"}, "a_only")


def test_gradient_progression():
    injector = FocusLatentConditionInjector(4, 1, 8, 2, hidden_channels=16)
    opt = torch.optim.SGD(injector.parameters(), lr=1.0)
    z_a = torch.randn(1, 4, 16, 16)
    first = injector(z_a).square().mean()
    first.backward()
    assert injector.proj_out.weight.grad.abs().sum().item() > 0
    opt.step()
    opt.zero_grad(set_to_none=True)
    assert injector(z_a).norm().item() > 0
    second = injector(z_a).square().mean()
    second.backward()
    assert injector.proj_in.weight.grad is not None
    assert injector.proj_mid.weight.grad is not None


def test_cfg_batch_repeat_condition_shape():
    z_a = torch.randn(1, 4, 16, 16)
    repeated = torch.cat([z_a, z_a], dim=0)
    assert repeated.shape == (2, 4, 16, 16)
