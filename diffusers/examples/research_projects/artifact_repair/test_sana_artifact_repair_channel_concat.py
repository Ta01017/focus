import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from torch import nn

from diffusers.models.embeddings import PatchEmbed

from sana_artifact_repair_channel_concat import (
    CONCAT_ORDER,
    expand_sana_patch_embedding_for_channel_concat,
    get_sana_patch_embedding,
    load_route2_patch_embedding,
)


class TinyTransformer(nn.Module):
    def __init__(self, channels=4, hidden=8):
        super().__init__()
        self.config = SimpleNamespace(in_channels=channels, out_channels=channels, patch_size=1)
        self.patch_embed = PatchEmbed(height=8, width=8, patch_size=1, in_channels=channels, embed_dim=hidden, pos_embed_type=None)
        self.proj_out = nn.Linear(hidden, channels)

    def register_to_config(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self.config, key, value)

    def forward(self, hidden_states):
        height, width = hidden_states.shape[-2:]
        tokens = self.patch_embed(hidden_states)
        tokens = self.proj_out(tokens)
        return tokens.transpose(1, 2).reshape(hidden_states.shape[0], -1, height, width)


def test_initialization_compatibility():
    torch.manual_seed(0)
    original = TinyTransformer()
    expanded = copy.deepcopy(original)
    z_t = torch.randn(2, 4, 8, 8)
    z_src = torch.randn(2, 4, 8, 8)
    out_original = original(z_t)
    expand_sana_patch_embedding_for_channel_concat(expanded)
    out_expanded = expanded(torch.cat([z_t, z_src], dim=1))
    assert (out_original - out_expanded).abs().max().item() < 1e-5


def test_condition_channel_gradient():
    torch.manual_seed(0)
    model = TinyTransformer()
    c = expand_sana_patch_embedding_for_channel_concat(model)
    out = model(torch.cat([torch.randn(2, c, 8, 8), torch.randn(2, c, 8, 8)], dim=1))
    loss = out.square().mean()
    loss.backward()
    _, proj = get_sana_patch_embedding(model)
    grad = proj.weight.grad[:, c : 2 * c]
    assert grad is not None
    assert grad.abs().sum().item() > 0


def test_condition_effect_after_one_update():
    torch.manual_seed(0)
    model = TinyTransformer()
    c = expand_sana_patch_embedding_for_channel_concat(model)
    optimizer = torch.optim.SGD(model.patch_embed.parameters(), lr=1e-1)
    z_t = torch.randn(2, c, 8, 8)
    z_src = torch.randn(2, c, 8, 8)
    loss = model(torch.cat([z_t, z_src], dim=1)).square().mean()
    loss.backward()
    optimizer.step()
    pred_real = model(torch.cat([z_t, z_src], dim=1))
    pred_zero = model(torch.cat([z_t, torch.zeros_like(z_src)], dim=1))
    assert (pred_real - pred_zero).abs().sum().item() > 0


def test_save_load_patch_embedding_consistency():
    torch.manual_seed(0)
    model = TinyTransformer()
    c = expand_sana_patch_embedding_for_channel_concat(model)
    z_t = torch.randn(1, c, 8, 8)
    z_src = torch.randn(1, c, 8, 8)
    expected = model(torch.cat([z_t, z_src], dim=1))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        save_file({k: v.detach().cpu() for k, v in model.patch_embed.state_dict().items()}, tmp / "i2i_patch_embedding.safetensors")
        loaded = TinyTransformer()
        expand_sana_patch_embedding_for_channel_concat(loaded)
        load_route2_patch_embedding(tmp, loaded)
        actual = loaded(torch.cat([z_t, z_src], dim=1))
    assert (expected - actual).abs().max().item() < 1e-6


def test_concat_order():
    assert CONCAT_ORDER == ["current_latent", "src_latent"]
