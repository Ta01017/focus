import types

import torch
from torch import nn

from sana_artifact_repair_wan_crossattn import (
    IMPLEMENTATION,
    Route3ImageCrossAttentionAdapter,
    Route3ImageProjector,
    Route3SanaModel,
)


class FakeAttn(nn.Module):
    def __init__(self, dim=32, heads=4):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.norm_q = None
        self.norm_k = None
        self.to_out = nn.ModuleList([nn.Linear(dim, dim), nn.Dropout(0.0)])
        self.rescale_output_factor = 1.0
        self.processor = None

    def set_processor(self, processor):
        self.processor = processor

    def prepare_attention_mask(self, attention_mask, sequence_length, batch_size):
        return attention_mask

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask)


class FakeBlock(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.attn2 = FakeAttn(dim)

    def forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, timestep, height, width):
        return hidden_states + self.attn2(hidden_states, encoder_hidden_states, encoder_attention_mask)


class FakeTransformer(nn.Module):
    def __init__(self, dim=32, layers=2):
        super().__init__()
        self.config = types.SimpleNamespace(patch_size=1, cross_attention_dim=dim, timestep_scale=1.0, in_channels=64)
        self.transformer_blocks = nn.ModuleList([FakeBlock(dim) for _ in range(layers)])


def test_projector_shape():
    projector = Route3ImageProjector(16, 32)
    x = torch.randn(2, 7, 16)
    assert projector(x).shape == (2, 7, 32)


def test_adapter_installs_independent_image_kv_and_gates():
    transformer = FakeTransformer()
    adapter = Route3ImageCrossAttentionAdapter(transformer, image_gate_init=1e-3)
    assert len(adapter.layers) == 2
    assert all(hasattr(layer, "to_k_img") and hasattr(layer, "to_v_img") for layer in adapter.layers)
    assert all(float(layer.image_gate.detach()) == 1e-3 for layer in adapter.layers)
    assert all(block.attn2.processor is not None for block in transformer.transformer_blocks)


def test_dynamic_image_token_lengths_change_runtime():
    transformer = FakeTransformer()
    adapter = Route3ImageCrossAttentionAdapter(transformer, image_gate_init=1e-3)
    hidden = torch.randn(1, 8, 32)
    text = torch.randn(1, 5, 32)
    img_short = torch.randn(1, 3, 32)
    img_long = torch.randn(1, 9, 32)
    adapter.set_runtime(3, image_cross_attention_scale=1.0)
    out_short = transformer.transformer_blocks[0].attn2(hidden, torch.cat([text, img_short], dim=1))
    adapter.set_runtime(9, image_cross_attention_scale=1.0)
    out_long = transformer.transformer_blocks[0].attn2(hidden, torch.cat([text, img_long], dim=1))
    assert out_short.shape == hidden.shape
    assert out_long.shape == hidden.shape
    assert not torch.allclose(out_short, out_long)


def test_disable_image_cross_attention_matches_text_only():
    transformer = FakeTransformer()
    adapter = Route3ImageCrossAttentionAdapter(transformer, image_gate_init=1e-3)
    hidden = torch.randn(1, 8, 32)
    text = torch.randn(1, 5, 32)
    img = torch.randn(1, 4, 32)
    adapter.set_runtime(4, image_cross_attention_scale=1.0, disable_image_cross_attention=True)
    out_disabled = transformer.transformer_blocks[0].attn2(hidden, torch.cat([text, img], dim=1))
    adapter.set_runtime(0, image_cross_attention_scale=1.0, disable_image_cross_attention=False)
    out_text = transformer.transformer_blocks[0].attn2(hidden, text)
    assert torch.allclose(out_disabled, out_text, atol=1e-6, rtol=1e-5)


def test_image_adapter_gradients_flow():
    transformer = FakeTransformer()
    adapter = Route3ImageCrossAttentionAdapter(transformer, image_gate_init=1e-3)
    hidden = torch.randn(1, 8, 32, requires_grad=True)
    text = torch.randn(1, 5, 32)
    img = torch.randn(1, 4, 32)
    adapter.set_runtime(4, image_cross_attention_scale=1.0)
    out = transformer.transformer_blocks[0].attn2(hidden, torch.cat([text, img], dim=1))
    out.square().mean().backward()
    layer = adapter.layers[0]
    assert layer.to_k_img.weight.grad is not None
    assert layer.to_v_img.weight.grad is not None
    assert layer.image_gate.grad is not None


def test_implementation_name():
    assert IMPLEMENTATION == "sana_artifact_repair_wan_crossattn_v1"
