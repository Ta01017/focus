import pytest
import torch
from torch import nn

from sana_focus_wan_crossattn import FocusWanImageProjector, expand_sana_patch_embedding_for_focus_wan, validate_focus_wan_checkpoint


class FakePatch(nn.Module):
    def __init__(self,c=4,d=8):
        super().__init__(); self.proj=nn.Conv2d(c,d,1)
    def forward(self,x): return self.proj(x).flatten(2).transpose(1,2)

class FakeTransformer(nn.Module):
    def __init__(self,c=4,d=8):
        super().__init__(); self.config=type('Cfg',(),{'in_channels':c,'patch_size':1,'cross_attention_dim':d})(); self.patch_embed=FakePatch(c,d)


def test_single_expands_c_to_2c_and_copies_target():
    t=FakeTransformer(); old=t.patch_embed.proj.weight.detach().clone(); c=expand_sana_patch_embedding_for_focus_wan(t,'single')
    assert c==4; assert t.patch_embed.proj.in_channels==8
    assert torch.allclose(t.patch_embed.proj.weight[:,:4], old)
    assert t.patch_embed.proj.weight[:,4:8].norm().item()==0


def test_dual_expands_c_to_3c_and_keeps_token_grid():
    t=FakeTransformer(); c=expand_sana_patch_embedding_for_focus_wan(t,'dual')
    assert c==4; assert t.patch_embed.proj.in_channels==12
    x=torch.randn(2,12,5,7); tokens=t.patch_embed(x)
    assert tokens.shape[1]==5*7


def test_projector_and_image_token_batch_shape():
    p=FocusWanImageProjector(16,8); x=torch.randn(1,11,16); y=p(x)
    assert y.shape==(1,11,8)
    assert torch.cat([y,y],0).shape[0]==2


def test_checkpoint_config_shape_checks():
    t=FakeTransformer(); expand_sana_patch_embedding_for_focus_wan(t,'single')
    validate_focus_wan_checkpoint({'condition_mode':'single','number_of_image_branches':1,'patch_embed_in_channels':8,'latent_channels':4}, 'single', t)
    with pytest.raises(ValueError):
        validate_focus_wan_checkpoint({'condition_mode':'single','number_of_image_branches':1,'patch_embed_in_channels':8,'latent_channels':4}, 'dual', t)


def test_zero_extra_channels_mean_no_condition_patch_effect_initially():
    t=FakeTransformer(); old=t.patch_embed.proj.weight.detach().clone(); expand_sana_patch_embedding_for_focus_wan(t,'single')
    zt=torch.randn(1,4,4,4); za=torch.randn(1,4,4,4)
    base=torch.nn.functional.conv2d(zt, old, t.patch_embed.proj.bias)
    expanded=t.patch_embed.proj(torch.cat([zt,za],1))
    assert torch.allclose(base, expanded, atol=1e-6)
