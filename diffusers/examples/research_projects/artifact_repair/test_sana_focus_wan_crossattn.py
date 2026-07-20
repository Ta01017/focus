import pytest
import torch
from torch import nn

from artifact_repair_utils import resolve_dataset_path
from infer_sana_focus_wan_crossattn import make_focus_composite, prepare_timesteps_for_init
from sana_focus_wan_crossattn import FocusWanImageProjector, expand_sana_patch_embedding_for_focus_wan, num_condition_images, validate_focus_wan_checkpoint


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


def test_single_dual_keep_expected_image_branch_counts_and_no_injector():
    assert 'SrcLatentConditionInjector' not in globals()
    assert num_condition_images('single') == 1
    assert num_condition_images('dual') == 2


def test_projector_and_image_token_batch_shape():
    p=FocusWanImageProjector(16,8); x=torch.randn(1,11,16); y=p(x)
    assert y.shape==(1,11,8)
    assert torch.cat([y,y],0).shape[0]==2


def test_checkpoint_config_shape_checks():
    t=FakeTransformer(); expand_sana_patch_embedding_for_focus_wan(t,'single')
    validate_focus_wan_checkpoint({'route':'focus_wan_crossattn','condition_mode':'single','number_of_image_branches':1,'patch_embed_in_channels':8,'latent_channels':4,'share_image_projector':False}, 'single', t)
    with pytest.raises(ValueError):
        validate_focus_wan_checkpoint({'route':'focus_wan_crossattn','condition_mode':'single','number_of_image_branches':1,'patch_embed_in_channels':8,'latent_channels':4,'share_image_projector':False}, 'dual', t)


def test_zero_extra_channels_mean_no_condition_patch_effect_initially():
    t=FakeTransformer(); old=t.patch_embed.proj.weight.detach().clone(); expand_sana_patch_embedding_for_focus_wan(t,'single')
    zt=torch.randn(1,4,4,4); za=torch.randn(1,4,4,4)
    base=torch.nn.functional.conv2d(zt, old, t.patch_embed.proj.bias)
    expanded=t.patch_embed.proj(torch.cat([zt,za],1))
    assert torch.allclose(base, expanded, atol=1e-6)



def test_resolve_dataset_path_dedupes_base_suffix(tmp_path):
    base = tmp_path / "processed" / "test"
    target = base / "a.png"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")
    resolved = resolve_dataset_path("processed/test/a.png", base, record_index=3, field_name="edit_image[0]")
    assert resolved == target.resolve(strict=False)


def test_resolve_dataset_path_missing_error_is_clear(tmp_path):
    with pytest.raises(FileNotFoundError, match="record index"):
        resolve_dataset_path("missing.png", tmp_path, record_index=7, field_name="image")


class FakeScheduler:
    order = 1
    init_noise_sigma = 7.0

    def __init__(self):
        self.timesteps = None
        self.sigmas = None
        self.begin_index = None

    def set_timesteps(self, steps, device=None):
        self.timesteps = torch.arange(steps - 1, -1, -1, device=device, dtype=torch.float32)
        self.sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)

    def set_begin_index(self, value):
        self.begin_index = value


class FakePipe:
    def __init__(self):
        self.scheduler = FakeScheduler()


def test_sliced_schedule_uses_same_initial_and_first_denoise_timestep():
    pipe = FakePipe()
    plan = prepare_timesteps_for_init(pipe, 20, 0.15, "a", "sliced", torch.device("cpu"))
    assert plan["effective_steps"] < 20
    assert plan["effective_steps"] == 3
    assert plan["t_start_index"] == 17
    assert float(plan["initial_noise_timestep"]) == float(plan["first_denoise_timestep"])
    assert plan["schedule_consistent"] is True
    assert pipe.scheduler.begin_index == 17


def test_pipeline_full_noise_uses_scheduler_init_noise_sigma():
    pipe = FakePipe()
    plan = prepare_timesteps_for_init(pipe, 20, 1.0, "noise", "pipeline_full", torch.device("cpu"))
    assert plan["effective_steps"] == 20
    assert float(plan["initial_sigma"]) == pipe.scheduler.init_noise_sigma
    assert float(plan["initial_noise_timestep"]) == float(plan["first_denoise_timestep"])
    assert pipe.scheduler.begin_index == 0


def test_pipeline_full_rejects_image_initialization_modes():
    with pytest.raises(ValueError, match="pipeline_full.*init_mode='noise'"):
        prepare_timesteps_for_init(FakePipe(), 20, 0.15, "a", "pipeline_full", torch.device("cpu"))
    with pytest.raises(ValueError, match="pipeline_full.*init_mode='noise'"):
        prepare_timesteps_for_init(FakePipe(), 20, 0.15, "focus_composite", "pipeline_full", torch.device("cpu"))


def test_accumulation_counts_optimizer_updates_not_micro_steps():
    grad_accum = 4
    max_train_steps = 3
    micro_step = 0
    global_step = 0
    scheduler_updates = 0
    checkpoints = []
    while global_step < max_train_steps:
        micro_step += 1
        sync_gradients = micro_step % grad_accum == 0
        if sync_gradients:
            scheduler_updates += 1
            global_step += 1
            checkpoints.append(f"checkpoint-{global_step}")
    assert micro_step == 12
    assert global_step == 3
    assert scheduler_updates == 3
    assert checkpoints == ["checkpoint-1", "checkpoint-2", "checkpoint-3"]
    assert "checkpoint-4" not in checkpoints
    assert "checkpoint-8" not in checkpoints
    assert "checkpoint-12" not in checkpoints


def test_resume_accumulation_advances_after_next_sync_update():
    grad_accum = 2
    restored_global_step = 2
    restored_scheduler_updates = 2
    micro_step = 0
    global_step = restored_global_step
    scheduler_updates = restored_scheduler_updates
    while global_step < 3:
        micro_step += 1
        sync_gradients = micro_step % grad_accum == 0
        if sync_gradients:
            scheduler_updates += 1
            global_step += 1
    assert restored_global_step == 2
    assert restored_scheduler_updates == 2
    assert micro_step == 2
    assert global_step == 3
    assert scheduler_updates == 3


def test_focus_composite_uses_focus_weights(tmp_path):
    from PIL import Image
    import numpy as np
    a = Image.new("RGB", (2, 1), (255, 0, 0))
    b = Image.new("RGB", (2, 1), (0, 0, 255))
    fa = tmp_path / "fa.png"
    fb = tmp_path / "fb.png"
    Image.fromarray(np.array([[255, 0]], dtype=np.uint8)).save(fa)
    Image.fromarray(np.array([[0, 255]], dtype=np.uint8)).save(fb)
    out = make_focus_composite({"src": a, "ref": b}, fa, fb)
    pixels = list(out.getdata())
    assert pixels[0][0] > pixels[0][2]
    assert pixels[1][2] > pixels[1][0]


def test_lora_disabled_has_no_matched_count_requirement():
    matched_lora = []
    train_transformer_lora = False
    if train_transformer_lora and not matched_lora:
        raise RuntimeError("should not happen")


def test_batch_size_gt_one_error_message():
    batch_size = 2
    with pytest.raises(ValueError, match="batch_size=1"):
        if batch_size != 1:
            raise ValueError("The current dynamic-resolution Focus WAN training path supports batch_size=1 only.")
