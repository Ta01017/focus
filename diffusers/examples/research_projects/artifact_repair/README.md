# SANA Artifact Repair

本目录包含 SANA artifact repair 的研究脚本。

## Route 2: channel concat I2I

Route 2 是一条 T2I → 单图条件 I2I 路线。它采用 Wan T2V→I2V、InstructPix2Pix 等路线中常见的输入方式：

```text
patch_embed(cat([current/noisy target latent, clean src latent], channel))
```

与 Route 1 的区别：

- Route 1：`src latent -> independent CNN injector -> token additive residual`
- Route 2：`cat([noisy/current target latent, clean src latent], channel) -> expanded native SANA patch embedding`

Route 2 不使用 ControlNet，不使用 adapter，不使用 `SrcLatentConditionInjector`，不把图片 token 拼到 text/caption tokens，也不使用 `H×3W` token 拼接。

训练：

```text
noisy target latent + clean source latent condition
```

推理：

```text
pure noise/current latent + clean source latent condition
```

每个 denoising step 都重新按通道拼接 source condition。concat 顺序固定为：

```text
第 0:C 通道：current/noisy target latent
第 C:2C 通道：clean src latent
```

`edit_image[1]` / ref 当前只为兼容 metadata 被读取，暂不参与 Route 2 计算。

Route 2 推理不使用 `strength`，不使用 src latent initialization，也不使用 sliced img2img schedule。初始化始终为 pure noise。

## 文件

- `sana_artifact_repair_channel_concat.py`
- `train_sana_artifact_repair_channel_concat_lora.py`
- `infer_sana_artifact_repair_channel_concat_lora.py`
- `batch_infer_sana_artifact_repair_channel_concat_lora.py`
- `run_sana_artifact_repair_channel_concat_lora.sh`
- `test_sana_artifact_repair_channel_concat.py`
- `test_sana_artifact_repair_channel_concat_real_sana.py`
- `test_sana_artifact_repair_channel_concat_pretrained.py`

## tiny1 训练

```bash
CUDA_VISIBLE_DEVICES=0 MODE=train \
MODEL=Efficient-Large-Model/Sana_600M_1024px_diffusers \
DATASET_METADATA_PATH=/path/to/metadata_train.json \
DATASET_BASE_PATH=/path/to/data-prompt-aug \
OUTPUT_DIR=/path/to/route2_channel_concat_tiny1 \
MAX_SAMPLES=1 \
MAX_TRAIN_STEPS=2000 \
BATCH_SIZE=1 \
TRAIN_MODE=patch_lora \
MIXED_PRECISION=no \
IMAGE_CONDITION_DROPOUT=0 \
MAX_PIXELS=1048576 \
DOWNSCALE_IF_EXCEEDS_MAX_PIXELS=1 \
LOCAL_FILES_ONLY=1 \
bash examples/research_projects/artifact_repair/run_sana_artifact_repair_channel_concat_lora.sh
```

## tiny2 条件敏感性验证

使用两个不同 src-target pair，固定 seed 做条件敏感性推理：

```bash
CUDA_VISIBLE_DEVICES=0 MODE=train \
DATASET_METADATA_PATH=/path/to/metadata_train.json \
DATASET_BASE_PATH=/path/to/data-prompt-aug \
OUTPUT_DIR=/path/to/route2_channel_concat_tiny2 \
MAX_SAMPLES=2 \
MAX_TRAIN_STEPS=2000 \
BATCH_SIZE=1 \
TRAIN_MODE=patch_lora \
MIXED_PRECISION=no \
IMAGE_CONDITION_DROPOUT=0 \
LOCAL_FILES_ONLY=1 \
bash examples/research_projects/artifact_repair/run_sana_artifact_repair_channel_concat_lora.sh
```

```bash
CUDA_VISIBLE_DEVICES=0 MODE=batch_infer \
CHECKPOINT=/path/to/route2_channel_concat_tiny2/checkpoint-2000 \
TEST_METADATA_PATH=/path/to/metadata_test.json \
DATASET_BASE_PATH=/path/to/data-prompt-aug \
OUTPUT_DIR=/path/to/route2_channel_concat_tiny2 \
MAX_SAMPLES=2 \
SEED=0 \
STEPS=20 \
GUIDANCE_SCALE=1.0 \
SAVE_CONDITION_SENSITIVITY_DEBUG=1 \
LOCAL_FILES_ONLY=1 \
bash examples/research_projects/artifact_repair/run_sana_artifact_repair_channel_concat_lora.sh
```

每个结果会保存：

- `output.png`
- `output.png.stats.json`

stats 中包含：

- `route=channel_concat_v1`
- `initialization=pure_noise`
- `condition_type=clean_src_vae_latent`
- `concat_order=["current_latent", "src_latent"]`
- condition sensitivity 指标

## train_mode

- `patch_lora`：默认。训练扩展后的 patch embedding 和 transformer LoRA。
- `full_transformer`：用于 tiny 验证。训练完整 transformer，包括扩展 patch embedding。checkpoint 会保存完整扩展后的 `SanaTransformer2DModel`，推理直接加载该 Transformer，不会回退为 base Transformer + patch-only。
- `patch_only`：只训练扩展 patch embedding，用于诊断 concat 路径。

## checkpoint

`patch_only` checkpoint：

- `route2_config.json`
- `i2i_patch_embedding.safetensors`
- `trainer_state/`，训练中间 checkpoint 保存

加载路径：

1. 加载原始 SANA T2I pipeline。
2. 扩展 patch embedding `C -> 2C`。
3. 加载 `i2i_patch_embedding.safetensors`。

`patch_lora` checkpoint：

- `route2_config.json`
- `i2i_patch_embedding.safetensors`
- `transformer_lora/`
- `trainer_state/`，训练中间 checkpoint 保存

加载路径：

1. 加载原始 SANA T2I pipeline。
2. 扩展 patch embedding `C -> 2C`。
3. 加载 `i2i_patch_embedding.safetensors`。
4. 加载 `transformer_lora/`。

`full_transformer` checkpoint：

- `route2_config.json`
- `i2i_patch_embedding.safetensors`，仅作为调试副本
- `transformer/config.json`
- `transformer/diffusion_pytorch_model.safetensors`
- `trainer_state/`，训练中间 checkpoint 保存

加载路径：

1. 使用 `SanaTransformer2DModel.from_pretrained(checkpoint/transformer)` 加载完整扩展后的 Transformer。
2. 使用 `SanaPipeline.from_pretrained(..., transformer=transformer)` 加载其他组件。
3. 验证 `config.in_channels`、patch projection `in_channels` 和输出通道与 `route2_config.json` 一致。
4. 禁止回退为 base Transformer + patch-only；如果 `transformer/` 缺失会直接报错。

## 测试

测试分三层：

1. `test_sana_artifact_repair_channel_concat.py`
   - 使用最小 TinyTransformer。
   - 快速验证通用 PatchEmbed 扩展 helper、zero-init 等价、condition 通道梯度、保存加载和 concat 顺序。
2. `test_sana_artifact_repair_channel_concat_real_sana.py`
   - 使用真实 `diffusers.SanaTransformer2DModel` 小模型。
   - 覆盖真实 SANA forward、GLUMBConv/unpatchify 路径、full_transformer save/load round-trip、三种 train mode 的 checkpoint 目录结构。
3. `test_sana_artifact_repair_channel_concat_pretrained.py`
   - 可选本地集成测试。
   - 默认跳过；只有 `RUN_SANA_PRETRAINED_TESTS=1` 时加载本地 `Sana_600M_1024px_diffusers` transformer。

快速测试命令：

```bash
PYTHONPATH="$PWD/src:$PWD/examples/research_projects/artifact_repair" \
pytest -q \
  examples/research_projects/artifact_repair/test_sana_artifact_repair_channel_concat.py \
  examples/research_projects/artifact_repair/test_sana_artifact_repair_channel_concat_real_sana.py
```

可选 600M 本地集成测试：

```bash
RUN_SANA_PRETRAINED_TESTS=1 \
SANA_TEST_MODEL="Efficient-Large-Model/Sana_600M_1024px_diffusers" \
SANA_TEST_DEVICE=cpu \
PYTHONPATH="$PWD/src:$PWD/examples/research_projects/artifact_repair" \
pytest -q -s \
  examples/research_projects/artifact_repair/test_sana_artifact_repair_channel_concat_pretrained.py
```


## Focus Fusion – Wan-style Latent Concat and Image Cross-Attention

新增 Focus WAN 路线，代码派生自现有 artifact-repair Wan-style latent-concat and image cross-attention route：

- `sana_artifact_repair_wan_crossattn.py`
- `train_sana_artifact_repair_wan_crossattn.py`
- `infer_sana_artifact_repair_wan_crossattn.py`
- `batch_infer_sana_artifact_repair_wan_crossattn.py`

新路线文件：

```text
sana_focus_wan_crossattn.py
train_sana_focus_wan_crossattn.py
infer_sana_focus_wan_crossattn.py
batch_infer_sana_focus_wan_crossattn.py
run_sana_focus_wan_crossattn.sh
```

### 架构

单图：

```text
condition_mode=single
GT -> z_gt
A/edit_image[0] -> z_a
model input = cat([z_t, z_a], channel)  # 2C patch input
A -> CLIP Vision -> image_projector -> per-block independent image K/V + image_gate
```

双图：

```text
condition_mode=dual
GT -> z_gt
A/edit_image[0] -> z_a
B/edit_image[1] -> z_b
model input = cat([z_t, z_a, z_b], channel)  # 3C patch input
A -> CLIP Vision -> image_projector_a -> image K/V A + gate A
B -> CLIP Vision -> image_projector_b -> image K/V B + gate B
```

文本 cross-attention 保持不变；image cross-attention 使用共享 query、独立 image K/V。focus map 当前只保留 metadata 兼容，不参与模型计算。

### 1. 单图 tiny16 fp32 20-step smoke

```bash
cd /mnt/d/work/vivo/focus/focus/diffusers
CONDITION_MODE=single \
MODE=train \
MAX_SAMPLES=16 \
MAX_TRAIN_STEPS=20 \
MIXED_PRECISION=no \
OUTPUT_DIR=/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/focus/models/train/sana_focus_wan_crossattn/single_tiny16_smoke \
bash examples/research_projects/artifact_repair/run_sana_focus_wan_crossattn.sh
```

### 2. 双图 tiny16 fp32 20-step smoke

```bash
cd /mnt/d/work/vivo/focus/focus/diffusers
CONDITION_MODE=dual \
MODE=train \
MAX_SAMPLES=16 \
MAX_TRAIN_STEPS=20 \
MIXED_PRECISION=no \
OUTPUT_DIR=/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/focus/models/train/sana_focus_wan_crossattn/dual_tiny16_smoke \
bash examples/research_projects/artifact_repair/run_sana_focus_wan_crossattn.sh
```

### 3. 单图正式训练

```bash
cd /mnt/d/work/vivo/focus/focus/diffusers
CONDITION_MODE=single \
MODE=train \
MAX_TRAIN_STEPS=10000 \
MAX_SAMPLES= \
OUTPUT_DIR=/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/focus/models/train/sana_focus_wan_crossattn/single_full \
bash examples/research_projects/artifact_repair/run_sana_focus_wan_crossattn.sh
```

### 4. 双图正式训练

```bash
cd /mnt/d/work/vivo/focus/focus/diffusers
CONDITION_MODE=dual \
MODE=train \
MAX_TRAIN_STEPS=10000 \
MAX_SAMPLES= \
OUTPUT_DIR=/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/focus/models/train/sana_focus_wan_crossattn/dual_full \
bash examples/research_projects/artifact_repair/run_sana_focus_wan_crossattn.sh
```

### 5. Batch inference

单图：

```bash
cd /mnt/d/work/vivo/focus/focus/diffusers
CONDITION_MODE=single \
MODE=batch_infer \
CHECKPOINT=/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/focus/models/train/sana_focus_wan_crossattn/single_full/checkpoint-10000 \
OUTPUT_DIR=/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/focus/outputs/focus_wan_single \
MAX_SAMPLES=16 \
INIT_MODE=a \
STRENGTH=0.3 \
bash examples/research_projects/artifact_repair/run_sana_focus_wan_crossattn.sh
```

双图：

```bash
cd /mnt/d/work/vivo/focus/focus/diffusers
CONDITION_MODE=dual \
MODE=batch_infer \
CHECKPOINT=/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/focus/models/train/sana_focus_wan_crossattn/dual_full/checkpoint-10000 \
OUTPUT_DIR=/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/focus/outputs/focus_wan_dual \
MAX_SAMPLES=16 \
INIT_MODE=a \
STRENGTH=0.3 \
bash examples/research_projects/artifact_repair/run_sana_focus_wan_crossattn.sh
```

Checkpoint 格式：

```text
checkpoint-N/
  focus_wan_config.json
  focus_wan_condition.safetensors
  transformer_lora/pytorch_lora_weights.safetensors
  trainer_state.json
```


### Focus WAN 工程修复说明

当前 Focus Fusion – Wan-style Latent Concat and Image Cross-Attention 路线保持原始结构不变：

```text
single: cat([z_t, z_a], channel), patch input = 2C, image branches = 1
dual:   cat([z_t, z_a, z_b], channel), patch input = 3C, image branches = 2
```

本路线来自已有 artifact repair WAN route。GT 只作为训练 target，推理阶段不输入 GT。

#### 动态尺寸 batch 限制

当前动态尺寸训练只支持：

```text
BATCH_SIZE=1
NUM_WORKERS=0
```

如果需要更大的有效 batch，请使用：

```text
GRADIENT_ACCUMULATION_STEPS > 1
```

脚本会在 `BATCH_SIZE>1` 时明确报错，不会静默丢弃 `samples[1:]`。

#### LoRA 开关

支持：

```bash
TRAIN_TRANSFORMER_LORA=1
TRAIN_TRANSFORMER_LORA=0
```

关闭 LoRA 时仍会训练：

```text
expanded patch embedding condition channels
image projector(s)
image K/V projections
image gate(s)
```

#### init_from_checkpoint 与 resume_from_checkpoint

```text
INIT_FROM_CHECKPOINT：只加载模型权重，optimizer/global_step 从 0 开始。
RESUME_FROM_CHECKPOINT：加载模型权重和 accelerator_state，恢复 optimizer/scaler/global_step。
```

如果 checkpoint 没有 `accelerator_state/`，不能用于 resume，应改用 `INIT_FROM_CHECKPOINT`。

#### sliced 与 pipeline_full

`IMG2IMG_SCHEDULE_MODE=sliced` 会按 strength 截取后半段 timesteps：

```text
steps=20, strength=0.15 -> effective_steps=3
```

`IMG2IMG_SCHEDULE_MODE=pipeline_full` 会执行完整 timestep 序列：

```text
effective_steps = steps
```

两者不再是同一个空参数。

#### focus_composite

`INIT_MODE=focus_composite` 已实现，仅允许 `CONDITION_MODE=dual`，并且必须提供 `focus_a/focus_b`。
它只影响初始 latent：

```text
focus composite RGB -> VAE -> z_init
```

A/B latent concat 条件和 A/B image cross-attention 条件保持不变。

#### checkpoint 自动查找

`run_sana_focus_wan_crossattn.sh` 中，如果没有显式设置 `CHECKPOINT`，推理会在 `OUTPUT_DIR` 下自动查找最新：

```text
checkpoint-*
```

不会再默认把 `OUTPUT_DIR` 当作 checkpoint。

#### 继续训练示例

```bash
cd /mnt/d/work/vivo/focus/focus/diffusers
CONDITION_MODE=single \
MODE=train \
RESUME_FROM_CHECKPOINT=/path/to/checkpoint-1000 \
MAX_TRAIN_STEPS=2000 \
bash examples/research_projects/artifact_repair/run_sana_focus_wan_crossattn.sh
```

#### 从 checkpoint 初始化重新训练示例

```bash
cd /mnt/d/work/vivo/focus/focus/diffusers
CONDITION_MODE=dual \
MODE=train \
INIT_FROM_CHECKPOINT=/path/to/checkpoint-1000 \
OUTPUT_DIR=/path/to/new_run \
bash examples/research_projects/artifact_repair/run_sana_focus_wan_crossattn.sh
```
