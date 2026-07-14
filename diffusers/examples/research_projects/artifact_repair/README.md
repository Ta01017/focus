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
- `full_transformer`：用于 tiny 验证。训练完整 transformer，包括扩展 patch embedding。
- `patch_only`：只训练扩展 patch embedding，用于诊断 concat 路径。

## checkpoint

每个 checkpoint 保存：

- `route2_config.json`
- `i2i_patch_embedding.safetensors`
- `transformer_lora/`，仅 `patch_lora` 模式保存
- `trainer_state/`，训练中间 checkpoint 保存

加载顺序固定为：

1. 加载原始 SANA T2I pipeline。
2. 扩展 patch embedding `C -> 2C`。
3. 加载 `i2i_patch_embedding.safetensors`。
4. 加载 transformer LoRA。
5. 切换 eval。
