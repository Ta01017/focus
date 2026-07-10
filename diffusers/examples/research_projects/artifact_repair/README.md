# SANA Artifact Repair / Bottom-Surface Repair

本目录提供一套面向 artifact repair / bottom-surface repair 的 SANA 训练与推理方案。任务目标不是景深融合，而是利用 `src` 保留结构，用 `ref` 提供底面外观、颜色和纹理参考，输出修复后的图像。

## 1. 项目概述

每条样本包含：

- `src`：结构参考图，可能包含 blur、artifacts、holes 或 missing regions。
- `ref`：appearance / color reference，只用于底面外观、颜色和纹理参考。
- `prompt`：来自 metadata 的文本描述。
- `image`：修复后的 GT 图像。

目标输出应尽量继承 `src` 的几何结构、物体形状、视角和空间关系；`ref` 只影响 appearance / color / texture，不应改变几何结构。

## 2. 当前路线

### A. SANA from-src / without ControlNet

入口文件：

- `train_sana_artifact_repair_froma.py`
- `infer_sana_artifact_repair_froma.py`
- `batch_infer_sana_artifact_repair_froma.py`
- `run_sana_artifact_repair_froma.sh`

特点：

- 训练和推理都使用 src latent initialization。
- 使用轻量 `[src_rgb, ref_rgb] -> latent residual` adapter。
- 支持 optional transformer LoRA。
- 默认使用 sliced img2img schedule。
- 不依赖 ControlNet，更轻量，适合优先做 tiny overfit 和快速验证。

### B. SANA ControlNet

入口文件：

- `train_sana_artifact_repair_controlnet.py`
- `infer_sana_artifact_repair_controlnet.py`
- `batch_infer_sana_artifact_repair_controlnet.py`
- `run_sana_artifact_repair_controlnet.sh`

特点：

- 使用 `[src_rgb, ref_rgb]` 作为 6-channel ControlNet condition。
- ControlNet 和 condition projection 可训练。
- 支持 optional transformer LoRA。
- 条件注入更强，但如果冻结主干，只训练 ControlNet，细节修复和去模糊能力可能有限。

## 3. 数据格式

metadata 示例：

```json
{
  "image": "processed/train/0103dc6af04d/109_gt.png",
  "edit_image": [
    "processed/train/0103dc6af04d/109_src.png",
    "processed/train/0103dc6af04d/109_ref.png"
  ],
  "prompt": "Image 1 is ...",
  "obj_name": "0103dc6af04d",
  "test_uid": 109,
  "src": "processed/train/0103dc6af04d/109_src.png",
  "ref": "processed/train/0103dc6af04d/109_ref.png"
}
```

读取规则：

- GT：优先读取 `image`。
- src：优先读取 `src`，否则读取 `edit_image[0]`。
- ref：优先读取 `ref`，否则读取 `edit_image[1]`。
- prompt：优先读取 `prompt`，否则使用默认 repair prompt。
- 输出命名：优先使用 `obj_name + "_" + test_uid + ".png"`，否则使用 index。

路径可以是绝对路径，也可以是相对 `DATASET_BASE_PATH` 的路径。

## 4. 为什么必须使用 src latent initialization

SANA 基模本质是 text-to-image 模型。若不使用 src latent initialization，仅依赖 prompt 或弱图像条件，模型不天然保证继承输入图像结构。

对 restoration / repair / fusion 这类 image-conditioned 任务，纯文本或弱条件容易导致：

- 输出结构与 src 无关；
- 内容漂移；
- 局部拼接感明显；
- 颜色或纹理不稳定；
- ref 影响 geometry，而不仅是 appearance。

src latent initialization 能提供强结构先验。当前推理默认使用 sliced img2img schedule，是因为它会让初始 latent 的噪声等级与实际 denoising timesteps 对齐，通常比 `pipeline_full` 更稳定。`pipeline_full` 保留为 debug 模式，不作为默认推荐。

## 5. 已观察到的实验现象

- 不使用 latent A / src latent init 时，经常学不到有效输入条件，输出容易和 src 无关。
- 使用 src latent init 后，结构保持和颜色稳定性明显改善。
- `pipeline_full` 在 from-A / from-src 推理中可能出现颜色异常或噪声等级不匹配。
- 当前默认推荐 `sliced` img2img schedule。
- 冻结 transformer、只训练 ControlNet 可以改善结构控制，但细节修复和去模糊能力有限。
- 如果需要更强修复能力，建议训练 transformer LoRA，而不是只依赖冻结主干。
- 高分辨率数据不建议固定成 1024x1024。本目录脚本保持原图比例，不裁边；只有超过 `MAX_PIXELS` 且开启 `DOWNSCALE_IF_EXCEEDS_MAX_PIXELS=1` 时才等比缩小，然后右/下 padding 到 `SIZE_DIVISOR`。

## 6. 安装与环境变量

建议从仓库根目录运行：

```bash
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/HF"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
```

需要安装 PyTorch、Accelerate、PEFT、safetensors、Pillow，以及当前 diffusers 源码环境。

## 7. 训练 without ControlNet

tiny1 overfit：

```bash
CUDA_VISIBLE_DEVICES=1 MODE=train \
DATASET_METADATA_PATH=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora/metadata_train.json \
DATASET_BASE_PATH=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora \
OUTPUT_DIR=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/focus/models/train/sana_artifact_repair/froma_tiny1_lora_$(date +%Y%m%d_%H%M%S) \
TRAIN_MAX_SAMPLES=1 \
MAX_TRAIN_STEPS=2000 \
SAVE_STEPS=500 \
LOG_STEPS=10 \
TRAIN_BATCH_SIZE=1 \
LEARNING_RATE=2e-5 \
LOSS_MODE=legacy_src_noise_gt_velocity \
TRAIN_TRANSFORMER_LORA=1 \
LORA_RANK=8 \
LORA_ALPHA=8 \
LORA_TARGET_MODULES="to_q,to_k,to_v,to_out.0" \
MAX_PIXELS=1048576 \
DOWNSCALE_IF_EXCEEDS_MAX_PIXELS=1 \
LOCAL_FILES_ONLY=1 \
bash examples/research_projects/artifact_repair/run_sana_artifact_repair_froma.sh
```

smoke test：

```bash
CUDA_VISIBLE_DEVICES=1 MODE=train \
DATASET_METADATA_PATH=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora/metadata_train.json \
DATASET_BASE_PATH=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora \
OUTPUT_DIR=/tmp/sana_artifact_repair_froma_smoke \
TRAIN_MAX_SAMPLES=1 \
MAX_TRAIN_STEPS=2 \
SAVE_STEPS=2 \
TRAIN_BATCH_SIZE=1 \
TRAIN_TRANSFORMER_LORA=0 \
MAX_PIXELS=1048576 \
DOWNSCALE_IF_EXCEEDS_MAX_PIXELS=1 \
LOCAL_FILES_ONLY=1 \
bash examples/research_projects/artifact_repair/run_sana_artifact_repair_froma.sh
```

full training 示例：

```bash
CUDA_VISIBLE_DEVICES=1 MODE=train \
DATASET_METADATA_PATH=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora/metadata_train.json \
DATASET_BASE_PATH=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora \
OUTPUT_DIR=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/focus/models/train/sana_artifact_repair/froma_full_lora \
MAX_TRAIN_STEPS=50000 \
SAVE_STEPS=1000 \
LOG_STEPS=10 \
TRAIN_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=4 \
LEARNING_RATE=2e-5 \
LOSS_MODE=legacy_src_noise_gt_velocity \
TRAIN_TRANSFORMER_LORA=1 \
LORA_RANK=8 \
LORA_ALPHA=8 \
LORA_TARGET_MODULES="to_q,to_k,to_v,to_out.0" \
MAX_PIXELS=1048576 \
DOWNSCALE_IF_EXCEEDS_MAX_PIXELS=1 \
LOCAL_FILES_ONLY=1 \
bash examples/research_projects/artifact_repair/run_sana_artifact_repair_froma.sh
```

## 8. 训练 ControlNet

tiny1 overfit：

```bash
CUDA_VISIBLE_DEVICES=1 MODE=train \
DATASET_METADATA_PATH=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora/metadata_train.json \
DATASET_BASE_PATH=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora \
OUTPUT_DIR=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/focus/models/train/sana_artifact_repair/controlnet_tiny1_lora_$(date +%Y%m%d_%H%M%S) \
TRAIN_MAX_SAMPLES=1 \
MAX_TRAIN_STEPS=2000 \
SAVE_STEPS=500 \
LOG_STEPS=10 \
TRAIN_BATCH_SIZE=1 \
LEARNING_RATE=2e-5 \
LOSS_MODE=legacy_src_noise_gt_velocity \
CONTROL_CONDITION_CHANNELS=6 \
TRAIN_TRANSFORMER_LORA=1 \
LORA_RANK=8 \
LORA_ALPHA=8 \
LORA_TARGET_MODULES="to_q,to_k,to_v,to_out.0" \
MAX_PIXELS=1048576 \
DOWNSCALE_IF_EXCEEDS_MAX_PIXELS=1 \
LOCAL_FILES_ONLY=1 \
bash examples/research_projects/artifact_repair/run_sana_artifact_repair_controlnet.sh
```

smoke test：

```bash
CUDA_VISIBLE_DEVICES=1 MODE=train \
DATASET_METADATA_PATH=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora/metadata_train.json \
DATASET_BASE_PATH=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora \
OUTPUT_DIR=/tmp/sana_artifact_repair_controlnet_smoke \
TRAIN_MAX_SAMPLES=1 \
MAX_TRAIN_STEPS=2 \
SAVE_STEPS=2 \
TRAIN_BATCH_SIZE=1 \
CONTROL_CONDITION_CHANNELS=6 \
TRAIN_TRANSFORMER_LORA=0 \
MAX_PIXELS=1048576 \
DOWNSCALE_IF_EXCEEDS_MAX_PIXELS=1 \
LOCAL_FILES_ONLY=1 \
bash examples/research_projects/artifact_repair/run_sana_artifact_repair_controlnet.sh
```

full training 示例：

```bash
CUDA_VISIBLE_DEVICES=1 MODE=train \
DATASET_METADATA_PATH=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora/metadata_train.json \
DATASET_BASE_PATH=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora \
OUTPUT_DIR=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/focus/models/train/sana_artifact_repair/controlnet_full_lora \
MAX_TRAIN_STEPS=50000 \
SAVE_STEPS=1000 \
LOG_STEPS=10 \
TRAIN_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=4 \
LEARNING_RATE=2e-5 \
LOSS_MODE=legacy_src_noise_gt_velocity \
CONTROL_CONDITION_CHANNELS=6 \
TRAIN_TRANSFORMER_LORA=1 \
LORA_RANK=8 \
LORA_ALPHA=8 \
LORA_TARGET_MODULES="to_q,to_k,to_v,to_out.0" \
MAX_PIXELS=1048576 \
DOWNSCALE_IF_EXCEEDS_MAX_PIXELS=1 \
LOCAL_FILES_ONLY=1 \
bash examples/research_projects/artifact_repair/run_sana_artifact_repair_controlnet.sh
```

## 9. 推理 without ControlNet

单图推理：

```bash
CUDA_VISIBLE_DEVICES=1 MODE=infer \
CHECKPOINT=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/focus/models/train/sana_artifact_repair/froma_tiny1_lora/checkpoint-2000 \
SRC_IMAGE=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora/processed/val/example_src.png \
REF_IMAGE=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora/processed/val/example_ref.png \
OUTPUT_PATH=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/focus/models/train/sana_artifact_repair/infer/froma_single_output.png \
STRENGTH=0.15 \
IMG2IMG_SCHEDULE_MODE=sliced \
ADAPTER_SCALE=1.0 \
STEPS=20 \
GUIDANCE_SCALE=1.0 \
MAX_PIXELS=1048576 \
DOWNSCALE_IF_EXCEEDS_MAX_PIXELS=1 \
LOCAL_FILES_ONLY=1 \
bash examples/research_projects/artifact_repair/run_sana_artifact_repair_froma.sh
```

batch 推理：

```bash
CUDA_VISIBLE_DEVICES=1 MODE=batch_infer \
CHECKPOINT=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/focus/models/train/sana_artifact_repair/froma_tiny1_lora/checkpoint-2000 \
DATASET_METADATA_PATH=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora/metadata_val.json \
DATASET_BASE_PATH=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora \
OUTPUT_DIR=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/focus/models/train/sana_artifact_repair/infer/froma_val \
STRENGTH=0.15 \
IMG2IMG_SCHEDULE_MODE=sliced \
ADAPTER_SCALE=1.0 \
STEPS=20 \
GUIDANCE_SCALE=1.0 \
MAX_PIXELS=1048576 \
DOWNSCALE_IF_EXCEEDS_MAX_PIXELS=1 \
LOCAL_FILES_ONLY=1 \
bash examples/research_projects/artifact_repair/run_sana_artifact_repair_froma.sh
```

strength sweep：

```bash
for S in 0.05 0.08 0.12 0.15 0.2 0.25 0.3; do
  CUDA_VISIBLE_DEVICES=1 MODE=batch_infer \
  CHECKPOINT=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/focus/models/train/sana_artifact_repair/froma_tiny1_lora/checkpoint-2000 \
  DATASET_METADATA_PATH=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora/metadata_val.json \
  DATASET_BASE_PATH=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora \
  OUTPUT_DIR=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/focus/models/train/sana_artifact_repair/infer/froma_s${S} \
  STRENGTH=${S} \
  IMG2IMG_SCHEDULE_MODE=sliced \
  ADAPTER_SCALE=1.0 \
  STEPS=20 \
  GUIDANCE_SCALE=1.0 \
  MAX_SAMPLES=20 \
  MAX_PIXELS=1048576 \
  DOWNSCALE_IF_EXCEEDS_MAX_PIXELS=1 \
  LOCAL_FILES_ONLY=1 \
  bash examples/research_projects/artifact_repair/run_sana_artifact_repair_froma.sh
done
```

adapter scale sweep：

```bash
for AS in 0.5 1.0 1.5 2.0; do
  CUDA_VISIBLE_DEVICES=1 MODE=batch_infer \
  CHECKPOINT=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/focus/models/train/sana_artifact_repair/froma_tiny1_lora/checkpoint-2000 \
  DATASET_METADATA_PATH=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora/metadata_val.json \
  DATASET_BASE_PATH=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora \
  OUTPUT_DIR=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/focus/models/train/sana_artifact_repair/infer/froma_as${AS} \
  STRENGTH=0.15 \
  ADAPTER_SCALE=${AS} \
  IMG2IMG_SCHEDULE_MODE=sliced \
  LOCAL_FILES_ONLY=1 \
  bash examples/research_projects/artifact_repair/run_sana_artifact_repair_froma.sh
done
```

## 10. 推理 ControlNet

单图推理：

```bash
CUDA_VISIBLE_DEVICES=1 MODE=infer \
CHECKPOINT=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/focus/models/train/sana_artifact_repair/controlnet_tiny1_lora/checkpoint-2000 \
SRC_IMAGE=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora/processed/val/example_src.png \
REF_IMAGE=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora/processed/val/example_ref.png \
OUTPUT_PATH=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/focus/models/train/sana_artifact_repair/infer/controlnet_single_output.png \
STRENGTH=0.2 \
IMG2IMG_SCHEDULE_MODE=sliced \
CONDITIONING_SCALE=1.0 \
STEPS=20 \
GUIDANCE_SCALE=1.0 \
MAX_PIXELS=1048576 \
DOWNSCALE_IF_EXCEEDS_MAX_PIXELS=1 \
LOCAL_FILES_ONLY=1 \
bash examples/research_projects/artifact_repair/run_sana_artifact_repair_controlnet.sh
```

batch 推理：

```bash
CUDA_VISIBLE_DEVICES=1 MODE=batch_infer \
CHECKPOINT=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/focus/models/train/sana_artifact_repair/controlnet_tiny1_lora/checkpoint-2000 \
DATASET_METADATA_PATH=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora/metadata_val.json \
DATASET_BASE_PATH=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora \
OUTPUT_DIR=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/focus/models/train/sana_artifact_repair/infer/controlnet_val \
STRENGTH=0.2 \
IMG2IMG_SCHEDULE_MODE=sliced \
CONDITIONING_SCALE=1.0 \
STEPS=20 \
GUIDANCE_SCALE=1.0 \
MAX_PIXELS=1048576 \
DOWNSCALE_IF_EXCEEDS_MAX_PIXELS=1 \
LOCAL_FILES_ONLY=1 \
bash examples/research_projects/artifact_repair/run_sana_artifact_repair_controlnet.sh
```

conditioning scale sweep：

```bash
for CS in 0.5 1.0 1.5 2.0 3.0; do
  CUDA_VISIBLE_DEVICES=1 MODE=batch_infer \
  CHECKPOINT=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/focus/models/train/sana_artifact_repair/controlnet_tiny1_lora/checkpoint-2000 \
  DATASET_METADATA_PATH=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora/metadata_val.json \
  DATASET_BASE_PATH=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora \
  OUTPUT_DIR=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/focus/models/train/sana_artifact_repair/infer/controlnet_cs${CS} \
  STRENGTH=0.2 \
  IMG2IMG_SCHEDULE_MODE=sliced \
  CONDITIONING_SCALE=${CS} \
  STEPS=20 \
  GUIDANCE_SCALE=1.0 \
  MAX_SAMPLES=20 \
  MAX_PIXELS=1048576 \
  DOWNSCALE_IF_EXCEEDS_MAX_PIXELS=1 \
  LOCAL_FILES_ONLY=1 \
  bash examples/research_projects/artifact_repair/run_sana_artifact_repair_controlnet.sh
done
```

## 11. 推荐实验流程

建议按以下顺序推进：

1. tiny1 overfit。
2. tiny16 overfit。
3. 2-step smoke test。
4. strength / scale sweep。
5. small validation batch。
6. full training。
7. full validation / evaluation。

优先建议先跑 without-ControlNet from-src 路线，因为它更轻、更容易定位问题。确认 src latent init、adapter 条件和 metadata 都正常后，再跑 ControlNet 路线做对比。

## 12. 输出目录结构

without-ControlNet from-src 路线：

- `adapter.safetensors`
- `adapter_config.json`
- `artifact_repair_config.json`
- `transformer_lora/`，开启 LoRA 时保存
- `checkpoint-500/`
- `checkpoint-1000/`
- `train.log`

ControlNet 路线：

- `controlnet/`
- `condition_projection.safetensors`
- `artifact_repair_config.json`
- `adapter_config.json`
- `transformer_lora/`，开启 LoRA 时保存
- `checkpoint-500/`
- `checkpoint-1000/`
- `train.log`

推理输出：

- PNG 结果图。
- batch 推理的 `metadata_results.json`。
- `debug_latent_dir/`，包含 raw images、VAE roundtrip、decoded init latents、final output、latent stats 和 condition stats。

## 13. Troubleshooting

- `LocalEntryNotFoundError`：本地 Hugging Face cache 中没有模型。关闭 `LOCAL_FILES_ONLY` 或准备正确的 cache。
- `getcwd: cannot access parent directories`：当前 shell 所在目录被删除或移动，重新 `cd` 到仓库根目录。
- `resolution exceeds max_pixels`：增大 `MAX_PIXELS`，或设置 `DOWNSCALE_IF_EXCEEDS_MAX_PIXELS=1`。
- `unrecognized arguments`：检查参数属于哪条路线。ControlNet 使用 `CONDITIONING_SCALE`，from-src 使用 `ADAPTER_SCALE`。
- `optimizer got an empty parameter list`：adapter、ControlNet 或 LoRA 都没有可训练参数。
- NCCL timeout：降低分辨率、减小 batch size，或检查分布式启动配置。
- `loss=nan`：先尝试 `MIXED_PRECISION=no`，降低 learning rate，并检查 debug tensor stats。
- ControlNet token mismatch，例如 `1080 vs 1105920`：ControlNet condition 没有在 projection 前下采样到 latent H/W。本目录 ControlNet wrapper 已包含该修复。
- 推理黑图 / 蓝黑偏色：优先使用 `IMG2IMG_SCHEDULE_MODE=sliced`，降低 `STRENGTH`，检查 VAE roundtrip debug 图。
- 输出和 src 无关：确认使用默认 sliced src latent init，降低 strength，并考虑开启 transformer LoRA。
- `zero_ref` / `zero_src` 对照实验：设置 `ZERO_REF_CONDITION=1` 或 `ZERO_SRC_CONDITION=1`。如果输出完全不变，说明对应条件可能没有被有效使用。
- `pipeline_full` 与 `sliced` 区别：`pipeline_full` 仅保留为 debug；`sliced` 能让初始 latent 噪声等级和实际 denoising timesteps 对齐，是当前默认推荐。

## 14. SANA Native Edit LoRA for Artifact Repair（cross-attention v2）

Native Edit LoRA 是一条独立路线，不使用 ControlNet，也不使用 Adapter。它参考 DiffSynth-Studio FLUX2 的 `edit_image` LoRA 训练范式，但模型仍然是 SANA。

重要：旧版 v1 的 `target/src/ref` 在 image stream 维度拼成 `H×3W` 已废弃。SANA block 内部有 GLUMBConv / spatial reshape，真实 image stream 必须保持 target 的 `H×W`，否则容易出现 token reshape 或卷积语义错误。

当前 v2 逻辑：

1. `image` / GT 作为 clean diffusion target。
2. `edit_image[0]` 固定为 `src`。
3. `edit_image[1]` 固定为 `ref`。
4. GT、src、ref 分别经过 VAE 编码为 latent。
5. noisy GT latent 只作为 SANA 的 target hidden/query stream，空间网格始终保持真实 `H×W`。
6. src/ref latent 经过 SANA 共享 `patch_embed`，加 role embedding，再可选 token norm / projection。
7. src/ref tokens 拼到 text encoder condition 后面，作为 cross-attention 的 K/V 条件。
8. 输出只有 target stream，不再做 target token slicing。

因此推理和训练都会保存并检查：

- `implementation=sana_cross_attention_edit_condition_v2`
- `version=2`
- `target_stream=target_only`
- `edit_stream=cross_attention_encoder_condition`
- `spatial_layout=target_HxW_only`

旧 v1 checkpoint 不兼容，加载时会直接报错，需要重新训练。

新增入口：

- `train_sana_artifact_repair_native_edit_lora.py`
- `infer_sana_artifact_repair_native_edit_lora.py`
- `batch_infer_sana_artifact_repair_native_edit_lora.py`
- `run_sana_artifact_repair_native_edit_lora.sh`

默认数据路径：

```bash
ART_BASE=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora/data-prompt-aug
ART_TRAIN_META=${ART_BASE}/metadata_train.json
ART_TEST_META=${ART_BASE}/metadata_test.json
```

注意：`DATASET_BASE_PATH` 必须是 `${ART_BASE}`，不要传到 `processed/train` 或 `processed/test`，否则 metadata 中的相对路径会重复拼接。

### 14.1 tiny1 fp32 overfit

```bash
CUDA_VISIBLE_DEVICES=1 MODE=train \
MIXED_PRECISION=no \
NATIVE_EDIT_IMPL=cross_attention_v2 \
LORA_SCOPE=cross_attention \
EDIT_CONDITION_SCALE=1.0 \
USE_EDIT_TOKEN_NORM=1 \
DEBUG_NAN=1 \
ART_BASE=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora/data-prompt-aug \
DATASET_METADATA_PATH=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora/data-prompt-aug/metadata_train.json \
DATASET_BASE_PATH=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora/data-prompt-aug \
OUTPUT_DIR=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/focus/models/train/sana_artifact_repair/native_edit_v2_tiny1_lora \
MAX_SAMPLES=1 \
MAX_TRAIN_STEPS=2000 \
SAVE_STEPS=500 \
LOG_STEPS=10 \
LORA_RANK=8 \
MAX_PIXELS=1048576 \
DOWNSCALE_IF_EXCEEDS_MAX_PIXELS=1 \
LOCAL_FILES_ONLY=1 \
bash examples/research_projects/artifact_repair/run_sana_artifact_repair_native_edit_lora.sh
```

### 14.2 tiny16 fp32 验证

```bash
CUDA_VISIBLE_DEVICES=1 MODE=train \
MIXED_PRECISION=no \
NATIVE_EDIT_IMPL=cross_attention_v2 \
LORA_SCOPE=cross_attention \
EDIT_CONDITION_SCALE=1.0 \
USE_EDIT_TOKEN_NORM=1 \
DEBUG_NAN=1 \
ART_BASE=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora/data-prompt-aug \
DATASET_METADATA_PATH=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora/data-prompt-aug/metadata_train.json \
DATASET_BASE_PATH=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora/data-prompt-aug \
OUTPUT_DIR=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/focus/models/train/sana_artifact_repair/native_edit_v2_tiny16_lora \
MAX_SAMPLES=16 \
MAX_TRAIN_STEPS=5000 \
SAVE_STEPS=1000 \
LOG_STEPS=10 \
LORA_RANK=16 \
MAX_PIXELS=1048576 \
DOWNSCALE_IF_EXCEEDS_MAX_PIXELS=1 \
LOCAL_FILES_ONLY=1 \
bash examples/research_projects/artifact_repair/run_sana_artifact_repair_native_edit_lora.sh
```

### 14.3 full train

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 MODE=train \
NUM_PROCESSES=4 \
MIXED_PRECISION=bf16 \
NATIVE_EDIT_IMPL=cross_attention_v2 \
LORA_SCOPE=cross_attention \
EDIT_CONDITION_SCALE=1.0 \
USE_EDIT_TOKEN_NORM=1 \
ART_BASE=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora/data-prompt-aug \
DATASET_METADATA_PATH=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora/data-prompt-aug/metadata_train.json \
DATASET_BASE_PATH=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora/data-prompt-aug \
OUTPUT_DIR=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/focus/models/train/sana_artifact_repair/native_edit_v2_full_lora \
MAX_TRAIN_STEPS=50000 \
SAVE_STEPS=5000 \
LOG_STEPS=10 \
LORA_RANK=32 \
MAX_PIXELS=1048576 \
DOWNSCALE_IF_EXCEEDS_MAX_PIXELS=1 \
LOCAL_FILES_ONLY=1 \
bash examples/research_projects/artifact_repair/run_sana_artifact_repair_native_edit_lora.sh
```

### 14.4 单图推理

```bash
CUDA_VISIBLE_DEVICES=1 MODE=infer \
DTYPE=fp32 \
CHECKPOINT=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/focus/models/train/sana_artifact_repair/native_edit_v2_tiny1_lora/checkpoint-2000 \
SRC_IMAGE=/path/to/src.png \
REF_IMAGE=/path/to/ref.png \
OUTPUT_PATH=/tmp/native_edit_v2_single.png \
INIT_MODE=src \
STRENGTH=0.15 \
STEPS=20 \
MAX_PIXELS=1048576 \
DOWNSCALE_IF_EXCEEDS_MAX_PIXELS=1 \
LOCAL_FILES_ONLY=1 \
bash examples/research_projects/artifact_repair/run_sana_artifact_repair_native_edit_lora.sh
```

### 14.5 batch 推理

```bash
CUDA_VISIBLE_DEVICES=1 MODE=batch_infer \
DTYPE=fp32 \
CHECKPOINT=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/focus/models/train/sana_artifact_repair/native_edit_v2_tiny1_lora/checkpoint-2000 \
ART_BASE=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora/data-prompt-aug \
TEST_METADATA_PATH=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora/data-prompt-aug/metadata_test.json \
DATASET_BASE_PATH=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora/data-prompt-aug \
OUTPUT_DIR=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/focus/models/train/sana_artifact_repair/native_edit_v2_tiny1_lora \
INIT_MODE=src \
STRENGTH=0.15 \
STEPS=20 \
MAX_SAMPLES=20 \
MAX_PIXELS=1048576 \
DOWNSCALE_IF_EXCEEDS_MAX_PIXELS=1 \
LOCAL_FILES_ONLY=1 \
bash examples/research_projects/artifact_repair/run_sana_artifact_repair_native_edit_lora.sh
```

### 14.6 ablation 推理

`ZERO_SRC=1` 和 `ZERO_REF=1` 现在只清空 cross-attention 条件 tokens，不会改变 `INIT_MODE=src` 使用的初始化 latent。这样可以更干净地判断 src/ref 条件是否真的被模型使用。

```bash
# zero src condition only
CUDA_VISIBLE_DEVICES=1 MODE=batch_infer \
DTYPE=fp32 \
CHECKPOINT=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/focus/models/train/sana_artifact_repair/native_edit_v2_tiny16_lora/checkpoint-5000 \
ZERO_SRC=1 MAX_SAMPLES=20 \
bash examples/research_projects/artifact_repair/run_sana_artifact_repair_native_edit_lora.sh

# zero ref condition only
CUDA_VISIBLE_DEVICES=1 MODE=batch_infer \
DTYPE=fp32 \
CHECKPOINT=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/focus/models/train/sana_artifact_repair/native_edit_v2_tiny16_lora/checkpoint-5000 \
ZERO_REF=1 MAX_SAMPLES=20 \
bash examples/research_projects/artifact_repair/run_sana_artifact_repair_native_edit_lora.sh

# swap src/ref condition roles, while keeping original src init latent
CUDA_VISIBLE_DEVICES=1 MODE=batch_infer \
DTYPE=fp32 \
CHECKPOINT=/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/focus/models/train/sana_artifact_repair/native_edit_v2_tiny16_lora/checkpoint-5000 \
SWAP_SRC_REF=1 MAX_SAMPLES=20 \
bash examples/research_projects/artifact_repair/run_sana_artifact_repair_native_edit_lora.sh
```
