# Sana / Sana-Sprint DiffSynth metadata training

本目录提供一条完整的通用图像条件训练链路：

1. 在多步 Sana teacher 上训练任务 LoRA 和多图条件 Adapter。
2. 从该 Sana teacher 在线执行官方 Sana-Sprint `sCM + LADD` 蒸馏。
3. 分别运行多步 Sana teacher 和 1-4 步 Sana-Sprint student 推理。

Sprint 训练主体来自 Diffusers 官方
`examples/research_projects/sana/train_sana_sprint_diffusers.py`。这里没有离线生成 teacher target，也没有使用
简化的图片回归蒸馏。

## 目录

- `train_sana.py`：Sana flow-matching LoRA SFT。
- `train_sana_sprint.py`：官方 TrigFlow、sCM JVP 和 LADD 蒸馏，增加 DiffSynth metadata 与图像条件。
- `infer.py`：Sana/Sprint 共用推理实现。
- `run_train_sana.sh`：默认四卡 Sana 训练。
- `run_distill_sana_sprint.sh`：默认四卡官方 Sprint 蒸馏。
- `run_infer_sana.sh`、`run_infer_sprint.sh`：两个模型的推理入口。

## Metadata

支持 JSON 数组和 JSONL。默认字段对应 DiffSynth Studio：

```json
[
  {
    "image": "images/000001_target.png",
    "edit_image": "images/000001_reference.png",
    "prompt": "turn the product red"
  }
]
```

多图输入写成列表：

```json
{
  "image": "images/000002_target.png",
  "edit_image": ["images/000002_a.png", "images/000002_b.png"],
  "prompt": "combine the supplied references"
}
```

约束：

- 每条样本必须恰好有一张 target。
- `edit_image` 可以是一张或多张，但整个数据集的参考图数量必须一致。
- 同一样本的 target 和参考图必须等尺寸，以便执行完全一致的 crop 和 flip。
- 相对路径均相对于 `DATASET_BASE_PATH`。

完整格式见 `metadata.example.json` 和 `metadata.multi_image.example.json`。

## 安装

在 `diffusers-sana` 根目录执行：

```bash
pip install -e .
pip install -r examples/research_projects/sana_diffsynth/requirements.txt
```

需要支持 BF16 的 CUDA GPU。Sprint 官方蒸馏会同时持有 student、冻结 teacher、JVP 中间状态和 LADD
判别器，显存需求明显高于普通 LoRA SFT。

## 模型配对

默认使用 0.6B 配对：

```text
Sana teacher/SFT:
Efficient-Large-Model/Sana_Sprint_0.6B_1024px_teacher_diffusers

Sprint inference components:
Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers
```

Sana SFT 与 Sprint 蒸馏的 `TEACHER_MODEL` 必须是同一个 checkpoint。切换 1.6B 时，两阶段都要切换为
对应的 1.6B teacher，并将 `SPRINT_BASE_MODEL` 切换为 1.6B Sprint。

## 1. 训练 Sana teacher

```bash
DATASET_BASE_PATH=/data/my_dataset \
METADATA=/data/my_dataset/metadata_train.json \
OUTPUT_DIR=outputs/sana_teacher \
NUM_GPUS=4 \
bash examples/research_projects/sana_diffsynth/run_train_sana.sh
```

常用覆盖项：

```bash
MODEL=/path/to/Sana_Sprint_0.6B_1024px_teacher_diffusers
DATA_FILE_KEYS=image,edit_image
EXTRA_INPUTS=edit_image
DATASET_REPEAT=10
RESOLUTION=1024
LEARNING_RATE=1e-4
MAX_TRAIN_STEPS=20000
BATCH_SIZE=1
GRAD_ACCUM_STEPS=4
LORA_RANK=32
SAVE_STEPS=500
```

输出包括标准 Diffusers transformer LoRA、`condition_adapter.safetensors` 和
`condition_adapter.json`。checkpoint 可用于推理；`RESUME_FROM_CHECKPOINT` 会恢复 LoRA、Adapter 和步数，但 Sana
LoRA 阶段会重新初始化 optimizer/LR scheduler。

## 2. Sana teacher 推理

单图条件：

```bash
ADAPTER_PATH=outputs/sana_teacher \
PROMPT="turn the product red" \
OUTPUT=sana_result.png \
bash examples/research_projects/sana_diffsynth/run_infer_sana.sh reference.png
```

多图条件必须按训练顺序传入：

```bash
ADAPTER_PATH=outputs/sana_teacher \
PROMPT="combine both references" \
bash examples/research_projects/sana_diffsynth/run_infer_sana.sh reference_a.png reference_b.png
```

默认执行 20 步，可通过 `STEPS`、`GUIDANCE_SCALE` 和 `SEED` 覆盖。

## 3. 官方 Sana-Sprint 蒸馏

```bash
TEACHER_MODEL=Efficient-Large-Model/Sana_Sprint_0.6B_1024px_teacher_diffusers \
TEACHER_ADAPTER_PATH=outputs/sana_teacher \
SPRINT_BASE_MODEL=Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers \
DATASET_BASE_PATH=/data/my_dataset \
METADATA=/data/my_dataset/metadata_train.json \
OUTPUT_DIR=outputs/sana_sprint \
NUM_GPUS=4 \
bash examples/research_projects/sana_diffsynth/run_distill_sana_sprint.sh
```

蒸馏时会执行：

- 将 Sana SFT LoRA 融合进 student 初始化和冻结 teacher。
- 从 teacher 复制条件 Adapter，teacher 侧冻结，student 侧参与全参数训练。
- teacher CFG 速度场指导 TrigFlow sCM，并通过 `torch.func.jvp` 计算切向一致性目标。
- LADD 使用冻结 teacher 特征和可训练判别头，区分真实 target latent 与 student 预测的 `x0`。
- 交替执行 generator/student 和 discriminator 更新，并强化最大时间步训练。

Sprint 输出为完整 `transformer/`、`condition_adapter.safetensors`、`condition_adapter.json` 和训练用
`disc_heads.pt`。这不是 LoRA 输出。

使用 checkpoint 恢复完整 optimizer、scheduler、student 和判别器状态：

```bash
RESUME_FROM_CHECKPOINT=latest \
TEACHER_ADAPTER_PATH=outputs/sana_teacher \
DATASET_BASE_PATH=/data/my_dataset \
OUTPUT_DIR=outputs/sana_sprint \
bash examples/research_projects/sana_diffsynth/run_distill_sana_sprint.sh
```

`misaligned_pairs_D` 只在单卡 local batch 大于 1 时实际增加错配样本；默认 batch 1 时该分支自动跳过。

## 4. Sana-Sprint 推理

```bash
ADAPTER_PATH=outputs/sana_sprint \
PROMPT="turn the product red" \
OUTPUT=sprint_result.png \
STEPS=2 \
bash examples/research_projects/sana_diffsynth/run_infer_sprint.sh reference.png
```

`STEPS` 可设为 1、2、3 或 4，默认 2。参考图数量必须与训练时一致。

## 快速数据检查

首次正式训练建议先跑少量样本和步骤：

```bash
MAX_SAMPLES=16 MAX_TRAIN_STEPS=10 NUM_GPUS=1 \
DATASET_BASE_PATH=/data/my_dataset \
OUTPUT_DIR=outputs/sana_smoke \
bash examples/research_projects/sana_diffsynth/run_train_sana.sh
```

确认 Sana teacher 能正确推理后，再启动 Sprint 蒸馏。官方 sCM+LADD 蒸馏成本较高，不适合作为第一步数据
格式检查。

## 参考

- [SANA-Sprint paper](https://arxiv.org/abs/2503.09641)
- [Diffusers SANA-Sprint documentation](https://huggingface.co/docs/diffusers/api/pipelines/sana_sprint)
- Diffusers 原始训练实现：`../sana/train_sana_sprint_diffusers.py`
