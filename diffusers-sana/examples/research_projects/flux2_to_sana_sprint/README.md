# DiffSynth Studio FLUX.2 to Sana-Sprint

本目录是一套独立的跨框架、跨架构蒸馏流程：

```text
DiffSynth Studio 训练的 FLUX.2 teacher
                 |
                 | 黑盒图片响应 + metadata
                 v
Diffusers Sana-Sprint student
```

FLUX.2 与 Sana-Sprint 的 DiT、VAE、latent 空间和时间参数化均不同，因此不能使用 Sana-Sprint 官方同构
teacher 的 sCM/JVP 速度场蒸馏。本方案采用黑盒行为蒸馏：FLUX.2 生成教师目标，Sana-Sprint 在自己的 VAE 和
TrigFlow 参数化中执行多时间步 `x0` 响应学习，并重点采样最大时间步以保留 1-4 步生成能力。

## 文件

- `export_diffsynth_teacher.py`：通过命令模板调用 DiffSynth FLUX.2，生成教师图片和新 metadata。
- `diffsynth_flux2_infer.py`：直接加载 DiffSynth FLUX.2 base 与训练 LoRA 的教师推理适配器。
- `train_sprint_cross_distill.py`：跨架构、多时间步 Sana-Sprint LoRA + 条件 Adapter 蒸馏。
- `infer_sprint.py`：蒸馏后 Sprint 推理。
- `run_export_teacher.sh`：教师数据导出。
- `run_train_sprint.sh`：默认四卡学生训练。
- `run_infer_sprint.sh`：1-4 步学生推理。
- `run_all.sh`：依次导出教师结果并训练学生。

## 输入 metadata

格式继续使用 DiffSynth Studio 的 JSON/JSONL：

```json
{
  "image": "images/target.png",
  "edit_image": "images/reference.png",
  "prompt": "turn the product red",
  "seed": 0
}
```

多图输入：

```json
{
  "image": "images/target.png",
  "edit_image": ["images/reference_a.png", "images/reference_b.png"],
  "prompt": "combine both references"
}
```

要求整个数据集的 `edit_image` 数量一致，不允许一部分单图、一部分多图。同一样本的条件图片必须等尺寸。

## 两个运行环境

教师导出在安装了 DiffSynth Studio 和已训练 FLUX.2 权重的环境中运行。学生训练在本目录所属的
`diffusers-sana` 环境中运行。两个环境仅通过 PNG 和 JSON 文件交接，不共享模型对象或 CUDA 进程。

安装学生环境：

```bash
cd /path/to/diffusers-sana
pip install -e .
pip install -r examples/research_projects/flux2_to_sana_sprint/requirements.txt
```

## 1. 准备 DiffSynth FLUX.2 推理入口

`TEACHER_COMMAND` 是逐样本执行的 argv 模板。你的推理脚本需要加载训练时相同的 FLUX.2 base、tokenizer、
VAE 和 LoRA checkpoint，然后读取参考图并将结果保存到 `--output`。

单图 teacher 示例：

```bash
export TEACHER_COMMAND='/path/to/diffsynth-env/bin/python your_flux2_infer.py \
  --image {condition_0} \
  --prompt {prompt} \
  --seed {seed} \
  --height {height} \
  --width {width} \
  --output {output}'
```

固定双图 teacher 示例：

```bash
export TEACHER_COMMAND='/path/to/diffsynth-env/bin/python your_flux2_infer.py \
  --image-a {condition_0} \
  --image-b {condition_1} \
  --prompt {prompt} \
  --seed {seed} \
  --height {height} \
  --width {width} \
  --output {output}'
```

如果推理脚本接受 JSON 图片列表：

```bash
export TEACHER_COMMAND='/path/to/diffsynth-env/bin/python your_flux2_infer.py \
  --images-json {condition_images_json} \
  --prompt {prompt} --seed {seed} \
  --height {height} --width {width} --output {output}'
```

可用占位符：

- `{condition_0}`、`{condition_1}` 等固定位置图片。
- `{condition_images_json}`：绝对路径组成的 JSON 列表。
- `{condition_images_csv}`：逗号分隔绝对路径。
- `{prompt}`、`{seed}`、`{height}`、`{width}`、`{output}`。
- `{index}` 和 `{id}`。

模板先由 `shlex` 拆分，再逐 argv 替换；带空格的 prompt 和路径不会被重新按空格拆开。不要在模板中添加
shell 重定向、管道或 `bash -c`。

### 直接使用随附的 DiffSynth 适配器

复制并修改 `diffsynth_teacher_config.example.json`，尤其是 `lora_checkpoint`。其余字段已经按照问题中给出的
FLUX.2 Klein 4B 训练参数填写。

```bash
export CROSS_DISTILL_DIR=/absolute/path/to/diffusers-sana/examples/research_projects/flux2_to_sana_sprint
export DIFFSYNTH_ROOT=/absolute/path/to/DiffSynth-Studio
export DIFFSYNTH_PYTHON=/path/to/diffsynth-env/bin/python
export TEACHER_CONFIG=/absolute/path/to/diffsynth_teacher_config.json

export TEACHER_COMMAND="$DIFFSYNTH_PYTHON \
  $CROSS_DISTILL_DIR/diffsynth_flux2_infer.py \
  --diffsynth_root $DIFFSYNTH_ROOT \
  --config $TEACHER_CONFIG \
  --images-json {condition_images_json} \
  --prompt {prompt} \
  --seed {seed} \
  --height {height} \
  --width {width} \
  --num_inference_steps 50 \
  --cfg_scale 1.0 \
  --embedded_guidance 1.0 \
  --output {output}"
```

该适配器复用 DiffSynth Studio 的 `Flux2ImageTrainingModule` 来重建 LoRA 层并加载 checkpoint，然后调用
`Flux2ImagePipeline(..., edit_image=[...])`。因此 config 中的 base、tokenizer、LoRA rank 和 target modules 必须
与训练命令一致。
示例按 `FLUX.2-klein-base-4B` 使用 50 步；若你的 teacher 本身是 step-distilled Klein 模型，应改成其推荐步数。

## 2. 导出 FLUX.2 教师响应

```bash
DIFFSYNTH_ROOT=/path/to/DiffSynth-Studio \
DATASET_BASE_PATH=/data/my_dataset \
METADATA=/data/my_dataset/metadata_train.json \
OUTPUT_ROOT=outputs/flux2_to_sana_sprint \
TEACHER_COMMAND="$TEACHER_COMMAND" \
bash examples/research_projects/flux2_to_sana_sprint/run_export_teacher.sh
```

输出：

```text
outputs/flux2_to_sana_sprint/
├── teacher_images/*.png
└── teacher_metadata.json
```

新 metadata 保留原字段，并增加绝对路径字段：

```json
"teacher_image": "/absolute/path/to/teacher_images/00000000_000000.png"
```

默认输出尺寸与参考图一致，且必须能被 16 整除。也可以同时设置 `TEACHER_HEIGHT` 和 `TEACHER_WIDTH`，但
此时应提前把条件图预处理到相同尺寸，否则学生的数据对齐校验会失败。

支持断点续导：默认 `SKIP_EXISTING=1`。可用 `MAX_SAMPLES=16` 先导出小样本。

## 3. 训练 Sana-Sprint student

```bash
DATASET_BASE_PATH=/data/my_dataset \
OUTPUT_ROOT=outputs/flux2_to_sana_sprint \
TEACHER_METADATA=outputs/flux2_to_sana_sprint/teacher_metadata.json \
OUTPUT_DIR=outputs/flux2_to_sana_sprint/sprint_student \
NUM_GPUS=4 \
bash examples/research_projects/flux2_to_sana_sprint/run_train_sprint.sh
```

训练内容：

1. 教师 PNG 经 Sana VAE 重新编码，不尝试映射 FLUX.2 latent。
2. 在 log-normal 随机 TrigFlow 时间步和最大时间步上混合加噪。
3. Sprint student 预测教师目标 `x0`。
4. 优化 MSE + L1 response loss、Sprint transformer LoRA 和固定数量多图条件 Adapter。
5. guidance scale 从 `GUIDANCE_SCALES` 中逐样本采样。

常用参数：

```bash
STUDENT_MODEL=Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers
DATASET_REPEAT=10
RESOLUTION=1024
LEARNING_RATE=1e-4
MAX_TRAIN_STEPS=20000
BATCH_SIZE=1
GRAD_ACCUM_STEPS=4
LORA_RANK=32
MAX_TIMESTEP_PROBABILITY=0.5
GUIDANCE_SCALES=4.0,4.5,5.0
SAVE_STEPS=500
```

使用 `RESUME_FROM_CHECKPOINT=latest` 可恢复权重和步数，但 optimizer/LR scheduler 会重新初始化。

## 4. Sprint 推理

单图：

```bash
CHECKPOINT=outputs/flux2_to_sana_sprint/sprint_student \
PROMPT="turn the product red" \
OUTPUT=result.png \
STEPS=2 \
bash examples/research_projects/flux2_to_sana_sprint/run_infer_sprint.sh reference.png
```

双图：

```bash
CHECKPOINT=outputs/flux2_to_sana_sprint/sprint_student \
PROMPT="combine both references" \
bash examples/research_projects/flux2_to_sana_sprint/run_infer_sprint.sh reference_a.png reference_b.png
```

`STEPS` 支持 1、2、3、4，默认 2。输入图片数量必须与蒸馏数据一致。

## 一键运行

教师命令、数据路径和输出路径准备好后：

```bash
DIFFSYNTH_ROOT=/path/to/DiffSynth-Studio \
DATASET_BASE_PATH=/data/my_dataset \
METADATA=/data/my_dataset/metadata_train.json \
OUTPUT_ROOT=outputs/flux2_to_sana_sprint \
TEACHER_COMMAND="$TEACHER_COMMAND" \
bash examples/research_projects/flux2_to_sana_sprint/run_all.sh
```

## 方法边界

这套方法蒸馏的是 Flux2 teacher 在给定参考图和 prompt 下的可观察行为。它无法直接继承 Flux2 的内部
score/velocity field，也不等价于 Sana-Sprint 论文中同构 Sana teacher 的 sCM + LADD。若 teacher 和 student
结构相同，应使用旁边 `sana_diffsynth` 目录的官方 sCM + LADD 流程。

参考：[DiffSynth Studio FLUX.2 文档](https://github.com/modelscope/DiffSynth-Studio/blob/main/docs/en/Model_Details/FLUX2.md)。
