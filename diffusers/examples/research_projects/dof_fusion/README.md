# 基于 Diffusers 的景深融合实验

本目录用于研究两张不同清晰区域图像的景深融合：

```text
A 图 + B 图 + 可选 focus map → 全清晰目标图（GT）
```

A/B 仅表示两张输入图，不对应固定的近焦或远焦关系。所有任务代码均位于
`examples/research_projects/dof_fusion`，不修改 `src/diffusers` 源文件。

## 环境安装

本项目直接基于当前 Diffusers 源码运行，要求 Python 3.10 或更高版本。推荐使用 Python 3.11 和独立
Conda 环境。

### 1. 创建环境

```bash
conda create -n dof-diffusers python=3.11 -y
conda activate dof-diffusers
python -m pip install --upgrade pip setuptools wheel
```

### 2. 安装 PyTorch

请根据机器的 CUDA 和驱动版本，从 PyTorch 官方安装页面选择对应命令。当前仓库依赖要求
`torch>=2.6`。一个不固定 CUDA wheel 来源的安装示例是：

```bash
python -m pip install "torch>=2.6" torchvision
```

安装后先确认 CUDA 可用：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

如果 `torch.cuda.is_available()` 为 `False`，不要开始下载模型或训练，应先修复 PyTorch、CUDA 或驱动
环境。

### 3. 从当前源码安装 Diffusers

在仓库根目录执行：

```bash
cd /mnt/d/work/vivo/diffusers
python -m pip install -e ".[training]"
```

`-e` 表示 editable install。后续修改本仓库中的 Diffusers 或景深融合代码后，无需反复重新安装。

### 4. 安装本实验额外依赖

```bash
python -m pip install --upgrade \
  "transformers>=4.51.0" \
  accelerate \
  "peft>=0.17.0" \
  torchvision \
  sentencepiece \
  safetensors \
  pillow \
  numpy \
  ftfy
```

FLUX.2 Klein 使用 `Qwen3ForCausalLM`，因此不能使用过旧的 Transformers。SANA 使用 Gemma tokenizer，
需要安装 `sentencepiece`。

### 5. 配置 Accelerate

单卡可以先使用默认配置：

```bash
accelerate config default
```

需要多卡、DeepSpeed 或其他分布式配置时，改用交互式配置：

```bash
accelerate config
```

### 6. Hugging Face 登录

部分模型需要先在 Hugging Face 页面同意许可证，再登录本机：

```bash
hf auth login
```

至少确认以下模型能够访问：

```text
black-forest-labs/FLUX.2-klein-4B
Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers
Efficient-Large-Model/Sana_600M_1024px_diffusers
```

### 7. 安装验证

```bash
python -c "import torch, diffusers, transformers, accelerate, peft; print('environment ok')"
python -c "from diffusers import Flux2KleinPipeline, SanaSprintPipeline, SanaSprintImg2ImgPipeline; print('pipelines ok')"
```

建议先运行小规模训练 smoke test，而不是直接训练 20,000 steps：

```bash
MODE=train \
MAX_TRAIN_STEPS=10 \
BATCH_SIZE=1 \
bash examples/research_projects/dof_fusion/run_sana_sprint.sh
```

## 方案总览

当前共有 6 套可训练方案：

| 方案 | 训练内容 | 基模是否冻结 | 默认推理 |
| --- | --- | --- | --- |
| FLUX.2 Klein LoRA | Transformer LoRA | 是 | 4 步 |
| FLUX.2 Klein + 轻量 ControlNet | 自定义 focus residual 分支 | 是 | 4 步 |
| SANA-Sprint A/B Adapter | 外置双图卷积 Adapter | 是 | 1 步 |
| SANA-Sprint + ControlNet | A/B Adapter + 完整 ControlNet | 是 | 1 步 |
| 普通 SANA + ControlNet | A/B Adapter + 完整 ControlNet | 是 | 20 步 |
| SANA-Sprint img2img | 外置双图卷积 Adapter | 是 | 4 步，strength 0.75 |

其中只有 FLUX.2 Klein LoRA 是纯 LoRA 微调。其余方案训练 Adapter、ControlNet 或二者组合。

相关入口：

1. `train_flux2_klein.py` / `infer_flux2_klein.py`
2. `train_flux2_klein_controlnet.py` / `infer_flux2_klein_controlnet.py`
3. `train_sana_sprint.py` / `infer_sana_sprint.py`
4. `train_sana_sprint_controlnet.py` / `infer_sana_sprint_controlnet.py`
5. `train_sana_controlnet.py` / `infer_sana_controlnet.py`
6. `train_sana_sprint_img2img.py` / `infer_sana_sprint_img2img.py`

普通 SANA ControlNet 和 Sprint img2img 的补充说明见
`README_sana_controlnet_and_img2img.md`，Sprint ControlNet 的补充说明见
`README_sana_sprint_controlnet.md`。

## 推荐实验顺序

### 第一阶段只运行 E1～E4

当前工程阶段不要启动普通/Sprint ControlNet、FLUX.2 LoRA 或 FLUX.2 control branch。第一阶段固定为：

| 实验 | 配置 | 目的 |
| --- | --- | --- |
| E1 | SANA-Sprint A/B Adapter，不使用 focus loss | 建立最小双图融合基线 |
| E2 | SANA-Sprint A/B Adapter + focus loss | 验证区域加权监督 |
| E3 | SANA-Sprint A/B+focus Adapter + focus loss | 验证 focus 直接进入网络 |
| E4 | SANA-Sprint img2img，A 初始化 | 验证结构和纹理保持能力 |

```bash
# E1
MODE=all ADAPTER_TYPE=ab USE_FOCUS_MAPS=0 FOCUS_LOSS_WEIGHT=0 \
OUTPUT_DIR=outputs/dof_fusion/e1_ab_no_focus \
bash examples/research_projects/dof_fusion/run_sana_sprint.sh

# E2
MODE=all ADAPTER_TYPE=ab USE_FOCUS_MAPS=1 FOCUS_LOSS_WEIGHT=1.0 \
OUTPUT_DIR=outputs/dof_fusion/e2_ab_focus_loss \
bash examples/research_projects/dof_fusion/run_sana_sprint.sh

# E3
MODE=all ADAPTER_TYPE=ab_focus USE_FOCUS_MAPS=1 FOCUS_LOSS_WEIGHT=1.0 \
FOCUS_A=/data/dof/focus_a.png FOCUS_B=/data/dof/focus_b.png \
OUTPUT_DIR=outputs/dof_fusion/e3_ab_focus_adapter \
bash examples/research_projects/dof_fusion/run_sana_sprint.sh

# E4
MODE=all INFER_STEPS=4 STRENGTH=0.75 \
OUTPUT_DIR=outputs/dof_fusion/e4_img2img \
bash examples/research_projects/dof_fusion/run_sana_sprint_img2img.sh
```

E1～E4 完成并形成可比较结果后，再决定是否运行 ControlNet 和 FLUX.2 路线。不要同时启动多个重模型
实验，以免把数据、显存和实现问题混在一起。

### 后续候选顺序

建议不要一开始同时验证多图注入、focus map 和 ControlNet。后续顺序如下：

1. **SANA-Sprint A/B Adapter**：最符合小模型、one-step 和后续端侧部署目标。
2. **FLUX.2 Klein LoRA**：作为原生多图条件下的质量上限和数据有效性基准。
3. **SANA-Sprint img2img**：当 one-step 方案出现结构变化或纹理重绘时，用 A 或初步融合结果初始化。
4. **普通 SANA + ControlNet**：focus map 稳定后，先验证多步 ControlNet 是否有效。
5. **SANA-Sprint + ControlNet**：普通 ControlNet 有收益后，再尝试迁移到 one-step。
6. **FLUX.2 Klein + 轻量 ControlNet**：自定义研究路线，建议最后验证。

E1 只使用 `[A, B] → GT` 且不开 focus loss；E2/E3 再逐项引入 focus，避免一次改变多个变量。

## 数据检查、smoke test 与评测

训练前检查 metadata、图片路径、尺寸和 focus map 值域，并生成 preview grid：

```bash
python examples/research_projects/dof_fusion/check_dof_metadata.py \
  --dataset_metadata_path /data/dof/metadata.json \
  --dataset_base_path /data/dof \
  --continue_on_error \
  --preview_output outputs/metadata_preview.jpg \
  --report_output outputs/metadata_report.json
```

工程 smoke test 会覆盖 E1～E4，并额外验证普通 SANA ControlNet、FLUX2 Klein LoRA 和 FLUX2 Klein
ControlNet 的动态分辨率闭环。新增三条路线各训练 2 steps、读取最多 2 个样本，随后运行单图和 batch
推理；脚本还会检查输出尺寸与 A 一致、`metadata_results.json` 包含 `generated_image`。任何失败都会
立即 `exit 1`：

```bash
DATASET_BASE_PATH=/data/dof \
METADATA=/data/dof/metadata.json \
IMAGE_A=/data/dof/a.png \
IMAGE_B=/data/dof/b.png \
FOCUS_A=/data/dof/focus_a.png \
FOCUS_B=/data/dof/focus_b.png \
bash examples/research_projects/dof_fusion/smoke_test_all.sh
```

评测生成结果：

```bash
python examples/research_projects/dof_fusion/eval_dof_results.py \
  --dataset_metadata_path outputs/sana/batch/metadata_results.json \
  --dataset_base_path /data/dof \
  --output_json outputs/sana/eval.json \
  --output_csv outputs/sana/eval.csv
```

评测项包括全图、keep 区域和 blur 区域的 PSNR、SSIM、L1，以及 Sobel edge L1。`focus_a` 白色表示
保留 A 的清晰区域，黑色表示需要从 B 恢复。

### 独立色差后处理

色差处理不接入推理 pipeline，而是读取已经生成的 `metadata_results.json`。默认有 focus_a 时用它构造
A/B 参考，没有时使用 A/B 平均参考；随后只在 CIELAB 的色度通道做空间配对的稳健中位数偏移，保留
生成结果的亮度和锐度结构。与传统均值/方差迁移相比，该默认
方法不依赖清晰度分布，因此 A/B 的模糊区域不容易把饱和度或对比度拉偏：

```bash
python examples/research_projects/dof_fusion/postprocess_dof_color.py \
  --dataset_metadata_path outputs/sana/batch/metadata_results.json \
  --dataset_base_path /data/dof \
  --output_dir outputs/sana/color_corrected \
  --reference_mode mean_ab \
  --method paired_offset \
  --channels chroma \
  --strength 0.8
```

输出 metadata 保留原字段并新增 `color_corrected_image`。`reference_mode` 支持 `a`、`b`、`mean_ab`、
`focus_composite`；`target` 仅用于带 GT 的离线分析，部署时不要用 GT 做颜色校正。

推荐优先使用 `paired_offset + chroma`。它使用 A/B 与生成图的空间对应关系、截尾统计和中位数，只修正
Lab 色度，并用 `--max_chroma_shift` 限制最大变化。`local_chroma` 可处理空间变化的色偏，但可能在景深
边界产生低频颜色过渡；`reinhard --match_std` 可能放大模糊图的颜色方差，默认不开启。

## SANA-Sprint 跨模型响应蒸馏

FLUX2 与 SANA 的 VAE、latent 和 Transformer 不同，不能直接共享官方 sCM 的教师速度场/JVP。这里提供
可运行的响应蒸馏：教师先离线生成 `teacher_image`，再用已经一步化的 SANA-Sprint 学习
`A/B(/focus) -> teacher_image`。这会迁移任务效果，但不是重新执行 SANA-Sprint 的底层 sCM+LADD。

```bash
# Diffusers FLUX2 base teacher
bash examples/research_projects/dof_fusion/run_distill_diffusers_flux2_to_sana_sprint.sh

# DiffSynth FLUX2 teacher：命令模板中的占位符会按样本替换
TEACHER_COMMAND='python your_diffsynth_infer.py --a {image_a} --b {image_b} --output {output}' \
bash examples/research_projects/dof_fusion/run_distill_diffsynth_flux2_to_sana_sprint.sh

# 任意已经训练好的 SANA DOF teacher
TEACHER_COMMAND='python your_sana_infer.py --image_a {image_a} --image_b {image_b} --output {output}' \
bash examples/research_projects/dof_fusion/run_distill_sana_to_sana_sprint.sh

# 本目录的普通 SANA ControlNet teacher
TEACHER_CHECKPOINT=outputs/sana_controlnet \
bash examples/research_projects/dof_fusion/run_distill_sana_controlnet_to_sana_sprint.sh
```

四个入口最终都调用 `generate_dof_teacher_targets.py` 和 `train_sana_sprint.py --target_key teacher_image`。
若要做严格的同架构 sCM，需要在官方 `examples/research_projects/sana/train_sana_sprint_diffusers.py`
中加入 A/B Adapter、ControlNet condition、DiffSynth dataset 和 valid mask；不要把 FLUX2 教师直接接到该
JVP loss。

如果只有原始推理 metadata、其中还没有 `generated_image`，可以额外指定结果目录；默认按 batch 推理的
`{index:06d}_{id}.png` 规则查找：

```bash
python examples/research_projects/dof_fusion/postprocess_dof_color.py \
  --dataset_metadata_path data/metadata.json \
  --dataset_base_path data \
  --generated_dir outputs/sana/batch \
  --output_dir outputs/sana/color_corrected
```

完全没有 JSON 时，也可以使用文件夹模式。每个子目录默认包含 `a.png`、`b.png`、`generated.png`，以及
可选的 `focus_a.png`；文件名可通过 `--a_filename` 等参数修改。如果输入目录只有 A/B，可同时传
`--generated_dir` 从单独的推理结果目录查找生成图：

```bash
python examples/research_projects/dof_fusion/postprocess_dof_color.py \
  --input_dir data/postprocess_samples \
  --output_dir outputs/color_corrected
```

## 断点恢复与离线加载

所有训练脚本支持：

```bash
--resume_from_checkpoint latest
```

训练会保存 `checkpoint-N`，其中包含模型增量权重、`optimizer.pt` 和 `trainer_state.json`。也可以传入
指定的 `checkpoint-N` 路径。

所有训练、单图推理和 batch 推理入口均支持：

```text
--local_files_only
--cache_dir /path/to/huggingface/cache
--revision main
```

## DiffSynth-Studio metadata 格式

数据格式保持 DiffSynth-Studio 现有约定，不重新设计字段：

- `metadata.json`：JSON 数组。
- `metadata.jsonl`：每行一个 JSON 对象。
- 相对路径基于 `--dataset_base_path` 解析，而不是 metadata 文件所在目录。
- GT 默认字段为 `image`，可通过 `--target_key` 修改。
- 条件字段默认为 `edit_image`，可通过 `--edit_key` 修改。
- 样本中的 `prompt` 优先；不存在时使用命令行 `--prompt`。

`edit_image` 定义：

```text
edit_image[0] = A
edit_image[1] = B
edit_image[2] = focus_a（可选）
edit_image[3] = focus_b 或 focus_b_warp（可选）
```

示例：

```json
[
  {
    "id": "000",
    "image": "images/000_gt.png",
    "edit_image": ["images/000_a.png", "images/000_b.png"],
    "prompt": "a photorealistic all-in-focus photograph"
  },
  {
    "id": "001",
    "image": "images/001_gt.png",
    "edit_image": [
      "images/001_a.png",
      "images/001_b.png",
      "images/001_focus_a.png",
      "images/001_focus_b.png"
    ]
  }
]
```

所有 DOF 路线（SANA/Sprint、FLUX2 Klein LoRA、FLUX2 Klein ControlNet）默认以 A 的原始宽高为准，
不强制变成 `1024x1024`：

- A 只在右侧和底部使用 zero padding 补齐到模型要求的倍数，模型输出后只移除这部分人工 padding。
- B、GT、focus_a、focus_b 如果像素尺寸不同但宽高比与 A 一致，会对齐到 A 的宽高。
- 宽高比误差超过 `--aspect_ratio_tolerance`（默认 1%）会报错，不会静默拉伸不匹配的数据。
- `--max_pixels` 默认不限制。设置后，小于上限时保持不变；超过上限默认报错。
- 只有显式增加 `--downscale_if_exceeds_max_pixels`，超限图像才会保持宽高比等比例缩小；输出默认仍恢复到 A 原始尺寸。
- 动态原尺寸训练和 batch 推理当前要求 `--batch_size 1`，可用梯度累积增大有效 batch。
- SANA/SANA-Sprint 的 `--size_divisor` 默认 32；FLUX2 Klein 默认 16。
- loss 只在 `valid_mask=1` 的 content 区域计算，padding 区域不参与 latent loss 或 focus loss。
- 显式传入 `--resolution 1024` 仍走旧固定正方形训练模式；推理对应传 `--height 1024 --width 1024`。

例如 `A=1920x1080` 时，SANA 模型画布仅 padding 为 `1920x1088`，最后输出仍严格为 `1920x1080`。

所有训练、单图推理和批量推理入口统一接受以下 metadata 参数（单图入口可只传 `--image_a`、
`--image_b`，这些 metadata 参数用于保持自动化调用接口一致）：

```text
--dataset_metadata_path  --dataset_base_path
--target_key image       --edit_key edit_image
--prompt_key prompt      --id_key id
--seed_key seed          --result_key generated_image
--start_index            --max_samples
--skip_existing          --continue_on_error
```

## 一键运行脚本

每套方案均提供一个 `.sh`，包含训练、单张推理和 metadata 批量推理：

| 方案 | 脚本 |
| --- | --- |
| FLUX.2 Klein LoRA | `run_flux2_klein.sh` |
| FLUX.2 Klein + 轻量 ControlNet | `run_flux2_klein_controlnet.sh` |
| SANA-Sprint A/B Adapter | `run_sana_sprint.sh` |
| SANA-Sprint + ControlNet | `run_sana_sprint_controlnet.sh` |
| 普通 SANA + ControlNet | `run_sana_controlnet.sh` |
| SANA-Sprint img2img | `run_sana_sprint_img2img.sh` |

`MODE` 支持：

- `MODE=train`：只训练。
- `MODE=infer`：只执行单张推理。
- `MODE=batch`：只执行 metadata 批量推理。
- `MODE=all`：依次执行训练、单张推理和批量推理，用于手动闭环检查。

第一阶段的 SANA-Sprint 脚本默认 `MODE=train`，避免一次正式训练结束后自动遍历整个数据集。

示例：

```bash
MODE=all \
DATASET_BASE_PATH=/data/dof \
METADATA=/data/dof/metadata.json \
IMAGE_A=/data/dof/example_a.png \
IMAGE_B=/data/dof/example_b.png \
bash examples/research_projects/dof_fusion/run_sana_sprint.sh
```

ControlNet 方案额外需要：

```bash
FOCUS_MAP=/data/dof/focus_a.png
CONTROL_INDEX=2
```

`CONTROL_INDEX=2` 使用 `edit_image[2]`，`CONTROL_INDEX=3` 使用 `edit_image[3]`。

普通 Sprint、Sprint img2img 和 FLUX LoRA 可选择只将 focus map 用于加权 loss：

```bash
USE_FOCUS_MAPS=1
FOCUS_LOSS_WEIGHT=1.0
```

## 方案一：SANA-Sprint A/B Adapter

这是建议优先运行的端侧候选方案。SANA-Sprint Transformer、VAE 和文本编码器均冻结，只训练一个外置
双图卷积 Adapter。Adapter 将 A/B 的 VAE latent 拼接后映射成 denoiser 的附加条件。

0.6B checkpoint 的单张 latent 为 32 通道，因此 Adapter 输入为 64 通道，输出为 32 通道。最后一层采用
零初始化，训练开始时不会破坏原始 SANA-Sprint 输出。

训练：

```bash
accelerate launch examples/research_projects/dof_fusion/train_sana_sprint.py \
  --dataset_base_path data \
  --dataset_metadata_path data/metadata.json \
  --output_dir outputs/sana_dof_adapter \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_pixels 2073600 \
  --max_train_steps 20000
```

单张推理：

```bash
python examples/research_projects/dof_fusion/infer_sana_sprint.py \
  --adapter outputs/sana_dof_adapter/adapter.safetensors \
  --image_a data/a.png \
  --image_b data/b.png \
  --output outputs/sana_fused.png \
  --steps 1
```

focus map 始终保持 `[0, 1]` 且不经过 VAE。`--adapter_type ab` 时它只用于可选 focus-aware loss；
`--adapter_type ab_focus` 时，Adapter 输入为 `[A_latent, B_latent, focus_a, focus_b]`，推理会根据
`adapter_config.json` 自动创建对应 Adapter。

## 方案二：FLUX.2 Klein LoRA

FLUX.2 Klein 原生支持多参考图。A/B 分别编码成 reference tokens，与目标 noisy tokens 一起进入
Transformer。训练遵循官方 FLUX.2 Klein img2img 的 latent packing 和 flow-matching 目标，保存标准
Diffusers LoRA。

零样本基线：

```bash
python examples/research_projects/dof_fusion/infer_flux2_klein.py \
  --image_a data/a.png \
  --image_b data/b.png \
  --output outputs/flux2_zero_shot.png \
  --height 1024 \
  --width 1024 \
  --steps 4
```

配对 LoRA 训练：

```bash
accelerate launch examples/research_projects/dof_fusion/train_flux2_klein.py \
  --dataset_base_path data \
  --dataset_metadata_path data/metadata.json \
  --output_dir outputs/flux2_dof_lora \
  --resolution 1024 \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --max_train_steps 20000
```

加载 LoRA 推理：

```bash
python examples/research_projects/dof_fusion/infer_flux2_klein.py \
  --lora outputs/flux2_dof_lora \
  --image_a data/a.png \
  --image_b data/b.png \
  --output outputs/flux2_fused.png
```

该方案适合验证数据和任务的质量上限，但 4B 基模不适合作为当前端侧最终模型。

## 方案三：SANA-Sprint img2img

该方案使用官方 `SanaSprintImg2ImgPipeline`。训练时默认从加噪后的 A latent 初始化，A/B 同时进入双图
Adapter，GT 作为监督目标。

推理时：

- 默认从 A 初始化。
- `--init_image` 可替换为初步融合结果。
- 批量推理可通过 `--init_key` 读取 metadata 中已有的初步融合图路径。
- 建议使用 2～4 步；一步且 `strength=1` 时，初始化图的信息几乎被完全抹除。

```bash
MODE=all \
DATASET_BASE_PATH=/data/dof \
METADATA=/data/dof/metadata.json \
IMAGE_A=/data/dof/a.png \
IMAGE_B=/data/dof/b.png \
INFER_STEPS=4 \
STRENGTH=0.75 \
bash examples/research_projects/dof_fusion/run_sana_sprint_img2img.sh
```

## 方案四：普通 SANA + ControlNet

普通 SANA 使用标准多步 flow-matching 路线，不需要 Sprint 的 one-step 蒸馏。训练内容为 A/B Adapter 和
完整 `SanaControlNetModel`，SANA 基模保持冻结。

需要注意：官方 `SanaControlNetPipeline` 会将 control image 当作 RGB 图像并通过 SANA VAE 编码。因此该
方案中的 focus map 会转成三通道并经过 VAE，训练和推理保持相同处理。

```bash
MODE=all \
DATASET_BASE_PATH=/data/dof \
METADATA=/data/dof/metadata.json \
IMAGE_A=/data/dof/a.png \
IMAGE_B=/data/dof/b.png \
FOCUS_MAP=/data/dof/focus_a.png \
CONTROL_INDEX=2 \
bash examples/research_projects/dof_fusion/run_sana_controlnet.sh
```

## 方案五：SANA-Sprint + ControlNet

Diffusers 没有官方 `SanaSprintControlNetPipeline`。本目录使用外部 wrapper 保留官方 Sprint sCM 推理循环，
同时将 focus map ControlNet residual 注入冻结的 Sprint Transformer。

这是实验路线，需要为当前景深融合任务重新训练 ControlNet，不能直接把普通 SANA ControlNet 权重当作
Sprint one-step 权重使用。

```bash
MODE=all \
DATASET_BASE_PATH=/data/dof \
METADATA=/data/dof/metadata.json \
IMAGE_A=/data/dof/a.png \
IMAGE_B=/data/dof/b.png \
FOCUS_MAP=/data/dof/focus_a.png \
CONTROL_INDEX=2 \
bash examples/research_projects/dof_fusion/run_sana_sprint_controlnet.sh
```

## 方案六：FLUX.2 Klein + 轻量 ControlNet

Diffusers 当前只有 FLUX.1 ControlNet，没有官方 `Flux2ControlNetModel` 或 FLUX.2 ControlNet pipeline。
FLUX.1 ControlNet 与 FLUX.2 Klein 的 Transformer 结构不兼容，因此这里提供的是外置轻量
ControlNet-style residual 分支：

- A/B 继续走 FLUX.2 Klein 原生多参考图 tokens。
- focus map 保持 `[0, 1]`，不经过 VAE。
- 小型 token MLP 为指定 double-stream/single-stream blocks 生成零初始化 residual。
- 4B Transformer、VAE 和文本编码器全部冻结，仅保存控制分支。

默认 residual 注入点为：

```text
double blocks: 0,1,2,3,4
single blocks: 0,4,8,12,16
```

```bash
MODE=all \
DATASET_BASE_PATH=/data/dof \
METADATA=/data/dof/metadata.json \
IMAGE_A=/data/dof/a.png \
IMAGE_B=/data/dof/b.png \
FOCUS_MAP=/data/dof/focus_a.png \
CONTROL_INDEX=2 \
bash examples/research_projects/dof_fusion/run_flux2_klein_controlnet.sh
```

该实现比复制完整 4B ControlNet 更轻，但属于自定义研究实现，不是官方预训练 FLUX.2 ControlNet。

## 批量推理

FLUX.2 Klein pipeline 会将一个 reference image list 视为整批 prompt 共享的参考图。因此，不同样本的
A/B 不能直接组成同一个 FLUX GPU batch。FLUX 批处理脚本只加载一次模型，然后逐条处理 metadata。

```bash
python examples/research_projects/dof_fusion/batch_infer.py \
  --backend flux2 \
  --lora outputs/flux2_dof_lora \
  --dataset_base_path data \
  --dataset_metadata_path data/metadata.json \
  --output_dir outputs/flux2 \
  --steps 4 \
  --skip_existing
```

SANA 的外置 Adapter 会为 batch 中每个样本保留独立 A/B 条件，因此支持真正的 mini-batch：

```bash
python examples/research_projects/dof_fusion/batch_infer.py \
  --backend sana \
  --adapter outputs/sana_dof_adapter/adapter.safetensors \
  --dataset_base_path data \
  --dataset_metadata_path data/metadata.json \
  --output_dir outputs/sana \
  --batch_size 4 \
  --steps 1 \
  --skip_existing
```

批量推理会在输出目录写入 `metadata_results.json`，保留原始 metadata 字段并新增
`generated_image`。`--start_index` 和 `--max_samples` 可用于切分大规模数据。

## 建议先做的检查

正式训练前建议依次执行：

1. 检查 metadata 中所有路径是否存在。
2. 检查 A/B/GT 是否几何配准，尤其是运动物体和边缘区域。
3. 单独测试 VAE encode/decode，观察清晰边界和细纹理损失。
4. 选择少量样本训练 200～500 steps，验证模型是否能过拟合。
5. 分别统计 focus 区域和非 focus 区域的 PSNR、SSIM、LPIPS 或边缘误差。

SANA 的 DC-AE 空间压缩较强，投入完整训练前，应特别检查焦点边缘经过 VAE 后是否出现模糊、光晕或
双边缘。

## 当前限制

- 当前运行环境没有安装 PyTorch 和 Accelerate，因此代码完成了 AST、`py_compile`、CLI 参数和 Bash
  静态检查，但尚未执行真实 GPU 前向和训练 smoke test。
- SANA-Sprint ControlNet 和 FLUX.2 ControlNet 都是自定义研究实现，不是 Diffusers 官方 pipeline。
- 仅训练小 Adapter 成本最低，但容量可能不足；确认欠拟合后，再考虑给 Transformer 增加 LoRA，而不是
  直接解冻整个基模。
- focus map 白色区域约定为保留对应清晰内容，黑色区域表示需要从另一张图恢复；制作和增强数据时应保持
  这一语义一致。
