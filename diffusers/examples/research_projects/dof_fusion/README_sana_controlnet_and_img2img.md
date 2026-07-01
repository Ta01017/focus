# Ordinary SANA ControlNet and SANA-Sprint img2img

These routes are additional experiments. They do not modify files under `src/diffusers` and they retain the existing
SANA-Sprint adapter and SANA-Sprint ControlNet code.

## 1. Ordinary SANA + focus ControlNet

This is the standard multi-step SANA flow-matching route. It does not require Sprint-specific one-step distillation.
It still requires task training: an edge/depth ControlNet checkpoint does not know that a white focus-map region means
"keep A" and a black region means "recover detail from B".

The training script freezes SANA, trains `SanaControlNetModel` plus the small A/B adapter, and uses
`edit_image[2]` by default as the ControlNet image. To use `focus_b`, pass `--control_index 3`.

```bash
accelerate launch examples/research_projects/dof_fusion/train_sana_controlnet.py \
  --dataset_metadata_path data/metadata.json \
  --dataset_base_path data \
  --output_dir outputs/sana_controlnet_dof \
  --control_index 2 \
  --resolution 1024 \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_train_steps 20000
```

Single inference:

```bash
python examples/research_projects/dof_fusion/infer_sana_controlnet.py \
  --checkpoint outputs/sana_controlnet_dof \
  --image_a data/a.png \
  --image_b data/b.png \
  --focus_map data/focus_a.png \
  --output outputs/sana_controlnet.png \
  --steps 20
```

Metadata batch inference:

```bash
python examples/research_projects/dof_fusion/batch_infer_sana_controlnet.py \
  --checkpoint outputs/sana_controlnet_dof \
  --dataset_metadata_path data/metadata.json \
  --dataset_base_path data \
  --output_dir outputs/sana_controlnet_batch \
  --batch_size 2
```

The official ordinary SANA ControlNet pipeline converts the control image to RGB-like image input and encodes it with
the SANA VAE. Training intentionally does the same. The raw `[0, 1]` focus map is retained separately for the optional
focus-weighted loss.

## 2. SANA-Sprint img2img

This route uses the official `SanaSprintImg2ImgPipeline`. During training, A is VAE-encoded and noised as the initial
state, while GT is the reconstruction target. A and B also enter the symmetric two-image adapter and have no fixed
focus-order correspondence. The default A initialization can later be replaced by a preliminary fusion image.

```bash
accelerate launch examples/research_projects/dof_fusion/train_sana_sprint_img2img.py \
  --dataset_metadata_path data/metadata.json \
  --dataset_base_path data \
  --output_dir outputs/sana_sprint_img2img \
  --resolution 1024 \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_train_steps 20000
```

Single inference, initialized from A:

```bash
python examples/research_projects/dof_fusion/infer_sana_sprint_img2img.py \
  --adapter outputs/sana_sprint_img2img/adapter.safetensors \
  --image_a data/a.png \
  --image_b data/b.png \
  --output outputs/sana_sprint_img2img.png \
  --steps 4 \
  --strength 0.75
```

Add `--init_image preliminary_fusion.png` to initialize from a preliminary fusion result. Batch inference defaults to
A and accepts `--init_key preliminary_fusion` when that path already exists in each metadata record:

```bash
python examples/research_projects/dof_fusion/batch_infer_sana_sprint_img2img.py \
  --adapter outputs/sana_sprint_img2img/adapter.safetensors \
  --dataset_metadata_path data/metadata.json \
  --dataset_base_path data \
  --output_dir outputs/sana_sprint_img2img_batch \
  --batch_size 2 \
  --steps 4 \
  --strength 0.75
```

With one inference step, `strength` must effectively select the only (maximum) timestep, where A is almost erased.
Use two to four steps when the purpose is to preserve useful information from the initialization image.
