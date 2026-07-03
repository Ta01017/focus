# SANA-Sprint + focus-map ControlNet

> Resolution: dynamic A-referenced content is the default, padded to a multiple of 32. Padding is excluded from
> loss and inference restores A's original size. Dynamic mode currently requires `batch_size=1`.

This is an additional route. It does not replace or modify the existing SANA dual-image adapter route.

Architecture:

```text
A -> SANA VAE --\
                 DualImageConditionAdapter -> SANA-Sprint noisy latent
B -> SANA VAE --/                              |
                                                  + frozen SANA-Sprint transformer -> target
focus map [0,1] -> resize to latent resolution -> SanaControlNetModel residuals
```

The official `SanaControlNetPipeline` uses the regular SANA scheduler. These scripts instead keep the official
`SanaSprintPipeline` sCM loop and invoke `SanaControlNetModel` from an external transformer wrapper.

## Metadata

The default control input is `edit_image[2]`:

```json
[
  {
    "image": "images/target.png",
    "edit_image": [
      "images/a.png",
      "images/b.png",
      "images/focus_a.png",
      "images/focus_b.png"
    ],
    "prompt": "a photorealistic all-in-focus photograph"
  }
]
```

Use `--control_index 3` when `edit_image[3]` should be the ControlNet input. Focus maps are loaded as grayscale,
resized with bilinear interpolation, normalized to `[0,1]`, and never passed through the VAE.

## Training

```bash
accelerate launch examples/research_projects/dof_fusion/train_sana_sprint_controlnet.py \
  --dataset_base_path data \
  --dataset_metadata_path data/metadata.json \
  --output_dir outputs/sana_sprint_focus_controlnet \
  --control_index 2 \
  --controlnet_layers 7 \
  --resolution 1024 \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --max_train_steps 20000
```

The checkpoint contains:

```text
outputs/sana_sprint_focus_controlnet/
  adapter.safetensors
  controlnet/
    config.json
    diffusion_pytorch_model.safetensors
  controlnet_config.json
```

## Single-sample inference

```bash
python examples/research_projects/dof_fusion/infer_sana_sprint_controlnet.py \
  --checkpoint outputs/sana_sprint_focus_controlnet \
  --image_a data/a.png \
  --image_b data/b.png \
  --focus_map data/focus_a.png \
  --output outputs/fused.png \
  --steps 1
```

## Metadata batch inference

```bash
python examples/research_projects/dof_fusion/batch_infer_sana_sprint_controlnet.py \
  --checkpoint outputs/sana_sprint_focus_controlnet \
  --dataset_base_path data \
  --dataset_metadata_path data/metadata.json \
  --output_dir outputs/controlnet_batch \
  --batch_size 4 \
  --steps 1 \
  --skip_existing
```
