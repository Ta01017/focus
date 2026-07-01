#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

MODE=${MODE:-all} # train | infer | batch | all
MODEL=${MODEL:-Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers}
DATASET_BASE_PATH=${DATASET_BASE_PATH:-data}
METADATA=${METADATA:-${DATASET_BASE_PATH}/metadata.json}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/dof_fusion/sana_sprint_img2img}
IMAGE_A=${IMAGE_A:-${DATASET_BASE_PATH}/a.png}
IMAGE_B=${IMAGE_B:-${DATASET_BASE_PATH}/b.png}
INIT_IMAGE=${INIT_IMAGE:-${IMAGE_A}}
RESOLUTION=${RESOLUTION:-1024}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-4}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-20000}
INFER_STEPS=${INFER_STEPS:-4}
STRENGTH=${STRENGTH:-0.75}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
USE_FOCUS_MAPS=${USE_FOCUS_MAPS:-0}
FOCUS_LOSS_WEIGHT=${FOCUS_LOSS_WEIGHT:-0.0}
SEED=${SEED:-0}

run_train() {
  local focus_args=()
  if [[ "${USE_FOCUS_MAPS}" == "1" ]]; then
    focus_args+=(--use_focus_maps --focus_loss_weight "${FOCUS_LOSS_WEIGHT}")
  fi
  accelerate launch "${SCRIPT_DIR}/train_sana_sprint_img2img.py" \
    --model "${MODEL}" \
    --dataset_metadata_path "${METADATA}" \
    --dataset_base_path "${DATASET_BASE_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --resolution "${RESOLUTION}" \
    --batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --max_train_steps "${MAX_TRAIN_STEPS}" \
    --mixed_precision "${MIXED_PRECISION}" \
    "${focus_args[@]}"
}

run_infer() {
  python "${SCRIPT_DIR}/infer_sana_sprint_img2img.py" \
    --adapter "${OUTPUT_DIR}/adapter.safetensors" \
    --model "${MODEL}" \
    --image_a "${IMAGE_A}" \
    --image_b "${IMAGE_B}" \
    --init_image "${INIT_IMAGE}" \
    --output "${OUTPUT_DIR}/single_result.png" \
    --height "${RESOLUTION}" \
    --width "${RESOLUTION}" \
    --steps "${INFER_STEPS}" \
    --strength "${STRENGTH}" \
    --seed "${SEED}"
}

run_batch() {
  python "${SCRIPT_DIR}/batch_infer_sana_sprint_img2img.py" \
    --adapter "${OUTPUT_DIR}/adapter.safetensors" \
    --model "${MODEL}" \
    --dataset_metadata_path "${METADATA}" \
    --dataset_base_path "${DATASET_BASE_PATH}" \
    --output_dir "${OUTPUT_DIR}/batch" \
    --height "${RESOLUTION}" \
    --width "${RESOLUTION}" \
    --batch_size "${BATCH_SIZE}" \
    --steps "${INFER_STEPS}" \
    --strength "${STRENGTH}" \
    --seed "${SEED}"
}

case "${MODE}" in
  train) run_train ;;
  infer) run_infer ;;
  batch) run_batch ;;
  all) run_train; run_infer; run_batch ;;
  *) echo "MODE must be train, infer, batch, or all" >&2; exit 2 ;;
esac
