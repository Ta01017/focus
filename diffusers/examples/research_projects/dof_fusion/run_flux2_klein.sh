#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

MODE=${MODE:-all} # train | infer | batch | all
# 非第一阶段方案：当前先完成 E1-E4 工程闭环。
MODEL=${MODEL:-black-forest-labs/FLUX.2-klein-4B}
DATASET_BASE_PATH=${DATASET_BASE_PATH:-data}
METADATA=${METADATA:-${DATASET_BASE_PATH}/metadata.json}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/dof_fusion/flux2_klein}
IMAGE_A=${IMAGE_A:-${DATASET_BASE_PATH}/a.png}
IMAGE_B=${IMAGE_B:-${DATASET_BASE_PATH}/b.png}
RESOLUTION=${RESOLUTION:-}
MAX_PIXELS=${MAX_PIXELS:-}
SIZE_DIVISOR=${SIZE_DIVISOR:-16}
RESTORE_TO_ORIGINAL_SIZE=${RESTORE_TO_ORIGINAL_SIZE:-1}
ASPECT_RATIO_TOLERANCE=${ASPECT_RATIO_TOLERANCE:-0.01}
DOWNSCALE_IF_EXCEEDS_MAX_PIXELS=${DOWNSCALE_IF_EXCEEDS_MAX_PIXELS:-0}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-4}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-20000}
INFER_STEPS=${INFER_STEPS:-4}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
USE_FOCUS_MAPS=${USE_FOCUS_MAPS:-0}
FOCUS_LOSS_WEIGHT=${FOCUS_LOSS_WEIGHT:-0.0}
SEED=${SEED:-0}

if [[ -z "${RESOLUTION}" && "${BATCH_SIZE}" != "1" ]]; then
  echo "Dynamic resolution requires BATCH_SIZE=1" >&2
  exit 2
fi
train_size_args=(--size_divisor "${SIZE_DIVISOR}" --aspect_ratio_tolerance "${ASPECT_RATIO_TOLERANCE}")
infer_size_args=(--size_divisor "${SIZE_DIVISOR}" --aspect_ratio_tolerance "${ASPECT_RATIO_TOLERANCE}")
if [[ -n "${RESOLUTION}" ]]; then
  train_size_args+=(--resolution "${RESOLUTION}")
  infer_size_args+=(--height "${RESOLUTION}" --width "${RESOLUTION}")
elif [[ -n "${MAX_PIXELS}" ]]; then
  train_size_args+=(--max_pixels "${MAX_PIXELS}")
  infer_size_args+=(--max_pixels "${MAX_PIXELS}")
fi
if [[ "${DOWNSCALE_IF_EXCEEDS_MAX_PIXELS}" == "1" ]]; then
  train_size_args+=(--downscale_if_exceeds_max_pixels)
  infer_size_args+=(--downscale_if_exceeds_max_pixels)
fi
if [[ "${RESTORE_TO_ORIGINAL_SIZE}" == "1" ]]; then
  infer_size_args+=(--restore_to_original_size)
else
  infer_size_args+=(--no_restore_to_original_size)
fi

run_train() {
  local focus_args=()
  if [[ "${USE_FOCUS_MAPS}" == "1" ]]; then
    focus_args+=(--use_focus_maps --focus_loss_weight "${FOCUS_LOSS_WEIGHT}")
  fi
  accelerate launch "${SCRIPT_DIR}/train_flux2_klein.py" \
    --model "${MODEL}" \
    --dataset_metadata_path "${METADATA}" \
    --dataset_base_path "${DATASET_BASE_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --gradient_checkpointing \
    --max_train_steps "${MAX_TRAIN_STEPS}" \
    --mixed_precision "${MIXED_PRECISION}" \
    "${focus_args[@]}" \
    "${train_size_args[@]}"
}

run_infer() {
  python "${SCRIPT_DIR}/infer_flux2_klein.py" \
    --model "${MODEL}" \
    --lora "${OUTPUT_DIR}" \
    --image_a "${IMAGE_A}" \
    --image_b "${IMAGE_B}" \
    --output "${OUTPUT_DIR}/single_result.png" \
    --steps "${INFER_STEPS}" \
    --seed "${SEED}" \
    "${infer_size_args[@]}"
}

run_batch() {
  python "${SCRIPT_DIR}/batch_infer.py" \
    --backend flux2 \
    --model "${MODEL}" \
    --lora "${OUTPUT_DIR}" \
    --dataset_metadata_path "${METADATA}" \
    --dataset_base_path "${DATASET_BASE_PATH}" \
    --output_dir "${OUTPUT_DIR}/batch" \
    --steps "${INFER_STEPS}" \
    --seed "${SEED}" \
    "${infer_size_args[@]}"
}

case "${MODE}" in
  train) run_train ;;
  infer) run_infer ;;
  batch) run_batch ;;
  all) run_train; run_infer; run_batch ;;
  *) echo "MODE must be train, infer, batch, or all" >&2; exit 2 ;;
esac
