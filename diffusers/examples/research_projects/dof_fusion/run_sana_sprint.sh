#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

MODE=${MODE:-train} # train | infer | batch | all
# 第一阶段仅用于 E1/E2/E3。E1/E2 使用 ab，E3 使用 ab_focus。
MODEL=${MODEL:-Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers}
DATASET_BASE_PATH=${DATASET_BASE_PATH:-data}
METADATA=${METADATA:-${DATASET_BASE_PATH}/metadata.json}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/dof_fusion/sana_sprint}
IMAGE_A=${IMAGE_A:-${DATASET_BASE_PATH}/a.png}
IMAGE_B=${IMAGE_B:-${DATASET_BASE_PATH}/b.png}
MAX_PIXELS=${MAX_PIXELS:-}
SIZE_DIVISOR=${SIZE_DIVISOR:-32}
ASPECT_RATIO_TOLERANCE=${ASPECT_RATIO_TOLERANCE:-0.01}
DOWNSCALE_IF_EXCEEDS_MAX_PIXELS=${DOWNSCALE_IF_EXCEEDS_MAX_PIXELS:-0}
RESTORE_TO_ORIGINAL_SIZE=${RESTORE_TO_ORIGINAL_SIZE:-1}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-4}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-20000}
INFER_STEPS=${INFER_STEPS:-1}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
USE_FOCUS_MAPS=${USE_FOCUS_MAPS:-0}
FOCUS_LOSS_WEIGHT=${FOCUS_LOSS_WEIGHT:-0.0}
ADAPTER_TYPE=${ADAPTER_TYPE:-ab}
FOCUS_A=${FOCUS_A:-${DATASET_BASE_PATH}/focus_a.png}
FOCUS_B=${FOCUS_B:-${DATASET_BASE_PATH}/focus_b.png}
SEED=${SEED:-0}
CACHE_DIR=${CACHE_DIR:-}
REVISION=${REVISION:-}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-0}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}

pretrained_args=()
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then pretrained_args+=(--local_files_only); fi
if [[ -n "${CACHE_DIR}" ]]; then pretrained_args+=(--cache_dir "${CACHE_DIR}"); fi
if [[ -n "${REVISION}" ]]; then pretrained_args+=(--revision "${REVISION}"); fi
resume_args=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  resume_args+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
size_args=(--size_divisor "${SIZE_DIVISOR}" --aspect_ratio_tolerance "${ASPECT_RATIO_TOLERANCE}")
if [[ -n "${MAX_PIXELS}" ]]; then size_args+=(--max_pixels "${MAX_PIXELS}"); fi
if [[ "${DOWNSCALE_IF_EXCEEDS_MAX_PIXELS}" == "1" ]]; then
  size_args+=(--downscale_if_exceeds_max_pixels)
fi
restore_args=(--restore_to_original_size)
if [[ "${RESTORE_TO_ORIGINAL_SIZE}" != "1" ]]; then restore_args=(--no_restore_to_original_size); fi

run_train() {
  local focus_args=()
  if [[ "${USE_FOCUS_MAPS}" == "1" ]]; then
    focus_args+=(--use_focus_maps --focus_loss_weight "${FOCUS_LOSS_WEIGHT}")
  fi
  accelerate launch "${SCRIPT_DIR}/train_sana_sprint.py" \
    --model "${MODEL}" \
    --dataset_metadata_path "${METADATA}" \
    --dataset_base_path "${DATASET_BASE_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --adapter_type "${ADAPTER_TYPE}" \
    --max_train_steps "${MAX_TRAIN_STEPS}" \
    --mixed_precision "${MIXED_PRECISION}" \
    "${focus_args[@]}" \
    "${resume_args[@]}" \
    "${size_args[@]}" \
    "${pretrained_args[@]}"
}

run_infer() {
  local focus_args=()
  if [[ "${ADAPTER_TYPE}" == "ab_focus" ]]; then
    focus_args+=(--focus_a "${FOCUS_A}" --focus_b "${FOCUS_B}")
  fi
  python "${SCRIPT_DIR}/infer_sana_sprint.py" \
    --model "${MODEL}" \
    --adapter "${OUTPUT_DIR}/adapter.safetensors" \
    --image_a "${IMAGE_A}" \
    --image_b "${IMAGE_B}" \
    --output "${OUTPUT_DIR}/single_result.png" \
    --steps "${INFER_STEPS}" \
    --seed "${SEED}" \
    "${focus_args[@]}" \
    "${size_args[@]}" \
    "${restore_args[@]}" \
    "${pretrained_args[@]}"
}

run_batch() {
  python "${SCRIPT_DIR}/batch_infer.py" \
    --backend sana \
    --model "${MODEL}" \
    --adapter "${OUTPUT_DIR}/adapter.safetensors" \
    --dataset_metadata_path "${METADATA}" \
    --dataset_base_path "${DATASET_BASE_PATH}" \
    --output_dir "${OUTPUT_DIR}/batch" \
    --batch_size "${BATCH_SIZE}" \
    --steps "${INFER_STEPS}" \
    --seed "${SEED}" \
    "${size_args[@]}" \
    "${restore_args[@]}" \
    "${pretrained_args[@]}"
}

case "${MODE}" in
  train) run_train ;;
  infer) run_infer ;;
  batch) run_batch ;;
  all) run_train; run_infer; run_batch ;;
  *) echo "MODE must be train, infer, batch, or all" >&2; exit 2 ;;
esac
