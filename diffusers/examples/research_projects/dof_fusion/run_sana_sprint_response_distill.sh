#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODE=${MODE:-all} # generate | train | all
TEACHER_BACKEND=${TEACHER_BACKEND:?设置 diffusers_flux2 或 command}
TEACHER_MODEL=${TEACHER_MODEL:-black-forest-labs/FLUX.2-klein-base-4B}
TEACHER_COMMAND=${TEACHER_COMMAND:-}
TEACHER_STEPS=${TEACHER_STEPS:-50}
TEACHER_GUIDANCE_SCALE=${TEACHER_GUIDANCE_SCALE:-1.0}
TEACHER_LORA=${TEACHER_LORA:-}
STUDENT_MODEL=${STUDENT_MODEL:-Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers}
DATASET_BASE_PATH=${DATASET_BASE_PATH:-data}
METADATA=${METADATA:-${DATASET_BASE_PATH}/metadata.json}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/dof_fusion/response_distill}
TEACHER_DIR=${TEACHER_DIR:-${OUTPUT_DIR}/teacher_targets}
TEACHER_METADATA=${TEACHER_METADATA:-${OUTPUT_DIR}/teacher_metadata.json}
ADAPTER_TYPE=${ADAPTER_TYPE:-ab}
MAX_PIXELS=${MAX_PIXELS:-}
SIZE_DIVISOR=${SIZE_DIVISOR:-32}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-4}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-20000}
MAX_SAMPLES=${MAX_SAMPLES:-}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}

if [[ "${BATCH_SIZE}" != "1" ]]; then echo "Dynamic response distillation requires BATCH_SIZE=1" >&2; exit 2; fi

generate_args=(
  --teacher_backend "${TEACHER_BACKEND}"
  --teacher_model "${TEACHER_MODEL}"
  --teacher_steps "${TEACHER_STEPS}"
  --teacher_guidance_scale "${TEACHER_GUIDANCE_SCALE}"
  --dataset_metadata_path "${METADATA}"
  --dataset_base_path "${DATASET_BASE_PATH}"
  --output_dir "${TEACHER_DIR}"
  --output_metadata_path "${TEACHER_METADATA}"
)
if [[ -n "${TEACHER_COMMAND}" ]]; then generate_args+=(--teacher_command "${TEACHER_COMMAND}"); fi
if [[ -n "${TEACHER_LORA}" ]]; then generate_args+=(--teacher_lora "${TEACHER_LORA}"); fi
if [[ -n "${MAX_PIXELS}" ]]; then generate_args+=(--max_pixels "${MAX_PIXELS}"); fi
if [[ -n "${MAX_SAMPLES}" ]]; then generate_args+=(--max_samples "${MAX_SAMPLES}"); fi
train_subset_args=()
if [[ -n "${MAX_SAMPLES}" ]]; then train_subset_args+=(--max_samples "${MAX_SAMPLES}"); fi

run_generate() {
  python "${SCRIPT_DIR}/generate_dof_teacher_targets.py" "${generate_args[@]}"
}

run_train() {
  local focus_args=()
  if [[ "${ADAPTER_TYPE}" == "ab_focus" ]]; then focus_args+=(--use_focus_maps); fi
  accelerate launch "${SCRIPT_DIR}/train_sana_sprint.py" \
    --model "${STUDENT_MODEL}" \
    --dataset_metadata_path "${TEACHER_METADATA}" \
    --dataset_base_path "${DATASET_BASE_PATH}" \
    --target_key teacher_image \
    --adapter_type "${ADAPTER_TYPE}" \
    --output_dir "${OUTPUT_DIR}/student" \
    --batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --max_train_steps "${MAX_TRAIN_STEPS}" \
    --mixed_precision "${MIXED_PRECISION}" \
    --size_divisor "${SIZE_DIVISOR}" \
    "${train_subset_args[@]}" \
    "${focus_args[@]}"
}

case "${MODE}" in
  generate) run_generate ;;
  train) run_train ;;
  all) run_generate; run_train ;;
  *) echo "MODE must be generate, train, or all" >&2; exit 2 ;;
esac
