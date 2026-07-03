#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

MODE=${MODE:-all} # train | infer | batch | all
# 非第一阶段方案：完成 E1-E4 前不要与第一阶段并行启动。
MODEL=${MODEL:-Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers}
DATASET_BASE_PATH=${DATASET_BASE_PATH:-data}
METADATA=${METADATA:-${DATASET_BASE_PATH}/metadata.json}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/dof_fusion/sana_sprint_controlnet}
IMAGE_A=${IMAGE_A:-${DATASET_BASE_PATH}/a.png}
IMAGE_B=${IMAGE_B:-${DATASET_BASE_PATH}/b.png}
FOCUS_MAP=${FOCUS_MAP:-${DATASET_BASE_PATH}/focus_a.png}
CONTROL_INDEX=${CONTROL_INDEX:-2}
RESOLUTION=${RESOLUTION:-}
MAX_PIXELS=${MAX_PIXELS:-}
SIZE_DIVISOR=${SIZE_DIVISOR:-32}
ASPECT_RATIO_TOLERANCE=${ASPECT_RATIO_TOLERANCE:-0.01}
DOWNSCALE_IF_EXCEEDS_MAX_PIXELS=${DOWNSCALE_IF_EXCEEDS_MAX_PIXELS:-0}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-4}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-20000}
INFER_STEPS=${INFER_STEPS:-1}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
SEED=${SEED:-0}

if [[ -z "${RESOLUTION}" && "${BATCH_SIZE}" != "1" ]]; then echo "Dynamic resolution requires BATCH_SIZE=1" >&2; exit 2; fi
train_size_args=(--size_divisor "${SIZE_DIVISOR}" --aspect_ratio_tolerance "${ASPECT_RATIO_TOLERANCE}")
infer_size_args=(--size_divisor "${SIZE_DIVISOR}" --aspect_ratio_tolerance "${ASPECT_RATIO_TOLERANCE}" --restore_to_original_size)
if [[ -n "${RESOLUTION}" ]]; then
  train_size_args+=(--resolution "${RESOLUTION}")
  infer_size_args+=(--height "${RESOLUTION}" --width "${RESOLUTION}")
elif [[ -n "${MAX_PIXELS}" ]]; then
  train_size_args+=(--max_pixels "${MAX_PIXELS}")
  infer_size_args+=(--max_pixels "${MAX_PIXELS}")
fi
if [[ "${DOWNSCALE_IF_EXCEEDS_MAX_PIXELS}" == "1" ]]; then train_size_args+=(--downscale_if_exceeds_max_pixels); infer_size_args+=(--downscale_if_exceeds_max_pixels); fi

run_train() {
  accelerate launch "${SCRIPT_DIR}/train_sana_sprint_controlnet.py" \
    --model "${MODEL}" \
    --dataset_metadata_path "${METADATA}" \
    --dataset_base_path "${DATASET_BASE_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --control_index "${CONTROL_INDEX}" \
    --batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --gradient_checkpointing \
    --max_train_steps "${MAX_TRAIN_STEPS}" \
    --mixed_precision "${MIXED_PRECISION}" \
    "${train_size_args[@]}"
}

run_infer() {
  python "${SCRIPT_DIR}/infer_sana_sprint_controlnet.py" \
    --checkpoint "${OUTPUT_DIR}" \
    --model "${MODEL}" \
    --image_a "${IMAGE_A}" \
    --image_b "${IMAGE_B}" \
    --focus_map "${FOCUS_MAP}" \
    --output "${OUTPUT_DIR}/single_result.png" \
    --steps "${INFER_STEPS}" \
    --seed "${SEED}" \
    "${infer_size_args[@]}"
}

run_batch() {
  python "${SCRIPT_DIR}/batch_infer_sana_sprint_controlnet.py" \
    --checkpoint "${OUTPUT_DIR}" \
    --model "${MODEL}" \
    --dataset_metadata_path "${METADATA}" \
    --dataset_base_path "${DATASET_BASE_PATH}" \
    --output_dir "${OUTPUT_DIR}/batch" \
    --control_index "${CONTROL_INDEX}" \
    --batch_size "${BATCH_SIZE}" \
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
