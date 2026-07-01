#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

MODE=${MODE:-all} # train | infer | batch | all
MODEL=${MODEL:-Efficient-Large-Model/Sana_600M_1024px_diffusers}
DATASET_BASE_PATH=${DATASET_BASE_PATH:-data}
METADATA=${METADATA:-${DATASET_BASE_PATH}/metadata.json}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/dof_fusion/sana_controlnet}
IMAGE_A=${IMAGE_A:-${DATASET_BASE_PATH}/a.png}
IMAGE_B=${IMAGE_B:-${DATASET_BASE_PATH}/b.png}
FOCUS_MAP=${FOCUS_MAP:-${DATASET_BASE_PATH}/focus_a.png}
CONTROL_INDEX=${CONTROL_INDEX:-2}
RESOLUTION=${RESOLUTION:-1024}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-4}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-20000}
INFER_STEPS=${INFER_STEPS:-20}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
SEED=${SEED:-0}

run_train() {
  accelerate launch "${SCRIPT_DIR}/train_sana_controlnet.py" \
    --model "${MODEL}" \
    --dataset_metadata_path "${METADATA}" \
    --dataset_base_path "${DATASET_BASE_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --control_index "${CONTROL_INDEX}" \
    --resolution "${RESOLUTION}" \
    --batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --gradient_checkpointing \
    --max_train_steps "${MAX_TRAIN_STEPS}" \
    --mixed_precision "${MIXED_PRECISION}"
}

run_infer() {
  python "${SCRIPT_DIR}/infer_sana_controlnet.py" \
    --checkpoint "${OUTPUT_DIR}" \
    --model "${MODEL}" \
    --image_a "${IMAGE_A}" \
    --image_b "${IMAGE_B}" \
    --focus_map "${FOCUS_MAP}" \
    --output "${OUTPUT_DIR}/single_result.png" \
    --height "${RESOLUTION}" \
    --width "${RESOLUTION}" \
    --steps "${INFER_STEPS}" \
    --seed "${SEED}"
}

run_batch() {
  python "${SCRIPT_DIR}/batch_infer_sana_controlnet.py" \
    --checkpoint "${OUTPUT_DIR}" \
    --model "${MODEL}" \
    --dataset_metadata_path "${METADATA}" \
    --dataset_base_path "${DATASET_BASE_PATH}" \
    --output_dir "${OUTPUT_DIR}/batch" \
    --control_index "${CONTROL_INDEX}" \
    --height "${RESOLUTION}" \
    --width "${RESOLUTION}" \
    --batch_size "${BATCH_SIZE}" \
    --steps "${INFER_STEPS}" \
    --seed "${SEED}"
}

case "${MODE}" in
  train) run_train ;;
  infer) run_infer ;;
  batch) run_batch ;;
  all) run_train; run_infer; run_batch ;;
  *) echo "MODE must be train, infer, batch, or all" >&2; exit 2 ;;
esac
