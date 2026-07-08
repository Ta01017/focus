#!/usr/bin/env bash
set -euo pipefail

MODE=${MODE:-batch_infer}
DATA_ROOT=${DATA_ROOT:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880}
MODEL=${MODEL:-Efficient-Large-Model/Sana_600M_1024px_diffusers}
DATASET_METADATA_PATH=${DATASET_METADATA_PATH:-}
DATASET_BASE_PATH=${DATASET_BASE_PATH:-}
OUTPUT_DIR=${OUTPUT_DIR:-${DATA_ROOT}/focus/models/train/sana_artifact_repair}
CHECKPOINT=${CHECKPOINT:-${OUTPUT_DIR}}

MAX_PIXELS=${MAX_PIXELS:-1048576}
DOWNSCALE_IF_EXCEEDS_MAX_PIXELS=${DOWNSCALE_IF_EXCEEDS_MAX_PIXELS:-1}
SIZE_DIVISOR=${SIZE_DIVISOR:-32}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-1}
CACHE_DIR=${CACHE_DIR:-}
REVISION=${REVISION:-}

TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-20000}
SAVE_STEPS=${SAVE_STEPS:-500}
LOG_STEPS=${LOG_STEPS:-10}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
LOSS_MODE=${LOSS_MODE:-legacy_src_noise_gt_velocity}
CONTROL_CONDITION_CHANNELS=${CONTROL_CONDITION_CHANNELS:-6}
TRAIN_TRANSFORMER_LORA=${TRAIN_TRANSFORMER_LORA:-1}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-8}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-to_q,to_k,to_v,to_out.0}
MIXED_PRECISION=${MIXED_PRECISION:-no}
NUM_WORKERS=${NUM_WORKERS:-4}

SRC_IMAGE=${SRC_IMAGE:-}
REF_IMAGE=${REF_IMAGE:-}
PROMPT=${PROMPT:-}
OUTPUT_PATH=${OUTPUT_PATH:-${OUTPUT_DIR}/artifact_repair.png}
STRENGTH=${STRENGTH:-0.15}
IMG2IMG_SCHEDULE_MODE=${IMG2IMG_SCHEDULE_MODE:-sliced}
CONDITIONING_SCALE=${CONDITIONING_SCALE:-1.0}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-1.0}
STEPS=${STEPS:-20}
SEED=${SEED:-0}
START_INDEX=${START_INDEX:-0}
MAX_SAMPLES=${MAX_SAMPLES:-}
ZERO_REF_CONDITION=${ZERO_REF_CONDITION:-0}
ZERO_SRC_CONDITION=${ZERO_SRC_CONDITION:-0}
SKIP_EXISTING=${SKIP_EXISTING:-0}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}
DEBUG_LATENT_DIR=${DEBUG_LATENT_DIR:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

common_pretrained_args=()
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  common_pretrained_args+=(--local_files_only)
fi
if [[ -n "${CACHE_DIR}" ]]; then
  common_pretrained_args+=(--cache_dir "${CACHE_DIR}")
fi
if [[ -n "${REVISION}" ]]; then
  common_pretrained_args+=(--revision "${REVISION}")
fi

size_args=(--max_pixels "${MAX_PIXELS}" --size_divisor "${SIZE_DIVISOR}")
if [[ "${DOWNSCALE_IF_EXCEEDS_MAX_PIXELS}" == "1" ]]; then
  size_args+=(--downscale_if_exceeds_max_pixels)
fi

echo "[MODE] ${MODE}"
echo "[MODEL] ${MODEL}"
echo "[DATASET_METADATA_PATH] ${DATASET_METADATA_PATH}"
echo "[DATASET_BASE_PATH] ${DATASET_BASE_PATH}"
echo "[OUTPUT_DIR] ${OUTPUT_DIR}"
echo "[MAX_PIXELS] ${MAX_PIXELS}"
echo "[DOWNSCALE_IF_EXCEEDS_MAX_PIXELS] ${DOWNSCALE_IF_EXCEEDS_MAX_PIXELS}"
echo "[SIZE_DIVISOR] ${SIZE_DIVISOR}"
echo "[LOCAL_FILES_ONLY] ${LOCAL_FILES_ONLY}"
echo "[LOSS_MODE] ${LOSS_MODE}"
echo "[CONTROL_CONDITION_CHANNELS] ${CONTROL_CONDITION_CHANNELS}"
echo "[TRAIN_TRANSFORMER_LORA] ${TRAIN_TRANSFORMER_LORA}"
echo "[LORA_RANK] ${LORA_RANK}"
echo "[LORA_ALPHA] ${LORA_ALPHA}"
echo "[LORA_TARGET_MODULES] ${LORA_TARGET_MODULES}"

run_train() {
  if [[ -z "${DATASET_METADATA_PATH}" || -z "${DATASET_BASE_PATH}" ]]; then
    echo "DATASET_METADATA_PATH and DATASET_BASE_PATH are required for MODE=train" >&2
    exit 1
  fi
  train_sample_args=()
  if [[ -n "${TRAIN_MAX_SAMPLES}" ]]; then
    train_sample_args+=(--max_samples "${TRAIN_MAX_SAMPLES}")
  fi
  lora_args=()
  if [[ "${TRAIN_TRANSFORMER_LORA}" == "1" ]]; then
    lora_args+=(--train_transformer_lora)
    lora_args+=(--lora_rank "${LORA_RANK}")
    lora_args+=(--lora_alpha "${LORA_ALPHA}")
    lora_args+=(--lora_dropout "${LORA_DROPOUT}")
    lora_args+=(--lora_target_modules "${LORA_TARGET_MODULES}")
  fi
  mkdir -p "${OUTPUT_DIR}"
  accelerate launch --mixed_precision "${MIXED_PRECISION}" \
    "${SCRIPT_DIR}/train_sana_artifact_repair_controlnet.py" \
    --model "${MODEL}" \
    --dataset_metadata_path "${DATASET_METADATA_PATH}" \
    --dataset_base_path "${DATASET_BASE_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --loss_mode "${LOSS_MODE}" \
    --control_condition_channels "${CONTROL_CONDITION_CHANNELS}" \
    --batch_size "${TRAIN_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --max_train_steps "${MAX_TRAIN_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --log_steps "${LOG_STEPS}" \
    --num_workers "${NUM_WORKERS}" \
    --mixed_precision "${MIXED_PRECISION}" \
    "${size_args[@]}" \
    "${common_pretrained_args[@]}" \
    "${train_sample_args[@]}" \
    "${lora_args[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"
}

run_infer() {
  if [[ -z "${SRC_IMAGE}" || -z "${REF_IMAGE}" ]]; then
    echo "SRC_IMAGE and REF_IMAGE are required for MODE=infer" >&2
    exit 1
  fi
  prompt_args=()
  if [[ -n "${PROMPT}" ]]; then
    prompt_args+=(--prompt "${PROMPT}")
  fi
  zero_args=()
  if [[ "${ZERO_REF_CONDITION}" == "1" ]]; then
    zero_args+=(--zero_ref_condition)
  fi
  if [[ "${ZERO_SRC_CONDITION}" == "1" ]]; then
    zero_args+=(--zero_src_condition)
  fi
  if [[ -n "${DEBUG_LATENT_DIR}" ]]; then
    zero_args+=(--debug_latent_dir "${DEBUG_LATENT_DIR}")
  fi
  python "${SCRIPT_DIR}/infer_sana_artifact_repair_controlnet.py" \
    --checkpoint "${CHECKPOINT}" \
    --model "${MODEL}" \
    --src_image "${SRC_IMAGE}" \
    --ref_image "${REF_IMAGE}" \
    --output_path "${OUTPUT_PATH}" \
    --strength "${STRENGTH}" \
    --img2img_schedule_mode "${IMG2IMG_SCHEDULE_MODE}" \
    --steps "${STEPS}" \
    --guidance_scale "${GUIDANCE_SCALE}" \
    --conditioning_scale "${CONDITIONING_SCALE}" \
    --seed "${SEED}" \
    "${size_args[@]}" \
    "${common_pretrained_args[@]}" \
    "${prompt_args[@]}" \
    "${zero_args[@]}"
}

run_batch_infer() {
  if [[ -z "${DATASET_METADATA_PATH}" || -z "${DATASET_BASE_PATH}" ]]; then
    echo "DATASET_METADATA_PATH and DATASET_BASE_PATH are required for MODE=batch_infer" >&2
    exit 1
  fi
  batch_args=()
  if [[ -n "${MAX_SAMPLES}" ]]; then
    batch_args+=(--max_samples "${MAX_SAMPLES}")
  fi
  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    batch_args+=(--skip_existing)
  fi
  if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
    batch_args+=(--continue_on_error)
  fi
  if [[ "${ZERO_REF_CONDITION}" == "1" ]]; then
    batch_args+=(--zero_ref_condition)
  fi
  if [[ "${ZERO_SRC_CONDITION}" == "1" ]]; then
    batch_args+=(--zero_src_condition)
  fi
  if [[ -n "${DEBUG_LATENT_DIR}" ]]; then
    batch_args+=(--debug_latent_dir "${DEBUG_LATENT_DIR}")
  fi
  python "${SCRIPT_DIR}/batch_infer_sana_artifact_repair_controlnet.py" \
    --checkpoint "${CHECKPOINT}" \
    --model "${MODEL}" \
    --dataset_metadata_path "${DATASET_METADATA_PATH}" \
    --dataset_base_path "${DATASET_BASE_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --start_index "${START_INDEX}" \
    --strength "${STRENGTH}" \
    --img2img_schedule_mode "${IMG2IMG_SCHEDULE_MODE}" \
    --steps "${STEPS}" \
    --guidance_scale "${GUIDANCE_SCALE}" \
    --conditioning_scale "${CONDITIONING_SCALE}" \
    --seed "${SEED}" \
    "${size_args[@]}" \
    "${common_pretrained_args[@]}" \
    "${batch_args[@]}"
}

case "${MODE}" in
  train)
    run_train
    ;;
  infer)
    run_infer
    ;;
  batch_infer)
    run_batch_infer
    ;;
  *)
    echo "Unsupported MODE=${MODE}; expected train, infer, or batch_infer." >&2
    exit 1
    ;;
esac
