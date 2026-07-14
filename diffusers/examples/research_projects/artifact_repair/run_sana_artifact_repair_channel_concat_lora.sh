#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE=${MODE:-train}
MODEL=${MODEL:-Efficient-Large-Model/Sana_600M_1024px_diffusers}
ART_BASE=${ART_BASE:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora/data-prompt-aug}
DATASET_METADATA_PATH=${DATASET_METADATA_PATH:-${ART_BASE}/metadata_train.json}
TEST_METADATA_PATH=${TEST_METADATA_PATH:-${ART_BASE}/metadata_test.json}
DATASET_BASE_PATH=${DATASET_BASE_PATH:-${ART_BASE}}
OUTPUT_DIR=${OUTPUT_DIR:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/focus/models/train/sana_artifact_repair/route2_channel_concat_lora}
CHECKPOINT=${CHECKPOINT:-${OUTPUT_DIR}}

NUM_PROCESSES=${NUM_PROCESSES:-1}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29501}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-1}
CACHE_DIR=${CACHE_DIR:-}
REVISION=${REVISION:-}

START_INDEX=${START_INDEX:-0}
MAX_SAMPLES=${MAX_SAMPLES:-1}
DATASET_REPEAT=${DATASET_REPEAT:-1}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-2000}
SAVE_STEPS=${SAVE_STEPS:-500}
LOG_STEPS=${LOG_STEPS:-10}
BATCH_SIZE=${BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
PATCH_LEARNING_RATE=${PATCH_LEARNING_RATE:-1e-4}
TRAIN_MODE=${TRAIN_MODE:-patch_lora}
LORA_SCOPE=${LORA_SCOPE:-wide}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-8}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
IMAGE_CONDITION_DROPOUT=${IMAGE_CONDITION_DROPOUT:-0.0}
MIXED_PRECISION=${MIXED_PRECISION:-no}
MAX_PIXELS=${MAX_PIXELS:-1048576}
SIZE_DIVISOR=${SIZE_DIVISOR:-32}
DOWNSCALE_IF_EXCEEDS_MAX_PIXELS=${DOWNSCALE_IF_EXCEEDS_MAX_PIXELS:-1}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-0}
DEBUG_CHECK_FINITE=${DEBUG_CHECK_FINITE:-0}
NUM_WORKERS=${NUM_WORKERS:-4}

IMAGE_SRC=${IMAGE_SRC:-${SRC_IMAGE:-}}
IMAGE_REF=${IMAGE_REF:-${REF_IMAGE:-}}
OUTPUT=${OUTPUT:-${OUTPUT_PATH:-${OUTPUT_DIR}/route2_single.png}}
PROMPT=${PROMPT:-}
DTYPE=${DTYPE:-auto}
STEPS=${STEPS:-20}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-1.0}
SEED=${SEED:-0}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}
SKIP_EXISTING=${SKIP_EXISTING:-0}
RESTORE_TO_ORIGINAL_SIZE=${RESTORE_TO_ORIGINAL_SIZE:-1}
DEBUG_LATENT_DIR=${DEBUG_LATENT_DIR:-}
SAVE_CONDITION_SENSITIVITY_DEBUG=${SAVE_CONDITION_SENSITIVITY_DEBUG:-0}

pretrained_args=()
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then pretrained_args+=(--local_files_only); fi
if [[ -n "${CACHE_DIR}" ]]; then pretrained_args+=(--cache_dir "${CACHE_DIR}"); fi
if [[ -n "${REVISION}" ]]; then pretrained_args+=(--revision "${REVISION}"); fi

size_args=(--max_pixels "${MAX_PIXELS}" --size_divisor "${SIZE_DIVISOR}")
if [[ "${DOWNSCALE_IF_EXCEEDS_MAX_PIXELS}" == "1" ]]; then size_args+=(--downscale_if_exceeds_max_pixels); fi

sample_args=()
if [[ -n "${MAX_SAMPLES}" ]]; then sample_args+=(--max_samples "${MAX_SAMPLES}"); fi

train_extra_args=()
if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then train_extra_args+=(--gradient_checkpointing); fi
if [[ "${DEBUG_CHECK_FINITE}" == "1" ]]; then train_extra_args+=(--debug_check_finite); fi

infer_extra_args=()
if [[ "${RESTORE_TO_ORIGINAL_SIZE}" == "1" ]]; then infer_extra_args+=(--restore_to_original_size); else infer_extra_args+=(--no-restore_to_original_size); fi
if [[ -n "${DEBUG_LATENT_DIR}" ]]; then infer_extra_args+=(--debug_latent_dir "${DEBUG_LATENT_DIR}"); fi
if [[ "${SAVE_CONDITION_SENSITIVITY_DEBUG}" == "1" ]]; then infer_extra_args+=(--save_condition_sensitivity_debug); fi

batch_extra_args=()
if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then batch_extra_args+=(--continue_on_error); fi
if [[ "${SKIP_EXISTING}" == "1" ]]; then batch_extra_args+=(--skip_existing); fi

prompt_args=()
if [[ -n "${PROMPT}" ]]; then prompt_args+=(--prompt "${PROMPT}"); fi

echo "[ROUTE2] MODE=${MODE}"
echo "[ROUTE2] MODEL=${MODEL}"
echo "[ROUTE2] TRAIN_MODE=${TRAIN_MODE}"
echo "[ROUTE2] DATASET_METADATA_PATH=${DATASET_METADATA_PATH}"
echo "[ROUTE2] DATASET_BASE_PATH=${DATASET_BASE_PATH}"
echo "[ROUTE2] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[ROUTE2] MIXED_PRECISION=${MIXED_PRECISION}"
echo "[ROUTE2] MAX_SAMPLES=${MAX_SAMPLES}"
echo "[ROUTE2] MAX_PIXELS=${MAX_PIXELS}"

run_train() {
  mkdir -p "${OUTPUT_DIR}"
  launch_args=(--mixed_precision "${MIXED_PRECISION}" --main_process_port "${MAIN_PROCESS_PORT}" --num_processes "${NUM_PROCESSES}")
  if [[ "${NUM_PROCESSES}" != "1" ]]; then launch_args+=(--multi_gpu); fi
  accelerate launch "${launch_args[@]}" \
    "${SCRIPT_DIR}/train_sana_artifact_repair_channel_concat_lora.py" \
    --model "${MODEL}" \
    --dataset_metadata_path "${DATASET_METADATA_PATH}" \
    --dataset_base_path "${DATASET_BASE_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --start_index "${START_INDEX}" \
    --dataset_repeat "${DATASET_REPEAT}" \
    --max_train_steps "${MAX_TRAIN_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --log_steps "${LOG_STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --patch_learning_rate "${PATCH_LEARNING_RATE}" \
    --train_mode "${TRAIN_MODE}" \
    --lora_scope "${LORA_SCOPE}" \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --image_condition_dropout "${IMAGE_CONDITION_DROPOUT}" \
    --mixed_precision "${MIXED_PRECISION}" \
    --num_workers "${NUM_WORKERS}" \
    "${size_args[@]}" \
    "${sample_args[@]}" \
    "${train_extra_args[@]}" \
    "${pretrained_args[@]}"
}

run_infer() {
  if [[ -z "${IMAGE_SRC}" || -z "${IMAGE_REF}" ]]; then
    echo "IMAGE_SRC and IMAGE_REF are required for MODE=infer." >&2
    exit 1
  fi
  python "${SCRIPT_DIR}/infer_sana_artifact_repair_channel_concat_lora.py" \
    --checkpoint "${CHECKPOINT}" \
    --model "${MODEL}" \
    --image_src "${IMAGE_SRC}" \
    --image_ref "${IMAGE_REF}" \
    --output "${OUTPUT}" \
    --steps "${STEPS}" \
    --guidance_scale "${GUIDANCE_SCALE}" \
    --seed "${SEED}" \
    --dtype "${DTYPE}" \
    "${size_args[@]}" \
    "${prompt_args[@]}" \
    "${infer_extra_args[@]}" \
    "${pretrained_args[@]}"
}

run_batch() {
  python "${SCRIPT_DIR}/batch_infer_sana_artifact_repair_channel_concat_lora.py" \
    --checkpoint "${CHECKPOINT}" \
    --model "${MODEL}" \
    --dataset_metadata_path "${TEST_METADATA_PATH}" \
    --dataset_base_path "${DATASET_BASE_PATH}" \
    --output_dir "${OUTPUT_DIR}/infer_route2_noise_steps${STEPS}_cfg${GUIDANCE_SCALE}" \
    --start_index "${START_INDEX}" \
    --steps "${STEPS}" \
    --guidance_scale "${GUIDANCE_SCALE}" \
    --seed "${SEED}" \
    --dtype "${DTYPE}" \
    "${size_args[@]}" \
    "${sample_args[@]}" \
    "${batch_extra_args[@]}" \
    "${prompt_args[@]}" \
    "${infer_extra_args[@]}" \
    "${pretrained_args[@]}"
}

case "${MODE}" in
  train) run_train ;;
  infer) run_infer ;;
  batch_infer) run_batch ;;
  *) echo "Unsupported MODE=${MODE}; expected train, infer, or batch_infer." >&2; exit 2 ;;
esac
