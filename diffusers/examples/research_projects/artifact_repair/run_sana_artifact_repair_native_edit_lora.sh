#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE=${MODE:-train}
ART_BASE=${ART_BASE:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora/data-prompt-aug}
MODEL=${MODEL:-Efficient-Large-Model/Sana_600M_1024px_diffusers}
DATASET_BASE_PATH=${DATASET_BASE_PATH:-${ART_BASE}}
DATASET_METADATA_PATH=${DATASET_METADATA_PATH:-${ART_BASE}/metadata_train.json}
TEST_METADATA_PATH=${TEST_METADATA_PATH:-${ART_BASE}/metadata_test.json}
OUTPUT_DIR=${OUTPUT_DIR:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/focus/models/train/sana_artifact_repair/native_edit_lora}
CHECKPOINT=${CHECKPOINT:-${OUTPUT_DIR}}

NUM_PROCESSES=${NUM_PROCESSES:-1}
MIXED_PRECISION=${MIXED_PRECISION:-no}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29501}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-1}
CACHE_DIR=${CACHE_DIR:-}
REVISION=${REVISION:-}

START_INDEX=${START_INDEX:-0}
MAX_SAMPLES=${MAX_SAMPLES:-}
DATASET_REPEAT=${DATASET_REPEAT:-1}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-2000}
SAVE_STEPS=${SAVE_STEPS:-500}
LOG_STEPS=${LOG_STEPS:-10}
BATCH_SIZE=${BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
NUM_WORKERS=${NUM_WORKERS:-4}
MAX_PIXELS=${MAX_PIXELS:-1048576}
SIZE_DIVISOR=${SIZE_DIVISOR:-32}
DOWNSCALE_IF_EXCEEDS_MAX_PIXELS=${DOWNSCALE_IF_EXCEEDS_MAX_PIXELS:-1}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-0}

LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-8}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-to_q,to_k,to_v,to_out.0,proj,linear,fc1,fc2}
LORA_SCOPE=${LORA_SCOPE:-wide}
NATIVE_EDIT_IMPL=${NATIVE_EDIT_IMPL:-cross_attention_v2}
EDIT_CONDITION_SCALE=${EDIT_CONDITION_SCALE:-1.0}
EDIT_ROLE_EMBEDDING=${EDIT_ROLE_EMBEDDING:-1}
USE_EDIT_TOKEN_NORM=${USE_EDIT_TOKEN_NORM:-1}
DEBUG_NAN=${DEBUG_NAN:-0}

SRC_IMAGE=${SRC_IMAGE:-}
REF_IMAGE=${REF_IMAGE:-}
PROMPT=${PROMPT:-}
OUTPUT_PATH=${OUTPUT_PATH:-${OUTPUT_DIR}/native_edit_single.png}
DTYPE=${DTYPE:-auto}
STEPS=${STEPS:-20}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-1.0}
SEED=${SEED:-0}
INIT_MODE=${INIT_MODE:-src}
STRENGTH=${STRENGTH:-0.15}
ZERO_SRC=${ZERO_SRC:-0}
ZERO_REF=${ZERO_REF:-0}
SWAP_SRC_REF=${SWAP_SRC_REF:-0}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}

pretrained_args=()
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then pretrained_args+=(--local_files_only); fi
if [[ -n "${CACHE_DIR}" ]]; then pretrained_args+=(--cache_dir "${CACHE_DIR}"); fi
if [[ -n "${REVISION}" ]]; then pretrained_args+=(--revision "${REVISION}"); fi

size_args=(--max_pixels "${MAX_PIXELS}" --size_divisor "${SIZE_DIVISOR}")
if [[ "${DOWNSCALE_IF_EXCEEDS_MAX_PIXELS}" == "1" ]]; then
  size_args+=(--downscale_if_exceeds_max_pixels)
fi

sample_args=()
if [[ -n "${MAX_SAMPLES}" ]]; then sample_args+=(--max_samples "${MAX_SAMPLES}"); fi

role_args=()
if [[ "${EDIT_ROLE_EMBEDDING}" == "1" ]]; then
  role_args+=(--edit_role_embedding)
else
  role_args+=(--no-edit_role_embedding)
fi

train_extra_args=()
if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then train_extra_args+=(--gradient_checkpointing); fi
if [[ "${DEBUG_NAN}" == "1" ]]; then train_extra_args+=(--debug_nan); fi

token_norm_args=()
if [[ "${USE_EDIT_TOKEN_NORM}" == "1" ]]; then
  token_norm_args+=(--use_edit_token_norm)
else
  token_norm_args+=(--no_edit_token_norm)
fi

ablation_args=()
if [[ "${ZERO_SRC}" == "1" ]]; then ablation_args+=(--zero_src); fi
if [[ "${ZERO_REF}" == "1" ]]; then ablation_args+=(--zero_ref); fi
if [[ "${SWAP_SRC_REF}" == "1" ]]; then ablation_args+=(--swap_src_ref); fi
if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then ablation_args+=(--continue_on_error); fi

echo "[NATIVE_EDIT] MODE=${MODE}"
echo "[NATIVE_EDIT] MODEL=${MODEL}"
echo "[NATIVE_EDIT] DATASET_BASE_PATH=${DATASET_BASE_PATH}"
echo "[NATIVE_EDIT] DATASET_METADATA_PATH=${DATASET_METADATA_PATH}"
echo "[NATIVE_EDIT] TEST_METADATA_PATH=${TEST_METADATA_PATH}"
echo "[NATIVE_EDIT] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[NATIVE_EDIT] CHECKPOINT=${CHECKPOINT}"
echo "[NATIVE_EDIT] MIXED_PRECISION=${MIXED_PRECISION}"
echo "[NATIVE_EDIT] DTYPE=${DTYPE}"
echo "[NATIVE_EDIT] LORA_RANK=${LORA_RANK}"
echo "[NATIVE_EDIT] LORA_SCOPE=${LORA_SCOPE}"
echo "[NATIVE_EDIT] NATIVE_EDIT_IMPL=${NATIVE_EDIT_IMPL}"
echo "[NATIVE_EDIT] EDIT_CONDITION_SCALE=${EDIT_CONDITION_SCALE}"
echo "[NATIVE_EDIT] USE_EDIT_TOKEN_NORM=${USE_EDIT_TOKEN_NORM}"
echo "[NATIVE_EDIT] DEBUG_NAN=${DEBUG_NAN}"
echo "[NATIVE_EDIT] MAX_PIXELS=${MAX_PIXELS}"
echo "[NATIVE_EDIT] INIT_MODE=${INIT_MODE}"
echo "[NATIVE_EDIT] STRENGTH=${STRENGTH}"

run_train() {
  mkdir -p "${OUTPUT_DIR}"
  launch_args=(--mixed_precision "${MIXED_PRECISION}" --main_process_port "${MAIN_PROCESS_PORT}" --num_processes "${NUM_PROCESSES}")
  if [[ "${NUM_PROCESSES}" != "1" ]]; then
    launch_args+=(--multi_gpu)
  fi
  accelerate launch "${launch_args[@]}" \
    "${SCRIPT_DIR}/train_sana_artifact_repair_native_edit_lora.py" \
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
    --mixed_precision "${MIXED_PRECISION}" \
    --num_workers "${NUM_WORKERS}" \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_target_modules "${LORA_TARGET_MODULES}" \
    --lora_scope "${LORA_SCOPE}" \
    --native_edit_impl "${NATIVE_EDIT_IMPL}" \
    --edit_condition_scale "${EDIT_CONDITION_SCALE}" \
    "${size_args[@]}" \
    "${sample_args[@]}" \
    "${role_args[@]}" \
    "${token_norm_args[@]}" \
    "${train_extra_args[@]}" \
    "${pretrained_args[@]}"
}

run_infer() {
  if [[ -z "${SRC_IMAGE}" || -z "${REF_IMAGE}" ]]; then
    echo "SRC_IMAGE and REF_IMAGE are required for MODE=infer." >&2
    exit 1
  fi
  prompt_args=()
  if [[ -n "${PROMPT}" ]]; then prompt_args+=(--prompt "${PROMPT}"); fi
  python "${SCRIPT_DIR}/infer_sana_artifact_repair_native_edit_lora.py" \
    --checkpoint "${CHECKPOINT}" \
    --model "${MODEL}" \
    --src_image "${SRC_IMAGE}" \
    --ref_image "${REF_IMAGE}" \
    --output_path "${OUTPUT_PATH}" \
    --steps "${STEPS}" \
    --guidance_scale "${GUIDANCE_SCALE}" \
    --seed "${SEED}" \
    --init_mode "${INIT_MODE}" \
    --strength "${STRENGTH}" \
    --dtype "${DTYPE}" \
    "${size_args[@]}" \
    "${prompt_args[@]}" \
    "${pretrained_args[@]}"
}

run_batch() {
  python "${SCRIPT_DIR}/batch_infer_sana_artifact_repair_native_edit_lora.py" \
    --checkpoint "${CHECKPOINT}" \
    --model "${MODEL}" \
    --dataset_metadata_path "${TEST_METADATA_PATH}" \
    --dataset_base_path "${DATASET_BASE_PATH}" \
    --output_dir "${OUTPUT_DIR}/infer_native_edit_${INIT_MODE}_s${STRENGTH}" \
    --start_index "${START_INDEX}" \
    --steps "${STEPS}" \
    --guidance_scale "${GUIDANCE_SCALE}" \
    --seed "${SEED}" \
    --init_mode "${INIT_MODE}" \
    --strength "${STRENGTH}" \
    --dtype "${DTYPE}" \
    "${size_args[@]}" \
    "${sample_args[@]}" \
    "${ablation_args[@]}" \
    "${pretrained_args[@]}"
}

case "${MODE}" in
  train) run_train ;;
  infer) run_infer ;;
  batch_infer) run_batch ;;
  *) echo "Unsupported MODE=${MODE}; expected train, infer, or batch_infer." >&2; exit 2 ;;
esac
