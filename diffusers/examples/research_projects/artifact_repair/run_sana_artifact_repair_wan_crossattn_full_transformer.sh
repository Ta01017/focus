#!/usr/bin/env bash
set -euo pipefail

GPU=${GPU:-0}
NUM_GPUS=${NUM_GPUS:-1}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29502}
MODEL=${MODEL:-Efficient-Large-Model/Sana_600M_1024px_diffusers}
IMAGE_ENCODER_MODEL=${IMAGE_ENCODER_MODEL:-openai/clip-vit-large-patch14}
IMAGE_ENCODER_SUBFOLDER=${IMAGE_ENCODER_SUBFOLDER:-}
IMAGE_ENCODER_REVISION=${IMAGE_ENCODER_REVISION:-}
IMAGE_ENCODER_LOCAL_FILES_ONLY=${IMAGE_ENCODER_LOCAL_FILES_ONLY:-0}
DATASET_BASE_PATH=${DATASET_BASE_PATH:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora/data-prompt-aug}
DATASET_METADATA_PATH=${DATASET_METADATA_PATH:-${DATASET_BASE_PATH}/metadata_train.json}
OUTPUT_DIR=${OUTPUT_DIR:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/focus/models/train/sana_artifact_repair/route3_wan_crossattn_full_transformer}
TRAIN_MODE=full_transformer
MIXED_PRECISION=${MIXED_PRECISION:-no}
MAX_PIXELS=${MAX_PIXELS:-1048576}
SIZE_DIVISOR=${SIZE_DIVISOR:-32}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-20000}
SAVE_STEPS=${SAVE_STEPS:-1000}
LOG_STEPS=${LOG_STEPS:-10}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
PATCH_LEARNING_RATE=${PATCH_LEARNING_RATE:-1e-4}
IMAGE_ADAPTER_LEARNING_RATE=${IMAGE_ADAPTER_LEARNING_RATE:-1e-4}
IMAGE_GATE_INIT=${IMAGE_GATE_INIT:-1e-3}
IMAGE_CROSS_ATTENTION_SCALE=${IMAGE_CROSS_ATTENTION_SCALE:-1.0}
IMAGE_CONDITION_DROPOUT=${IMAGE_CONDITION_DROPOUT:-0.0}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-4}
DATASET_REPEAT=${DATASET_REPEAT:-1}
MAX_SAMPLES=${MAX_SAMPLES:-}
INIT_FROM_ROUTE2_CHECKPOINT=${INIT_FROM_ROUTE2_CHECKPOINT:-}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-0}
CACHE_DIR=${CACHE_DIR:-}
REVISION=${REVISION:-}
DOWNSCALE_IF_EXCEEDS_MAX_PIXELS=${DOWNSCALE_IF_EXCEEDS_MAX_PIXELS:-1}

echo "[ROUTE3 FULL] MODEL=${MODEL}"
echo "[ROUTE3 FULL] IMAGE_ENCODER_MODEL=${IMAGE_ENCODER_MODEL}"
echo "[ROUTE3 FULL] DATASET_BASE_PATH=${DATASET_BASE_PATH}"
echo "[ROUTE3 FULL] DATASET_METADATA_PATH=${DATASET_METADATA_PATH}"
echo "[ROUTE3 FULL] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[ROUTE3 FULL] TRAIN_MODE=${TRAIN_MODE}"
echo "[ROUTE3 FULL] MIXED_PRECISION=${MIXED_PRECISION}"
echo "[ROUTE3 FULL] INIT_FROM_ROUTE2_CHECKPOINT=${INIT_FROM_ROUTE2_CHECKPOINT}"
echo "[ROUTE3 FULL] RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT}"

mkdir -p "${OUTPUT_DIR}"

extra_args=()
if [[ "${DOWNSCALE_IF_EXCEEDS_MAX_PIXELS}" == "1" ]]; then extra_args+=(--downscale_if_exceeds_max_pixels); fi
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then extra_args+=(--local_files_only); fi
if [[ -n "${CACHE_DIR}" ]]; then extra_args+=(--cache_dir "${CACHE_DIR}"); fi
if [[ -n "${REVISION}" ]]; then extra_args+=(--revision "${REVISION}"); fi
if [[ -n "${MAX_SAMPLES}" ]]; then extra_args+=(--max_samples "${MAX_SAMPLES}"); fi
if [[ -n "${IMAGE_ENCODER_SUBFOLDER}" ]]; then extra_args+=(--image_encoder_subfolder "${IMAGE_ENCODER_SUBFOLDER}"); fi
if [[ -n "${IMAGE_ENCODER_REVISION}" ]]; then extra_args+=(--image_encoder_revision "${IMAGE_ENCODER_REVISION}"); fi
if [[ "${IMAGE_ENCODER_LOCAL_FILES_ONLY}" == "1" ]]; then extra_args+=(--image_encoder_local_files_only); fi
if [[ -n "${INIT_FROM_ROUTE2_CHECKPOINT}" ]]; then extra_args+=(--init_from_route2_checkpoint "${INIT_FROM_ROUTE2_CHECKPOINT}"); fi
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then extra_args+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}"); fi

CUDA_VISIBLE_DEVICES="${GPU}" python -m accelerate.commands.launch \
  --mixed_precision "${MIXED_PRECISION}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  --num_processes "${NUM_GPUS}" \
  examples/research_projects/artifact_repair/train_sana_artifact_repair_wan_crossattn.py \
  --model "${MODEL}" \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${DATASET_METADATA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --train_mode "${TRAIN_MODE}" \
  --mixed_precision "${MIXED_PRECISION}" \
  --max_pixels "${MAX_PIXELS}" \
  --size_divisor "${SIZE_DIVISOR}" \
  --max_train_steps "${MAX_TRAIN_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --log_steps "${LOG_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --patch_learning_rate "${PATCH_LEARNING_RATE}" \
  --image_adapter_learning_rate "${IMAGE_ADAPTER_LEARNING_RATE}" \
  --image_gate_init "${IMAGE_GATE_INIT}" \
  --image_cross_attention_scale "${IMAGE_CROSS_ATTENTION_SCALE}" \
  --image_condition_dropout "${IMAGE_CONDITION_DROPOUT}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --dataset_repeat "${DATASET_REPEAT}" \
  --image_encoder_model "${IMAGE_ENCODER_MODEL}" \
  --debug_check_finite \
  "${extra_args[@]}"
