#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE=${MODE:-train}
ART_BASE=${ART_BASE:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/dataset/obj-curated-v2-lora/data-prompt-aug}
MODEL=${MODEL:-Efficient-Large-Model/Sana_600M_1024px_diffusers}
DATASET_BASE_PATH=${DATASET_BASE_PATH:-${ART_BASE}}
DATASET_METADATA_PATH=${DATASET_METADATA_PATH:-${ART_BASE}/metadata_train.json}
TEST_METADATA_PATH=${TEST_METADATA_PATH:-${ART_BASE}/metadata_test.json}
OUTPUT_DIR=${OUTPUT_DIR:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11188633/focus/models/train/sana_artifact_repair/route1_latent_concat_lora}
CHECKPOINT=${CHECKPOINT:-${OUTPUT_DIR}}

NUM_PROCESSES=${NUM_PROCESSES:-1}
MIXED_PRECISION=${MIXED_PRECISION:-no}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29501}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-1}
CACHE_DIR=${CACHE_DIR:-}
REVISION=${REVISION:-}

TARGET_KEY=${TARGET_KEY:-image}
PROMPT_KEY=${PROMPT_KEY:-prompt}
ID_KEY=${ID_KEY:-id}
SEED_KEY=${SEED_KEY:-seed}
RESULT_KEY=${RESULT_KEY:-generated_image}
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
DEBUG_CHECK_FINITE=${DEBUG_CHECK_FINITE:-0}

TRAIN_TRANSFORMER_LORA=${TRAIN_TRANSFORMER_LORA:-1}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-8}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
LORA_SCOPE=${LORA_SCOPE:-wide}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out,linear_in,linear_out,proj,linear,fc1,fc2}
INJECTOR_HIDDEN_CHANNELS=${INJECTOR_HIDDEN_CHANNELS:-128}

IMAGE_SRC=${IMAGE_SRC:-${SRC_IMAGE:-}}
IMAGE_REF=${IMAGE_REF:-${REF_IMAGE:-}}
PROMPT=${PROMPT:-}
OUTPUT=${OUTPUT:-${OUTPUT_PATH:-${OUTPUT_DIR}/route1_single.png}}
DTYPE=${DTYPE:-auto}
STEPS=${STEPS:-20}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-1.0}
STRENGTH=${STRENGTH:-0.15}
IMG2IMG_SCHEDULE_MODE=${IMG2IMG_SCHEDULE_MODE:-sliced}
USE_SRC_LATENT_INIT=${USE_SRC_LATENT_INIT:-1}
SEED=${SEED:-0}
SKIP_EXISTING=${SKIP_EXISTING:-0}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}
RESTORE_TO_ORIGINAL_SIZE=${RESTORE_TO_ORIGINAL_SIZE:-1}
DEBUG_LATENT_DIR=${DEBUG_LATENT_DIR:-}

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

train_extra_args=()
if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then train_extra_args+=(--gradient_checkpointing); fi
if [[ "${DEBUG_CHECK_FINITE}" == "1" ]]; then train_extra_args+=(--debug_check_finite); fi
if [[ "${TRAIN_TRANSFORMER_LORA}" == "1" ]]; then
  train_extra_args+=(--train_transformer_lora)
else
  train_extra_args+=(--no-train_transformer_lora)
fi

infer_extra_args=()
if [[ "${USE_SRC_LATENT_INIT}" == "1" ]]; then
  infer_extra_args+=(--use_src_latent_init)
else
  infer_extra_args+=(--no-use_src_latent_init)
fi
if [[ "${RESTORE_TO_ORIGINAL_SIZE}" == "1" ]]; then
  infer_extra_args+=(--restore_to_original_size)
else
  infer_extra_args+=(--no-restore_to_original_size)
fi
if [[ -n "${DEBUG_LATENT_DIR}" ]]; then infer_extra_args+=(--debug_latent_dir "${DEBUG_LATENT_DIR}"); fi

batch_extra_args=()
if [[ "${SKIP_EXISTING}" == "1" ]]; then batch_extra_args+=(--skip_existing); fi
if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then batch_extra_args+=(--continue_on_error); fi

prompt_args=()
if [[ -n "${PROMPT}" ]]; then prompt_args+=(--prompt "${PROMPT}"); fi

echo "[ROUTE1] MODE=${MODE}"
echo "[ROUTE1] MODEL=${MODEL}"
echo "[ROUTE1] DATASET_BASE_PATH=${DATASET_BASE_PATH}"
echo "[ROUTE1] DATASET_METADATA_PATH=${DATASET_METADATA_PATH}"
echo "[ROUTE1] TEST_METADATA_PATH=${TEST_METADATA_PATH}"
echo "[ROUTE1] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[ROUTE1] CHECKPOINT=${CHECKPOINT}"
echo "[ROUTE1] MIXED_PRECISION=${MIXED_PRECISION}"
echo "[ROUTE1] DTYPE=${DTYPE}"
echo "[ROUTE1] TRAIN_TRANSFORMER_LORA=${TRAIN_TRANSFORMER_LORA}"
echo "[ROUTE1] LORA_SCOPE=${LORA_SCOPE}"
echo "[ROUTE1] LORA_RANK=${LORA_RANK}"
echo "[ROUTE1] INJECTOR_HIDDEN_CHANNELS=${INJECTOR_HIDDEN_CHANNELS}"
echo "[ROUTE1] MAX_PIXELS=${MAX_PIXELS}"
echo "[ROUTE1] IMG2IMG_SCHEDULE_MODE=${IMG2IMG_SCHEDULE_MODE}"
echo "[ROUTE1] USE_SRC_LATENT_INIT=${USE_SRC_LATENT_INIT}"
echo "[ROUTE1] STRENGTH=${STRENGTH}"

run_train() {
  mkdir -p "${OUTPUT_DIR}"
  launch_args=(--mixed_precision "${MIXED_PRECISION}" --main_process_port "${MAIN_PROCESS_PORT}" --num_processes "${NUM_PROCESSES}")
  if [[ "${NUM_PROCESSES}" != "1" ]]; then
    launch_args+=(--multi_gpu)
  fi
  accelerate launch "${launch_args[@]}" \
    "${SCRIPT_DIR}/train_sana_artifact_repair_latent_concat_lora.py" \
    --model "${MODEL}" \
    --dataset_metadata_path "${DATASET_METADATA_PATH}" \
    --dataset_base_path "${DATASET_BASE_PATH}" \
    --target_key "${TARGET_KEY}" \
    --prompt_key "${PROMPT_KEY}" \
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
    --lora_dropout "${LORA_DROPOUT}" \
    --lora_scope "${LORA_SCOPE}" \
    --lora_target_modules "${LORA_TARGET_MODULES}" \
    --injector_hidden_channels "${INJECTOR_HIDDEN_CHANNELS}" \
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
  python "${SCRIPT_DIR}/infer_sana_artifact_repair_latent_concat_lora.py" \
    --checkpoint "${CHECKPOINT}" \
    --model "${MODEL}" \
    --image_src "${IMAGE_SRC}" \
    --image_ref "${IMAGE_REF}" \
    --output "${OUTPUT}" \
    --steps "${STEPS}" \
    --guidance_scale "${GUIDANCE_SCALE}" \
    --strength "${STRENGTH}" \
    --img2img_schedule_mode "${IMG2IMG_SCHEDULE_MODE}" \
    --seed "${SEED}" \
    --dtype "${DTYPE}" \
    "${size_args[@]}" \
    "${prompt_args[@]}" \
    "${infer_extra_args[@]}" \
    "${pretrained_args[@]}"
}

run_batch() {
  python "${SCRIPT_DIR}/batch_infer_sana_artifact_repair_latent_concat_lora.py" \
    --checkpoint "${CHECKPOINT}" \
    --model "${MODEL}" \
    --dataset_metadata_path "${TEST_METADATA_PATH}" \
    --dataset_base_path "${DATASET_BASE_PATH}" \
    --target_key "${TARGET_KEY}" \
    --prompt_key "${PROMPT_KEY}" \
    --id_key "${ID_KEY}" \
    --seed_key "${SEED_KEY}" \
    --result_key "${RESULT_KEY}" \
    --output_dir "${OUTPUT_DIR}/infer_route1_${IMG2IMG_SCHEDULE_MODE}_init${USE_SRC_LATENT_INIT}_s${STRENGTH}" \
    --start_index "${START_INDEX}" \
    --steps "${STEPS}" \
    --guidance_scale "${GUIDANCE_SCALE}" \
    --strength "${STRENGTH}" \
    --img2img_schedule_mode "${IMG2IMG_SCHEDULE_MODE}" \
    --seed "${SEED}" \
    --dtype "${DTYPE}" \
    "${size_args[@]}" \
    "${sample_args[@]}" \
    "${prompt_args[@]}" \
    "${infer_extra_args[@]}" \
    "${batch_extra_args[@]}" \
    "${pretrained_args[@]}"
}

case "${MODE}" in
  train) run_train ;;
  infer) run_infer ;;
  batch_infer) run_batch ;;
  *) echo "Unsupported MODE=${MODE}; expected train, infer, or batch_infer." >&2; exit 2 ;;
esac
