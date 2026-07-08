#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

MODE=${MODE:-train} # train | infer | batch_infer
DATA_ROOT=${DATA_ROOT:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880}
MODEL=${MODEL:-Efficient-Large-Model/Sana_600M_1024px_diffusers}
TRAIN_META=${TRAIN_META:-${DATA_ROOT}/dataset/0626_focus_merged_q25_shot_train_val_test/train/metadata_with_focus_masks.json}
TRAIN_BASE=${TRAIN_BASE:-${DATA_ROOT}/dataset/0626_focus_merged_q25_shot_train_val_test/train}
OUTPUT_DIR=${OUTPUT_DIR:-${DATA_ROOT}/focus/models/train/sana_controlnet_ab_fusion}

export HF_HOME=${HF_HOME:-${DATA_ROOT}/HF}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}/hub}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-2000}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
SAVE_STEPS=${SAVE_STEPS:-1000}
NUM_WORKERS=${NUM_WORKERS:-0}
LOSS_MODE=${LOSS_MODE:-legacy_a_noise_gt_velocity}
CONTROL_CONDITION_CHANNELS=${CONTROL_CONDITION_CHANNELS:-8}
USE_FOCUS_CONDITIONS=${USE_FOCUS_CONDITIONS:-1}
MAX_PIXELS=${MAX_PIXELS:-1048576}
SIZE_DIVISOR=${SIZE_DIVISOR:-32}
DOWNSCALE_IF_EXCEEDS_MAX_PIXELS=${DOWNSCALE_IF_EXCEEDS_MAX_PIXELS:-1}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-0}
CONDITIONING_SCALE=${CONDITIONING_SCALE:-1.0}
TRAIN_TRANSFORMER_LORA=${TRAIN_TRANSFORMER_LORA:-0}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-8}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-to_q,to_k,to_v}
STRENGTH=${STRENGTH:-0.2}
IMG2IMG_SCHEDULE_MODE=${IMG2IMG_SCHEDULE_MODE:-sliced}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-1}
MIXED_PRECISION=${MIXED_PRECISION:-no}
LOG_STEPS=${LOG_STEPS:-10}
IMAGE_A=${IMAGE_A:-${TRAIN_BASE}/a.png}
IMAGE_B=${IMAGE_B:-${TRAIN_BASE}/b.png}
FOCUS_A=${FOCUS_A:-}
FOCUS_B=${FOCUS_B:-}
DEBUG_LATENT_DIR=${DEBUG_LATENT_DIR:-}

pretrained_args=()
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  pretrained_args+=(--local_files_only)
fi
sample_args=()
if [[ -n "${TRAIN_MAX_SAMPLES}" ]]; then
  sample_args+=(--max_samples "${TRAIN_MAX_SAMPLES}")
fi
focus_args=()
if [[ "${USE_FOCUS_CONDITIONS}" == "1" ]]; then
  focus_args+=(--use_focus_conditions)
else
  focus_args+=(--no-use_focus_conditions)
fi
debug_args=()
if [[ -n "${DEBUG_LATENT_DIR}" ]]; then
  debug_args+=(--debug_latent_dir "${DEBUG_LATENT_DIR}")
fi
size_args=(--size_divisor "${SIZE_DIVISOR}")
if [[ "${DOWNSCALE_IF_EXCEEDS_MAX_PIXELS}" == "1" ]]; then
  size_args+=(--downscale_if_exceeds_max_pixels)
fi
train_extra_args=()
if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
  train_extra_args+=(--gradient_checkpointing)
fi
if [[ "${TRAIN_TRANSFORMER_LORA}" == "1" ]]; then
  train_extra_args+=(--train_transformer_lora)
  train_extra_args+=(--lora_rank "${LORA_RANK}")
  train_extra_args+=(--lora_alpha "${LORA_ALPHA}")
  train_extra_args+=(--lora_target_modules "${LORA_TARGET_MODULES}")
fi

echo "[MODE] ${MODE}"
echo "[LOSS_MODE] ${LOSS_MODE}"
echo "[CONTROL_CONDITION_CHANNELS] ${CONTROL_CONDITION_CHANNELS}"
echo "[USE_FOCUS_CONDITIONS] ${USE_FOCUS_CONDITIONS}"
echo "[MAX_PIXELS] ${MAX_PIXELS}"
echo "[TRAIN_MAX_SAMPLES] ${TRAIN_MAX_SAMPLES}"
echo "[MAX_TRAIN_STEPS] ${MAX_TRAIN_STEPS}"
echo "[OUTPUT_DIR] ${OUTPUT_DIR}"
echo "[MODEL] ${MODEL}"
echo "[DATASET_METADATA_PATH] ${TRAIN_META}"
echo "[IMG2IMG_SCHEDULE_MODE] ${IMG2IMG_SCHEDULE_MODE}"
echo "[DEBUG_LATENT_DIR] ${DEBUG_LATENT_DIR}"
echo "[LEARNING_RATE] ${LEARNING_RATE}"
echo "[TRAIN_BATCH_SIZE] ${TRAIN_BATCH_SIZE}"
echo "[GRADIENT_ACCUMULATION_STEPS] ${GRADIENT_ACCUMULATION_STEPS}"
echo "[SAVE_STEPS] ${SAVE_STEPS}"
echo "[NUM_WORKERS] ${NUM_WORKERS}"
echo "[SIZE_DIVISOR] ${SIZE_DIVISOR}"
echo "[DOWNSCALE_IF_EXCEEDS_MAX_PIXELS] ${DOWNSCALE_IF_EXCEEDS_MAX_PIXELS}"
echo "[GRADIENT_CHECKPOINTING] ${GRADIENT_CHECKPOINTING}"
echo "[CONDITIONING_SCALE] ${CONDITIONING_SCALE}"
echo "[TRAIN_TRANSFORMER_LORA] ${TRAIN_TRANSFORMER_LORA}"
echo "[LORA_RANK] ${LORA_RANK}"
echo "[LORA_ALPHA] ${LORA_ALPHA}"
echo "[LORA_TARGET_MODULES] ${LORA_TARGET_MODULES}"

run_train() {
  accelerate launch --num_processes 1 --mixed_precision "${MIXED_PRECISION}" \
    "${SCRIPT_DIR}/train_sana_controlnet_ab_fusion.py" \
    --model "${MODEL}" \
    --dataset_metadata_path "${TRAIN_META}" \
    --dataset_base_path "${TRAIN_BASE}" \
    --output_dir "${OUTPUT_DIR}" \
    --loss_mode "${LOSS_MODE}" \
    --control_condition_channels "${CONTROL_CONDITION_CHANNELS}" \
    --batch_size "${TRAIN_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --max_pixels "${MAX_PIXELS}" \
    --max_train_steps "${MAX_TRAIN_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --log_steps "${LOG_STEPS}" \
    --num_workers "${NUM_WORKERS}" \
    --conditioning_scale "${CONDITIONING_SCALE}" \
    --mixed_precision "${MIXED_PRECISION}" \
    "${focus_args[@]}" \
    "${size_args[@]}" \
    "${train_extra_args[@]}" \
    "${sample_args[@]}" \
    "${pretrained_args[@]}"
}

run_infer() {
  infer_focus_args=()
  if [[ -n "${FOCUS_A}" ]]; then infer_focus_args+=(--focus_a "${FOCUS_A}"); fi
  if [[ -n "${FOCUS_B}" ]]; then infer_focus_args+=(--focus_b "${FOCUS_B}"); fi
  python "${SCRIPT_DIR}/infer_sana_controlnet_ab_fusion.py" \
    --checkpoint "${OUTPUT_DIR}" \
    --model "${MODEL}" \
    --image_a "${IMAGE_A}" \
    --image_b "${IMAGE_B}" \
    --output "${OUTPUT_DIR}/verify_controlnet_tiny1_sliced_s${STRENGTH}_gs10.png" \
    --strength "${STRENGTH}" \
    --img2img_schedule_mode "${IMG2IMG_SCHEDULE_MODE}" \
    --max_pixels "${MAX_PIXELS}" \
    --conditioning_scale "${CONDITIONING_SCALE}" \
    "${infer_focus_args[@]}" \
    "${size_args[@]}" \
    "${debug_args[@]}" \
    "${pretrained_args[@]}"
}

run_batch() {
  python "${SCRIPT_DIR}/batch_infer_sana_controlnet_ab_fusion.py" \
    --checkpoint "${OUTPUT_DIR}" \
    --model "${MODEL}" \
    --dataset_metadata_path "${TRAIN_META}" \
    --dataset_base_path "${TRAIN_BASE}" \
    --output_dir "${OUTPUT_DIR}/verify_controlnet_tiny1_sliced_s${STRENGTH}_gs10" \
    --strength "${STRENGTH}" \
    --img2img_schedule_mode "${IMG2IMG_SCHEDULE_MODE}" \
    --max_pixels "${MAX_PIXELS}" \
    --conditioning_scale "${CONDITIONING_SCALE}" \
    "${size_args[@]}" \
    "${debug_args[@]}" \
    "${pretrained_args[@]}"
}

case "${MODE}" in
  train) run_train ;;
  infer) run_infer ;;
  batch_infer) run_batch ;;
  *) echo "MODE must be train, infer, or batch_infer" >&2; exit 2 ;;
esac
