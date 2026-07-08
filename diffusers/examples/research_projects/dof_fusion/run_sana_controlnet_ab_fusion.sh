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
LOSS_MODE=${LOSS_MODE:-legacy_a_noise_gt_velocity}
CONTROL_CONDITION_CHANNELS=${CONTROL_CONDITION_CHANNELS:-8}
USE_FOCUS_CONDITIONS=${USE_FOCUS_CONDITIONS:-1}
MAX_PIXELS=${MAX_PIXELS:-1048576}
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

run_train() {
  accelerate launch --num_processes 1 --mixed_precision "${MIXED_PRECISION}" \
    "${SCRIPT_DIR}/train_sana_controlnet_ab_fusion.py" \
    --model "${MODEL}" \
    --dataset_metadata_path "${TRAIN_META}" \
    --dataset_base_path "${TRAIN_BASE}" \
    --output_dir "${OUTPUT_DIR}" \
    --loss_mode "${LOSS_MODE}" \
    --control_condition_channels "${CONTROL_CONDITION_CHANNELS}" \
    --max_pixels "${MAX_PIXELS}" \
    --max_train_steps "${MAX_TRAIN_STEPS}" \
    --log_steps "${LOG_STEPS}" \
    --mixed_precision "${MIXED_PRECISION}" \
    "${focus_args[@]}" \
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
    "${infer_focus_args[@]}" \
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
    "${debug_args[@]}" \
    "${pretrained_args[@]}"
}

case "${MODE}" in
  train) run_train ;;
  infer) run_infer ;;
  batch_infer) run_batch ;;
  *) echo "MODE must be train, infer, or batch_infer" >&2; exit 2 ;;
esac
