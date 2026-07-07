#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/HF}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}/hub}
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

cleanup_grabgpu() {
  local grabgpu="${SCRIPT_DIR}/../GrabGPU/gg"
  if [[ -x "${grabgpu}" ]]; then
    "${grabgpu}" 80 150 "${CUDA_VISIBLE_DEVICES:-0}" 0.7 || true
  else
    echo "Warning: GrabGPU cleanup tool not found at ${grabgpu}" >&2
  fi
}
trap cleanup_grabgpu EXIT

MODE=${MODE:-all} # train | infer | batch | all
DATA_ROOT=${DATA_ROOT:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880}
DATASET_ROOT=${DATASET_ROOT:-${DATA_ROOT}/dataset/0626_focus_merged_q25_shot_train_val_test}
TRAIN_BASE=${TRAIN_BASE:-${DATASET_ROOT}/train}
TRAIN_METADATA=${TRAIN_METADATA:-${TRAIN_BASE}/metadata_with_focus_masks.json}
MODEL=${MODEL:-Efficient-Large-Model/Sana_600M_1024px_diffusers}
OUTPUT_DIR=${OUTPUT_DIR:-${DATA_ROOT}/focus/models/train/sana_ab_adapter/sana600m_ab_dynamic_$(date +%Y%m%d_%H%M%S)}
PROMPT=${PROMPT:-a photorealistic all-in-focus photograph}

NUM_PROCESSES=${NUM_PROCESSES:-1}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
NUM_WORKERS=${NUM_WORKERS:-0}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-3000}
SAVE_STEPS=${SAVE_STEPS:-500}
MIXED_PRECISION=${MIXED_PRECISION:-no}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
ADAPTER_HIDDEN_CHANNELS=${ADAPTER_HIDDEN_CHANNELS:-128}
MAX_PIXELS=${MAX_PIXELS:-1048576}
DOWNSCALE_IF_EXCEEDS_MAX_PIXELS=${DOWNSCALE_IF_EXCEEDS_MAX_PIXELS:-1}
SIZE_DIVISOR=${SIZE_DIVISOR:-32}
ASPECT_RATIO_TOLERANCE=${ASPECT_RATIO_TOLERANCE:-0.01}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-1}
TRAIN_START_INDEX=${TRAIN_START_INDEX:-0}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-}
INFER_STEPS=${INFER_STEPS:-20}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-4.5}
SEED=${SEED:-0}
RESTORE_TO_ORIGINAL_SIZE=${RESTORE_TO_ORIGINAL_SIZE:-1}
IMAGE_A=${IMAGE_A:-${TRAIN_BASE}/a.png}
IMAGE_B=${IMAGE_B:-${TRAIN_BASE}/b.png}

if [[ "${BATCH_SIZE}" != "1" ]]; then
  echo "Dynamic resolution requires BATCH_SIZE=1" >&2
  exit 2
fi

pretrained_args=()
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  pretrained_args+=(--local_files_only)
fi

train_size_args=(--size_divisor "${SIZE_DIVISOR}" --aspect_ratio_tolerance "${ASPECT_RATIO_TOLERANCE}")
infer_size_args=(--size_divisor "${SIZE_DIVISOR}" --aspect_ratio_tolerance "${ASPECT_RATIO_TOLERANCE}")
if [[ -n "${MAX_PIXELS}" ]]; then
  train_size_args+=(--max_pixels "${MAX_PIXELS}")
  infer_size_args+=(--max_pixels "${MAX_PIXELS}")
fi
if [[ "${DOWNSCALE_IF_EXCEEDS_MAX_PIXELS}" == "1" ]]; then
  train_size_args+=(--downscale_if_exceeds_max_pixels)
  infer_size_args+=(--downscale_if_exceeds_max_pixels)
fi
if [[ "${RESTORE_TO_ORIGINAL_SIZE}" == "1" ]]; then
  infer_size_args+=(--restore_to_original_size)
else
  infer_size_args+=(--no_restore_to_original_size)
fi

sample_args=(--start_index "${TRAIN_START_INDEX}")
if [[ -n "${TRAIN_MAX_SAMPLES}" ]]; then
  sample_args+=(--max_samples "${TRAIN_MAX_SAMPLES}")
fi

run_train() {
  accelerate launch \
    --num_processes "${NUM_PROCESSES}" \
    --mixed_precision "${MIXED_PRECISION}" \
    "${SCRIPT_DIR}/train_sana_ab_adapter.py" \
    --model "${MODEL}" \
    --dataset_metadata_path "${TRAIN_METADATA}" \
    --dataset_base_path "${TRAIN_BASE}" \
    --output_dir "${OUTPUT_DIR}" \
    --prompt "${PROMPT}" \
    --batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --max_train_steps "${MAX_TRAIN_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --num_workers "${NUM_WORKERS}" \
    --adapter_hidden_channels "${ADAPTER_HIDDEN_CHANNELS}" \
    --debug_check_finite \
    "${sample_args[@]}" \
    "${pretrained_args[@]}" \
    "${train_size_args[@]}"
}

run_infer() {
  python "${SCRIPT_DIR}/infer_sana_ab_adapter.py" \
    --checkpoint "${OUTPUT_DIR}" \
    --model "${MODEL}" \
    --image_a "${IMAGE_A}" \
    --image_b "${IMAGE_B}" \
    --output "${OUTPUT_DIR}/single_result.png" \
    --prompt "${PROMPT}" \
    --steps "${INFER_STEPS}" \
    --guidance_scale "${GUIDANCE_SCALE}" \
    --seed "${SEED}" \
    "${pretrained_args[@]}" \
    "${infer_size_args[@]}"
}

run_batch() {
  python "${SCRIPT_DIR}/batch_infer_sana_ab_adapter.py" \
    --checkpoint "${OUTPUT_DIR}" \
    --model "${MODEL}" \
    --dataset_metadata_path "${TRAIN_METADATA}" \
    --dataset_base_path "${TRAIN_BASE}" \
    --output_dir "${OUTPUT_DIR}/batch" \
    --prompt "${PROMPT}" \
    --batch_size "${BATCH_SIZE}" \
    --steps "${INFER_STEPS}" \
    --guidance_scale "${GUIDANCE_SCALE}" \
    --seed "${SEED}" \
    "${sample_args[@]}" \
    "${pretrained_args[@]}" \
    "${infer_size_args[@]}"
}

case "${MODE}" in
  train) run_train ;;
  infer) run_infer ;;
  batch) run_batch ;;
  all) run_train; run_infer; run_batch ;;
  *) echo "MODE must be train, infer, batch, or all" >&2; exit 2 ;;
esac
