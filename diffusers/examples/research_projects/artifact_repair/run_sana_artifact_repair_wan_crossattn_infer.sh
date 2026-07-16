#!/usr/bin/env bash
set -euo pipefail

GPU=${GPU:-0}
CHECKPOINT=${CHECKPOINT:?Please set CHECKPOINT to a Route3 checkpoint directory.}
METADATA=${METADATA:-}
DATASET_BASE_PATH=${DATASET_BASE_PATH:-.}
SRC_IMAGE=${SRC_IMAGE:-}
REF_IMAGE=${REF_IMAGE:-}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/route3_wan_crossattn}
MODEL=${MODEL:-}
INIT_MODE=${INIT_MODE:-all}
SRC_INIT_STRENGTH=${SRC_INIT_STRENGTH:-0.30}
INFER_STEPS=${INFER_STEPS:-20}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-1.0}
MAX_PIXELS=${MAX_PIXELS:-1048576}
SIZE_DIVISOR=${SIZE_DIVISOR:-32}
DTYPE=${DTYPE:-auto}
MAX_SAMPLES=${MAX_SAMPLES:-1}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-0}
CACHE_DIR=${CACHE_DIR:-}
DOWNSCALE_IF_EXCEEDS_MAX_PIXELS=${DOWNSCALE_IF_EXCEEDS_MAX_PIXELS:-1}

echo "[ROUTE3 INFER] CHECKPOINT=${CHECKPOINT}"
echo "[ROUTE3 INFER] METADATA=${METADATA}"
echo "[ROUTE3 INFER] DATASET_BASE_PATH=${DATASET_BASE_PATH}"
echo "[ROUTE3 INFER] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[ROUTE3 INFER] GPU=${GPU}"
echo "[ROUTE3 INFER] INIT_MODE=${INIT_MODE}"
echo "[ROUTE3 INFER] SRC_INIT_STRENGTH=${SRC_INIT_STRENGTH}"
echo "[ROUTE3 INFER] INFER_STEPS=${INFER_STEPS}"
echo "[ROUTE3 INFER] GUIDANCE_SCALE=${GUIDANCE_SCALE}"
echo "[ROUTE3 INFER] MAX_PIXELS=${MAX_PIXELS}"
echo "[ROUTE3 INFER] DTYPE=${DTYPE}"

mkdir -p "${OUTPUT_DIR}"

common_args=(
  --checkpoint "${CHECKPOINT}"
  --steps "${INFER_STEPS}"
  --guidance_scale "${GUIDANCE_SCALE}"
  --max_pixels "${MAX_PIXELS}"
  --size_divisor "${SIZE_DIVISOR}"
  --dtype "${DTYPE}"
)
if [[ -n "${MODEL}" ]]; then common_args+=(--model "${MODEL}"); fi
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then common_args+=(--local_files_only); fi
if [[ -n "${CACHE_DIR}" ]]; then common_args+=(--cache_dir "${CACHE_DIR}"); fi
if [[ "${DOWNSCALE_IF_EXCEEDS_MAX_PIXELS}" == "1" ]]; then common_args+=(--downscale_if_exceeds_max_pixels); fi

run_batch() {
  local mode="$1"
  local strength="$2"
  CUDA_VISIBLE_DEVICES="${GPU}" python examples/research_projects/artifact_repair/batch_infer_sana_artifact_repair_wan_crossattn.py \
    "${common_args[@]}" \
    --dataset_metadata_path "${METADATA}" \
    --dataset_base_path "${DATASET_BASE_PATH}" \
    --output_dir "${OUTPUT_DIR}/${mode}_${strength}" \
    --max_samples "${MAX_SAMPLES}" \
    --init_mode "${mode}" \
    --src_init_strength "${strength}"
}

run_single() {
  local mode="$1"
  local strength="$2"
  CUDA_VISIBLE_DEVICES="${GPU}" python examples/research_projects/artifact_repair/infer_sana_artifact_repair_wan_crossattn.py \
    "${common_args[@]}" \
    --image_src "${SRC_IMAGE}" \
    --image_ref "${REF_IMAGE}" \
    --output "${OUTPUT_DIR}/${mode}_${strength}.png" \
    --init_mode "${mode}" \
    --src_init_strength "${strength}"
}

if [[ -n "${METADATA}" ]]; then
  if [[ "${INIT_MODE}" == "all" ]]; then
    run_batch pure_noise 0.0
    run_batch src_latent 0.15
    run_batch src_latent 0.30
    run_batch src_latent 0.50
    run_batch src_latent 0.70
  else
    run_batch "${INIT_MODE}" "${SRC_INIT_STRENGTH}"
  fi
else
  if [[ -z "${SRC_IMAGE}" || -z "${REF_IMAGE}" ]]; then
    echo "Please set either METADATA or both SRC_IMAGE and REF_IMAGE." >&2
    false
  fi
  if [[ "${INIT_MODE}" == "all" ]]; then
    run_single pure_noise 0.0
    run_single src_latent 0.15
    run_single src_latent 0.30
    run_single src_latent 0.50
    run_single src_latent 0.70
  else
    run_single "${INIT_MODE}" "${SRC_INIT_STRENGTH}"
  fi
fi
