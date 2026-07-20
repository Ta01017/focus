#!/usr/bin/env bash
set -euo pipefail

CONDITION_MODE=ab
MODE=${MODE:-train}
GPU=${GPU:-0}
MODEL=${MODEL:-Efficient-Large-Model/Sana_600M_1024px_diffusers}
TRAIN_BASE=${TRAIN_BASE:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/dataset/0626_focus_merged_q25_shot_train_val_test/train}
TRAIN_META=${TRAIN_META:-${TRAIN_BASE}/metadata_with_focus_masks.json}
VAL_BASE=${VAL_BASE:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/dataset/0626_focus_merged_q25_shot_train_val_test/val}
VAL_META=${VAL_META:-${VAL_BASE}/metadata_with_focus_masks.json}
OUTPUT_DIR=${OUTPUT_DIR:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/focus/models/train/sana_focus_latent_injection/ab}
CHECKPOINT=${CHECKPOINT:-${OUTPUT_DIR}}
INFER_OUTPUT_DIR=${INFER_OUTPUT_DIR:-${OUTPUT_DIR}/infer}
IMAGE_A=${IMAGE_A:-}
IMAGE_B=${IMAGE_B:-}
OUTPUT=${OUTPUT:-${INFER_OUTPUT_DIR}/single.png}
MIXED_PRECISION=${MIXED_PRECISION:-no}
DTYPE=${DTYPE:-fp32}
MAX_PIXELS=${MAX_PIXELS:-1048576}
SIZE_DIVISOR=${SIZE_DIVISOR:-32}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-20000}
SAVE_STEPS=${SAVE_STEPS:-1000}
LOG_STEPS=${LOG_STEPS:-10}
BATCH_SIZE=${BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_SCOPE=${LORA_SCOPE:-wide}
INJECTOR_HIDDEN_CHANNELS=${INJECTOR_HIDDEN_CHANNELS:-128}
CONDITION_SCALE=${CONDITION_SCALE:-1.0}
MAX_SAMPLES=${MAX_SAMPLES:-16}
START_INDEX=${START_INDEX:-0}
STEPS=${STEPS:-20}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-1.0}
STRENGTH=${STRENGTH:-0.30}
SEED=${SEED:-0}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-0}
CACHE_DIR=${CACHE_DIR:-}
DOWNSCALE_IF_EXCEEDS_MAX_PIXELS=${DOWNSCALE_IF_EXCEEDS_MAX_PIXELS:-1}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-1}
TRAIN_TRANSFORMER_LORA=${TRAIN_TRANSFORMER_LORA:-1}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}

echo "[FOCUS_ROUTE1_AB] MODE=${MODE}"
echo "[FOCUS_ROUTE1_AB] CONDITION_MODE=${CONDITION_MODE}"
echo "[FOCUS_ROUTE1_AB] TRAIN_BASE=${TRAIN_BASE}"
echo "[FOCUS_ROUTE1_AB] TRAIN_META=${TRAIN_META}"
echo "[FOCUS_ROUTE1_AB] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[FOCUS_ROUTE1_AB] CHECKPOINT=${CHECKPOINT}"

common_pretrained=()
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then common_pretrained+=(--local_files_only); fi
if [[ -n "${CACHE_DIR}" ]]; then common_pretrained+=(--cache_dir "${CACHE_DIR}"); fi
common_size=()
if [[ "${DOWNSCALE_IF_EXCEEDS_MAX_PIXELS}" == "1" ]]; then common_size+=(--downscale_if_exceeds_max_pixels); fi

if [[ "${MODE}" == "train" ]]; then
  mkdir -p "${OUTPUT_DIR}"
  train_extra=()
  if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then train_extra+=(--gradient_checkpointing); fi
  if [[ "${TRAIN_TRANSFORMER_LORA}" == "1" ]]; then train_extra+=(--train_transformer_lora); fi
  if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then train_extra+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}"); fi
  CUDA_VISIBLE_DEVICES="${GPU}" python examples/research_projects/dof_fusion/train_sana_focus_latent_injection_lora.py \
    --model "${MODEL}" \
    --condition_mode "${CONDITION_MODE}" \
    --dataset_base_path "${TRAIN_BASE}" \
    --dataset_metadata_path "${TRAIN_META}" \
    --output_dir "${OUTPUT_DIR}" \
    --mixed_precision "${MIXED_PRECISION}" \
    --max_pixels "${MAX_PIXELS}" \
    --size_divisor "${SIZE_DIVISOR}" \
    --max_train_steps "${MAX_TRAIN_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --log_steps "${LOG_STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_scope "${LORA_SCOPE}" \
    --injector_hidden_channels "${INJECTOR_HIDDEN_CHANNELS}" \
    --condition_scale "${CONDITION_SCALE}" \
    --start_index "${START_INDEX}" \
    --max_samples "${MAX_SAMPLES}" \
    --debug_check_finite \
    "${train_extra[@]}" \
    "${common_size[@]}" \
    "${common_pretrained[@]}"
elif [[ "${MODE}" == "infer" || "${MODE}" == "sanity" ]]; then
  mkdir -p "${INFER_OUTPUT_DIR}"
  infer_extra=()
  if [[ "${MODE}" == "sanity" ]]; then
    infer_extra+=(--allow_untrained_injector)
  else
    infer_extra+=(--checkpoint "${CHECKPOINT}")
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" python examples/research_projects/dof_fusion/infer_sana_focus_latent_injection_lora.py \
    --model "${MODEL}" \
    --condition_mode "${CONDITION_MODE}" \
    --image_a "${IMAGE_A}" \
    --image_b "${IMAGE_B}" \
    --output "${OUTPUT}" \
    --dtype "${DTYPE}" \
    --max_pixels "${MAX_PIXELS}" \
    --size_divisor "${SIZE_DIVISOR}" \
    --steps "${STEPS}" \
    --guidance_scale "${GUIDANCE_SCALE}" \
    --strength "${STRENGTH}" \
    --condition_scale "${CONDITION_SCALE}" \
    --seed "${SEED}" \
    "${infer_extra[@]}" \
    "${common_size[@]}" \
    "${common_pretrained[@]}"
elif [[ "${MODE}" == "batch_infer" ]]; then
  mkdir -p "${INFER_OUTPUT_DIR}"
  CUDA_VISIBLE_DEVICES="${GPU}" python examples/research_projects/dof_fusion/batch_infer_sana_focus_latent_injection_lora.py \
    --model "${MODEL}" \
    --condition_mode "${CONDITION_MODE}" \
    --checkpoint "${CHECKPOINT}" \
    --dataset_base_path "${TRAIN_BASE}" \
    --dataset_metadata_path "${TRAIN_META}" \
    --output_dir "${INFER_OUTPUT_DIR}" \
    --dtype "${DTYPE}" \
    --max_pixels "${MAX_PIXELS}" \
    --size_divisor "${SIZE_DIVISOR}" \
    --steps "${STEPS}" \
    --guidance_scale "${GUIDANCE_SCALE}" \
    --strength "${STRENGTH}" \
    --condition_scale "${CONDITION_SCALE}" \
    --seed "${SEED}" \
    --start_index "${START_INDEX}" \
    --max_samples "${MAX_SAMPLES}" \
    --save_comparison \
    "${common_size[@]}" \
    "${common_pretrained[@]}"
else
  echo "Unsupported MODE=${MODE}" >&2
  false
fi
