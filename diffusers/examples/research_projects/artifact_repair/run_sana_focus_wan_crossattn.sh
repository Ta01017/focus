#!/usr/bin/env bash
set -euo pipefail

MODE=${MODE:-train}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
MODEL=${MODEL:-Efficient-Large-Model/Sana_600M_1024px_diffusers}
CONDITION_MODE=${CONDITION_MODE:-single}
TRAIN_META=${TRAIN_META:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/dataset/0626_focus_merged_q25_shot_train_val_test/train/metadata_with_focus_masks.json}
TEST_META=${TEST_META:-${TRAIN_META}}
DATASET_BASE_PATH=${DATASET_BASE_PATH:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/dataset/0626_focus_merged_q25_shot_train_val_test/train}
OUTPUT_DIR=${OUTPUT_DIR:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/focus/models/train/sana_focus_wan_crossattn/${CONDITION_MODE}}
CHECKPOINT=${CHECKPOINT:-}
MAX_SAMPLES=${MAX_SAMPLES:-16}
START_INDEX=${START_INDEX:-0}
DATASET_REPEAT=${DATASET_REPEAT:-1}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-10000}
SAVE_STEPS=${SAVE_STEPS:-1000}
LOG_STEPS=${LOG_STEPS:-10}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-0}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
LEARNING_RATE=${LEARNING_RATE:-${LR:-1e-4}}
LR_SCHEDULER=${LR_SCHEDULER:-constant}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-0}
LR_NUM_CYCLES=${LR_NUM_CYCLES:-1}
LR_POWER=${LR_POWER:-1.0}
MIXED_PRECISION=${MIXED_PRECISION:-no}
DTYPE=${DTYPE:-fp32}
TRAIN_TRANSFORMER_LORA=${TRAIN_TRANSFORMER_LORA:-1}
LORA_SCOPE=${LORA_SCOPE:-wide}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
MAX_PIXELS=${MAX_PIXELS:-1048576}
SIZE_DIVISOR=${SIZE_DIVISOR:-32}
DOWNSCALE_IF_EXCEEDS_MAX_PIXELS=${DOWNSCALE_IF_EXCEEDS_MAX_PIXELS:-1}
STEPS=${STEPS:-20}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-1.0}
STRENGTH=${STRENGTH:-0.3}
INIT_MODE=${INIT_MODE:-a}
IMG2IMG_SCHEDULE_MODE=${IMG2IMG_SCHEDULE_MODE:-sliced}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-0}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-0}
DEBUG_CHECK_FINITE=${DEBUG_CHECK_FINITE:-1}
INIT_FROM_CHECKPOINT=${INIT_FROM_CHECKPOINT:-}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
IMAGE_ENCODER_MODEL=${IMAGE_ENCODER_MODEL:-openai/clip-vit-large-patch14}
IMAGE_A=${IMAGE_A:-}
IMAGE_B=${IMAGE_B:-}
FOCUS_A=${FOCUS_A:-}
FOCUS_B=${FOCUS_B:-}
OUTPUT=${OUTPUT:-${OUTPUT_DIR}/single.png}

export CUDA_VISIBLE_DEVICES

echo "[FOCUS_WAN_RUN] MODE=${MODE}"
echo "[FOCUS_WAN_RUN] CONDITION_MODE=${CONDITION_MODE}"
echo "[FOCUS_WAN_RUN] MODEL=${MODEL}"
echo "[FOCUS_WAN_RUN] TRAIN_META=${TRAIN_META}"
echo "[FOCUS_WAN_RUN] TEST_META=${TEST_META}"
echo "[FOCUS_WAN_RUN] DATASET_BASE_PATH=${DATASET_BASE_PATH}"
echo "[FOCUS_WAN_RUN] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[FOCUS_WAN_RUN] CHECKPOINT=${CHECKPOINT}"
echo "[FOCUS_WAN_RUN] NUM_WORKERS=${NUM_WORKERS}"
echo "[FOCUS_WAN_RUN] INIT_FROM_CHECKPOINT=${INIT_FROM_CHECKPOINT}"
echo "[FOCUS_WAN_RUN] RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT}"
echo "[FOCUS_WAN_RUN] LEARNING_RATE=${LEARNING_RATE}"
echo "[FOCUS_WAN_RUN] LR_SCHEDULER=${LR_SCHEDULER}"
echo "[FOCUS_WAN_RUN] LR_WARMUP_STEPS=${LR_WARMUP_STEPS}"
echo "[FOCUS_WAN_RUN] GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
echo "[FOCUS_WAN_RUN] MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS}"


find_latest_checkpoint() {
  local root="$1"
  find "$root" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1
}

resolve_checkpoint() {
  if [[ -n "${CHECKPOINT}" ]]; then
    echo "[CHECKPOINT] user provided" >&2
    echo "[CHECKPOINT] resolved path=${CHECKPOINT}" >&2
    printf '%s' "${CHECKPOINT}"
    return 0
  fi
  local latest
  latest=$(find_latest_checkpoint "${OUTPUT_DIR}")
  if [[ -z "${latest}" ]]; then
    echo "[CHECKPOINT] no CHECKPOINT provided and no checkpoint-* found under ${OUTPUT_DIR}" >&2
    return 1
  fi
  echo "[CHECKPOINT] auto detected" >&2
  echo "[CHECKPOINT] resolved path=${latest}" >&2
  printf '%s' "${latest}"
}

common_pretrained=()
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then common_pretrained+=(--local_files_only); fi
common_size=()
if [[ "${DOWNSCALE_IF_EXCEEDS_MAX_PIXELS}" == "1" ]]; then common_size+=(--downscale_if_exceeds_max_pixels); fi

case "${MODE}" in
  train)
    mkdir -p "${OUTPUT_DIR}"
    lora_args=()
    if [[ "${TRAIN_TRANSFORMER_LORA}" == "1" ]]; then lora_args+=(--train_transformer_lora); else lora_args+=(--no-train_transformer_lora); fi
    debug_args=()
    if [[ "${DEBUG_CHECK_FINITE}" == "1" ]]; then debug_args+=(--debug_check_finite); fi
    train_resume_args=()
    if [[ -n "${INIT_FROM_CHECKPOINT}" ]]; then train_resume_args+=(--init_from_checkpoint "${INIT_FROM_CHECKPOINT}"); fi
    if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then train_resume_args+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}"); fi
    sample_args=()
    if [[ -n "${MAX_SAMPLES}" ]]; then sample_args+=(--max_samples "${MAX_SAMPLES}"); fi
    python examples/research_projects/artifact_repair/train_sana_focus_wan_crossattn.py \
      --model "${MODEL}" --condition_mode "${CONDITION_MODE}" \
      --dataset_metadata_path "${TRAIN_META}" --dataset_base_path "${DATASET_BASE_PATH}" \
      --output_dir "${OUTPUT_DIR}" "${sample_args[@]}" --start_index "${START_INDEX}" \
      --dataset_repeat "${DATASET_REPEAT}" --max_train_steps "${MAX_TRAIN_STEPS}" --save_steps "${SAVE_STEPS}" --log_steps "${LOG_STEPS}" \
      --batch_size "${BATCH_SIZE}" --num_workers "${NUM_WORKERS}" --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
      --learning_rate "${LEARNING_RATE}" --lr_scheduler "${LR_SCHEDULER}" --lr_warmup_steps "${LR_WARMUP_STEPS}" \
      --lr_num_cycles "${LR_NUM_CYCLES}" --lr_power "${LR_POWER}" --mixed_precision "${MIXED_PRECISION}" \
      --lora_scope "${LORA_SCOPE}" --lora_rank "${LORA_RANK}" --lora_alpha "${LORA_ALPHA}" --lora_dropout "${LORA_DROPOUT}" \
      --max_pixels "${MAX_PIXELS}" --size_divisor "${SIZE_DIVISOR}" --image_encoder_model "${IMAGE_ENCODER_MODEL}" \
      "${lora_args[@]}" "${debug_args[@]}" "${train_resume_args[@]}" "${common_size[@]}" "${common_pretrained[@]}"
    ;;
  infer)
    RESOLVED_CHECKPOINT=$(resolve_checkpoint)
    python examples/research_projects/artifact_repair/infer_sana_focus_wan_crossattn.py \
      --model "${MODEL}" --condition_mode "${CONDITION_MODE}" --checkpoint "${RESOLVED_CHECKPOINT}" \
      --image_a "${IMAGE_A}" --image_b "${IMAGE_B}" --focus_a "${FOCUS_A}" --focus_b "${FOCUS_B}" --output "${OUTPUT}" \
      --steps "${STEPS}" --guidance_scale "${GUIDANCE_SCALE}" --strength "${STRENGTH}" --init_mode "${INIT_MODE}" --img2img_schedule_mode "${IMG2IMG_SCHEDULE_MODE}" \
      --dtype "${DTYPE}" --max_pixels "${MAX_PIXELS}" --size_divisor "${SIZE_DIVISOR}" \
      "${common_size[@]}" "${common_pretrained[@]}"
    ;;
  batch_infer)
    RESOLVED_CHECKPOINT=$(resolve_checkpoint)
    cont_args=()
    if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then cont_args+=(--continue_on_error); fi
    sample_args=()
    if [[ -n "${MAX_SAMPLES}" ]]; then sample_args+=(--max_samples "${MAX_SAMPLES}"); fi
    python examples/research_projects/artifact_repair/batch_infer_sana_focus_wan_crossattn.py \
      --model "${MODEL}" --condition_mode "${CONDITION_MODE}" --checkpoint "${RESOLVED_CHECKPOINT}" \
      --metadata_path "${TEST_META}" --dataset_base_path "${DATASET_BASE_PATH}" --output_dir "${OUTPUT_DIR}/batch_infer" \
      --start_index "${START_INDEX}" "${sample_args[@]}" \
      --steps "${STEPS}" --guidance_scale "${GUIDANCE_SCALE}" --strength "${STRENGTH}" --init_mode "${INIT_MODE}" --img2img_schedule_mode "${IMG2IMG_SCHEDULE_MODE}" \
      --dtype "${DTYPE}" --max_pixels "${MAX_PIXELS}" --size_divisor "${SIZE_DIVISOR}" --save_comparison \
      "${cont_args[@]}" "${common_size[@]}" "${common_pretrained[@]}"
    ;;
  *)
    echo "Unsupported MODE=${MODE}" >&2
    false
    ;;
esac
