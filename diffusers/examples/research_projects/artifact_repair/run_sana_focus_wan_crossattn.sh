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
CHECKPOINT=${CHECKPOINT:-${OUTPUT_DIR}}
MAX_SAMPLES=${MAX_SAMPLES:-16}
START_INDEX=${START_INDEX:-0}
DATASET_REPEAT=${DATASET_REPEAT:-1}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-10000}
SAVE_STEPS=${SAVE_STEPS:-1000}
LOG_STEPS=${LOG_STEPS:-10}
BATCH_SIZE=${BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
LEARNING_RATE=${LEARNING_RATE:-${LR:-1e-4}}
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
IMAGE_ENCODER_MODEL=${IMAGE_ENCODER_MODEL:-openai/clip-vit-large-patch14}
IMAGE_A=${IMAGE_A:-}
IMAGE_B=${IMAGE_B:-}
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
echo "[FOCUS_WAN_RUN] LEARNING_RATE=${LEARNING_RATE}"

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
    python examples/research_projects/artifact_repair/train_sana_focus_wan_crossattn.py \
      --model "${MODEL}" --condition_mode "${CONDITION_MODE}" \
      --dataset_metadata_path "${TRAIN_META}" --dataset_base_path "${DATASET_BASE_PATH}" \
      --output_dir "${OUTPUT_DIR}" --max_samples "${MAX_SAMPLES}" --start_index "${START_INDEX}" \
      --dataset_repeat "${DATASET_REPEAT}" --max_train_steps "${MAX_TRAIN_STEPS}" --save_steps "${SAVE_STEPS}" --log_steps "${LOG_STEPS}" \
      --batch_size "${BATCH_SIZE}" --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
      --learning_rate "${LEARNING_RATE}" --mixed_precision "${MIXED_PRECISION}" \
      --lora_scope "${LORA_SCOPE}" --lora_rank "${LORA_RANK}" --lora_alpha "${LORA_ALPHA}" --lora_dropout "${LORA_DROPOUT}" \
      --max_pixels "${MAX_PIXELS}" --size_divisor "${SIZE_DIVISOR}" --image_encoder_model "${IMAGE_ENCODER_MODEL}" \
      "${lora_args[@]}" "${debug_args[@]}" "${common_size[@]}" "${common_pretrained[@]}"
    ;;
  infer)
    python examples/research_projects/artifact_repair/infer_sana_focus_wan_crossattn.py \
      --model "${MODEL}" --condition_mode "${CONDITION_MODE}" --checkpoint "${CHECKPOINT}" \
      --image_a "${IMAGE_A}" --image_b "${IMAGE_B}" --output "${OUTPUT}" \
      --steps "${STEPS}" --guidance_scale "${GUIDANCE_SCALE}" --strength "${STRENGTH}" --init_mode "${INIT_MODE}" --img2img_schedule_mode "${IMG2IMG_SCHEDULE_MODE}" \
      --dtype "${DTYPE}" --max_pixels "${MAX_PIXELS}" --size_divisor "${SIZE_DIVISOR}" \
      "${common_size[@]}" "${common_pretrained[@]}"
    ;;
  batch_infer)
    cont_args=()
    if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then cont_args+=(--continue_on_error); fi
    python examples/research_projects/artifact_repair/batch_infer_sana_focus_wan_crossattn.py \
      --model "${MODEL}" --condition_mode "${CONDITION_MODE}" --checkpoint "${CHECKPOINT}" \
      --metadata_path "${TEST_META}" --dataset_base_path "${DATASET_BASE_PATH}" --output_dir "${OUTPUT_DIR}/batch_infer" \
      --start_index "${START_INDEX}" --max_samples "${MAX_SAMPLES}" \
      --steps "${STEPS}" --guidance_scale "${GUIDANCE_SCALE}" --strength "${STRENGTH}" --init_mode "${INIT_MODE}" --img2img_schedule_mode "${IMG2IMG_SCHEDULE_MODE}" \
      --dtype "${DTYPE}" --max_pixels "${MAX_PIXELS}" --size_divisor "${SIZE_DIVISOR}" --save_comparison \
      "${cont_args[@]}" "${common_size[@]}" "${common_pretrained[@]}"
    ;;
  *)
    echo "Unsupported MODE=${MODE}" >&2
    false
    ;;
esac
