#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODEL=${MODEL:-Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers}
DATASET_BASE_PATH=${DATASET_BASE_PATH:?请设置 DATASET_BASE_PATH}
METADATA=${METADATA:?请设置 METADATA}
IMAGE_A=${IMAGE_A:?请设置 IMAGE_A}
IMAGE_B=${IMAGE_B:?请设置 IMAGE_B}
FOCUS_A=${FOCUS_A:?请设置 FOCUS_A}
FOCUS_B=${FOCUS_B:?请设置 FOCUS_B}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/dof_fusion/smoke}
MAX_PIXELS=${MAX_PIXELS:-}
SIZE_DIVISOR=${SIZE_DIVISOR:-32}
CACHE_DIR=${CACHE_DIR:-}
REVISION=${REVISION:-}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-0}

pretrained_args=()
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  pretrained_args+=(--local_files_only)
fi
size_args=(--size_divisor "${SIZE_DIVISOR}")
if [[ -n "${MAX_PIXELS}" ]]; then size_args+=(--max_pixels "${MAX_PIXELS}"); fi
if [[ -n "${CACHE_DIR}" ]]; then
  pretrained_args+=(--cache_dir "${CACHE_DIR}")
fi
if [[ -n "${REVISION}" ]]; then
  pretrained_args+=(--revision "${REVISION}")
fi

python "${SCRIPT_DIR}/check_dof_metadata.py" \
  --dataset_metadata_path "${METADATA}" \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --require_focus_maps \
  --max_samples 1 \
  --preview_count 1 \
  --preview_output "${OUTPUT_ROOT}/metadata_preview.jpg" \
  --report_output "${OUTPUT_ROOT}/metadata_report.json"

train_adapter() {
  local name=$1
  local adapter_type=$2
  local focus_weight=$3
  local output_dir="${OUTPUT_ROOT}/${name}"
  local focus_args=()
  if [[ "${focus_weight}" != "0" || "${adapter_type}" == "ab_focus" ]]; then
    focus_args+=(--use_focus_maps --min_edit_images 4 --focus_loss_weight "${focus_weight}")
  fi
  accelerate launch "${SCRIPT_DIR}/train_sana_sprint.py" \
    --model "${MODEL}" \
    --dataset_metadata_path "${METADATA}" \
    --dataset_base_path "${DATASET_BASE_PATH}" \
    --output_dir "${output_dir}" \
    --adapter_type "${adapter_type}" \
    --batch_size 1 \
    --num_workers 0 \
    --max_samples 1 \
    --max_train_steps 2 \
    --save_steps 1 \
    "${focus_args[@]}" \
    "${size_args[@]}" \
    "${pretrained_args[@]}"

  local infer_focus_args=()
  if [[ "${adapter_type}" == "ab_focus" ]]; then
    infer_focus_args+=(--focus_a "${FOCUS_A}" --focus_b "${FOCUS_B}")
  fi
  python "${SCRIPT_DIR}/infer_sana_sprint.py" \
    --adapter "${output_dir}/adapter.safetensors" \
    --image_a "${IMAGE_A}" \
    --image_b "${IMAGE_B}" \
    --output "${output_dir}/single.png" \
    --steps 1 \
    "${infer_focus_args[@]}" \
    "${size_args[@]}" \
    "${pretrained_args[@]}"

  python "${SCRIPT_DIR}/batch_infer.py" \
    --backend sana \
    --adapter "${output_dir}/adapter.safetensors" \
    --dataset_metadata_path "${METADATA}" \
    --dataset_base_path "${DATASET_BASE_PATH}" \
    --output_dir "${output_dir}/batch" \
    --batch_size 1 \
    --max_samples 1 \
    --steps 1 \
    "${size_args[@]}" \
    "${pretrained_args[@]}"
}

train_adapter e1_ab_no_focus ab 0
train_adapter e2_ab_focus_loss ab 1.0
train_adapter e3_ab_focus_adapter ab_focus 1.0

E4_DIR="${OUTPUT_ROOT}/e4_img2img"
accelerate launch "${SCRIPT_DIR}/train_sana_sprint_img2img.py" \
  --model "${MODEL}" \
  --dataset_metadata_path "${METADATA}" \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --output_dir "${E4_DIR}" \
  --batch_size 1 \
  --num_workers 0 \
  --max_samples 1 \
  --max_train_steps 2 \
  --save_steps 1 \
  "${size_args[@]}" \
  "${pretrained_args[@]}"

python "${SCRIPT_DIR}/infer_sana_sprint_img2img.py" \
  --adapter "${E4_DIR}/adapter.safetensors" \
  --image_a "${IMAGE_A}" \
  --image_b "${IMAGE_B}" \
  --output "${E4_DIR}/single.png" \
  --steps 4 \
  --strength 0.75 \
  "${size_args[@]}" \
  "${pretrained_args[@]}"

python "${SCRIPT_DIR}/batch_infer_sana_sprint_img2img.py" \
  --adapter "${E4_DIR}/adapter.safetensors" \
  --dataset_metadata_path "${METADATA}" \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --output_dir "${E4_DIR}/batch" \
  --batch_size 1 \
  --max_samples 1 \
  --steps 4 \
  --strength 0.75 \
  "${size_args[@]}" \
  "${pretrained_args[@]}"

echo "E1-E4 smoke tests passed."
