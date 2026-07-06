#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DATASET_BASE_PATH=${DATASET_BASE_PATH:?Set DATASET_BASE_PATH}
METADATA=${METADATA:-${DATASET_BASE_PATH}/metadata.json}
TEACHER_COMMAND=${TEACHER_COMMAND:?Set TEACHER_COMMAND to the DiffSynth FLUX.2 inference command template}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/flux2_to_sana_sprint}
TEACHER_OUTPUT_DIR=${TEACHER_OUTPUT_DIR:-${OUTPUT_ROOT}/teacher_images}
TEACHER_METADATA=${TEACHER_METADATA:-${OUTPUT_ROOT}/teacher_metadata.json}
extra_args=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then extra_args+=(--max_samples "${MAX_SAMPLES}"); fi
if [[ "${SKIP_EXISTING:-1}" == "1" ]]; then extra_args+=(--skip_existing); fi
if [[ -n "${TEACHER_HEIGHT:-}" ]]; then
  extra_args+=(--height "${TEACHER_HEIGHT}" --width "${TEACHER_WIDTH:?Set TEACHER_WIDTH}")
fi
if [[ -n "${DIFFSYNTH_ROOT:-}" ]]; then extra_args+=(--command_cwd "${DIFFSYNTH_ROOT}"); fi

"${EXPORT_PYTHON:-python3}" "${SCRIPT_DIR}/export_diffsynth_teacher.py" \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${METADATA}" \
  --condition_key "${CONDITION_KEY:-edit_image}" \
  --prompt_key "${PROMPT_KEY:-prompt}" \
  --default_prompt "${DEFAULT_PROMPT:-}" \
  --teacher_command "${TEACHER_COMMAND}" \
  --output_dir "${TEACHER_OUTPUT_DIR}" \
  --output_metadata_path "${TEACHER_METADATA}" \
  --seed "${SEED:-0}" \
  "${extra_args[@]}"
