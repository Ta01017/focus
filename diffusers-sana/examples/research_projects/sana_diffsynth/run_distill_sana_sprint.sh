#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export NUM_NODES=${NUM_NODES:-1}
export NUM_GPUS=${NUM_GPUS:-4}

TEACHER_MODEL=${TEACHER_MODEL:-Efficient-Large-Model/Sana_Sprint_0.6B_1024px_teacher_diffusers}
TEACHER_ADAPTER_PATH=${TEACHER_ADAPTER_PATH:?Set TEACHER_ADAPTER_PATH to the trained Sana output}
SPRINT_BASE_MODEL=${SPRINT_BASE_MODEL:-Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers}
DATASET_BASE_PATH=${DATASET_BASE_PATH:?Set DATASET_BASE_PATH}
METADATA=${METADATA:-${DATASET_BASE_PATH}/metadata.json}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/sana_diffsynth/sprint_student}
extra_args=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then extra_args+=(--max_samples "${MAX_SAMPLES}"); fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  extra_args+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

launch_args=(
  --mixed_precision bf16
  --num_machines "${NUM_NODES}"
  --num_processes "${NUM_GPUS}"
  --main_process_port "${MAIN_PROCESS_PORT:-29502}"
)
if ((NUM_GPUS > 1)); then launch_args+=(--multi_gpu); fi

accelerate launch \
  "${launch_args[@]}" \
  "${SCRIPT_DIR}/train_sana_sprint.py" \
  --pretrained_model_name_or_path "${TEACHER_MODEL}" \
  --teacher_adapter_path "${TEACHER_ADAPTER_PATH}" \
  --sprint_base_model "${SPRINT_BASE_MODEL}" \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${METADATA}" \
  --data_file_keys "${DATA_FILE_KEYS:-image,edit_image}" \
  --extra_inputs "${EXTRA_INPUTS:-edit_image}" \
  --prompt_key "${PROMPT_KEY:-prompt}" \
  --dataset_repeat "${DATASET_REPEAT:-1}" \
  --resolution "${RESOLUTION:-1024}" \
  --learning_rate "${LEARNING_RATE:-1e-6}" \
  --max_train_steps "${MAX_TRAIN_STEPS:-30000}" \
  --train_batch_size "${BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS:-1}" \
  --dataloader_num_workers "${NUM_WORKERS:-8}" \
  --checkpointing_steps "${SAVE_STEPS:-500}" \
  --checkpoints_total_limit "${CHECKPOINTS_TOTAL_LIMIT:-10}" \
  --output_dir "${OUTPUT_DIR}" \
  --train_largest_timestep \
  --misaligned_pairs_D \
  --gradient_checkpointing \
  --allow_tf32 \
  "${extra_args[@]}"
