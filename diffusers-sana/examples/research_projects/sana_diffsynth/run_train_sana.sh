#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export NUM_NODES=${NUM_NODES:-1}
export NUM_GPUS=${NUM_GPUS:-4}

MODEL=${MODEL:-Efficient-Large-Model/Sana_Sprint_0.6B_1024px_teacher_diffusers}
DATASET_BASE_PATH=${DATASET_BASE_PATH:?Set DATASET_BASE_PATH}
METADATA=${METADATA:-${DATASET_BASE_PATH}/metadata.json}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/sana_diffsynth/sana_teacher}
extra_args=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then extra_args+=(--max_samples "${MAX_SAMPLES}"); fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  extra_args+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

launch_args=(
  --mixed_precision bf16
  --num_machines "${NUM_NODES}"
  --num_processes "${NUM_GPUS}"
  --main_process_port "${MAIN_PROCESS_PORT:-29501}"
)
if ((NUM_GPUS > 1)); then launch_args+=(--multi_gpu); fi

accelerate launch \
  "${launch_args[@]}" \
  "${SCRIPT_DIR}/train_sana.py" \
  --pretrained_model_name_or_path "${MODEL}" \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${METADATA}" \
  --data_file_keys "${DATA_FILE_KEYS:-image,edit_image}" \
  --extra_inputs "${EXTRA_INPUTS:-edit_image}" \
  --dataset_repeat "${DATASET_REPEAT:-10}" \
  --resolution "${RESOLUTION:-1024}" \
  --learning_rate "${LEARNING_RATE:-1e-4}" \
  --max_train_steps "${MAX_TRAIN_STEPS:-20000}" \
  --train_batch_size "${BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS:-1}" \
  --rank "${LORA_RANK:-32}" \
  --lora_alpha "${LORA_ALPHA:-32}" \
  --adapter_hidden_channels "${ADAPTER_HIDDEN_CHANNELS:-128}" \
  --checkpointing_steps "${SAVE_STEPS:-500}" \
  --output_dir "${OUTPUT_DIR}" \
  --gradient_checkpointing \
  "${extra_args[@]}"
