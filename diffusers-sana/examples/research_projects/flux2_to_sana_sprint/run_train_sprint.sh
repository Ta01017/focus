#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export NUM_NODES=${NUM_NODES:-1}
export NUM_GPUS=${NUM_GPUS:-4}

DATASET_BASE_PATH=${DATASET_BASE_PATH:?Set DATASET_BASE_PATH}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/flux2_to_sana_sprint}
TEACHER_METADATA=${TEACHER_METADATA:-${OUTPUT_ROOT}/teacher_metadata.json}
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/sprint_student}
STUDENT_MODEL=${STUDENT_MODEL:-Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers}
extra_args=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then extra_args+=(--max_samples "${MAX_SAMPLES}"); fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  extra_args+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
launch_args=(
  --mixed_precision bf16
  --num_machines "${NUM_NODES}"
  --num_processes "${NUM_GPUS}"
  --main_process_port "${MAIN_PROCESS_PORT:-29503}"
)
if ((NUM_GPUS > 1)); then launch_args+=(--multi_gpu); fi

accelerate launch \
  "${launch_args[@]}" \
  "${SCRIPT_DIR}/train_sprint_cross_distill.py" \
  --pretrained_model_name_or_path "${STUDENT_MODEL}" \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${TEACHER_METADATA}" \
  --data_file_keys "${DATA_FILE_KEYS:-teacher_image,edit_image}" \
  --extra_inputs "${EXTRA_INPUTS:-edit_image}" \
  --prompt_key "${PROMPT_KEY:-prompt}" \
  --dataset_repeat "${DATASET_REPEAT:-10}" \
  --resolution "${RESOLUTION:-1024}" \
  --learning_rate "${LEARNING_RATE:-1e-4}" \
  --max_train_steps "${MAX_TRAIN_STEPS:-20000}" \
  --train_batch_size "${BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS:-1}" \
  --dataloader_num_workers "${NUM_WORKERS:-4}" \
  --rank "${LORA_RANK:-32}" \
  --lora_alpha "${LORA_ALPHA:-32}" \
  --adapter_hidden_channels "${ADAPTER_HIDDEN_CHANNELS:-128}" \
  --guidance_scales "${GUIDANCE_SCALES:-4.0,4.5,5.0}" \
  --max_timestep_probability "${MAX_TIMESTEP_PROBABILITY:-0.5}" \
  --checkpointing_steps "${SAVE_STEPS:-500}" \
  --output_dir "${OUTPUT_DIR}" \
  --teacher_description "${TEACHER_DESCRIPTION:-DiffSynth-Studio FLUX.2}" \
  --gradient_checkpointing \
  "${extra_args[@]}"
