#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

ADAPTER_PATH=${ADAPTER_PATH:?Set ADAPTER_PATH to the Sprint distillation output}
if (($# == 0)); then
  echo "Usage: ADAPTER_PATH=... PROMPT=... $0 reference1.png [reference2.png ...]" >&2
  exit 2
fi

python3 "${SCRIPT_DIR}/infer.py" \
  --pipeline_type sana_sprint \
  --adapter_path "${ADAPTER_PATH}" \
  --condition_images "$@" \
  --prompt "${PROMPT:-}" \
  --output "${OUTPUT:-sprint_output.png}" \
  --num_inference_steps "${STEPS:-2}" \
  --guidance_scale "${GUIDANCE_SCALE:-4.5}" \
  --seed "${SEED:-0}"
