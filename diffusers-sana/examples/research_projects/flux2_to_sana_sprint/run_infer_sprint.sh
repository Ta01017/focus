#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

CHECKPOINT=${CHECKPOINT:?Set CHECKPOINT to the cross-distilled Sprint output}
if (($# == 0)); then
  echo "Usage: CHECKPOINT=... PROMPT=... $0 reference1.png [reference2.png ...]" >&2
  exit 2
fi

python3 "${SCRIPT_DIR}/infer_sprint.py" \
  --checkpoint "${CHECKPOINT}" \
  --condition_images "$@" \
  --prompt "${PROMPT:-}" \
  --output "${OUTPUT:-flux2_distilled_sprint.png}" \
  --num_inference_steps "${STEPS:-2}" \
  --guidance_scale "${GUIDANCE_SCALE:-4.5}" \
  --seed "${SEED:-0}"
