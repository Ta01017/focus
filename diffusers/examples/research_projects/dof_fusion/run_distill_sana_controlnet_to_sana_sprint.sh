#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TEACHER_CHECKPOINT=${TEACHER_CHECKPOINT:?请设置 SANA ControlNet teacher checkpoint}
TEACHER_MODEL=${TEACHER_MODEL:-Efficient-Large-Model/Sana_600M_1024px_diffusers}
export TEACHER_BACKEND=command
export TEACHER_COMMAND="python '${SCRIPT_DIR}/infer_sana_controlnet.py' --checkpoint '${TEACHER_CHECKPOINT}' --model '${TEACHER_MODEL}' --image_a {image_a} --image_b {image_b} --focus_map {focus_a} --output {output} --prompt {prompt} --seed {seed}"
exec bash "${SCRIPT_DIR}/run_sana_sprint_response_distill.sh"
