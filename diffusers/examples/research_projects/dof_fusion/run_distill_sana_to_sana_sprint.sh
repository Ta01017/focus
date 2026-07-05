#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export TEACHER_BACKEND=command
TEACHER_COMMAND=${TEACHER_COMMAND:?请提供已训练 SANA DOF 教师的推理命令模板}
export TEACHER_COMMAND
exec bash "${SCRIPT_DIR}/run_sana_sprint_response_distill.sh"
