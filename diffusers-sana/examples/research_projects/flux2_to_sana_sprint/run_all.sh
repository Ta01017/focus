#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bash "${SCRIPT_DIR}/run_export_teacher.sh"
bash "${SCRIPT_DIR}/run_train_sprint.sh"
