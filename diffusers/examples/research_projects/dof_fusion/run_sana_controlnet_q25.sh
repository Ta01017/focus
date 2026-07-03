#!/usr/bin/env bash
set -euo pipefail

# Q25 checkpoint names differ between local deployments. Supply the concrete model path/id explicitly.
MODEL=${MODEL:?请设置 Q25 的 MODEL 路径或模型 ID}
export MODEL

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "${SCRIPT_DIR}/run_sana_controlnet.sh"
