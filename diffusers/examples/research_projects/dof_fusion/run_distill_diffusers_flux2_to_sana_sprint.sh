#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export TEACHER_BACKEND=diffusers_flux2
export TEACHER_MODEL=${TEACHER_MODEL:-black-forest-labs/FLUX.2-klein-base-4B}
exec bash "${SCRIPT_DIR}/run_sana_sprint_response_distill.sh"
