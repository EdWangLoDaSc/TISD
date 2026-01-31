#!/bin/bash
set -e

# ============================================================
# TISD: Trajectory-Informed Self-Distillation for ALFWorld
# ============================================================
# Usage:
#   bash run.sh setup          — install deps + download alfworld data
#   bash run.sh collect        — collect trajectories only
#   bash run.sh eval           — evaluate model only
#   bash run.sh train          — full TISD training pipeline
#   bash run.sh train --resume_iter N  — resume from iteration N
# ============================================================

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${PROJECT_ROOT}/agent_distill/configs/default.yaml"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# ---------- helpers ----------
log() { echo -e "\n>>> $1\n"; }

# ---------- setup ----------
cmd_setup() {
    log "Installing Python dependencies"
    pip install -r "${PROJECT_ROOT}/requirements.txt"

    log "Downloading ALFWorld game data"
    python -c "import alfworld.agents; alfworld.agents.download()" 2>/dev/null || \
    python -c "
import alfworld
import os, subprocess
data_dir = os.path.dirname(alfworld.__file__)
print(f'alfworld installed at {data_dir}')
" && alfworld-download 2>/dev/null || echo "ALFWorld data may need manual download: export ALFWORLD_DATA=<path>"

    log "Setup complete"
}

# ---------- collect ----------
cmd_collect() {
    NUM=${1:-64}
    SPLIT=${2:-train}
    OUTPUT=${3:-trajectories/collected}

    log "Collecting ${NUM} trajectories from ${SPLIT} split"
    python -m agent_distill.scripts.collect_trajectories \
        --config "${CONFIG}" \
        --num "${NUM}" \
        --split "${SPLIT}" \
        --output "${OUTPUT}" \
        "${@:4}"
}

# ---------- evaluate ----------
cmd_eval() {
    SPLIT=${1:-test}
    NUM=${2:-134}

    log "Evaluating on ${SPLIT} split (${NUM} tasks)"
    python -m agent_distill.scripts.evaluate \
        --config "${CONFIG}" \
        --split "${SPLIT}" \
        --num "${NUM}" \
        "${@:3}"
}

# ---------- train ----------
cmd_train() {
    log "Starting TISD training pipeline"
    python -m agent_distill.scripts.run_tisd \
        --config "${CONFIG}" \
        "$@"
}

# ---------- dispatch ----------
CMD="${1:-train}"
shift 2>/dev/null || true

case "${CMD}" in
    setup)    cmd_setup "$@" ;;
    collect)  cmd_collect "$@" ;;
    eval)     cmd_eval "$@" ;;
    train)    cmd_train "$@" ;;
    *)
        echo "Unknown command: ${CMD}"
        echo "Usage: bash run.sh {setup|collect|eval|train}"
        exit 1
        ;;
esac
