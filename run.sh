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

# Always resolve to project root regardless of where this script is called from
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# ---------- helpers ----------
log() { echo -e "\n>>> $1\n"; }

# ---------- setup ----------
cmd_setup() {
    log "Installing Python dependencies"
    pip install -r requirements.txt

    log "Downloading ALFWorld game data"
    alfworld-download 2>/dev/null || \
    python -c "import alfworld.agents; alfworld.agents.download()" 2>/dev/null || \
    echo "WARNING: Could not auto-download ALFWorld data. Run 'alfworld-download' manually."

    log "Setup complete"
}

# ---------- collect ----------
cmd_collect() {
    log "Collecting trajectories"
    python -m agent_distill.scripts.collect_trajectories "$@"
}

# ---------- evaluate ----------
cmd_eval() {
    log "Evaluating"
    python -m agent_distill.scripts.evaluate "$@"
}

# ---------- train ----------
cmd_train() {
    log "Starting TISD training pipeline"
    python -m agent_distill.scripts.run_tisd "$@"
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
        echo "Usage: bash run.sh {setup|collect|eval|train} [options]"
        echo ""
        echo "Examples:"
        echo "  bash run.sh setup"
        echo "  bash run.sh train"
        echo "  bash run.sh train --resume_iter 1"
        echo "  bash run.sh collect --num 64 --split train"
        echo "  bash run.sh eval --split test --num 134"
        exit 1
        ;;
esac
