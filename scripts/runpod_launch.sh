#!/usr/bin/env bash
# StrAIght Shot — RunPod Launch Script
# ======================================
# Drop this into a RunPod RTX 6000 Ada 48GB pod (or any 24GB+ GPU).
# Runs on boot: installs deps, downloads model, places lens, launches middleware.
#
# Usage:
#   curl -sL https://raw.githubusercontent.com/noirsys/straightshot/main/scripts/runpod_launch.sh | bash
#
# Or upload directly to RunPod as the container start command.

set -euo pipefail

echo "╔══════════════════════════════════════════════╗"
echo "║   StrAIght Shot — RunPod Launch v1.0.0      ║"
echo "╚══════════════════════════════════════════════╝"

WORKDIR="${WORKDIR:-/workspace}"
PORT="${PORT:-8099}"
API_KEY="${STRAIGHTSHOT_API_KEY:-}"

cd "$WORKDIR"

# ── 1. Clone the repo ──────────────────────────────────
if [ ! -d straightshot ]; then
    echo "[1/6] Cloning repo..."
    git clone https://github.com/noirsys/straightshot.git
else
    echo "[1/6] Repo exists, pulling..."
    cd straightshot && git pull && cd ..
fi

cd straightshot

# ── 2. Install Python deps ─────────────────────────────
echo "[2/6] Installing dependencies..."
pip install -e . -q
pip install transformers jlens fastapi uvicorn pydantic torch accelerate -q

# ── 3. Download model ──────────────────────────────────
MODEL_DIR="$WORKDIR/models/Qwen3-4B"
if [ ! -d "$MODEL_DIR" ]; then
    echo "[3/6] Downloading Qwen 3.5 4B (this takes ~10 min)..."
    HF_HUB_DISABLE_XET=1 TMPDIR="$WORKDIR/tmp" \
        huggingface-cli download Qwen/Qwen3-4B --local-dir "$MODEL_DIR"
else
    echo "[3/6] Model already cached."
fi

# ── 4. Place Jacobian lens ─────────────────────────────
LENS_PATH="$WORKDIR/models/jacobian_lens_4b.pt"
if [ ! -f "$LENS_PATH" ]; then
    echo "[4/6] Placing Jacobian lens..."
    cp models/jacobian_lens_4b.pt "$LENS_PATH" 2>/dev/null || {
        echo "Lens not in repo — downloading..."
        curl -sL "https://github.com/noirsys/straightshot/releases/download/v1.0.0/jacobian_lens_4b.pt" \
            -o "$LENS_PATH" || echo "WARNING: Lens download failed — manual placement needed"
    }
fi

# ── 5. Launch middleware ────────────────────────────────
echo "[5/6] Starting StrAIght Shot middleware..."
echo "       Port: $PORT"
echo "       Dashboard: http://0.0.0.0:$PORT/"
echo "       API:       http://0.0.0.0:$PORT/v1/chat/completions"
echo ""

CMD="python middleware/shim.py \
    --model '$MODEL_DIR' \
    --lens '$LENS_PATH' \
    --device cuda \
    --port $PORT \
    --host 0.0.0.0"

if [ -n "$API_KEY" ]; then
    export STRAIGHTSHOT_API_KEY="$API_KEY"
    echo "[6/6] API key auth enabled."
else
    echo "[6/6] No API key set — open access mode (demo only)."
fi

exec $CMD
