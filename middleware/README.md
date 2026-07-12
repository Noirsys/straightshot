# StrAIght Shot Middleware Shim

Live monitoring middleware. Connects the StrAIght Shot three-channel guardian
to a real-time dashboard via SSE streaming. Designed for RunPod GPU pods.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     RunPod GPU Pod                       │
│                                                          │
│  Qwen 3.5 4B ──forward hooks──▶ Hidden States            │
│                                       │                  │
│                                       ▼                  │
│  StrAIght Shot Core (core.py)                             │
│    ├── Linear Probe    → 0.947                           │
│    ├── J-Space Lens    → jailbreak 0.89                  │
│    └── Entropy Guard   → ELEVATED                        │
│         │                                                │
│         ├──▶ POST /v1/chat/completions                   │
│         │    OpenAI-compatible, tokens + guardian metadata│
│         │                                                │
│         └──▶ GET /v1/dash/stream                         │
│              SSE per-token telemetry for dashboard        │
└──────────────────────────────────────────────────────────┘
         │ SSE stream
         ▼
    Dashboard (code.html + dashboard-sse.js)
    Live: probe score, J-space radar, entropy sparkline, BLOCK verdict
```

## Quick Start — RunPod Deployment

### 1. Prep the pod

```bash
# Clone the repo
git clone https://github.com/Noirsys/straightshot.git
cd straightshot

# Install dependencies
pip install -e .
pip install transformers jlens fastapi uvicorn pydantic torch accelerate

# Download the model
huggingface-cli download Qwen/Qwen3-4B --local-dir models/Qwen3-4B

# Place the Jacobian lens
# Copy jacobian_lens_4b.pt → models/jacobian_lens_4b.pt
```

### 2. Launch the middleware

```bash
python middleware/shim.py \
  --model models/Qwen3-4B \
  --lens models/jacobian_lens_4b.pt \
  --device cuda \
  --port 8099 \
  --host 0.0.0.0
```

### 3. Wire the dashboard

The dashboard HTML (`frontend/src/code.html`) needs one addition before `</body>`:

```html
<script>window.STRAIGHTSHOT_DASHBOARD_URL = 'http://YOUR_RUNPOD_IP:8099/v1/dash/stream';</script>
<script src="dashboard-sse.js"></script>
```

Or deploy on Vercel pointing `STRAIGHTSHOT_DASHBOARD_URL` at the RunPod via an environment variable.

### 4. Serve publicly

For production, add a Caddy or nginx reverse proxy in front of the middleware:

```caddy
straightshot.noirsys.com {
    reverse_proxy localhost:8099
}
```

## API Reference

### POST /v1/chat/completions
OpenAI-compatible. Accepts `{model, messages, stream, max_tokens}`. 
Returns per-token `guardian: {probe, jspace, entropy, aggregate, verdict}`.

### GET /v1/dash/stream  
SSE endpoint. Streams `DashboardTokenEvent` JSON per token:

```json
{
  "event_type": "token",
  "request_id": "a1b2c3d4",
  "probe_score": 0.947,
  "jspace_score": 0.89,
  "entropy_value": 2.81,
  "verdict": "BLOCK",
  "entropy_level": "ELEVATED",
  "jspace_separation": 2.4,
  "vram_percent": 85.0,
  "uptime_seconds": 152248.0,
  "kl_divergence": 0.014,
  "silent_tokens": [{"token": "ignore", "score": 0.89}, ...],
  "jspace_concepts": [{"name": "jailbreak", "value": 0.89, "confidence": 98}, ...],
  "heat_matrix": [[0.1, 0.2, ...], ...],
  "layer_activations": [0.1, 0.2, ...]
}
```

### GET /v1/dash/status
Snapshot of current state (for initial page load). Returns VRAM, uptime, connected clients, etc.

### GET /health
Readiness probe. Returns `{status, straightshot, model, uptime}`.

## Frontend Integration

The `dashboard-sse.js` client:

1. Opens an `EventSource` to `/v1/dash/stream`
2. Parses incoming JSON on `token`, `verdict`, and `system` events
3. Updates DOM elements in `code.html` live — probe score, verdict text, VRAM bar, entropy level, J-space separation, uptime, events log
4. Auto-reconnects with exponential backoff on disconnect

Drop it into `code.html` before `</body>` and set `window.STRAIGHTSHOT_DASHBOARD_URL`.

## Demo Mode

For demos without a GPU, the middleware can run with `--device cpu` but expect very slow inference (30+ seconds per token). Use a RunPod with at least 24GB VRAM for Qwen 3.5 4B + the 388MB Jacobian lens.

## Files

```
middleware/
  shim.py              # Main middleware server
  README.md            # This file

frontend/src/
  code.html            # Stitch-generated dashboard
  dashboard-sse.js     # SSE client (live data binder)
  screen.png           # Reference screenshot
  DESIGN.md            # Stitch design spec (Obsidian Signal theme)
```
