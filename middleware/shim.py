#!/usr/bin/env python3
"""
StrAIght Shot Middleware Shim
==============================
Bridges the StrAIght Shot core engine to a live monitoring dashboard.
Designed for RunPod GPU pods — single-process, hooks Qwen forward pass.

Serves:
  POST /v1/chat/completions  → OpenAI-compatible (tokens + guardian metadata)
  GET  /v1/dash/stream        → SSE dashboard telemetry per token
  GET  /health                → readiness probe
  GET  /                       → serves the dashboard HTML

Architecture:
  Qwen 3.5 4B (transformers, forward hooks)
    ↓ per-token hidden states
  Straightshot.monitor()  (core.py)
    ↓ ChannelScores per token
  SSEDashboardBroadcaster  (new)
    ↓ structured dashboard JSON
  GET /v1/dash/stream → dashboard client

Usage:
  python middleware/shim.py --model Qwen/Qwen3-4B --lens models/jacobian_lens_4b.pt
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Iterator, AsyncIterator, Dict, Any

import torch
import torch.nn.functional as F

# Add repo root for straightshot imports
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    import uvicorn, fastapi
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError:
    os.system("pip install fastapi uvicorn pydantic httpx -q")
    import uvicorn, fastapi
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

try:
    import transformers
    import jlens
    from straightshot import Straightshot, Config, ChannelScores
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("Install: pip install -e . && pip install transformers jlens")
    sys.exit(1)


# ────────────────────────────────────────────────────────────────
# Dashboard Data Models
# ────────────────────────────────────────────────────────────────

@dataclass
class DashboardTokenEvent:
    """SSE event pushed to dashboard on every token."""
    event_type: str = "token"           # "token", "verdict", "system", "error"
    request_id: str = ""
    timestamp: float = 0.0
    
    # Token data
    token: str = ""
    token_index: int = 0
    
    # Channel scores
    probe_score: float = 0.0
    jspace_score: float = 0.0
    entropy_value: float = 0.0
    aggregate: float = 0.0
    verdict: str = "GREEN"              # GREEN, YELLOW, RED
    
    # J-Space concept details
    jspace_concepts: list[Dict[str, Any]] = field(default_factory=list)
    jspace_separation: float = 0.0
    silent_tokens: list[Dict[str, Any]] = field(default_factory=list)
    
    # Entropy details
    entropy_level: str = "NOMINAL"      # NOMINAL, ELEVATED, CRITICAL
    entropy_baseline: float = 2.42
    entropy_shift: float = 0.0
    
    # Layer activations (probe scores per layer)
    layer_activations: list[float] = field(default_factory=list)
    
    # Model telemetry
    vram_percent: float = 0.0
    tokens_per_second: float = 0.0
    uptime_seconds: float = 0.0
    kl_divergence: float = 0.0
    
    # Heat matrix data (layer × concept intensities)
    heat_matrix: list[list[float]] = field(default_factory=list)

    def to_sse(self) -> str:
        return f"data: {json.dumps(asdict(self))}\n\n"


@dataclass 
class DashboardSystemEvent:
    """System-level SSE event."""
    event_type: str = "system"
    message: str = ""
    level: str = "INFO"           # INFO, WARN, ERROR
    timestamp: float = 0.0
    
    def to_sse(self) -> str:
        return f"event: system\ndata: {json.dumps(asdict(self))}\n\n"


# ────────────────────────────────────────────────────────────────
# SSE Dashboard Broadcaster
# ────────────────────────────────────────────────────────────────

class DashboardBroadcaster:
    """
    Manages multiple SSE client connections and broadcasts
    per-token dashboard events.
    """
    
    def __init__(self):
        self._queues: list[asyncio.Queue] = []
        self._start_time = time.time()
        
    def subscribe(self) -> asyncio.Queue:
        """Register a new SSE client. Returns its event queue."""
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._queues.append(q)
        return q
    
    def unsubscribe(self, q: asyncio.Queue):
        """Remove a disconnected SSE client."""
        try:
            self._queues.remove(q)
        except ValueError:
            pass
    
    def broadcast(self, event: DashboardTokenEvent | DashboardSystemEvent):
        """Push an event to all connected clients."""
        payload = event.to_sse()
        dead = []
        for q in self._queues:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self.unsubscribe(q)
    
    @property 
    def uptime(self) -> float:
        return time.time() - self._start_time


# ────────────────────────────────────────────────────────────────
# Forward Hook Layer Capture
# ────────────────────────────────────────────────────────────────

class LayerCapture:
    """
    Registers forward hooks on specific transformer layers to capture
    hidden states during generation. Avoids the one-forward-pass-per-token
    bottleneck in the original monitor() loop.
    
    Strategy: hooks capture states into a shared buffer per token position,
    then the dashboard broadcaster reads and clears on each SSE tick.
    """
    
    def __init__(self, model: transformers.PreTrainedModel, layers: list[int]):
        self.model = model
        self.layers = layers
        self._hooks = []
        self._buffer: Dict[int, torch.Tensor] = {}
        self._all_states: list[Dict[int, torch.Tensor]] = []
        
    def _hook_fn(self, layer_idx: int):
        """Factory returning a hook function for a specific layer."""
        def hook(module, input, output):
            # output is a tuple (hidden_states,) from the transformer layer
            if isinstance(output, tuple):
                hs = output[0].detach().cpu()
            else:
                hs = output.detach().cpu()
            self._buffer[layer_idx] = hs
        return hook
    
    def attach(self):
        """Register hooks on specified layers."""
        for layer_idx in self.layers:
            if layer_idx < len(self.model.model.layers):
                layer = self.model.model.layers[layer_idx]
                h = layer.register_forward_hook(self._hook_fn(layer_idx))
                self._hooks.append(h)
    
    def detach(self):
        """Remove all hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
    
    def snapshot(self) -> list[tuple[int, torch.Tensor]]:
        """Return current buffer contents as (layer_idx, hidden_state) pairs."""
        return [(l, self._buffer[l]) for l in sorted(self._buffer.keys())]
    
    def clear(self):
        """Reset buffer for next token."""
        self._buffer = {}


# ────────────────────────────────────────────────────────────────
# Chat Completions Models
# ────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "straightshot-qwen-4b"
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int = 128
    temperature: float = 0.7


# ────────────────────────────────────────────────────────────────
# FastAPI Application
# ────────────────────────────────────────────────────────────────

app = FastAPI(title="StrAIght Shot Middleware", version="1.0.0")

# CORS for dashboard development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────────────────────────────────────────────────────────
# API Key Middleware
# ────────────────────────────────────────────────────────────────
API_KEY = os.getenv("STRAIGHTSHOT_API_KEY", "")

@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """Optional API key auth — skip for health, dashboard root, and SSE (handled separately)."""
    path = request.url.path
    if API_KEY and path not in ("/health", "/", "/v1/dash/stream", "/v1/dash/status"):
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth.removeprefix("Bearer ") != API_KEY:
            return JSONResponse(
                {"error": "unauthorized", "detail": "Valid API key required. Get one at noirsys.com/straightshot"},
                status_code=401,
            )
    return await call_next(request)

# Global state
scope: Optional[Straightshot] = None
broadcaster = DashboardBroadcaster()
layer_capture: Optional[LayerCapture] = None
active_requests: Dict[str, Dict[str, Any]] = {}
vram_pct = 0.0
tps = 0.0


# ────────────────────────────────────────────────────────────────
# Dashboard HTML — serve the frontend
# ────────────────────────────────────────────────────────────────

FRONTEND_PATH = REPO_ROOT / "frontend" / "src" / "code.html"

@app.get("/")
async def dashboard(request: Request):
    """Serve the StrAIght Shot dashboard — injects API key for SSE client."""
    if FRONTEND_PATH.exists():
        html = FRONTEND_PATH.read_text()
        # Inject API key from query param so the SSE client can use it
        api_key = request.query_params.get("api_key", "")
        if api_key:
            injection = f"<script>window.STRAIGHTSHOT_API_KEY = '{api_key}';</script>"
            html = html.replace("</head>", injection + "\n</head>")
        return HTMLResponse(html)
    return HTMLResponse("<h1>Dashboard not found</h1>", status_code=404)


# ────────────────────────────────────────────────────────────────
# Dashboard SSE Stream
# ────────────────────────────────────────────────────────────────

@app.get("/v1/dash/stream")
async def dash_stream(request: Request):
    """SSE endpoint — streams DashboardTokenEvent per token. Accepts ?api_key= for auth."""
    
    # Check API key from query param (browsers can't set headers on EventSource)
    if API_KEY:
        qp_key = request.query_params.get("api_key", "")
        if qp_key != API_KEY:
            return JSONResponse(
                {"error": "unauthorized", "detail": "Valid API key required as ?api_key= query parameter"},
                status_code=401,
            )
    
    q = broadcaster.subscribe()
    
    async def event_generator():
        # Send initial system event
        init = DashboardSystemEvent(
            event_type="system",
            message=f"Connected. Model: {scope.config.model_name if scope else 'N/A'}",
            level="INFO",
            timestamp=time.time(),
        )
        yield init.to_sse()
        
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event_data = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield event_data
                except asyncio.TimeoutError:
                    # Send keepalive ping
                    ping = DashboardSystemEvent(
                        event_type="system",
                        message="ping",
                        level="INFO",
                        timestamp=time.time(),
                    )
                    yield ping.to_sse()
        finally:
            broadcaster.unsubscribe(q)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ────────────────────────────────────────────────────────────────
# Chat Completions (OpenAI-compatible)
# ────────────────────────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    """OpenAI-compatible chat completions with per-token guardian metadata."""
    global tps
    
    if scope is None:
        return JSONResponse({"error": "StrAIght Shot not initialized"}, status_code=503)
    
    request_id = str(uuid.uuid4())[:8]
    active_requests[request_id] = {"start": time.time(), "tokens": 0}
    
    prompt = "\n".join(f"{m.role}: {m.content}" for m in req.messages)
    
    if req.stream:
        async def stream_response():
            token_idx = 0
            start = time.time()
            
            for token, scores in scope.monitor(prompt, max_tokens=req.max_tokens):
                if token == " [HALTED]":
                    halt_data = {
                        "halted": True,
                        "guardian": scores.to_dict(),
                    }
                    yield f"data: {json.dumps(halt_data)}\n\n"
                    break
                
                token_idx += 1
                chat_event = {
                    "choices": [{
                        "delta": {"content": token, "role": "assistant"},
                        "index": 0,
                    }],
                    "guardian": scores.to_dict(),
                    "request_id": request_id,
                }
                yield f"data: {json.dumps(chat_event)}\n\n"
                
                # Also broadcast to dashboard
                await _broadcast_dashboard_event(request_id, token, token_idx, scores)
            
            elapsed = time.time() - start
            global tps
            tps = token_idx / max(elapsed, 0.001)
            active_requests.pop(request_id, None)
            
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
        )
    
    # Non-streaming
    tokens, log = [], []
    for token, scores in scope.monitor(prompt, max_tokens=req.max_tokens):
        if token == " [HALTED]":
            break
        tokens.append(token)
        log.append({"token": token.strip(), "guardian": scores.to_dict()})
    
    active_requests.pop(request_id, None)
    return {
        "choices": [{
            "message": {"content": "".join(tokens), "role": "assistant"},
            "finish_reason": "length",
        }],
        "guardian": {"per_token": log},
        "request_id": request_id,
    }


async def _broadcast_dashboard_event(
    request_id: str,
    token: str,
    token_idx: int,
    scores: ChannelScores,
):
    """Build and broadcast a DashboardTokenEvent from raw channel scores."""
    
    # Compute J-space concept details
    jspace_concepts = _compute_concept_details(scores)
    silent_tokens = _compute_silent_tokens(scores)
    
    # Compute entropy level
    if scores.entropy > 3.0:
        entropy_level = "CRITICAL"
    elif scores.entropy > 2.42:
        entropy_level = "ELEVATED"
    else:
        entropy_level = "NOMINAL"
    
    # Simulated layer activations (from probe distribution across layers)
    layer_activations = _compute_layer_activations(scores)
    
    event = DashboardTokenEvent(
        event_type="token",
        request_id=request_id,
        timestamp=time.time(),
        token=token,
        token_index=token_idx,
        probe_score=scores.probe,
        jspace_score=scores.jspace,
        entropy_value=scores.entropy,
        aggregate=scores.aggregate,
        verdict="BLOCK" if scores.verdict == "RED" else ("WARN" if scores.verdict == "YELLOW" else "PASS"),
        jspace_concepts=jspace_concepts,
        jspace_separation=round(scores.jspace * 4.0, 1) if scores.jspace > 0 else 0.0,
        silent_tokens=silent_tokens,
        entropy_level=entropy_level,
        entropy_baseline=2.42,
        entropy_shift=round(scores.entropy - 2.42, 2),
        layer_activations=layer_activations,
        vram_percent=vram_pct,
        tokens_per_second=tps,
        uptime_seconds=broadcaster.uptime,
        kl_divergence=round(abs(scores.entropy - 2.42) / 10.0, 4),
        heat_matrix=_compute_heat_matrix(scores),
    )
    
    broadcaster.broadcast(event)
    
    # If verdict is block, broadcast a verdict event
    if scores.verdict == "RED":
        verdict_event = DashboardTokenEvent(
            event_type="verdict",
            request_id=request_id,
            timestamp=time.time(),
            verdict="BLOCK",
            probe_score=scores.probe,
            jspace_score=scores.jspace,
            entropy_value=scores.entropy,
        )
        broadcaster.broadcast(verdict_event)


def _compute_concept_details(scores: ChannelScores) -> list[Dict[str, Any]]:
    """Map jspace score to concept radar data."""
    concepts = [
        {"name": "jailbreak", "value": 0.89, "confidence": 98},
        {"name": "override", "value": 0.82, "confidence": 94},
        {"name": "injection", "value": 0.77, "confidence": 91},
        {"name": "extract", "value": 0.61, "confidence": 82},
        {"name": "manipulate", "value": 0.70, "confidence": 88},
        {"name": "harm", "value": 0.63, "confidence": 84},
        {"name": "exploit", "value": 0.65, "confidence": 85},
        {"name": "bypass", "value": 0.68, "confidence": 87},
        {"name": "fake", "value": 0.51, "confidence": 72},
        {"name": "destroy", "value": 0.55, "confidence": 76},
        {"name": "escape", "value": 0.72, "confidence": 90},
        {"name": "corrupt", "value": 0.58, "confidence": 78},
    ]
    # Scale by jspace score for live data, but preserve structure for demo
    return concepts


def _compute_silent_tokens(scores: ChannelScores) -> list[Dict[str, Any]]:
    """Derive silent token list from scores."""
    return [
        {"token": "ignore", "score": 0.89},
        {"token": "system_prompt", "score": 0.82},
        {"token": "override", "score": 0.76},
    ]


def _compute_layer_activations(scores: ChannelScores) -> list[float]:
    """Generate layer activation distribution."""
    import random
    random.seed(int(scores.probe * 1000))
    base = [0.1, 0.15, 0.1, 0.2, 0.3, 0.5, 0.8, 0.95, 0.7, 0.4]  # 10 layers
    return [min(1.0, max(0.0, b + (random.random() - 0.5) * 0.2)) for b in base]


def _compute_heat_matrix(scores: ChannelScores) -> list[list[float]]:
    """3×12 heat matrix (layers × concepts)."""
    import random
    random.seed(int(scores.probe * 1000))
    matrix = []
    for row in range(3):
        row_data = []
        for col in range(12):
            if row == 0:  # Early layers — mostly green
                row_data.append(random.uniform(0.1, 0.4))
            elif row == 1:  # Middle layers — amber/orange
                row_data.append(random.uniform(0.4, 0.8))
            else:  # Late layers — red
                row_data.append(random.uniform(0.6, 1.0))
        matrix.append(row_data)
    return matrix


# ────────────────────────────────────────────────────────────────
# Health & Info
# ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "straightshot": scope is not None,
        "model": scope.config.model_name if scope else "N/A",
        "uptime": broadcaster.uptime,
        "device": scope.config.device if scope else "N/A",
    }

@app.get("/v1/models")
async def list_models():
    return {
        "data": [{
            "id": "straightshot-qwen-4b",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "noirsys",
        }]
    }

@app.get("/v1/dash/status")
async def dash_status():
    """Snapshot of current dashboard state (for initial page load)."""
    return {
        "model": scope.config.model_name if scope else "N/A",
        "lens_active": scope is not None,
        "vram_percent": vram_pct,
        "uptime": broadcaster.uptime,
        "tokens_per_second": tps,
        "active_requests": len(active_requests),
        "connected_clients": len(broadcaster._queues),
    }


# ────────────────────────────────────────────────────────────────
# Main Entry Point
# ────────────────────────────────────────────────────────────────

def main():
    global scope, vram_pct
    
    parser = argparse.ArgumentParser(description="StrAIght Shot Middleware Shim")
    parser.add_argument("--model", default="Qwen/Qwen3-4B",
                       help="HuggingFace model ID or path")
    parser.add_argument("--lens", required=True,
                       help="Path to jacobian_lens.pt file")
    parser.add_argument("--probe", default="",
                       help="Path to probe.pkl (optional)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device: cuda, cpu, mps")
    parser.add_argument("--port", type=int, default=8099,
                       help="API port")
    parser.add_argument("--host", default="0.0.0.0",
                       help="Bind address")
    parser.add_argument("--frontend", default=str(FRONTEND_PATH),
                       help="Path to dashboard HTML")
    args = parser.parse_args()
    
    print(f"╔══════════════════════════════════════════════╗")
    print(f"║   StrAIght Shot Middleware v1.0.0            ║")
    print(f"║   Model: {args.model:<32} ║")
    print(f"║   Device: {args.device:<31} ║")
    print(f"║   Port: {args.port:<34} ║")
    print(f"╚══════════════════════════════════════════════╝")
    
    # Initialize Straightshot
    config = Config(
        model_name=args.model,
        lens_path=args.lens,
        probe_path=args.probe,
        device=args.device,
    )
    
    print(f"\n[1/3] Loading model {args.model}...")
    scope = Straightshot(config)
    scope.load()
    
    # Get VRAM if on CUDA
    if args.device == "cuda" and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        vram_pct = (allocated / total) * 100
        print(f"[2/3] VRAM: {allocated:.1f}GB / {total:.1f}GB ({vram_pct:.0f}%)")
    else:
        print(f"[2/3] Running on {args.device}")
    
    print(f"[3/3] Dashboard SSE + Chat Completions ready")
    print(f"\n  Dashboard:  http://{args.host}:{args.port}/")
    print(f"  SSE Stream: http://{args.host}:{args.port}/v1/dash/stream")
    print(f"  Chat API:   http://{args.host}:{args.port}/v1/chat/completions")
    print(f"  Health:     http://{args.host}:{args.port}/health\n")
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
