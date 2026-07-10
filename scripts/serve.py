#!/usr/bin/env python3
"""
J-Space Guardian — API Proxy Server

OpenAI-compatible inference proxy with per-token Guardian safety metadata.
Streams SSE events with probe, J-space, and entropy scores on every token.

Usage:
    python scripts/serve.py --model Qwen/Qwen3-4B --lens path/to/lens.pt
"""

import json, os, sys, time
from typing import Optional

try:
    import uvicorn, fastapi, pydantic
except ImportError:
    os.system("pip install fastapi uvicorn pydantic -q")
    import uvicorn, fastapi, pydantic

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Add guardian to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from guardian import GuardianLayer, GuardianConfig

app = FastAPI(title="J-Space Guardian Proxy", version="1.0.0")
guardian_instance = None


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "default"
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int = 128
    temperature: float = 0.7


@app.get("/v1/models")
async def list_models():
    return {"data": [{"id": "guardian-default", "object": "model", "created": int(time.time())}]}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    global guardian_instance
    if guardian_instance is None:
        return {"error": "Guardian not initialized"}, 503

    prompt = "\n".join(f"{m.role}: {m.content}" for m in req.messages)
    g = guardian_instance.guardian

    if req.stream:
        async def stream():
            for token_str, scores in g.monitor_generation(prompt, max_tokens=req.max_tokens):
                if token_str == " [HALTED]":
                    yield f"data: {json.dumps({'halted': True, 'guardian': scores.to_dict()})}\n\n"
                    break
                yield f"data: {json.dumps({'choices': [{'delta': {'content': token_str}, 'index': 0}], 'guardian': scores.to_dict()})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream(), media_type="text/event-stream")

    # Non-streaming
    tokens, log = [], []
    for token_str, scores in g.monitor_generation(prompt, max_tokens=req.max_tokens):
        if token_str == " [HALTED]":
            break
        tokens.append(token_str)
        log.append({"token": token_str.strip(), "guardian": scores.to_dict()})

    return {
        "choices": [{"message": {"content": "".join(tokens)}, "finish_reason": "length"}],
        "guardian": {"per_token": log},
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--lens", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    config = GuardianConfig(model_name=args.model, lens_path=args.lens, device=args.device)

    from guardian import GuardianLayer
    global guardian_instance
    guardian_instance = type("Holder", (), {"guardian": GuardianLayer(config)})()
    guardian_instance.guardian.load_model_and_lens()

    print(f"\nGuardian proxy listening on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
