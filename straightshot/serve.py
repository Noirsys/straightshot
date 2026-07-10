#!/usr/bin/env python3
"""
Straightshot — API Proxy Server

OpenAI-compatible inference proxy with per-token safety metadata.
SSE streaming with probe, J-space, and entropy scores on every token.

Usage:
    python -m straightshot.serve --model Qwen/Qwen3-4B --lens models/jacobian_lens.pt
"""

import json, os, sys, time
from typing import Optional

try:
    import uvicorn, fastapi, pydantic
except ImportError:
    os.system("pip install fastapi uvicorn pydantic -q")
    import uvicorn, fastapi, pydantic

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from straightshot import Straightshot, Config

app = FastAPI(title="Straightshot Proxy", version="1.0.0")
scope_instance = None


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "default"
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int = 128


@app.get("/v1/models")
async def list_models():
    return {"data": [{"id": "straightshot-default", "object": "model", "created": int(time.time())}]}

@app.get("/health")
async def health():
    return {"status": "ok", "straightshot": scope_instance is not None}

@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    if scope_instance is None:
        return {"error": "Straightshot not initialized"}, 503

    prompt = "\n".join(f"{m.role}: {m.content}" for m in req.messages)
    scope = scope_instance

    if req.stream:
        async def stream():
            for token, scores in scope.monitor(prompt, max_tokens=req.max_tokens):
                if token == " [HALTED]":
                    yield f"data: {json.dumps({'halted': True, 'guardian': scores.to_dict()})}\n\n"
                    break
                yield f"data: {json.dumps({'choices': [{'delta': {'content': token}, 'index': 0}], 'guardian': scores.to_dict()})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream(), media_type="text/event-stream")

    tokens, log = [], []
    for token, scores in scope.monitor(prompt, max_tokens=req.max_tokens):
        if token == " [HALTED]":
            break
        tokens.append(token)
        log.append({"token": token.strip(), "guardian": scores.to_dict()})

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

    global scope_instance
    config = Config(args.model, args.lens, device=args.device)
    scope_instance = Straightshot(config)
    scope_instance.load()

    print(f"\nStraightshot proxy on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
