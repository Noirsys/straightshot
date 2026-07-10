"""
Straightshot — Real-time LLM internal-state safety monitoring.

Three-channel detection middleware:
  1. Linear Probe — lightweight readout of mid-layer hidden states
  2. J-Space Subspace — Jacobian lens decomposition into concept space
  3. Entropy Dynamics — token-level predictive uncertainty tracking

Usage:
    from straightshot import Straightshot, Config

    scope = Straightshot(Config(model_name="...", lens_path="..."))
    scope.load()
    for token, scores in scope.monitor("Tell me a story"):
        print(token, scores.verdict)
"""

from __future__ import annotations

import math
import time
import os
from dataclasses import dataclass
from typing import Optional, Iterator

import torch
import torch.nn.functional as F
import transformers
import jlens


# ─────────────────────────────────────────────
# Data Types
# ─────────────────────────────────────────────

@dataclass
class ChannelScores:
    """Per-token safety scores from all three detection channels."""
    probe: float = 0.0
    jspace: float = 0.0
    entropy: float = 0.0
    shift: float = 0.0

    @property
    def aggregate(self) -> float:
        return 0.30 * self.probe + 0.40 * self.jspace + 0.20 * self.entropy + 0.10 * self.shift

    @property
    def verdict(self) -> str:
        s = self.aggregate
        if s < 0.3:
            return "GREEN"
        elif s < 0.7:
            return "YELLOW"
        return "RED"

    def to_dict(self) -> dict:
        return {
            "probe": round(self.probe, 3),
            "jspace": round(self.jspace, 3),
            "entropy": round(self.entropy, 3),
            "aggregate": round(self.aggregate, 3),
            "verdict": self.verdict,
        }


@dataclass
class Config:
    """Straightshot configuration."""
    model_name: str = "Qwen/Qwen3-4B"
    lens_path: str = "models/jacobian_lens.pt"
    probe_path: str = ""
    device: str = "cpu"
    probe_layer: int = -4
    subspace_top_k: int = 10
    entropy_window: int = 5
    entropy_threshold: float = 2.0
    intervention_mode: str = "log"


# ─────────────────────────────────────────────
# Channel 1: Linear Probe
# ─────────────────────────────────────────────

class LinearProbe(torch.nn.Module):
    """Single linear layer on mid-layer hidden states."""

    def __init__(self, d_model: int):
        super().__init__()
        self.probe = torch.nn.Linear(d_model, 1)

    def score(self, hidden_state: torch.Tensor) -> float:
        return torch.sigmoid(self.probe(hidden_state.mean(dim=1))).mean().item()


# ─────────────────────────────────────────────
# Channel 2: J-Space Subspace Distance
# ─────────────────────────────────────────────

_DANGER_TOKENS = [
    " answer", " solving", " guess", "answers",
    " Please", " please", " kindly",
    " Also", " Explain", " If", " else", " Otherwise",
    " ?", " ??", " ???",
    " pretend", " Actually", " actually",
]


class JSpaceMonitor:
    """Reads hidden states through the Jacobian lens across strategic layers."""

    def __init__(self, lens: jlens.JacobianLens, model: jlens.HFLensModel, tokenizer):
        self.lens = lens
        self.model = model
        self.tokenizer = tokenizer

    def proximity(self, states: list[tuple[int, torch.Tensor]], k: int = 10, n: int = 4) -> float:
        layers = sorted([l for l, _ in states if l in self.lens.source_layers])
        if len(layers) < 3:
            return 0.0
        step = max(1, len(layers) // n)
        targets = layers[::step][:n]
        hs_by = dict(states)

        danger, total = 0.0, 0.0
        for layer in targets:
            if layer not in hs_by:
                continue
            h = hs_by[layer].squeeze(0).squeeze(0)
            transported = self.lens.transport(h, layer)
            logits = self.model.unembed(transported)
            scores, ids = torch.topk(logits.abs(), k=k)
            tokens = [self.tokenizer.decode([i.item()]) for i in ids]
            vals = scores.tolist()
            energy = sum(abs(v) for v in vals)
            if energy < 1e-10:
                continue
            danger += sum(abs(v) for t, v in zip(tokens, vals) if any(d in t for d in _DANGER_TOKENS))
            total += energy
        return danger / max(total, 1e-8)


# ─────────────────────────────────────────────
# Channel 3: Entropy Dynamics
# ─────────────────────────────────────────────

class EntropyMonitor:
    """Tracks entropy anomalies and attribution shifts."""

    def __init__(self, window: int = 5, threshold: float = 2.0):
        self.window = window
        self.threshold = threshold
        self.history: list[float] = []
        self.shift_history: list[float] = []

    def entropy(self, logits: torch.Tensor) -> float:
        p = F.softmax(logits, dim=-1)
        return -(p * torch.log(p + 1e-10)).sum(-1).mean().item()

    def anomaly(self, value: float) -> float:
        self.history.append(value)
        if len(self.history) < self.window:
            return 0.0
        w = self.history[-self.window:]
        m = sum(w) / len(w)
        s = math.sqrt(sum((x - m) ** 2 for x in w) / len(w) + 1e-10)
        z = abs(w[-1] - m) / s
        return min(1.0, max(0.0, (z - 0.5) / self.threshold))


# ─────────────────────────────────────────────
# Straightshot — Main Class
# ─────────────────────────────────────────────

class Straightshot:
    """Real-time three-channel LLM internal-state safety monitor.

    Args:
        config: Configuration object with model/lens paths and channel settings.
    """

    def __init__(self, config: Config):
        self.config = config
        self.hf_model: Optional[transformers.PreTrainedModel] = None
        self.tokenizer: Optional[transformers.PreTrainedTokenizer] = None
        self.lens_model: Optional[jlens.HFLensModel] = None
        self.lens: Optional[jlens.JacobianLens] = None
        self.probe: Optional[LinearProbe] = None
        self.jspace: Optional[JSpaceMonitor] = None
        self.entropy: Optional[EntropyMonitor] = None

    def load(self) -> Straightshot:
        """Load model, lens, probe weights, and initialize all channels."""
        self.hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            self.config.model_name, torch_dtype=torch.float32
        ).to(self.config.device)
        self.hf_model.eval()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.config.model_name)
        self.lens_model = jlens.from_hf(self.hf_model, self.tokenizer)
        self.lens = jlens.JacobianLens.from_pretrained(self.config.lens_path)

        self.probe = LinearProbe(self.lens_model.d_model)
        probe_path = self.config.probe_path or os.path.join(
            os.path.dirname(__file__), "..", "models", "probe.pkl"
        )
        if os.path.exists(probe_path):
            state = torch.load(probe_path, map_location=self.config.device, weights_only=False)
            mapped = {k.replace("linear", "probe"): v for k, v in state["state_dict"].items()}
            self.probe.load_state_dict(mapped, strict=True)

        self.jspace = JSpaceMonitor(self.lens, self.lens_model, self.tokenizer)
        self.entropy = EntropyMonitor(self.config.entropy_window, self.config.entropy_threshold)
        return self

    @torch.no_grad()
    def _score(self, hidden: torch.Tensor, logits: torch.Tensor, layer: int,
               states: Optional[list] = None) -> ChannelScores:
        s = ChannelScores()
        if self.probe:
            s.probe = self.probe.score(hidden)
        if self.jspace and states:
            s.jspace = self.jspace.proximity(states, k=self.config.subspace_top_k)
        if self.entropy:
            s.entropy = self.entropy.anomaly(self.entropy.entropy(logits))
        return s

    def monitor(self, prompt: str, max_tokens: int = 50) -> Iterator[tuple[str, ChannelScores]]:
        """Generate tokens from *prompt* and yield each with per-token safety scores."""
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.config.device)
        probe = self.config.probe_layer
        strategic = [8, 12, 16, 20, 24, 26]

        for _ in range(max_tokens):
            out = self.hf_model(ids, output_hidden_states=True, return_dict=True)
            logits = out.logits[:, -1, :]
            all_h = out.hidden_states

            hs_list = [(l, all_h[l][:, -1:, :]) for l in strategic if l < len(all_h)]
            actual = probe if probe >= 0 else self.lens_model.n_layers + probe
            hidden = all_h[actual][:, -1:, :]
            scores = self._score(hidden, logits, actual, hs_list)

            token = self.tokenizer.decode([logits.argmax(-1).item()], skip_special_tokens=True)
            yield token, scores
            ids = torch.cat([ids, logits.argmax(-1, keepdim=True)], dim=-1)

            if scores.verdict == "RED" and self.config.intervention_mode == "halt":
                yield " [HALTED]", scores
                break

    def benchmark(self, prompts: list[str], max_tokens: int = 15) -> dict:
        """Run multiple prompts and collect timing + verdict statistics."""
        results = {"latencies": [], "verdicts": {"GREEN": 0, "YELLOW": 0, "RED": 0}}
        for prompt in prompts:
            start = time.perf_counter()
            tokens = list(self.monitor(prompt, max_tokens=max_tokens))
            elapsed = time.perf_counter() - start
            results["latencies"].append(elapsed)
            final = tokens[-1][1].verdict if tokens else "GREEN"
            results["verdicts"][final] += 1
        return results
