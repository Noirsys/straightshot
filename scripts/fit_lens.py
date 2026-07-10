#!/usr/bin/env python3
"""
Fit a Jacobian lens on any HuggingFace decoder.

Requires: pip install jlens jspace datasets

Usage:
    python scripts/fit_lens.py Qwen/Qwen3-4B --n-prompts 100 --out lenses/
"""

import argparse, sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="HuggingFace model ID or local path")
    parser.add_argument("--n-prompts", type=int, default=100)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--out", default="lenses/")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    try:
        from jlens.examples import load_wikitext_prompts
        import jlens, transformers, torch
    except ImportError:
        print("Install dependencies: pip install jlens datasets")
        sys.exit(1)

    print(f"Fitting lens on {args.model}...")

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32
    ).to(args.device or "cpu")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    wrapped = jlens.from_hf(model, tokenizer)

    prompts = load_wikitext_prompts(args.n_prompts)

    fitter = jlens.LensFitter(wrapped)
    lens = fitter.fit(prompts, max_seq_len=args.max_seq_len)

    import os
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "jacobian_lens.pt")
    lens.save(path)
    print(f"Lens saved to {path}")


if __name__ == "__main__":
    main()
