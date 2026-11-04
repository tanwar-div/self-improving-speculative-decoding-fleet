"""Quick reality check on a real model: does speculation actually speed this up,
and is the output still byte-identical to greedy?

Not the benchmark -- just a fast signal before wiring the serving fleet.
Usage: PYTHONPATH=src .venv/bin/python scripts/smoke_specdec.py [--device cuda]
"""

from __future__ import annotations

import argparse
import statistics
import time

from fleet.specdec.drafters.base import NullDrafter
from fleet.specdec.drafters.ngram import NgramDrafter
from fleet.specdec.engine import SpeculativeEngine
from fleet.specdec.target import TargetModel

TRAIN_PROMPTS = [
    "Write a Python function that reverses a linked list.",
    "Write a Python function to check whether a string is a palindrome.",
    "Write a Python function that merges two sorted lists.",
    "Explain what a hash table is in two sentences.",
    "Summarize: Speculative decoding drafts several tokens then verifies them.",
    "What is the time complexity of binary search?",
]

EVAL_PROMPTS = [
    "Write a Python function that reverses a string.",
    "Write a Python function to compute the nth Fibonacci number.",
    "Explain what a binary search tree is in two sentences.",
    "Summarize: speculative decoding verifies drafted tokens against the target model.",
    "What is the time complexity of inserting into a hash table?",
    "Write a Python function that sorts a list of dictionaries by a key.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--gamma", type=int, default=4)
    args = ap.parse_args()

    t0 = time.perf_counter()
    target = TargetModel.load(args.model, device=args.device)
    print(f"loaded {args.model} on {args.device} in {time.perf_counter() - t0:.1f}s")

    engine = SpeculativeEngine(target, NullDrafter(), gamma=args.gamma)

    # Train the drafter the way the production trainer does: on the target's
    # own greedy continuations of real prompts.
    print("\nharvesting training sequences (target-only greedy)...")
    seqs = []
    for p in TRAIN_PROMPTS:
        ids = target.encode_chat(p)
        res = engine.greedy_reference(ids, max_new_tokens=args.max_new_tokens)
        seqs.append(ids + res.generated_ids)
    t_fit = time.perf_counter()
    drafter = NgramDrafter.fit(seqs, order=3, min_count=1, min_confidence=0.15)
    print(
        f"fit drafter in {(time.perf_counter() - t_fit) * 1000:.0f}ms, "
        f"{len(drafter.table)} table entries"
    )

    # Warmup so the first CUDA kernels don't land in the measurement.
    engine.decode(target.encode_chat("hello"), max_new_tokens=8, drafter=drafter)

    print(
        "\n{:<52} {:>9} {:>9} {:>7} {:>6}".format(
            "prompt", "greedy t/s", "spec t/s", "accept", "same"
        )
    )
    print("-" * 90)
    greedy_tps, spec_tps, ratios = [], [], []
    all_match = True
    for p in EVAL_PROMPTS:
        ids = target.encode_chat(p)
        ref = engine.greedy_reference(ids, max_new_tokens=args.max_new_tokens)
        spec = engine.decode(
            ids, max_new_tokens=args.max_new_tokens, gamma=args.gamma, drafter=drafter
        )
        match = spec.generated_ids == ref.generated_ids
        all_match &= match
        greedy_tps.append(ref.tokens_per_s)
        spec_tps.append(spec.tokens_per_s)
        ratios.append(spec.tokens_per_s / ref.tokens_per_s)
        acc = spec.acceptance_rate
        print(
            "{:<52} {:>9.1f} {:>9.1f} {:>7} {:>6}".format(
                p[:50],
                ref.tokens_per_s,
                spec.tokens_per_s,
                f"{acc:.0%}" if acc is not None else "-",
                "yes" if match else "NO",
            )
        )

    print("-" * 90)
    print(f"mean greedy : {statistics.mean(greedy_tps):.1f} tok/s")
    print(f"mean spec   : {statistics.mean(spec_tps):.1f} tok/s")
    print(
        f"speedup     : {statistics.mean(spec_tps) / statistics.mean(greedy_tps):.3f}x "
        f"(per-prompt mean {statistics.mean(ratios):.3f}x)"
    )
    print(f"greedy-identical outputs: {'ALL' if all_match else 'MISMATCH DETECTED'}")
    print(f"drafter proposal sources: {dict(drafter.stats())}")


if __name__ == "__main__":
    main()
