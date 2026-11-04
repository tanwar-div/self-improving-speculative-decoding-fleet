"""Precision study backing the correctness claim.

Speculative decoding with greedy verification is output-identical to greedy
decoding *in exact arithmetic*. In fp16 that identity can break, and this
script establishes exactly why, with three measurements:

  A. fp32 speculative vs fp32 greedy
        -> expected: 100% identical. Shows the verification logic is correct.

  B. fp16 speculative vs fp16 greedy
        -> some divergence. For every divergence, record the target's top-2
           logit gap at that position and compare it to one fp16 ULP.

  C. fp16 greedy-with-padding vs fp16 greedy  (NO SPECULATION AT ALL)
        -> each forward pass carries `gamma` extra copies of the last token
           whose logits are discarded: mathematically a no-op that changes only
           the matmul shape. Divergence here is caused by fp16 alone.

C is the control that matters: if plain greedy decoding disagrees with itself
when only the batch shape changes, then divergence in B is a property of fp16
inference rather than of speculation, and the correctness guarantee should be
stated as "exact in fp32; in fp16, exact up to argmax ties finer than fp16 can
represent".

Run one dtype per process -- a 4GB GPU cannot hold two copies of the target.
    PYTHONPATH=src .venv/bin/python scripts/precision_study.py --dtype float32
    PYTHONPATH=src .venv/bin/python scripts/precision_study.py --dtype float16
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import DynamicCache

from fleet.bench.prompts import PRECISION_STUDY_PROMPTS, build_pool, stratified_sample
from fleet.specdec.drafters.base import NullDrafter
from fleet.specdec.drafters.ngram import NgramDrafter
from fleet.specdec.engine import SpeculativeEngine
from fleet.specdec.target import TargetModel


def fp16_ulp(x: float) -> float:
    """Spacing between representable fp16 values at magnitude |x|."""
    if x == 0:
        return 2.0**-24
    exp = math.floor(math.log2(abs(x)))
    return 2.0 ** (exp - 10)


def logits_at(target: TargetModel, context: list[int]) -> torch.Tensor:
    """Target logits for the token following `context`, via the incremental
    single-token cache path -- i.e. what plain greedy decoding computes."""
    with torch.inference_mode():
        cache = DynamicCache()
        out = target.model(
            input_ids=torch.tensor([context], device=target.device),
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.arange(len(context), device=target.device),
        )
        return out.logits[0, -1].float()


def greedy_padded(target: TargetModel, prompt_ids: list[int], pad: int, max_new: int) -> list[int]:
    """Plain greedy decoding, except every forward pass carries `pad` extra
    copies of the last committed token whose logits are thrown away.

    No speculation: the emitted token is always the target's own argmax. Only
    the shape of the matmuls differs from ordinary greedy decoding.
    """
    committed = list(prompt_ids)
    cache = DynamicCache()
    recorder = getattr(cache, "activate_past_recording", None)
    if callable(recorder):
        recorder()
    with torch.inference_mode():
        out = target.model(
            input_ids=torch.tensor([committed], device=target.device),
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.arange(len(committed), device=target.device),
        )
        cache = out.past_key_values
        gen = [int(out.logits[0, -1].argmax())]
        committed.append(gen[-1])

        while len(gen) < max_new and gen[-1] not in target.eos_ids:
            offset = len(committed) - 1
            if cache.get_seq_length() > offset:
                cache.crop(offset - cache.get_seq_length())
            step = [committed[-1]] * (1 + pad)
            out = target.model(
                input_ids=torch.tensor([step], device=target.device),
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.arange(offset, offset + len(step), device=target.device),
            )
            cache = out.past_key_values
            nxt = int(out.logits[0, 0].argmax())  # only the first slot is real
            gen.append(nxt)
            committed.append(nxt)
    return gen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--out", default="runs/precision_study")
    ap.add_argument(
        "--n",
        type=int,
        default=160,
        help="prompts drawn from the multi-domain pool; 0 uses the small built-in set",
    )
    ap.add_argument("--pool-total", type=int, default=40_000)
    args = ap.parse_args()

    target = TargetModel.load(args.model, device=args.device, dtype=args.dtype)
    engine = SpeculativeEngine(target, NullDrafter(), gamma=args.gamma)

    # Draw from the real multi-domain pool so the divergence rate is measured
    # over genuinely varied traffic rather than a handful of hand-written
    # prompts. Falls back to the built-in set when the pool is unavailable.
    domains: list[str] = []
    if args.n > 0:
        try:
            pool, _ = build_pool(total=args.pool_total)
            sampled = stratified_sample(pool, args.n, seed=31)
            prompts = [p.text for p in sampled]
            domains = [p.domain for p in sampled]
        except Exception as exc:  # noqa: BLE001
            print(f"prompt pool unavailable ({exc}); using the built-in set")
            prompts = PRECISION_STUDY_PROMPTS
    else:
        prompts = PRECISION_STUDY_PROMPTS

    split_at = max(1, len(prompts) // 4)
    train, evalset = prompts[:split_at], prompts[split_at:]
    eval_domains = domains[split_at:] if domains else ["builtin"] * len(evalset)

    seqs = []
    for p in train:
        ids = target.encode_chat(p)
        seqs.append(
            ids + engine.greedy_reference(ids, max_new_tokens=args.max_new_tokens).generated_ids
        )
    drafter = NgramDrafter.fit(seqs, order=3, min_count=1, min_confidence=0.15)

    spec_match = 0
    pad_match = 0
    total_tokens = 0
    divergences: list[dict] = []
    pad_divergences: list[dict] = []

    for p_idx, p in enumerate(evalset):
        ids = target.encode_chat(p)
        ref = engine.greedy_reference(ids, max_new_tokens=args.max_new_tokens).generated_ids
        spec = engine.decode(
            ids, max_new_tokens=args.max_new_tokens, gamma=args.gamma, drafter=drafter
        ).generated_ids
        padded = greedy_padded(target, ids, args.gamma, args.max_new_tokens)
        total_tokens += len(ref)

        if spec == ref:
            spec_match += 1
        else:
            i = next((k for k in range(min(len(spec), len(ref))) if spec[k] != ref[k]), -1)
            lg = logits_at(target, ids + ref[:i])
            top2 = torch.topk(lg, 2).values.tolist()
            gap = top2[0] - top2[1]
            ulp = fp16_ulp(top2[0])
            divergences.append(
                {
                    "prompt": p[:70],
                    "domain": eval_domains[p_idx],
                    "index": i,
                    "tokens_generated": len(ref),
                    "top1_logit": round(top2[0], 6),
                    "top2_gap": round(gap, 8),
                    "fp16_ulp_at_that_magnitude": round(ulp, 8),
                    "gap_in_ulps": round(gap / ulp, 3) if ulp else None,
                    "tie_below_fp16_resolution": gap <= ulp,
                }
            )

        if padded == ref:
            pad_match += 1
        else:
            i = next((k for k in range(min(len(padded), len(ref))) if padded[k] != ref[k]), -1)
            pad_divergences.append({"prompt": p[:70], "index": i})

    n = len(evalset)
    result = {
        "dtype": args.dtype,
        "device": args.device,
        "model": args.model,
        "gamma": args.gamma,
        "max_new_tokens": args.max_new_tokens,
        "eval_prompts": n,
        "train_prompts": len(train),
        "total_reference_tokens": total_tokens,
        "token_level_divergence_rate": (
            round(len(divergences) / total_tokens, 8) if total_tokens else None
        ),
        "A_or_B_speculative_vs_greedy": {
            "prompts_identical": spec_match,
            "prompts_diverged": n - spec_match,
            "sequence_match_rate": round(spec_match / n, 4),
            "divergences": divergences,
            "all_divergences_are_subulp_ties": all(
                d["tie_below_fp16_resolution"] for d in divergences
            ),
        },
        "C_control_padded_greedy_vs_greedy_no_speculation": {
            "prompts_identical": pad_match,
            "prompts_diverged": n - pad_match,
            "sequence_match_rate": round(pad_match / n, 4),
            "divergences": pad_divergences,
            "note": (
                "No speculation involved. Each forward pass carries gamma extra "
                "copies of the last token whose logits are discarded, so only the "
                "matmul shape differs from plain greedy decoding."
            ),
        },
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.dtype}.json"
    path.write_text(json.dumps(result, indent=2))

    print(f"\n=== precision study: {args.dtype} on {args.device} ===")
    print(
        f"prompts: {n} eval ({len(train)} used to fit the drafter), {total_tokens} reference tokens"
    )
    print(f"speculative vs greedy       : {spec_match}/{n} sequences identical")
    print(
        f"padded greedy vs greedy     : {pad_match}/{n} sequences identical   "
        f"(NO speculation -- batch shape change only)"
    )
    if divergences:
        print("\nspeculative divergences (target top-2 logit gap at the divergent position):")
        for d in divergences:
            print(
                f"  idx {d['index']:>3}  gap={d['top2_gap']:.6f}  "
                f"fp16 ULP={d['fp16_ulp_at_that_magnitude']:.6f}  "
                f"({d['gap_in_ulps']} ULPs)  sub-ULP tie={d['tie_below_fp16_resolution']}"
            )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
