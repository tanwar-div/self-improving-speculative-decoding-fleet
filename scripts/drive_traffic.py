"""Drive traffic at the running fleet.

Sends prompts from the multi-domain pool through the router, optionally walking
the workload through domain regimes so the drift detector fires and the whole
loop -- retrain, canary, SLO gate, promote or roll back, hot-swap -- can be
watched happening. Prints the fleet's own view of itself at the end.

    PYTHONPATH=src .venv/bin/python scripts/drive_traffic.py --requests 400
    PYTHONPATH=src .venv/bin/python scripts/drive_traffic.py --shift --requests 600
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import time

import httpx

from fleet.bench.prompts import build_pool, stratified_sample

REGIMES = (("code", "chat"), ("medical", "legal"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--pool-total", type=int, default=40_000)
    ap.add_argument(
        "--shift",
        action="store_true",
        help="walk the workload from code+chat to medical+legal halfway through",
    )
    args = ap.parse_args()

    pool, _ = build_pool(total=args.pool_total)
    if args.shift:
        by_domain: dict[str, list] = {}
        for p in pool:
            by_domain.setdefault(p.domain, []).append(p)
        half = args.requests // 2
        stream = []
        for regime, n in zip(REGIMES, (half, args.requests - half), strict=True):
            per = n // len(regime) + 1
            picked = []
            for d in regime:
                picked.extend(by_domain.get(d, [])[:per])
            stream.extend(picked[:n])
    else:
        stream = stratified_sample(pool, args.requests, seed=5)

    print(
        f"driving {len(stream)} requests at {args.url} "
        f"(concurrency {args.concurrency}, shift={args.shift})"
    )

    latencies: list[float] = []
    tps: list[float] = []
    by_version: collections.Counter = collections.Counter()
    canary_hits = 0
    failures = 0
    started = time.perf_counter()

    with httpx.Client(timeout=180.0) as client:
        for i, prompt in enumerate(stream):
            try:
                r = client.post(
                    f"{args.url}/generate",
                    json={
                        "prompt": prompt.text,
                        "domain": prompt.domain,
                        "max_new_tokens": args.max_new_tokens,
                    },
                )
                r.raise_for_status()
                body = r.json()
            except Exception as exc:  # noqa: BLE001
                failures += 1
                if failures <= 5:
                    print(f"  request {i} failed: {exc}")
                continue

            latencies.append(body["total_ms"])
            tps.append(body["tokens_per_s"])
            by_version[body["draft_version"]] += 1
            canary_hits += bool(body["is_canary"])

            if (i + 1) % 50 == 0:
                recent = statistics.fmean(tps[-50:])
                print(
                    f"  {i + 1:>5}/{len(stream)}  {recent:6.1f} tok/s  "
                    f"versions {dict(by_version)}  canary {canary_hits}"
                )

    wall = time.perf_counter() - started
    print(f"\n{len(latencies)} ok, {failures} failed in {wall:.1f}s")
    if latencies:
        ordered = sorted(latencies)
        p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
        print(
            f"mean {statistics.fmean(tps):.1f} tok/s   "
            f"p50 {statistics.median(ordered):.0f}ms   p95 {p95:.0f}ms"
        )
        print(f"requests per draft version: {dict(by_version)}")
        print(f"served by a canary version: {canary_hits}")

    try:
        with httpx.Client(timeout=30.0) as client:
            status = client.get(f"{args.url}/fleet/status").json()
        print("\n--- fleet status ---")
        print(json.dumps(status["registry"], indent=2))
        print("versions:", json.dumps(status["versions"], indent=2))
        print("recent deployment events:")
        for e in status["history"][:12]:
            print(
                f"  {e['ts']}  {e['action']:<14} "
                f"v{e.get('from_version')}->v{e.get('to_version')}  {e.get('reason')}"
            )
    except Exception as exc:  # noqa: BLE001
        print(f"could not read fleet status: {exc}")


if __name__ == "__main__":
    main()
