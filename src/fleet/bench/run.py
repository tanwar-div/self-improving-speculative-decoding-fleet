"""The benchmark experiments.

E1 -- Throughput, run on two different workloads, because which one you pick
      determines the answer and only one of them resembles production.

      E1a NON-STATIONARY (the real case). The serving mix moves through a
          sequence of domain regimes, as production traffic does. Arms:
            no_speculation : target-only greedy; the correctness oracle
            static_initial : draft model fitted once on the traffic available
                             at deploy time (the first regime), then frozen --
                             what "train a drafter and ship it" actually gets
                             you a week later
            static_broad   : draft model fitted once on a stratified sample of
                             *every* domain, including ones not yet seen in
                             production. A deliberately strong static baseline:
                             it is handed data the self-improving arm has to
                             earn from live traffic
            self_improving : cold start, retrained from its own live traffic,
                             promoted through the SLO gate, hot-swapped
          Headline figure: self_improving / static_broad.

      E1b STATIONARY (control). One fixed stratified mix throughout. Here a
          static drafter trained in-distribution is close to optimal and the
          self-improving arm cannot do better than tie -- it pays a cold-start
          cost and catches up. Reported to show the loop does not *regress*
          throughput when there is nothing to adapt to. Quoting E1b as the
          headline would understate the mechanism; quoting E1a without E1b
          would hide its cost.

E2 -- Workload-shift recovery. The serving mix changes abruptly from one set of
      domains to a disjoint set, and two retraining *policies* race on the
      identical prompt stream:
        periodic        : retrain every N requests regardless of the workload
        drift_triggered : retrain when the drift detector says the distribution
                          moved
      The only difference between the two is *when* retraining fires -- same
      trainer, same promotion machinery, same `require_improvement` setting --
      so the comparison isolates trigger timing rather than training quality.

      Recovery is measured against an oracle: a drafter trained directly on
      post-shift traffic, serving the post-shift stream. "Recovered" means a
      policy's rolling throughput reached `--recovery-frac` of that oracle and
      stayed there. Measuring against pre-shift throughput instead would be
      meaningless, because the new domains have their own intrinsic speed.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from fleet import db
from fleet.bench import harness
from fleet.bench import recovery as recovery_metric
from fleet.bench.prompts import Prompt, build_pool, stratified_sample
from fleet.config import REPO_ROOT, settings
from fleet.serving.hotswap import DeploymentWatcher
from fleet.specdec.engine import SpeculativeEngine
from fleet.specdec.target import TargetModel

log = logging.getLogger(__name__)

PRE_SHIFT_DOMAINS = ("code", "chat")
POST_SHIFT_DOMAINS = ("medical", "legal")


# ---------------------------------------------------------------------------
# E1: throughput
# ---------------------------------------------------------------------------


REGIMES: tuple[tuple[str, ...], ...] = (
    ("code", "chat"),
    ("summarization",),
    ("medical", "legal"),
    ("math", "code"),
)


def build_regime_stream(
    pool: list[Prompt], regimes: tuple[tuple[str, ...], ...], per_regime: int, *, offset: int = 0
) -> tuple[list[Prompt], list[int]]:
    """A stream that walks through domain regimes, with the boundaries returned
    so the report can line throughput up against the shifts."""
    by_domain: dict[str, list[Prompt]] = {}
    for p in pool:
        by_domain.setdefault(p.domain, []).append(p)

    cursor = dict.fromkeys(by_domain, offset)
    stream: list[Prompt] = []
    boundaries: list[int] = []
    for regime in regimes:
        boundaries.append(len(stream))
        picked: list[Prompt] = []
        per_domain = per_regime // len(regime) + 1
        for d in regime:
            items = by_domain.get(d, [])
            start = cursor.get(d, 0)
            chunk = items[start : start + per_domain]
            cursor[d] = start + len(chunk)
            picked.extend(chunk)
        stream.extend(picked[:per_regime])
    return stream, boundaries


def _run_arm(
    engine: SpeculativeEngine,
    stream: list[Prompt],
    *,
    name: str,
    description: str,
    resolver,
    max_new_tokens: int,
    audit_every: int,
    write_telemetry: bool,
    trainer: harness.BackgroundTrainer | None = None,
) -> harness.ArmResult:
    arm = harness.ArmResult(name, description)
    if trainer is not None:
        trainer.start()
    try:
        harness.serve_stream(
            engine,
            stream,
            arm=arm,
            drafter_for=resolver,
            max_new_tokens=max_new_tokens,
            audit_every=audit_every,
            write_telemetry=write_telemetry,
        )
    finally:
        if trainer is not None:
            trainer.stop()
            trainer.join(timeout=60)
            arm.train_wall_s = trainer.train_wall_s
            arm.retrains = trainer.retrains
            arm.promotions = trainer.promotions
            arm.rollbacks = trainer.rollbacks
    log.info(
        "%s: %.1f tok/s, %.3f tok/forward, accept %s, versions %s",
        name,
        arm.tokens_per_s,
        arm.tokens_per_target_forward,
        f"{arm.acceptance_rate:.3f}" if arm.acceptance_rate else "n/a",
        sorted(set(arm.versions_served)),
    )
    return arm


def _steady_state_tps(arm: harness.ArmResult, skip: int) -> float:
    """Throughput excluding the first `skip` requests.

    Reported alongside the full-stream figure because the self-improving arm
    starts cold by construction: the full-stream number includes its warm-up
    cost, and the steady-state number says what it converges to. Both belong in
    the result; only one of them is the honest headline, and it is the full one.
    """
    tail = arm.metrics[skip:]
    if not tail:
        return 0.0
    total_tokens = sum(m.generated_tokens for m in tail)
    total_ms = sum(m.total_ms for m in tail)
    return total_tokens / (total_ms / 1000.0) if total_ms else 0.0


def run_throughput_workload(
    engine: SpeculativeEngine,
    *,
    label: str,
    stream: list[Prompt],
    boundaries: list[int],
    initial_warmup: list[Prompt],
    broad_warmup: list[Prompt],
    max_new_tokens: int,
    audit_every: int,
    steady_skip: int,
) -> dict:
    """Run all four arms over one workload."""
    arms: dict[str, harness.ArmResult] = {}

    # --- no speculation ---------------------------------------------------
    harness.reset_registry()
    version, _ = harness.bootstrap_cold()
    arms["no_speculation"] = _run_arm(
        engine,
        stream,
        name="no_speculation",
        description="target-only greedy decoding; no draft model",
        resolver=harness.null_drafter_resolver(version),
        max_new_tokens=max_new_tokens,
        audit_every=0,  # it is the reference; auditing it against itself is vacuous
        write_telemetry=False,
    )

    # --- static, fitted on deploy-time traffic only ------------------------
    harness.reset_registry()
    t0 = time.perf_counter()
    sv, static_initial = harness.train_static_baseline(
        initial_warmup, engine, max_new_tokens=max_new_tokens
    )
    initial_train_s = time.perf_counter() - t0
    arms["static_initial"] = _run_arm(
        engine,
        stream,
        name="static_initial",
        description=(
            f"fitted once on {len(initial_warmup)} prompts from the traffic available "
            "at deploy time, then frozen"
        ),
        resolver=harness.fixed_drafter_resolver(sv, static_initial),
        max_new_tokens=max_new_tokens,
        audit_every=audit_every,
        write_telemetry=False,
    )
    arms["static_initial"].train_wall_s = initial_train_s

    # --- static, fitted on every domain (strong baseline) ------------------
    harness.reset_registry()
    t0 = time.perf_counter()
    bv, static_broad = harness.train_static_baseline(
        broad_warmup, engine, max_new_tokens=max_new_tokens
    )
    broad_train_s = time.perf_counter() - t0
    arms["static_broad"] = _run_arm(
        engine,
        stream,
        name="static_broad",
        description=(
            f"fitted once on {len(broad_warmup)} prompts stratified across every domain, "
            "including domains not yet seen in production; frozen"
        ),
        resolver=harness.fixed_drafter_resolver(bv, static_broad),
        max_new_tokens=max_new_tokens,
        audit_every=audit_every,
        write_telemetry=False,
    )
    arms["static_broad"].train_wall_s = broad_train_s

    # --- the self-improving fleet -----------------------------------------
    harness.reset_registry()
    harness.bootstrap_cold()
    watcher = DeploymentWatcher(engine, interval=1e9)  # synced per request instead
    watcher.sync(force=True)
    arms["self_improving"] = _run_arm(
        engine,
        stream,
        name="self_improving",
        description=(
            "cold start; drafter retrained from live traffic, gated by canary SLOs, "
            "hot-swapped mid-stream"
        ),
        resolver=harness.registry_drafter_resolver(watcher),
        max_new_tokens=max_new_tokens,
        audit_every=audit_every,
        write_telemetry=True,
        trainer=harness.BackgroundTrainer(poll_s=0.5, auto_promote=True),
    )

    def tps(name: str) -> float:
        return arms[name].tokens_per_s

    base = tps("no_speculation")
    out = {
        "label": label,
        "stream_requests": len(stream),
        "regime_boundaries": boundaries,
        "steady_state_skip": steady_skip,
        "arms": {k: v.summary() for k, v in arms.items()},
        "steady_state_tokens_per_s": {
            k: round(_steady_state_tps(v, steady_skip), 3) for k, v in arms.items()
        },
        "comparisons": {
            "static_initial_vs_no_speculation": round(tps("static_initial") / base, 4)
            if base
            else None,
            "static_broad_vs_no_speculation": round(tps("static_broad") / base, 4)
            if base
            else None,
            "self_improving_vs_no_speculation": round(tps("self_improving") / base, 4)
            if base
            else None,
            "self_improving_vs_static_initial": (
                round(tps("self_improving") / tps("static_initial"), 4)
                if tps("static_initial")
                else None
            ),
            "self_improving_vs_static_broad": (
                round(tps("self_improving") / tps("static_broad"), 4)
                if tps("static_broad")
                else None
            ),
            "steady_state_self_improving_vs_static_broad": (
                round(
                    _steady_state_tps(arms["self_improving"], steady_skip)
                    / _steady_state_tps(arms["static_broad"], steady_skip),
                    4,
                )
                if _steady_state_tps(arms["static_broad"], steady_skip)
                else None
            ),
            "p95_self_improving_vs_static_broad": (
                round(arms["self_improving"].p95_ms / arms["static_broad"].p95_ms, 4)
                if arms["static_broad"].p95_ms
                else None
            ),
        },
    }
    return out


# ---------------------------------------------------------------------------
# E2: workload-shift recovery
# ---------------------------------------------------------------------------


class PeriodicTrainer(harness.BackgroundTrainer):
    """Retrains on a fixed request cadence, ignoring the workload entirely.

    The policy the drift detector is being compared against.
    """

    def __init__(self, *, every_n_requests: int, counter, **kw) -> None:
        super().__init__(**kw)
        self.every_n_requests = every_n_requests
        self.counter = counter
        self._last_trained_at = 0

    def step(self, detector) -> int | None:
        from fleet.registry import store
        from fleet.trainer.worker import train_candidate

        seen = self.counter()
        if seen - self._last_trained_at < self.every_n_requests:
            return None
        dep = store.get_deployment()
        if dep.canary_version is not None or dep.active_version is None:
            return None
        self._last_trained_at = seen
        return train_candidate(
            parent_version=dep.active_version,
            reason=f"periodic retrain at request {seen}",
            require_improvement=False,
        )


class DriftTrainer(harness.BackgroundTrainer):
    """Retrains only when the live distribution has left the training one."""

    def step(self, detector) -> int | None:
        from fleet.registry import store
        from fleet.trainer import worker

        dep = store.get_deployment()
        if dep.canary_version is not None or dep.active_version is None:
            return None
        # worker.tick() applies the detector and records the check either way.
        return worker.tick_with_policy(detector, require_improvement=False)


@dataclass
class RecoveryTrace:
    policy: str
    arm: harness.ArmResult
    shift_index: int
    rolling_tps: list[float]
    recovery_index: int | None
    retrain_events: list[int]

    def requests_to_recover(self) -> int | None:
        if self.recovery_index is None:
            return None
        return self.recovery_index - self.shift_index


def rolling_fleet_tps(metrics: list, window: int) -> list[float]:
    """Rolling throughput as tokens / elapsed time over the window.

    Aggregated the same way as the oracle target it is compared against
    (total tokens / total time), not as a mean of per-request rates. Those two
    are not the same number, and mixing them would bias the recovery point by
    over-weighting short requests.
    """
    out: list[float] = []
    dq: collections.deque = collections.deque()
    tokens = 0
    ms = 0.0
    for m in metrics:
        dq.append(m)
        tokens += m.generated_tokens
        ms += m.total_ms
        if len(dq) > window:
            old = dq.popleft()
            tokens -= old.generated_tokens
            ms -= old.total_ms
        out.append(tokens / (ms / 1000.0) if ms > 0 else 0.0)
    return out


def find_recovery(
    rolling_tps: list[float], *, shift_index: int, window: int, target: float, sustain: int
) -> int | None:
    """First index where the rolling throughput reaches `target` and stays
    there for `sustain` consecutive requests.

    The search starts at `shift_index + window`, not at the shift itself: until
    the window has fully rolled past the boundary it still contains pre-shift
    requests, and with pre-shift throughput typically above the post-shift
    target that would report instant recovery for every policy.
    """
    n = len(rolling_tps)
    start = min(shift_index + window, n)
    for i in range(start, n):
        if i + sustain > n:
            break
        if all(rolling_tps[j] >= target for j in range(i, i + sustain)):
            return i
    return None


def experiment_recovery(
    engine: SpeculativeEngine,
    *,
    pool: list[Prompt],
    pre_n: int,
    post_n: int,
    oracle_train_n: int,
    max_new_tokens: int,
    window: int,
    recovery_frac: float,
    sustain: int,
    periodic_every: int,
    audit_every: int,
) -> dict:
    by_domain: dict[str, list[Prompt]] = {}
    for p in pool:
        by_domain.setdefault(p.domain, []).append(p)

    def take(domains, n, offset=0):
        out: list[Prompt] = []
        per = n // len(domains) + 1
        for d in domains:
            out.extend(by_domain.get(d, [])[offset : offset + per])
        return out[:n]

    pre_stream = take(PRE_SHIFT_DOMAINS, pre_n)
    post_stream = take(POST_SHIFT_DOMAINS, post_n)
    oracle_train = take(POST_SHIFT_DOMAINS, oracle_train_n, offset=post_n + 10)
    stream = pre_stream + post_stream
    shift_index = len(pre_stream)

    if len(pre_stream) < pre_n or len(post_stream) < post_n:
        log.warning(
            "prompt pool thin for the requested phases: pre=%d/%d post=%d/%d",
            len(pre_stream),
            pre_n,
            len(post_stream),
            post_n,
        )

    # --- oracle: what throughput is achievable on the post-shift workload --
    harness.reset_registry()
    ov, oracle_drafter = harness.train_static_baseline(
        oracle_train, engine, max_new_tokens=max_new_tokens
    )
    oracle_arm = harness.ArmResult(
        "oracle_post_shift",
        f"drafter trained directly on {len(oracle_train)} post-shift prompts",
    )
    harness.serve_stream(
        engine,
        post_stream,
        arm=oracle_arm,
        drafter_for=harness.fixed_drafter_resolver(ov, oracle_drafter),
        max_new_tokens=max_new_tokens,
        audit_every=audit_every,
        write_telemetry=False,
    )
    oracle_tps = oracle_arm.tokens_per_s
    target_tps = recovery_frac * oracle_tps
    log.info(
        "oracle post-shift throughput %.1f tok/s -> recovery target %.1f tok/s",
        oracle_tps,
        target_tps,
    )

    # --- the two policies on the identical stream -------------------------
    traces: list[RecoveryTrace] = []
    for policy in ("periodic", "drift_triggered"):
        harness.reset_registry()
        harness.bootstrap_cold()
        watcher = DeploymentWatcher(engine, interval=1e9)
        watcher.sync(force=True)

        served = {"n": 0}
        retrain_events: list[int] = []

        if policy == "periodic":
            trainer = PeriodicTrainer(
                every_n_requests=periodic_every,
                counter=lambda served=served: served["n"],
                poll_s=0.4,
                auto_promote=True,
            )
        else:
            trainer = DriftTrainer(poll_s=0.4, auto_promote=True)

        last_retrains = {"n": 0}

        # Default-bound rather than captured: the closure is only used inside
        # this iteration, but binding explicitly keeps it correct if the loop
        # ever starts overlapping policies.
        def on_request(
            i: int,
            _metric,
            *,
            served=served,
            trainer=trainer,
            last_retrains=last_retrains,
            retrain_events=retrain_events,
        ) -> None:
            served["n"] = i + 1
            if trainer.retrains != last_retrains["n"]:
                last_retrains["n"] = trainer.retrains
                retrain_events.append(i)

        arm = harness.ArmResult(policy, f"{policy} retraining policy")
        trainer.start()
        try:
            harness.serve_stream(
                engine,
                stream,
                arm=arm,
                drafter_for=harness.registry_drafter_resolver(watcher),
                max_new_tokens=max_new_tokens,
                audit_every=audit_every,
                write_telemetry=True,
                on_request=on_request,
            )
        finally:
            trainer.stop()
            trainer.join(timeout=30)

        arm.retrains = trainer.retrains
        arm.promotions = trainer.promotions
        arm.rollbacks = trainer.rollbacks
        arm.train_wall_s = trainer.train_wall_s

        roll = rolling_fleet_tps(arm.metrics, window)
        rec = find_recovery(
            roll,
            shift_index=shift_index,
            window=window,
            target=target_tps,
            sustain=sustain,
        )
        traces.append(
            RecoveryTrace(
                policy=policy,
                arm=arm,
                shift_index=shift_index,
                rolling_tps=[round(x, 3) for x in roll],
                recovery_index=rec,
                retrain_events=retrain_events,
            )
        )
        log.info(
            "%s: recovered after %s requests (%d retrains)",
            policy,
            rec - shift_index if rec is not None else "never",
            trainer.retrains,
        )

    out = {
        "setup": {
            "pre_shift_domains": list(PRE_SHIFT_DOMAINS),
            "post_shift_domains": list(POST_SHIFT_DOMAINS),
            "pre_shift_requests": len(pre_stream),
            "post_shift_requests": len(post_stream),
            "shift_index": shift_index,
            "rolling_window": window,
            "recovery_frac_of_oracle": recovery_frac,
            "sustain_requests": sustain,
            "periodic_every_n_requests": periodic_every,
            "oracle_train_prompts": len(oracle_train),
        },
        "oracle": oracle_arm.summary(),
        "oracle_tokens_per_s": round(oracle_tps, 3),
        "recovery_target_tokens_per_s": round(target_tps, 3),
        "policies": {},
    }
    for t in traces:
        out["policies"][t.policy] = {
            **t.arm.summary(),
            "requests_to_recover": t.requests_to_recover(),
            "retrain_triggered_at_requests": t.retrain_events,
            "rolling_tokens_per_s": t.rolling_tps,
        }

    # Recovery against a reachable, policy-independent target. The oracle
    # comparison above is kept for context but cannot be met by a live adaptive
    # system -- see fleet/bench/recovery.py.
    out["recovery_vs_shared_plateau"] = recovery_metric.analyse(
        out["policies"],
        shift_index=shift_index,
        window=window,
        recovery_frac=recovery_frac,
        sustain=sustain,
        oracle_tokens_per_s=oracle_tps,
    )

    periodic = out["policies"].get("periodic", {}).get("requests_to_recover")
    drift = out["policies"].get("drift_triggered", {}).get("requests_to_recover")
    out["comparison"] = {
        "periodic_requests_to_recover": periodic,
        "drift_requests_to_recover": drift,
        "drift_faster_by_fraction": (
            round((periodic - drift) / periodic, 4) if periodic and drift and periodic > 0 else None
        ),
        "periodic_retrains": out["policies"].get("periodic", {}).get("retrains"),
        "drift_retrains": out["policies"].get("drift_triggered", {}).get("retrains"),
    }
    return out


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default=None, help="defaults to FLEET_TARGET_MODEL")
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--pool-total", type=int, default=40_000)
    ap.add_argument("--decode-n", type=int, default=1200, help="prompts decoded per throughput arm")
    ap.add_argument(
        "--warmup-n", type=int, default=300, help="prompts used to fit the static baseline"
    )
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--gamma", type=int, default=None)
    ap.add_argument("--audit-every", type=int, default=20)
    # recovery experiment
    ap.add_argument("--pre-n", type=int, default=240)
    ap.add_argument("--post-n", type=int, default=360)
    ap.add_argument("--oracle-train-n", type=int, default=120)
    ap.add_argument("--window", type=int, default=25)
    ap.add_argument("--recovery-frac", type=float, default=0.95)
    ap.add_argument("--sustain", type=int, default=15)
    ap.add_argument("--periodic-every", type=int, default=150)
    ap.add_argument(
        "--skip",
        default="",
        help="comma-separated subset of: nonstationary,stationary,recovery",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    db.wait_for_db()
    db.apply_schema()

    pool, manifest = build_pool(total=args.pool_total)
    log.info("prompt pool: %d prompts, fully_real=%s", len(pool), manifest["fully_real"])

    target = TargetModel.load(args.model, device=args.device, dtype=args.dtype)
    engine = SpeculativeEngine(target, gamma=args.gamma or settings.gamma)
    harness.warm_up(engine)

    # --- build the two workloads -----------------------------------------
    per_regime = max(1, args.decode_n // len(REGIMES))
    nonstationary_stream, boundaries = build_regime_stream(pool, REGIMES, per_regime)

    # The static_initial baseline sees only what production had at deploy time:
    # traffic from the first regime.
    initial_warmup, _ = build_regime_stream(
        pool, (REGIMES[0],), args.warmup_n, offset=per_regime + 50
    )
    # The static_broad baseline is handed a stratified sample of every domain,
    # including the ones the fleet has not served yet.
    broad_warmup = stratified_sample(pool, args.warmup_n, seed=99)

    stationary_stream = stratified_sample(pool, len(nonstationary_stream), seed=7)

    started = datetime.now(UTC)
    report: dict = {
        "started_at": started.isoformat(),
        "config": {
            "model": target.name,
            "device": args.device,
            "dtype": str(target.dtype),
            "gamma": engine.gamma,
            "max_new_tokens": args.max_new_tokens,
            "decode_prompts_per_arm": len(nonstationary_stream),
            "warmup_prompts": args.warmup_n,
            "regimes": [list(r) for r in REGIMES],
            "audit_every": args.audit_every,
            "drafter_kind": settings.drafter_kind,
            "canary_pct": settings.canary_pct,
            "canary_min_requests": settings.canary_min_requests,
            "slo_min_tps_ratio": settings.slo_min_tps_ratio,
            "slo_max_p95_ratio": settings.slo_max_p95_ratio,
            "drift_threshold": settings.drift_threshold,
        },
        "prompt_pool": manifest,
    }

    steady_skip = max(1, len(nonstationary_stream) // 5)

    if "nonstationary" not in skip:
        log.info("=== E1a: throughput on a non-stationary workload ===")
        report["throughput_nonstationary"] = run_throughput_workload(
            engine,
            label="non-stationary (regime sequence)",
            stream=nonstationary_stream,
            boundaries=boundaries,
            initial_warmup=initial_warmup,
            broad_warmup=broad_warmup,
            max_new_tokens=args.max_new_tokens,
            audit_every=args.audit_every,
            steady_skip=steady_skip,
        )

    if "stationary" not in skip:
        log.info("=== E1b: throughput on a stationary workload (control) ===")
        report["throughput_stationary"] = run_throughput_workload(
            engine,
            label="stationary (fixed stratified mix)",
            stream=stationary_stream,
            boundaries=[0],
            initial_warmup=broad_warmup,
            broad_warmup=broad_warmup,
            max_new_tokens=args.max_new_tokens,
            audit_every=args.audit_every,
            steady_skip=steady_skip,
        )

    if "recovery" not in skip:
        log.info("=== E2: workload-shift recovery ===")
        report["recovery"] = experiment_recovery(
            engine,
            pool=pool,
            pre_n=args.pre_n,
            post_n=args.post_n,
            oracle_train_n=args.oracle_train_n,
            max_new_tokens=args.max_new_tokens,
            window=args.window,
            recovery_frac=args.recovery_frac,
            sustain=args.sustain,
            periodic_every=args.periodic_every,
            audit_every=args.audit_every,
        )

    report["finished_at"] = datetime.now(UTC).isoformat()
    report["duration_s"] = round((datetime.now(UTC) - started).total_seconds(), 1)

    out_dir = Path(args.out) if args.out else REPO_ROOT / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"bench-{stamp}.json"
    path.write_text(json.dumps(report, indent=2))

    from fleet.bench.report import render

    print("\n" + render(report))
    latest = out_dir / "latest.json"
    latest.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {path}\nwrote {latest}")


if __name__ == "__main__":
    main()
