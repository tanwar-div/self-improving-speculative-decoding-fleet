"""Shared machinery for the benchmark experiments.

Both experiments drive the *real* control plane -- real engine, real registry,
real drift detector, real trainer -- rather than a simulation of it. Training
runs in a background thread exactly as it does in the fleet, and the serving
loop picks up new versions through the same `DeploymentWatcher` a replica uses.

That matters for the throughput numbers: if training were run inline, its cost
would land inside the serving measurement and the comparison between policies
would be measuring the trainer rather than the drafter. Training wall-time is
reported separately.
"""

from __future__ import annotations

import hashlib
import logging
import statistics
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from fleet import db
from fleet.bench.prompts import Prompt
from fleet.config import settings
from fleet.serving.hotswap import DeploymentWatcher, load_drafter
from fleet.serving.telemetry import RequestRecord, TelemetryWriter
from fleet.specdec.drafters.base import Drafter, NullDrafter
from fleet.specdec.engine import SpeculativeEngine

log = logging.getLogger(__name__)


@dataclass
class RequestMetric:
    index: int
    domain: str
    draft_version: int
    is_canary: bool
    generated_tokens: int
    total_ms: float
    tokens_per_s: float
    draft_proposed: int
    draft_accepted: int
    target_forwards: int
    greedy_match: bool | None


@dataclass
class ArmResult:
    name: str
    description: str
    requests: int = 0
    generated_tokens: int = 0
    serve_wall_s: float = 0.0
    train_wall_s: float = 0.0
    retrains: int = 0
    promotions: int = 0
    rollbacks: int = 0
    versions_served: list[int] = field(default_factory=list)
    metrics: list[RequestMetric] = field(default_factory=list)

    # -- aggregates ---------------------------------------------------------

    @property
    def tokens_per_s(self) -> float:
        """Fleet-level throughput: tokens produced per second of serving wall
        time. This is the number that matters operationally -- a per-request
        mean would hide the cost of slow requests."""
        return self.generated_tokens / self.serve_wall_s if self.serve_wall_s else 0.0

    @property
    def mean_request_tps(self) -> float:
        return statistics.fmean(m.tokens_per_s for m in self.metrics) if self.metrics else 0.0

    @property
    def p50_ms(self) -> float:
        return self._pct(50)

    @property
    def p95_ms(self) -> float:
        return self._pct(95)

    @property
    def p99_ms(self) -> float:
        return self._pct(99)

    def _pct(self, pct: float) -> float:
        if not self.metrics:
            return 0.0
        xs = sorted(m.total_ms for m in self.metrics)
        k = (len(xs) - 1) * pct / 100.0
        lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
        return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)

    @property
    def acceptance_rate(self) -> float | None:
        prop = sum(m.draft_proposed for m in self.metrics)
        acc = sum(m.draft_accepted for m in self.metrics)
        return acc / prop if prop else None

    @property
    def tokens_per_target_forward(self) -> float:
        fwd = sum(m.target_forwards for m in self.metrics)
        return self.generated_tokens / fwd if fwd else 0.0

    @property
    def greedy_audit(self) -> dict[str, int | float | None]:
        audited = [m for m in self.metrics if m.greedy_match is not None]
        matched = [m for m in audited if m.greedy_match]
        return {
            "audited": len(audited),
            "matched": len(matched),
            "mismatched": len(audited) - len(matched),
            "match_rate": (len(matched) / len(audited)) if audited else None,
        }

    def per_domain(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for m in self.metrics:
            d = out.setdefault(
                m.domain, {"requests": 0, "tokens": 0, "ms": 0.0, "proposed": 0, "accepted": 0}
            )
            d["requests"] += 1
            d["tokens"] += m.generated_tokens
            d["ms"] += m.total_ms
            d["proposed"] += m.draft_proposed
            d["accepted"] += m.draft_accepted
        return {
            k: {
                "requests": int(v["requests"]),
                "tokens_per_s": round(v["tokens"] / (v["ms"] / 1000.0), 2) if v["ms"] else 0.0,
                "acceptance": round(v["accepted"] / v["proposed"], 4) if v["proposed"] else None,
            }
            for k, v in out.items()
        }

    def summary(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "requests": self.requests,
            "generated_tokens": self.generated_tokens,
            "serve_wall_s": round(self.serve_wall_s, 2),
            "train_wall_s": round(self.train_wall_s, 2),
            "fleet_tokens_per_s": round(self.tokens_per_s, 3),
            "mean_request_tokens_per_s": round(self.mean_request_tps, 3),
            "p50_ms": round(self.p50_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
            "acceptance_rate": (
                round(self.acceptance_rate, 4) if self.acceptance_rate is not None else None
            ),
            "tokens_per_target_forward": round(self.tokens_per_target_forward, 4),
            "retrains": self.retrains,
            "promotions": self.promotions,
            "rollbacks": self.rollbacks,
            "distinct_versions_served": sorted(set(self.versions_served)),
            "greedy_audit": self.greedy_audit,
            "per_domain": self.per_domain(),
        }


def reset_registry() -> None:
    """Clear registry state so each arm starts from an identical cold start."""
    db.execute(
        """
        TRUNCATE traffic_samples, traffic_events, slo_evaluations,
                 deployment_events, drift_checks, draft_versions
            RESTART IDENTITY CASCADE
        """
    )
    db.execute("INSERT INTO active_deployment (id) VALUES (TRUE) ON CONFLICT (id) DO NOTHING")
    db.execute(
        """
        UPDATE active_deployment
           SET active_version = NULL, canary_version = NULL, canary_pct = 0,
               last_known_good = NULL, generation = 0, updated_reason = NULL
         WHERE id = TRUE
        """
    )


class BackgroundTrainer(threading.Thread):
    """The trainer, running off the serving path exactly as in the fleet.

    Holds no reference to the engine and never touches it: it writes a new
    version into the registry, and the serving side discovers it by polling.
    The only shared state is the database and the artifact store.
    """

    def __init__(self, *, poll_s: float = 1.0, auto_promote: bool = True) -> None:
        super().__init__(name="bench-trainer", daemon=True)
        # NOT `self._stop`: threading.Thread uses a private _stop() method
        # internally from join(), and shadowing it with an Event breaks join.
        self._stop_event = threading.Event()
        self.poll_s = poll_s
        self.auto_promote = auto_promote
        self.retrains = 0
        self.promotions = 0
        self.rollbacks = 0
        self.train_wall_s = 0.0
        self.trigger = threading.Event()
        self.enabled = threading.Event()
        self.enabled.set()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        from fleet.canary import controller
        from fleet.drift.detector import DriftDetector

        detector = DriftDetector()
        while not self._stop_event.wait(self.poll_s):
            if not self.enabled.is_set():
                continue
            try:
                t0 = time.perf_counter()
                version = self.step(detector)
                if version is not None:
                    self.retrains += 1
                    self.train_wall_s += time.perf_counter() - t0

                if self.auto_promote:
                    before = _deployment_versions()
                    controller.tick()
                    after = _deployment_versions()
                    if after["active"] != before["active"]:
                        if after["active"] == before["last_known_good"]:
                            self.rollbacks += 1
                        else:
                            self.promotions += 1
            except Exception:  # noqa: BLE001
                log.exception("background trainer step failed")

    def step(self, detector) -> int | None:
        from fleet.trainer import worker

        return worker.tick(detector)


def _deployment_versions() -> dict[str, int | None]:
    from fleet.registry import store

    dep = store.get_deployment()
    return {
        "active": dep.active_version,
        "canary": dep.canary_version,
        "last_known_good": dep.last_known_good,
    }


def serve_stream(
    engine: SpeculativeEngine,
    prompts: list[Prompt],
    *,
    arm: ArmResult,
    drafter_for: Resolver,
    max_new_tokens: int,
    audit_every: int,
    write_telemetry: bool,
    replica_id: str = "bench-replica",
    on_request: Callable[[int, RequestMetric], None] | None = None,
) -> None:
    """Drive one arm's prompt stream and record what happened.

    Serving wall time excludes the correctness audit: the audit is an extra
    target-only decode that production runs on a small sample, and folding it
    into the measurement would understate throughput for whichever arm happened
    to get audited more.
    """
    tok = engine.target.tokenizer
    serve_s = 0.0

    for i, prompt in enumerate(prompts):
        version, drafter, is_canary = drafter_for(i)
        ids = engine.target.encode_chat(prompt.text)

        t0 = time.perf_counter()
        result = engine.decode(ids, max_new_tokens=max_new_tokens, drafter=drafter)
        serve_s += time.perf_counter() - t0

        greedy_match: bool | None = None
        if audit_every and i % audit_every == 0:
            reference = engine.greedy_reference(ids, max_new_tokens=max_new_tokens)
            greedy_match = reference.generated_ids == result.generated_ids

        metric = RequestMetric(
            index=i,
            domain=prompt.domain,
            draft_version=version,
            is_canary=is_canary,
            generated_tokens=len(result.generated_ids),
            total_ms=result.total_ms,
            tokens_per_s=result.tokens_per_s,
            draft_proposed=result.draft_proposed,
            draft_accepted=result.draft_accepted,
            target_forwards=result.target_forwards,
            greedy_match=greedy_match,
        )
        arm.metrics.append(metric)
        arm.requests += 1
        arm.generated_tokens += len(result.generated_ids)
        arm.versions_served.append(version)

        if write_telemetry:
            TelemetryWriter._write(
                RequestRecord(
                    request_id=uuid.uuid4(),
                    replica_id=replica_id,
                    draft_version=version,
                    is_canary=is_canary,
                    domain=prompt.domain,
                    prompt_sha256=hashlib.sha256(prompt.text.encode()).hexdigest(),
                    prompt_tokens=len(ids),
                    generated_tokens=len(result.generated_ids),
                    draft_proposed=result.draft_proposed,
                    draft_accepted=result.draft_accepted,
                    target_forwards=result.target_forwards,
                    ttft_ms=result.ttft_ms,
                    total_ms=result.total_ms,
                    tokens_per_s=result.tokens_per_s,
                    greedy_match=greedy_match,
                    prompt_token_ids=ids,
                    target_token_ids=result.generated_ids,
                )
            )

        if on_request is not None:
            on_request(i, metric)

    arm.serve_wall_s += serve_s
    _ = tok  # tokenizer kept for debugging convenience


Resolver = Callable[[int], tuple[int, Drafter, bool]]


def registry_drafter_resolver(watcher: DeploymentWatcher) -> Resolver:
    """Resolve each request against whatever the registry currently says,
    including the canary split.

    The canary arm has to actually receive traffic here: the SLO gate compares
    candidate against baseline on live requests, so forcing every request to
    the active version would leave every candidate stuck in canary forever and
    the self-improving arm would never promote anything.

    Syncing per request also means hot-swaps land mid-stream, which is the
    behaviour being measured.
    """

    def _resolve(_: int) -> tuple[int, Drafter, bool]:
        watcher.sync()
        return watcher.resolve()

    return _resolve


def fixed_drafter_resolver(version: int, drafter: Drafter) -> Resolver:
    def _resolve(_: int) -> tuple[int, Drafter, bool]:
        return version, drafter, False

    return _resolve


def null_drafter_resolver(version: int) -> Resolver:
    null = NullDrafter()

    def _resolve(_: int) -> tuple[int, Drafter, bool]:
        return version, null, False

    return _resolve


def warm_up(engine: SpeculativeEngine, prompt: str = "hello", tokens: int = 8) -> None:
    """Pay CUDA/kernel initialisation before anything is timed."""
    ids = engine.target.encode_chat(prompt)
    engine.decode(ids, max_new_tokens=tokens, drafter=NullDrafter())


def train_static_baseline(
    prompts: list[Prompt], engine: SpeculativeEngine, *, max_new_tokens: int
) -> tuple[int, Drafter]:
    """Train a drafter once on a warm-up slice and freeze it.

    This is the 'static speculative decoding' baseline: a draft model fitted
    once, offline, and never updated -- the thing the self-improving fleet is
    being compared against.
    """
    from fleet.specdec.drafters.ngram import NgramDrafter
    from fleet.trainer.dataset import TrainingSample
    from fleet.trainer.worker import bootstrap, select_drafter_config

    bootstrap()
    samples: list[TrainingSample] = []
    for i, p in enumerate(prompts):
        ids = engine.target.encode_chat(p.text)
        res = engine.greedy_reference(ids, max_new_tokens=max_new_tokens)
        samples.append(
            TrainingSample(
                event_id=i, domain=p.domain, prompt_ids=ids, target_ids=res.generated_ids
            )
        )

    # Same chronological train/holdout split and same blend-policy selection
    # the live trainer uses. The static baseline has to be fitted as well as it
    # can be on its own data, or the headline ratio would be measuring a
    # handicapped baseline rather than the value of retraining.
    n_holdout = max(1, len(samples) // 5)
    train_set, holdout = samples[:-n_holdout], samples[-n_holdout:]

    drafter = NgramDrafter.fit(
        (s.full_sequence for s in train_set),
        order=settings.ngram_order,
        min_count=settings.ngram_min_count,
        min_confidence=0.15,
    )
    best_cfg, scores = select_drafter_config(drafter, holdout)
    drafter.policy = best_cfg["policy"]
    drafter.depth_penalty = best_cfg["depth_penalty"]
    drafter.meta.version = 1
    drafter.meta.hyperparams.update(
        {
            "blend_policy": best_cfg["policy"],
            "depth_penalty": best_cfg["depth_penalty"],
            "config_search_scores": scores,
        }
    )
    log.info(
        "static baseline fitted on %d prompts (holdout %d), config=%s",
        len(train_set),
        len(holdout),
        best_cfg,
    )
    return 1, drafter


def bootstrap_cold() -> tuple[int, Drafter]:
    from fleet.trainer.worker import bootstrap

    version = bootstrap()
    return version, load_drafter(version)
