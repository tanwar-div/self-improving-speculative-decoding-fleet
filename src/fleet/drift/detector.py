"""Domain-aware drift detection on the live request distribution.

Retraining on a fixed schedule is wasteful in both directions: it burns compute
when the workload is stationary, and it leaves a stale drafter in production for
up to a full period when the workload moves. This module decides *whether the
serving distribution has actually left the current version's training
distribution*, and only then does the trainer run.

A version's `training_manifest` stores a fingerprint of what it was trained on:
the domain mixture and a token-unigram histogram. A check compares the most
recent window of live traffic against that fingerprint on three axes:

  domain_js       Jensen-Shannon divergence between the live and training
                  domain mixtures. Catches the obvious case: the fleet is now
                  serving clinical questions and the drafter learned Python.
  token_psi       Population Stability Index over token unigrams. Catches
                  within-domain shift that the domain label alone would miss
                  (same "code" label, but Rust instead of Python).
  acceptance_drop Realised acceptance rate versus what this version achieved
                  when it was promoted. The outcome measure -- it is the one
                  that directly predicts lost throughput.

The three are combined into one score against one threshold, and every check is
written to `drift_checks` whether or not it fired, so "why did/didn't it
retrain" is answerable after the fact.
"""

from __future__ import annotations

import logging
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from fleet import db
from fleet.config import settings

log = logging.getLogger(__name__)

TOKEN_BUCKETS = 256  # token ids are hashed into this many buckets for the PSI
_EPS = 1e-9


@dataclass
class TrafficFingerprint:
    """What a draft version was trained on, or what is being served now."""

    domain_counts: dict[str, int] = field(default_factory=dict)
    token_bucket_counts: dict[str, int] = field(default_factory=dict)
    n_events: int = 0
    mean_acceptance: float | None = None

    def to_manifest(self) -> dict[str, Any]:
        return {
            "domain_counts": self.domain_counts,
            "token_bucket_counts": self.token_bucket_counts,
            "n_events": self.n_events,
            "mean_acceptance": self.mean_acceptance,
        }

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any]) -> TrafficFingerprint:
        return cls(
            domain_counts={
                str(k): int(v) for k, v in (manifest.get("domain_counts") or {}).items()
            },
            token_bucket_counts={
                str(k): int(v) for k, v in (manifest.get("token_bucket_counts") or {}).items()
            },
            n_events=int(manifest.get("n_events") or 0),
            mean_acceptance=manifest.get("mean_acceptance"),
        )

    @property
    def is_empty(self) -> bool:
        return self.n_events == 0 or not self.domain_counts


def token_bucket(token_id: int) -> str:
    """Hash a token id into a fixed number of buckets.

    Fixed buckets keep the fingerprint small and, more importantly, make two
    fingerprints comparable even when they contain entirely disjoint vocabulary
    -- which is exactly the situation after a domain shift.
    """
    return str(token_id % TOKEN_BUCKETS)


def _normalise(counts: dict[str, int], keys: list[str]) -> list[float]:
    total = sum(counts.get(k, 0) for k in keys)
    if total <= 0:
        return [1.0 / len(keys)] * len(keys)
    return [(counts.get(k, 0) / total) for k in keys]


def jensen_shannon(p_counts: dict[str, int], q_counts: dict[str, int]) -> float:
    """JS divergence in [0, 1] (log base 2)."""
    keys = sorted(set(p_counts) | set(q_counts))
    if not keys:
        return 0.0
    p = _normalise(p_counts, keys)
    q = _normalise(q_counts, keys)

    def kl(a: list[float], b: list[float]) -> float:
        return sum(
            ai * math.log2((ai + _EPS) / (bi + _EPS)) for ai, bi in zip(a, b, strict=True) if ai > 0
        )

    m = [(ai + bi) / 2 for ai, bi in zip(p, q, strict=True)]
    jsd = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return max(0.0, min(1.0, jsd))


def population_stability_index(
    reference: dict[str, int], live: dict[str, int], *, smoothing: float = 1e-4
) -> float:
    """PSI over shared buckets. Conventional reading: <0.1 stable,
    0.1-0.25 moderate shift, >0.25 significant shift."""
    keys = sorted(set(reference) | set(live))
    if not keys:
        return 0.0
    ref = _normalise(reference, keys)
    cur = _normalise(live, keys)
    psi = 0.0
    for r, c in zip(ref, cur, strict=True):
        r = max(r, smoothing)
        c = max(c, smoothing)
        psi += (c - r) * math.log(c / r)
    return max(0.0, psi)


@dataclass
class DriftResult:
    triggered: bool
    score: float
    threshold: float
    domain_js: float
    token_psi: float
    acceptance_drop: float
    window_events: int
    per_domain: dict[str, Any]
    reason: str

    def as_row(self, reference_version: int) -> dict[str, Any]:
        return {
            "reference_version": reference_version,
            "window_events": self.window_events,
            "domain_js": self.domain_js,
            "token_psi": self.token_psi,
            "acceptance_drop": self.acceptance_drop,
            "score": self.score,
            "threshold": self.threshold,
            "triggered": self.triggered,
            "per_domain": self.per_domain,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# fingerprint construction
# ---------------------------------------------------------------------------


def fingerprint_from_sequences(
    sequences: list[tuple[str, list[int]]], *, mean_acceptance: float | None = None
) -> TrafficFingerprint:
    """Fingerprint a training set: (domain, token_ids) pairs."""
    domains: Counter[str] = Counter()
    buckets: Counter[str] = Counter()
    for domain, ids in sequences:
        domains[domain] += 1
        for tid in ids:
            buckets[token_bucket(tid)] += 1
    return TrafficFingerprint(
        domain_counts=dict(domains),
        token_bucket_counts=dict(buckets),
        n_events=len(sequences),
        mean_acceptance=mean_acceptance,
    )


def live_fingerprint(window: int | None = None) -> TrafficFingerprint:
    """Fingerprint the most recent window of real serving traffic."""
    window = window or settings.drift_window
    rows = db.fetchall(
        """
        SELECT s.domain, s.prompt_token_ids, s.target_token_ids
          FROM traffic_samples s
          JOIN traffic_events e ON e.event_id = s.event_id
         ORDER BY s.event_id DESC
         LIMIT %s
        """,
        (window,),
    )
    seqs = [(r["domain"], list(r["prompt_token_ids"]) + list(r["target_token_ids"])) for r in rows]
    acc = db.fetchone(
        """
        SELECT SUM(draft_accepted)::double precision / NULLIF(SUM(draft_proposed), 0) AS acc
          FROM (SELECT draft_accepted, draft_proposed FROM traffic_events
                 ORDER BY event_id DESC LIMIT %s) t
        """,
        (window,),
    )
    return fingerprint_from_sequences(seqs, mean_acceptance=acc["acc"] if acc else None)


def per_domain_acceptance(window: int | None = None) -> dict[str, dict[str, float]]:
    window = window or settings.drift_window
    rows = db.fetchall(
        """
        SELECT domain,
               COUNT(*)                                                      AS n,
               AVG(tokens_per_s)                                             AS tps,
               SUM(draft_accepted)::double precision
                   / NULLIF(SUM(draft_proposed), 0)                          AS acceptance
          FROM (SELECT domain, tokens_per_s, draft_accepted, draft_proposed
                  FROM traffic_events ORDER BY event_id DESC LIMIT %s) t
         GROUP BY domain
        """,
        (window,),
    )
    return {
        r["domain"]: {
            "requests": int(r["n"]),
            "tokens_per_s": round(float(r["tps"] or 0), 2),
            "acceptance": round(float(r["acceptance"]), 4) if r["acceptance"] is not None else None,
        }
        for r in rows
    }


# ---------------------------------------------------------------------------
# the check
# ---------------------------------------------------------------------------


class DriftDetector:
    """Decides whether the live workload has left the active version's
    training distribution."""

    # Weights chosen so that any single axis going strongly out of range can
    # trigger on its own, rather than requiring all three to agree.
    W_DOMAIN = 0.5
    W_PSI = 0.3
    W_ACCEPTANCE = 0.6

    def __init__(self, threshold: float | None = None) -> None:
        self.threshold = settings.drift_threshold if threshold is None else threshold
        self._last_check_at = 0.0

    def ready(self) -> bool:
        """Rate-limit: a drift check reads a window of traffic, and firing one
        retrain per request would defeat the purpose."""
        return (time.monotonic() - self._last_check_at) >= settings.drift_min_interval_s

    def check(
        self,
        reference: TrafficFingerprint,
        live: TrafficFingerprint,
        *,
        reference_acceptance: float | None = None,
    ) -> DriftResult:
        self._last_check_at = time.monotonic()

        if live.n_events < max(20, settings.drift_window // 10):
            return DriftResult(
                triggered=False,
                score=0.0,
                threshold=self.threshold,
                domain_js=0.0,
                token_psi=0.0,
                acceptance_drop=0.0,
                window_events=live.n_events,
                per_domain={},
                reason=f"insufficient traffic ({live.n_events} samples)",
            )

        domain_js = jensen_shannon(reference.domain_counts, live.domain_counts)
        token_psi = population_stability_index(
            reference.token_bucket_counts, live.token_bucket_counts
        )

        baseline_acc = (
            reference_acceptance if reference_acceptance is not None else reference.mean_acceptance
        )
        acceptance_drop = 0.0
        if baseline_acc and live.mean_acceptance is not None and baseline_acc > 0:
            acceptance_drop = max(0.0, (baseline_acc - live.mean_acceptance) / baseline_acc)

        score = (
            self.W_DOMAIN * domain_js
            + self.W_PSI * min(token_psi, 1.0)
            + self.W_ACCEPTANCE * acceptance_drop
        )
        triggered = score >= self.threshold

        drivers = []
        if self.W_DOMAIN * domain_js >= self.threshold * 0.4:
            drivers.append(f"domain mixture shifted (JS={domain_js:.3f})")
        if self.W_PSI * min(token_psi, 1.0) >= self.threshold * 0.4:
            drivers.append(f"token distribution shifted (PSI={token_psi:.3f})")
        if self.W_ACCEPTANCE * acceptance_drop >= self.threshold * 0.4:
            drivers.append(f"acceptance rate down {acceptance_drop:.0%}")
        reason = (
            "; ".join(drivers)
            if triggered
            else f"workload stationary (score {score:.3f} < {self.threshold:.3f})"
        )

        return DriftResult(
            triggered=triggered,
            score=round(score, 5),
            threshold=self.threshold,
            domain_js=round(domain_js, 5),
            token_psi=round(token_psi, 5),
            acceptance_drop=round(acceptance_drop, 5),
            window_events=live.n_events,
            per_domain=per_domain_acceptance(),
            reason=reason,
        )
