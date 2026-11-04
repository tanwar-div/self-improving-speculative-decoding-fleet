"""The SLO gate.

A candidate draft version is promoted or rejected by this computation and
nothing else -- no human reads a dashboard and decides. Three properties make
the comparison trustworthy:

  Concurrent baseline. Candidate and baseline requests are drawn from the same
  time window, on the same replicas, under the same load. The gate never
  compares "the candidate now" against "the incumbent an hour ago", which would
  fold load and time-of-day into the verdict.

  Random assignment. Variant choice is per-request, so prompt mix and output
  length are balanced across arms in expectation instead of by construction.

  Correctness is a gate, not a metric. Any audited greedy mismatch on the
  candidate fails it outright, whatever it did to throughput.

Thresholds are deliberately asymmetric: a candidate must not *lose* throughput
(>= 0.98x) and must not inflate tail latency (<= 1.05x). A candidate that is
merely neutral is still allowed through, because the drafter is retrained to
track a moving workload -- holding position on current traffic is the point.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from fleet import db
from fleet.canary.stats import ComparisonTest, bootstrap_p95_ratio_test, welch_throughput_test
from fleet.config import settings

log = logging.getLogger(__name__)


@dataclass
class ArmStats:
    requests: int
    tokens_per_s: float
    p95_ms: float
    p50_ms: float
    acceptance: float | None
    greedy_audited: int
    greedy_mismatches: int
    mean_generated_tokens: float


@dataclass
class GateDecision:
    result: str  # pass | fail | insufficient_data
    failed_checks: list[dict[str, Any]]
    tests: dict[str, Any]
    decision_note: str
    candidate: ArmStats
    baseline: ArmStats
    tps_ratio: float
    p95_ratio: float
    thresholds: dict[str, Any]
    window_start: datetime
    window_end: datetime

    @property
    def passed(self) -> bool:
        return self.result == "pass"

    def summary(self) -> str:
        if self.result == "insufficient_data":
            return f"undecided: {self.decision_note}"
        verdict = "PASS" if self.passed else "FAIL"
        detail = (
            f"tps {self.candidate.tokens_per_s:.1f} vs {self.baseline.tokens_per_s:.1f} "
            f"({self.tps_ratio:.3f}x), p95 {self.candidate.p95_ms:.0f}ms vs "
            f"{self.baseline.p95_ms:.0f}ms ({self.p95_ratio:.3f}x)"
        )
        if self.failed_checks:
            detail += " | " + "; ".join(c["check"] for c in self.failed_checks)
        elif self.decision_note:
            detail += f" | {self.decision_note}"
        return f"{verdict}: {detail}"


def arm_samples(version: int, since: datetime) -> tuple[list[float], list[float]]:
    """Per-request (tokens_per_s, total_ms) for one arm.

    The gate needs the raw samples, not just aggregates: a mean and a p95
    cannot tell you whether a difference is distinguishable from noise.
    """
    rows = db.fetchall(
        """
        SELECT tokens_per_s, total_ms
          FROM traffic_events
         WHERE draft_version = %s AND ts >= %s
         ORDER BY event_id
        """,
        (version, since),
    )
    return (
        [float(r["tokens_per_s"]) for r in rows],
        [float(r["total_ms"]) for r in rows],
    )


def arm_stats(version: int, since: datetime, *, is_canary: bool | None = None) -> ArmStats:
    clause = "" if is_canary is None else " AND is_canary = %(is_canary)s"
    row = db.fetchone(
        f"""
        SELECT COUNT(*)                                            AS requests,
               COALESCE(AVG(tokens_per_s), 0)                      AS tps,
               COALESCE(
                 PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_ms), 0)  AS p95_ms,
               COALESCE(
                 PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY total_ms), 0)  AS p50_ms,
               SUM(draft_accepted)::double precision
                   / NULLIF(SUM(draft_proposed), 0)                AS acceptance,
               COUNT(greedy_match)                                 AS audited,
               COUNT(*) FILTER (WHERE greedy_match IS FALSE)       AS mismatches,
               COALESCE(AVG(generated_tokens), 0)                  AS mean_gen
          FROM traffic_events
         WHERE draft_version = %(version)s AND ts >= %(since)s{clause}
        """,
        {"version": version, "since": since, "is_canary": is_canary},
    )
    return ArmStats(
        requests=int(row["requests"]),
        tokens_per_s=float(row["tps"]),
        p95_ms=float(row["p95_ms"]),
        p50_ms=float(row["p50_ms"]),
        acceptance=float(row["acceptance"]) if row["acceptance"] is not None else None,
        greedy_audited=int(row["audited"]),
        greedy_mismatches=int(row["mismatches"]),
        mean_generated_tokens=float(row["mean_gen"]),
    )


def evaluate(
    *,
    candidate_version: int,
    baseline_version: int,
    window_start: datetime,
    window_end: datetime,
    min_requests: int | None = None,
) -> GateDecision:
    min_requests = settings.canary_min_requests if min_requests is None else min_requests
    candidate = arm_stats(candidate_version, window_start)
    baseline = arm_stats(baseline_version, window_start)

    thresholds = {
        "min_tps_ratio": settings.slo_min_tps_ratio,
        "max_p95_ratio": settings.slo_max_p95_ratio,
        "min_requests_per_arm": min_requests,
        "require_greedy_match": settings.slo_require_greedy_match,
        "alpha": settings.slo_alpha,
        "canary_max_requests": settings.canary_max_requests,
    }

    tps_ratio = candidate.tokens_per_s / baseline.tokens_per_s if baseline.tokens_per_s > 0 else 0.0
    p95_ratio = candidate.p95_ms / baseline.p95_ms if baseline.p95_ms > 0 else 0.0

    # --- raw samples, so the comparison can be tested rather than asserted --
    cand_tps, cand_ms = arm_samples(candidate_version, window_start)
    base_tps, base_ms = arm_samples(baseline_version, window_start)
    tps_test = welch_throughput_test(cand_tps, base_tps, alpha=settings.slo_alpha)
    p95_test = bootstrap_p95_ratio_test(cand_ms, base_ms, alpha=settings.slo_alpha)
    tests: dict[str, Any] = {
        "tokens_per_s": tps_test.as_dict(),
        "p95_ms": p95_test.as_dict(),
        "alpha": settings.slo_alpha,
    }

    if candidate.requests < min_requests or baseline.requests < min_requests:
        return GateDecision(
            result="insufficient_data",
            failed_checks=[],
            tests=tests,
            decision_note=(
                f"waiting for {min_requests} requests per arm "
                f"(candidate {candidate.requests}, baseline {baseline.requests})"
            ),
            candidate=candidate,
            baseline=baseline,
            tps_ratio=round(tps_ratio, 4),
            p95_ratio=round(p95_ratio, 4),
            thresholds=thresholds,
            window_start=window_start,
            window_end=window_end,
        )

    failed: list[dict[str, Any]] = []
    undecided: list[str] = []

    def _record(
        violated: bool, test: ComparisonTest, metric: str, check: str, threshold: float
    ) -> None:
        """Fail only on a regression that is both present and measurable."""
        if not violated:
            return
        if test.significant:
            failed.append(
                {
                    "check": check,
                    "metric": metric,
                    "observed": round(test.point_ratio, 4),
                    "threshold": threshold,
                    "p_value": test.p_value,
                    "ci": [test.ci_low, test.ci_high],
                    "significant": True,
                }
            )
        else:
            undecided.append(
                f"{metric} looks worse ({test.point_ratio:.3f}x) but is not "
                f"distinguishable from the incumbent ({test.note})"
            )

    _record(
        tps_ratio < settings.slo_min_tps_ratio,
        tps_test,
        "tokens_per_s",
        f"throughput regression: {tps_ratio:.3f}x < {settings.slo_min_tps_ratio:.3f}x "
        f"(p={tps_test.p_value})",
        settings.slo_min_tps_ratio,
    )
    _record(
        p95_ratio > settings.slo_max_p95_ratio,
        p95_test,
        "p95_ms",
        f"p95 latency regression: {p95_ratio:.3f}x > {settings.slo_max_p95_ratio:.3f}x "
        f"(bootstrap CI {p95_test.ci_low}-{p95_test.ci_high})",
        settings.slo_max_p95_ratio,
    )

    # Correctness is never subject to a significance test. One audited mismatch
    # is one user who got different output, and there is no sample size at
    # which that becomes acceptable.
    if settings.slo_require_greedy_match and candidate.greedy_mismatches > 0:
        failed.append(
            {
                "check": f"greedy-output mismatch on {candidate.greedy_mismatches} of "
                f"{candidate.greedy_audited} audited requests",
                "metric": "greedy_match",
                "observed": candidate.greedy_mismatches,
                "threshold": 0,
                "significant": True,
            }
        )

    # An apparent regression that cannot yet be measured keeps the canary
    # running rather than killing the candidate -- up to a cap, after which a
    # still-indistinguishable candidate is allowed through.
    if not failed and undecided:
        at_cap = candidate.requests >= settings.canary_max_requests
        if not at_cap:
            return GateDecision(
                result="insufficient_data",
                failed_checks=[],
                tests=tests,
                decision_note="; ".join(undecided) + "; extending canary",
                candidate=candidate,
                baseline=baseline,
                tps_ratio=round(tps_ratio, 4),
                p95_ratio=round(p95_ratio, 4),
                thresholds=thresholds,
                window_start=window_start,
                window_end=window_end,
            )
        note = (
            "; ".join(undecided)
            + f"; canary cap of {settings.canary_max_requests} requests reached, "
            "promoting an indistinguishable candidate"
        )
        return GateDecision(
            result="pass",
            failed_checks=[],
            tests=tests,
            decision_note=note,
            candidate=candidate,
            baseline=baseline,
            tps_ratio=round(tps_ratio, 4),
            p95_ratio=round(p95_ratio, 4),
            thresholds=thresholds,
            window_start=window_start,
            window_end=window_end,
        )

    return GateDecision(
        result="fail" if failed else "pass",
        failed_checks=failed,
        tests=tests,
        decision_note=(
            "; ".join(c["check"] for c in failed)
            if failed
            else "no measurable regression on throughput, tail latency or output"
        ),
        candidate=candidate,
        baseline=baseline,
        tps_ratio=round(tps_ratio, 4),
        p95_ratio=round(p95_ratio, 4),
        thresholds=thresholds,
        window_start=window_start,
        window_end=window_end,
    )


def to_row(decision: GateDecision, candidate_version: int, baseline_version: int) -> dict:
    return {
        "candidate_version": candidate_version,
        "baseline_version": baseline_version,
        "window_start": decision.window_start,
        "window_end": decision.window_end,
        "candidate_requests": decision.candidate.requests,
        "baseline_requests": decision.baseline.requests,
        "candidate_tps": decision.candidate.tokens_per_s,
        "baseline_tps": decision.baseline.tokens_per_s,
        "candidate_p95_ms": decision.candidate.p95_ms,
        "baseline_p95_ms": decision.baseline.p95_ms,
        "candidate_acceptance": decision.candidate.acceptance,
        "baseline_acceptance": decision.baseline.acceptance,
        "candidate_greedy_mismatches": decision.candidate.greedy_mismatches,
        "tps_ratio": decision.tps_ratio,
        "p95_ratio": decision.p95_ratio,
        "gate_result": decision.result,
        "failed_checks": decision.failed_checks,
        "thresholds": decision.thresholds,
        "statistical_tests": decision.tests,
        "decision_note": decision.decision_note,
    }
