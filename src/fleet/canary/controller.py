"""Automated canary rollout controller.

Owns every promotion decision in the fleet. The trainer can only produce
candidates; this process is the only thing that can put one into traffic or
take one out, and it does so on the SLO gate's verdict.

State machine per candidate:

    candidate --canary_start(pct)--> canary --gate pass--> active
                                        |
                                        +---gate fail---> rejected

and independently, for a version that already took full traffic:

    active --regression vs its own promotion-time numbers--> rolled_back
            (fleet reverts to last_known_good)

The post-promotion watch matters: a candidate can pass a short canary and still
regress once it is serving the full prompt mix, so the gate is not a one-time
admission test. Both paths push a `/admin/sync` to each replica so a rollout or
a revert lands in milliseconds rather than waiting out the poll interval; the
replica poll loop stays as the fallback if a push is lost.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

import httpx

from fleet import db
from fleet.canary import slo, stats
from fleet.config import settings
from fleet.registry import store

log = logging.getLogger(__name__)


def push_sync() -> dict[str, bool]:
    """Tell every replica to reconcile now. Best-effort: polling is the
    guarantee, this is just the latency optimisation."""
    results: dict[str, bool] = {}
    for url in settings.replica_url_list:
        try:
            r = httpx.post(f"{url}/admin/sync", timeout=20.0)
            results[url] = r.status_code == 200
        except Exception as exc:  # noqa: BLE001
            log.warning("sync push to %s failed: %s", url, exc)
            results[url] = False
    return results


def pending_candidate() -> int | None:
    row = db.fetchone(
        "SELECT version FROM draft_versions WHERE status = 'candidate' ORDER BY version ASC LIMIT 1"
    )
    return int(row["version"]) if row else None


def canary_started_at(version: int) -> datetime | None:
    row = db.fetchone(
        """
        SELECT ts FROM deployment_events
         WHERE action = 'canary_start' AND to_version = %s
         ORDER BY id DESC LIMIT 1
        """,
        (version,),
    )
    return row["ts"] if row else None


def start_canary(version: int, pct: int | None = None) -> None:
    pct = settings.canary_pct if pct is None else pct
    store.start_canary(version, pct, reason=f"automated canary at {pct}%")
    push_sync()


def evaluate_canary(*, min_requests: int | None = None) -> slo.GateDecision | None:
    """Gate the in-flight canary. Promotes, rejects, or waits for more data."""
    dep = store.get_deployment()
    if dep.canary_version is None or dep.active_version is None:
        return None

    started = canary_started_at(dep.canary_version)
    if started is None:
        log.warning("canary v%s has no start event; aborting it", dep.canary_version)
        store.abort_canary(dep.canary_version, reason="missing canary_start event")
        push_sync()
        return None

    now = datetime.now(UTC)
    decision = slo.evaluate(
        candidate_version=dep.canary_version,
        baseline_version=dep.active_version,
        window_start=started,
        window_end=now,
        min_requests=min_requests,
    )

    if decision.result == "insufficient_data":
        log.info("canary v%s: %s", dep.canary_version, decision.summary())
        return decision

    eval_id = store.record_slo_evaluation(
        slo.to_row(decision, dep.canary_version, dep.active_version)
    )

    if decision.passed:
        log.info("canary v%s PASSED gate: %s", dep.canary_version, decision.summary())
        store.promote(
            dep.canary_version,
            reason=f"SLO gate passed ({decision.tps_ratio:.3f}x tps, "
            f"{decision.p95_ratio:.3f}x p95)",
            slo_evaluation_id=eval_id,
        )
    else:
        log.warning("canary v%s FAILED gate: %s", dep.canary_version, decision.summary())
        store.abort_canary(
            dep.canary_version,
            reason="; ".join(c["check"] for c in decision.failed_checks),
            slo_evaluation_id=eval_id,
        )
    push_sync()
    return decision


def check_active_regression(*, min_requests: int | None = None) -> bool:
    """Watch the version currently taking full traffic.

    Compares its live throughput against the throughput it demonstrated in the
    canary that promoted it. A candidate can look fine on a 20% slice and
    regress on the full prompt mix; this is what catches that, and it reverts to
    `last_known_good` without anyone being paged.
    """
    min_requests = settings.canary_min_requests if min_requests is None else min_requests
    dep = store.get_deployment()
    if dep.active_version is None or dep.canary_version is not None:
        return False
    if dep.last_known_good is None or dep.last_known_good == dep.active_version:
        return False  # nothing safe to fall back to

    promo = db.fetchone(
        """
        SELECT e.id, e.candidate_tps, e.candidate_p95_ms,
               e.window_start AS canary_started_at, d.ts AS promoted_at
          FROM deployment_events d
          JOIN slo_evaluations e ON e.id = d.slo_evaluation_id
         WHERE d.action = 'promote' AND d.to_version = %s
         ORDER BY d.id DESC LIMIT 1
        """,
        (dep.active_version,),
    )
    if promo is None:
        return False

    live = slo.arm_stats(dep.active_version, promo["promoted_at"])
    if live.requests < min_requests:
        return False

    # Same guard as the canary gate: the live arm is compared against the
    # throughput distribution it demonstrated during its canary, and a
    # difference that is not measurable must not trigger a rollback.
    live_tps, live_ms = slo.arm_samples(dep.active_version, promo["promoted_at"])
    canary_tps, canary_ms = slo.arm_samples(dep.active_version, promo["canary_started_at"])
    tps_test = stats.welch_throughput_test(live_tps, canary_tps, alpha=settings.slo_alpha)
    p95_test = stats.bootstrap_p95_ratio_test(live_ms, canary_ms, alpha=settings.slo_alpha)
    regression_tests = {
        "tokens_per_s": tps_test.as_dict(),
        "p95_ms": p95_test.as_dict(),
        "alpha": settings.slo_alpha,
        "scope": "post-promotion regression watch",
    }

    expected_tps = float(promo["candidate_tps"])
    expected_p95 = float(promo["candidate_p95_ms"])
    tps_ratio = live.tokens_per_s / expected_tps if expected_tps > 0 else 1.0
    p95_ratio = live.p95_ms / expected_p95 if expected_p95 > 0 else 1.0

    failed = []
    if tps_ratio < settings.slo_min_tps_ratio and tps_test.significant:
        failed.append(
            {
                "check": f"post-promotion throughput regression: {tps_ratio:.3f}x of "
                f"canary-measured {expected_tps:.1f} tok/s (p={tps_test.p_value})",
                "metric": "tokens_per_s",
                "observed": round(tps_ratio, 4),
                "threshold": settings.slo_min_tps_ratio,
                "p_value": tps_test.p_value,
            }
        )
    if p95_ratio > settings.slo_max_p95_ratio and p95_test.significant:
        failed.append(
            {
                "check": f"post-promotion p95 regression: {p95_ratio:.3f}x of "
                f"canary-measured {expected_p95:.0f}ms "
                f"(bootstrap CI {p95_test.ci_low}-{p95_test.ci_high})",
                "metric": "p95_ms",
                "observed": round(p95_ratio, 4),
                "threshold": settings.slo_max_p95_ratio,
            }
        )
    if settings.slo_require_greedy_match and live.greedy_mismatches > 0:
        failed.append(
            {
                "check": f"greedy mismatch on {live.greedy_mismatches} audited requests "
                "while serving full traffic",
                "metric": "greedy_match",
                "observed": live.greedy_mismatches,
                "threshold": 0,
            }
        )

    if not failed:
        return False

    eval_id = store.record_slo_evaluation(
        {
            "candidate_version": dep.active_version,
            "baseline_version": dep.last_known_good,
            "window_start": promo["promoted_at"],
            "window_end": datetime.now(UTC),
            "candidate_requests": live.requests,
            "baseline_requests": live.requests,
            "candidate_tps": live.tokens_per_s,
            "baseline_tps": expected_tps,
            "candidate_p95_ms": live.p95_ms,
            "baseline_p95_ms": expected_p95,
            "candidate_acceptance": live.acceptance,
            "baseline_acceptance": None,
            "candidate_greedy_mismatches": live.greedy_mismatches,
            "tps_ratio": round(tps_ratio, 4),
            "p95_ratio": round(p95_ratio, 4),
            "gate_result": "fail",
            "failed_checks": failed,
            "thresholds": {
                "min_tps_ratio": settings.slo_min_tps_ratio,
                "max_p95_ratio": settings.slo_max_p95_ratio,
                "scope": "post-promotion regression watch",
            },
            "statistical_tests": regression_tests,
            "decision_note": "; ".join(c["check"] for c in failed),
        }
    )
    log.error(
        "regression on active v%s: %s -- rolling back to v%s",
        dep.active_version,
        "; ".join(c["check"] for c in failed),
        dep.last_known_good,
    )
    store.rollback(reason="; ".join(c["check"] for c in failed), slo_evaluation_id=eval_id)
    push_sync()
    return True


def tick() -> None:
    """One controller iteration."""
    dep = store.get_deployment()
    if dep.canary_version is not None:
        evaluate_canary()
        return
    candidate = pending_candidate()
    if candidate is not None:
        log.info("promoting candidate v%s into canary", candidate)
        start_canary(candidate)
        return
    check_active_regression()


def run_forever(interval: float = 5.0) -> None:
    db.wait_for_db()
    db.apply_schema()
    log.info(
        "canary controller up: interval %.1fs, canary %s%%, gate >= %.3fx tps, <= %.3fx p95",
        interval,
        settings.canary_pct,
        settings.slo_min_tps_ratio,
        settings.slo_max_p95_ratio,
    )
    while True:
        try:
            tick()
        except Exception:  # noqa: BLE001 - controller must not die on one bad tick
            log.exception("controller tick failed")
        time.sleep(interval)
