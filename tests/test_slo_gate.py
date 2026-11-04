"""The SLO gate decides promotions, so its failure modes matter more than its
happy path. These tests drive it with synthetic traffic so each verdict is
forced deterministically rather than hoped for."""

from __future__ import annotations

import random
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from fleet import db
from fleet.canary import controller, slo
from fleet.config import settings
from fleet.registry import store


def _register(fake_artifact, **kw) -> int:
    stored = fake_artifact()
    return store.register_version(
        drafter_kind="ngram",
        target_model="test/model",
        artifact_uri=stored.uri,
        artifact_sha256=stored.sha256,
        artifact_bytes=stored.size_bytes,
        **kw,
    )


def emit(
    version: int,
    *,
    n: int,
    tps: float,
    total_ms: float,
    is_canary: bool = False,
    domain: str = "chat",
    greedy_match: bool | None = None,
    cv: float = 0.25,
    seed: int = 0,
) -> None:
    """Insert n synthetic served requests for `version`.

    `cv` is the coefficient of variation. Real serving latency is noisy, and
    the gate now runs significance tests, so emitting constants would both make
    the tests degenerate and hide whether the statistics work.
    """
    rng = random.Random(f"{version}-{is_canary}-{seed}-{n}")
    rows = []
    for _ in range(n):
        jitter = max(0.15, rng.gauss(1.0, cv))
        rows.append(
            (
                str(uuid.uuid4()),
                "r-test",
                version,
                is_canary,
                domain,
                "a" * 64,
                20,
                32,
                40,
                20,
                10,
                5.0,
                total_ms * jitter,
                tps / jitter,
                greedy_match,
            )
        )
    with db.conn() as c, c.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO traffic_events
                (request_id, replica_id, draft_version, is_canary, domain,
                 prompt_sha256, prompt_tokens, generated_tokens, draft_proposed,
                 draft_accepted, target_forwards, ttft_ms, total_ms, tokens_per_s,
                 greedy_match)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            rows,
        )


@pytest.fixture
def two_versions(clean_db, fake_artifact):
    baseline = _register(fake_artifact, activate_immediately=True)
    candidate = _register(fake_artifact)
    return baseline, candidate


def _evaluate(
    baseline: int, candidate: int, *, min_requests: int | None = None
) -> slo.GateDecision:
    start = datetime.now(UTC) - timedelta(minutes=5)
    return slo.evaluate(
        candidate_version=candidate,
        baseline_version=baseline,
        window_start=start,
        window_end=datetime.now(UTC),
        min_requests=min_requests,
    )


def test_gate_waits_for_enough_data(two_versions):
    baseline, candidate = two_versions
    emit(baseline, n=5, tps=100, total_ms=300)
    emit(candidate, n=5, tps=120, total_ms=280, is_canary=True)
    d = _evaluate(baseline, candidate)
    assert d.result == "insufficient_data"
    assert not d.passed


def test_faster_candidate_passes(two_versions):
    baseline, candidate = two_versions
    n = settings.canary_min_requests + 10
    emit(baseline, n=n, tps=100, total_ms=320)
    emit(candidate, n=n, tps=118, total_ms=300, is_canary=True)
    d = _evaluate(baseline, candidate)
    assert d.passed, d.summary()
    assert d.tps_ratio > 1.0


def test_throughput_regression_fails_when_measurable(two_versions):
    """A real regression, on enough requests to prove it, must fail."""
    baseline, candidate = two_versions
    n = settings.canary_min_requests + 60
    emit(baseline, n=n, tps=100, total_ms=320, cv=0.12)
    emit(candidate, n=n, tps=70, total_ms=330, is_canary=True, cv=0.12)  # 0.70x
    d = _evaluate(baseline, candidate)
    assert d.result == "fail", d.summary()
    check = next(c for c in d.failed_checks if c["metric"] == "tokens_per_s")
    assert check["significant"] is True
    assert check["p_value"] is not None and check["p_value"] < 0.05


def test_tail_latency_regression_fails_even_when_throughput_holds(two_versions):
    """A candidate can keep mean throughput while wrecking the tail; the gate
    has to catch that independently."""
    baseline, candidate = two_versions
    n = settings.canary_min_requests + 10
    emit(baseline, n=n, tps=100, total_ms=300)
    emit(candidate, n=n, tps=101, total_ms=400, is_canary=True)  # 1.33x p95
    d = _evaluate(baseline, candidate)
    assert not d.passed
    assert any(c["metric"] == "p95_ms" for c in d.failed_checks), d.failed_checks


def test_any_greedy_mismatch_fails_regardless_of_speed(two_versions):
    """Correctness is a gate, not a trade-off: a candidate that is faster but
    changed the output must not be promoted."""
    baseline, candidate = two_versions
    n = settings.canary_min_requests + 10
    emit(baseline, n=n, tps=100, total_ms=300, greedy_match=True)
    emit(candidate, n=n - 1, tps=150, total_ms=250, is_canary=True, greedy_match=True)
    emit(candidate, n=1, tps=150, total_ms=250, is_canary=True, greedy_match=False)
    d = _evaluate(baseline, candidate)
    assert not d.passed
    assert any(c["metric"] == "greedy_match" for c in d.failed_checks), d.failed_checks


def test_controller_promotes_on_pass(two_versions):
    baseline, candidate = two_versions
    controller.start_canary(candidate, 50)
    n = settings.canary_min_requests + 10
    emit(baseline, n=n, tps=100, total_ms=320)
    emit(candidate, n=n, tps=120, total_ms=300, is_canary=True)

    decision = controller.evaluate_canary()
    assert decision is not None and decision.passed
    dep = store.get_deployment()
    assert dep.active_version == candidate
    assert dep.last_known_good == baseline
    assert dep.canary_version is None

    # The promotion must be linked to the evaluation that justified it.
    event = next(e for e in store.deployment_history(20) if e["action"] == "promote")
    assert event["slo_evaluation_id"] is not None


def test_controller_rejects_on_fail_and_keeps_serving_baseline(two_versions):
    baseline, candidate = two_versions
    controller.start_canary(candidate, 50)
    n = settings.canary_min_requests + 10
    emit(baseline, n=n, tps=100, total_ms=300)
    emit(candidate, n=n, tps=60, total_ms=500, is_canary=True)

    decision = controller.evaluate_canary()
    assert decision is not None and not decision.passed
    dep = store.get_deployment()
    assert dep.active_version == baseline, "a failed canary must not disturb production"
    assert store.get_version(candidate).status == "rejected"


def test_post_promotion_regression_triggers_rollback(two_versions):
    """A candidate that passes a narrow canary and then regresses on the full
    traffic mix must be reverted without human involvement."""
    baseline, candidate = two_versions
    controller.start_canary(candidate, 50)
    n = settings.canary_min_requests + 10
    emit(baseline, n=n, tps=100, total_ms=320)
    emit(candidate, n=n, tps=120, total_ms=300, is_canary=True)
    assert controller.evaluate_canary().passed
    assert store.get_deployment().active_version == candidate

    # Now it serves full traffic and collapses.
    emit(candidate, n=n * 2, tps=55, total_ms=700)
    rolled_back = controller.check_active_regression()

    assert rolled_back
    dep = store.get_deployment()
    assert dep.active_version == baseline
    assert store.get_version(candidate).status == "rolled_back"
    event = next(e for e in store.deployment_history(30) if e["action"] == "rollback")
    assert "regression" in (event["reason"] or "")


def test_healthy_active_version_is_not_rolled_back(two_versions):
    """The regression watch must not fire on a version that is behaving."""
    baseline, candidate = two_versions
    controller.start_canary(candidate, 50)
    n = settings.canary_min_requests + 10
    emit(baseline, n=n, tps=100, total_ms=320)
    emit(candidate, n=n, tps=120, total_ms=300, is_canary=True)
    controller.evaluate_canary()

    emit(candidate, n=n * 2, tps=121, total_ms=298)
    assert controller.check_active_regression() is False
    assert store.get_deployment().active_version == candidate


# ---------------------------------------------------------------------------
# Significance guard. These are the regression tests for a bug found by running
# the real containerized fleet: the gate rejected three consecutive candidates
# at 0.88-0.92x throughput measured on 15 candidate vs 23-28 baseline requests,
# where that ratio is indistinguishable from noise.
# ---------------------------------------------------------------------------


def test_apparent_regression_on_a_small_noisy_sample_does_not_fail(two_versions):
    """The exact shape of the observed bug.

    Both arms are drawn from the SAME underlying distribution, with the sample
    sizes and noise level the container run actually produced. The gate must
    not fail the candidate -- there is no regression to find.
    """
    baseline, candidate = two_versions
    emit(baseline, n=27, tps=8.2, total_ms=3200, cv=0.35, seed=1)
    emit(candidate, n=15, tps=8.2, total_ms=3200, is_canary=True, cv=0.35, seed=2)

    # min_requests lowered so the verdict comes from the statistics rather than
    # the sample-size short circuit: the point is that even when the gate is
    # willing to decide, it must not call this a regression.
    d = _evaluate(baseline, candidate, min_requests=10)
    assert d.result != "fail", (
        f"gate failed a candidate drawn from the same distribution: {d.summary()}"
    )
    assert d.tests["tokens_per_s"]["significant"] is False


def test_undecidable_candidate_keeps_the_canary_running(two_versions, monkeypatch):
    """An apparent regression that cannot be measured extends the canary rather
    than killing the candidate.

    The point estimate here breaches the 0.98x threshold (0.88x), but with this
    noise and sample size a Welch test cannot distinguish it from no
    difference -- so the correct action is to collect more traffic.
    """
    monkeypatch.setattr(settings, "canary_max_requests", 10_000)
    baseline, candidate = two_versions
    emit(baseline, n=20, tps=100, total_ms=320, cv=0.45, seed=3)
    emit(candidate, n=20, tps=88, total_ms=360, is_canary=True, cv=0.45, seed=4)

    d = _evaluate(baseline, candidate, min_requests=10)
    assert d.tps_ratio < settings.slo_min_tps_ratio, (
        f"test setup should breach the threshold, got {d.tps_ratio}"
    )
    assert d.tests["tokens_per_s"]["significant"] is False, d.tests
    assert d.result == "insufficient_data", d.summary()
    assert "not distinguishable" in d.decision_note
    assert "extending canary" in d.decision_note


def test_indistinguishable_candidate_is_promoted_at_the_canary_cap(two_versions, monkeypatch):
    """A neutral candidate must eventually go through: the drafter is retrained
    to track a moving workload, so holding position is success, and an
    undecidable canary cannot run forever."""
    monkeypatch.setattr(settings, "canary_max_requests", 15)
    baseline, candidate = two_versions
    emit(baseline, n=20, tps=100, total_ms=320, cv=0.45, seed=3)
    emit(candidate, n=20, tps=88, total_ms=360, is_canary=True, cv=0.45, seed=4)

    d = _evaluate(baseline, candidate, min_requests=10)
    assert d.tests["tokens_per_s"]["significant"] is False
    assert d.result == "pass", d.summary()
    assert "cap" in d.decision_note


def test_correctness_failure_ignores_significance_entirely(two_versions):
    """One audited mismatch is one user who got different output. There is no
    sample size at which that becomes acceptable, so it must fail even on a
    tiny, noisy, otherwise-faster sample."""
    baseline, candidate = two_versions
    n = settings.canary_min_requests + 2
    emit(baseline, n=n, tps=100, total_ms=320, greedy_match=True, cv=0.4)
    emit(candidate, n=n - 1, tps=200, total_ms=160, is_canary=True, greedy_match=True, cv=0.4)
    emit(candidate, n=1, tps=200, total_ms=160, is_canary=True, greedy_match=False)

    d = _evaluate(baseline, candidate)
    assert d.result == "fail"
    check = next(c for c in d.failed_checks if c["metric"] == "greedy_match")
    assert check["threshold"] == 0


def test_statistical_tests_are_persisted_for_audit(two_versions):
    """The gate's reasoning has to be reconstructable after the fact, not just
    its verdict."""
    baseline, candidate = two_versions
    controller.start_canary(candidate, 50)
    n = settings.canary_min_requests + 60
    emit(baseline, n=n, tps=100, total_ms=320, cv=0.12)
    emit(candidate, n=n, tps=70, total_ms=330, is_canary=True, cv=0.12)

    decision = controller.evaluate_canary()
    assert decision is not None and not decision.passed

    row = db.fetchall("SELECT * FROM slo_evaluations ORDER BY id DESC LIMIT 1")[0]
    tests = row["statistical_tests"]
    assert tests["alpha"] == settings.slo_alpha
    assert tests["tokens_per_s"]["p_value"] is not None
    assert tests["tokens_per_s"]["candidate_n"] == n
    assert tests["p95_ms"]["ci_low"] is not None
    assert row["decision_note"]
