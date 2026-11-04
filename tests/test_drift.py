"""Drift detection.

The claim being tested is two-sided, and the negative half is the one that
matters: a detector that fires constantly would be indistinguishable from the
fixed-schedule retraining it is supposed to replace.
"""

from __future__ import annotations

import random

from fleet.drift.detector import (
    DriftDetector,
    TrafficFingerprint,
    fingerprint_from_sequences,
    jensen_shannon,
    population_stability_index,
)

# Each domain gets its own token-id band, standing in for genuinely different
# vocabulary -- Python identifiers vs clinical terminology vs case law.
DOMAIN_BANDS = {
    "code": (1000, 2000),
    "math": (2000, 3000),
    "chat": (3000, 4000),
    "medical": (8000, 9000),
    "legal": (9000, 10000),
}


def make_traffic(mixture: dict[str, int], *, seed: int = 0, length: int = 40):
    rng = random.Random(seed)
    out = []
    for domain, count in mixture.items():
        lo, hi = DOMAIN_BANDS[domain]
        for _ in range(count):
            out.append((domain, [rng.randrange(lo, hi) for _ in range(length)]))
    rng.shuffle(out)
    return out


def test_js_and_psi_bounds():
    assert jensen_shannon({}, {}) == 0.0
    assert jensen_shannon({"a": 10}, {"a": 10}) == 0.0
    assert jensen_shannon({"a": 10}, {"b": 10}) > 0.99
    assert population_stability_index({"a": 10}, {"a": 10}) == 0.0


def test_stationary_traffic_does_not_trigger():
    """Same workload, different sample: must not retrain."""
    train = make_traffic({"code": 120, "chat": 150, "math": 90}, seed=1)
    live = make_traffic({"code": 118, "chat": 155, "math": 92}, seed=2)

    reference = fingerprint_from_sequences(train, mean_acceptance=0.45)
    current = fingerprint_from_sequences(live, mean_acceptance=0.44)

    result = DriftDetector().check(reference, current, reference_acceptance=0.45)
    assert not result.triggered, result.reason
    assert "stationary" in result.reason


def test_domain_shift_triggers():
    """The fleet was serving code and chat; now it serves clinical and legal."""
    train = make_traffic({"code": 200, "chat": 160}, seed=3)
    live = make_traffic({"medical": 190, "legal": 170}, seed=4)

    reference = fingerprint_from_sequences(train, mean_acceptance=0.45)
    current = fingerprint_from_sequences(live, mean_acceptance=0.12)

    result = DriftDetector().check(reference, current, reference_acceptance=0.45)
    assert result.triggered
    assert result.domain_js > 0.5
    assert "domain mixture shifted" in result.reason


def test_within_domain_shift_is_caught_by_token_distribution():
    """Same domain labels, different token distribution -- the case a
    label-only detector would miss entirely."""
    rng = random.Random(11)
    train = [("code", [rng.randrange(1000, 1100) for _ in range(40)]) for _ in range(250)]
    live = [("code", [rng.randrange(1800, 1900) for _ in range(40)]) for _ in range(250)]

    reference = fingerprint_from_sequences(train, mean_acceptance=0.5)
    current = fingerprint_from_sequences(live, mean_acceptance=0.5)

    result = DriftDetector().check(reference, current, reference_acceptance=0.5)
    assert result.domain_js == 0.0, "domain mixture is identical by construction"
    assert result.token_psi > 0.25, "token-level shift should be visible in the PSI"
    assert result.triggered, result.reason


def test_acceptance_collapse_triggers_on_its_own():
    """Even if inputs look similar, a drafter that stopped working should be
    retrained -- acceptance rate is the outcome measure."""
    train = make_traffic({"chat": 200}, seed=5)
    live = make_traffic({"chat": 200}, seed=6)
    reference = fingerprint_from_sequences(train, mean_acceptance=0.50)
    current = fingerprint_from_sequences(live, mean_acceptance=0.10)

    result = DriftDetector().check(reference, current, reference_acceptance=0.50)
    assert result.acceptance_drop > 0.7
    assert result.triggered
    assert "acceptance rate down" in result.reason


def test_insufficient_traffic_never_triggers():
    reference = fingerprint_from_sequences(make_traffic({"chat": 200}, seed=7))
    current = fingerprint_from_sequences(make_traffic({"legal": 5}, seed=8))
    result = DriftDetector().check(reference, current)
    assert not result.triggered
    assert "insufficient traffic" in result.reason


def test_gradual_shift_crosses_the_threshold_eventually():
    """Monotonic: as the live mixture moves further from training, the score
    rises, and it crosses the threshold rather than oscillating."""
    train = make_traffic({"code": 200, "chat": 200}, seed=9)
    reference = fingerprint_from_sequences(train, mean_acceptance=0.45)

    scores = []
    for medical_share in (0, 10, 25, 50, 75, 100):
        n_med = medical_share * 4
        n_old = (100 - medical_share) * 4
        live = make_traffic(
            {"code": n_old // 2, "chat": n_old // 2, "medical": n_med}, seed=20 + medical_share
        )
        current = fingerprint_from_sequences(live, mean_acceptance=0.45)
        scores.append(DriftDetector().check(reference, current, reference_acceptance=0.45).score)

    assert scores == sorted(scores), f"drift score should rise monotonically, got {scores}"
    assert scores[0] < 0.15 <= scores[-1]


def test_empty_reference_is_recognised_as_cold_start():
    assert TrafficFingerprint().is_empty
    assert not fingerprint_from_sequences(make_traffic({"chat": 5}, seed=1)).is_empty
