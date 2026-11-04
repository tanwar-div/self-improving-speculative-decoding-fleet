"""Significance testing for the canary gate.

Why this module exists: an earlier version of the gate compared point
estimates against fixed thresholds, and in a live container run it rejected
three consecutive candidates at 0.88-0.92x throughput measured on **15
candidate requests against 23-28 baseline requests**. At that sample size a
0.92x ratio is indistinguishable from noise, so the gate was not enforcing an
SLO -- it was flipping a weighted coin and writing the result to an audit
table.

A threshold alone cannot express "this candidate is worse"; it can only express
"this sample looked worse". So the gate now needs both:

  * the point estimate violates the threshold, **and**
  * the violation is statistically distinguishable from no difference.

A regression that is real but not yet measurable keeps the canary running
instead of failing it, up to a cap -- and a candidate that still cannot be
distinguished from the incumbent after that cap is allowed through, because a
neutral candidate passing is the intended behaviour (the drafter is retrained
to track a moving workload; holding position is success).

Throughput uses Welch's t-test, which does not assume equal variance -- the
arms genuinely have different variance, since a better drafter both raises the
mean and changes the spread. Tail latency uses a percentile bootstrap, because
a p95 has no usable closed-form standard error.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from scipy import stats as _scipy_stats

log = logging.getLogger(__name__)

MIN_SAMPLES_FOR_TEST = 8
DEFAULT_ALPHA = 0.05
BOOTSTRAP_RESAMPLES = 2000


@dataclass
class ComparisonTest:
    """Outcome of comparing one metric between the two canary arms."""

    metric: str
    candidate_n: int
    baseline_n: int
    point_ratio: float
    p_value: float | None
    significant: bool
    ci_low: float | None = None
    ci_high: float | None = None
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "metric": self.metric,
            "candidate_n": self.candidate_n,
            "baseline_n": self.baseline_n,
            "point_ratio": round(self.point_ratio, 4),
            "p_value": round(self.p_value, 5) if self.p_value is not None else None,
            "significant": self.significant,
            "ci_low": round(self.ci_low, 4) if self.ci_low is not None else None,
            "ci_high": round(self.ci_high, 4) if self.ci_high is not None else None,
            "note": self.note,
        }


def _percentile(xs: list[float], pct: float) -> float:
    if not xs:
        return 0.0
    ordered = sorted(xs)
    k = (len(ordered) - 1) * pct / 100.0
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def welch_throughput_test(
    candidate: list[float], baseline: list[float], *, alpha: float = DEFAULT_ALPHA
) -> ComparisonTest:
    """Two-sided Welch's t-test on per-request throughput."""
    n_c, n_b = len(candidate), len(baseline)
    mean_c = sum(candidate) / n_c if n_c else 0.0
    mean_b = sum(baseline) / n_b if n_b else 0.0
    ratio = mean_c / mean_b if mean_b > 0 else 0.0

    if n_c < MIN_SAMPLES_FOR_TEST or n_b < MIN_SAMPLES_FOR_TEST:
        return ComparisonTest(
            metric="tokens_per_s",
            candidate_n=n_c,
            baseline_n=n_b,
            point_ratio=ratio,
            p_value=None,
            significant=False,
            note=f"too few samples to test (need {MIN_SAMPLES_FOR_TEST} per arm)",
        )

    result = _scipy_stats.ttest_ind(candidate, baseline, equal_var=False)
    p = float(result.pvalue)
    ci = result.confidence_interval(confidence_level=1 - alpha)
    # Express the difference CI as a ratio of the baseline mean, so it is
    # directly comparable to the configured threshold.
    ci_low = 1 + ci.low / mean_b if mean_b > 0 else None
    ci_high = 1 + ci.high / mean_b if mean_b > 0 else None

    return ComparisonTest(
        metric="tokens_per_s",
        candidate_n=n_c,
        baseline_n=n_b,
        point_ratio=ratio,
        p_value=p,
        significant=p < alpha,
        ci_low=ci_low,
        ci_high=ci_high,
        note=f"Welch t-test, alpha={alpha}",
    )


def bootstrap_p95_ratio_test(
    candidate: list[float],
    baseline: list[float],
    *,
    alpha: float = DEFAULT_ALPHA,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 20260101,
) -> ComparisonTest:
    """Percentile bootstrap on the ratio of p95 latencies.

    A p95 has no usable closed-form standard error, so the sampling
    distribution is estimated by resampling both arms with replacement.
    """
    n_c, n_b = len(candidate), len(baseline)
    p95_c = _percentile(candidate, 95)
    p95_b = _percentile(baseline, 95)
    ratio = p95_c / p95_b if p95_b > 0 else 0.0

    if n_c < MIN_SAMPLES_FOR_TEST or n_b < MIN_SAMPLES_FOR_TEST:
        return ComparisonTest(
            metric="p95_ms",
            candidate_n=n_c,
            baseline_n=n_b,
            point_ratio=ratio,
            p_value=None,
            significant=False,
            note=f"too few samples to bootstrap (need {MIN_SAMPLES_FOR_TEST} per arm)",
        )

    rng = random.Random(seed)
    ratios: list[float] = []
    for _ in range(resamples):
        rc = _percentile([candidate[rng.randrange(n_c)] for _ in range(n_c)], 95)
        rb = _percentile([baseline[rng.randrange(n_b)] for _ in range(n_b)], 95)
        if rb > 0:
            ratios.append(rc / rb)

    if not ratios:
        return ComparisonTest(
            metric="p95_ms",
            candidate_n=n_c,
            baseline_n=n_b,
            point_ratio=ratio,
            p_value=None,
            significant=False,
            note="bootstrap produced no usable resamples",
        )

    ci_low = _percentile(ratios, 100 * alpha / 2)
    ci_high = _percentile(ratios, 100 * (1 - alpha / 2))
    # "Significant" here means the interval excludes 1.0, i.e. the two arms'
    # tail latencies are distinguishable at all.
    significant = ci_low > 1.0 or ci_high < 1.0

    return ComparisonTest(
        metric="p95_ms",
        candidate_n=n_c,
        baseline_n=n_b,
        point_ratio=ratio,
        p_value=None,
        significant=significant,
        ci_low=ci_low,
        ci_high=ci_high,
        note=f"percentile bootstrap, {len(ratios)} resamples, alpha={alpha}",
    )
