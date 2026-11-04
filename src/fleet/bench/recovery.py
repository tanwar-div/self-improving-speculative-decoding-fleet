"""Measuring recovery from a workload shift.

The first version of this metric compared each retraining policy against an
**oracle**: a drafter trained directly on post-shift traffic, serving with a
frozen version and no canary. Both policies then "recovered after never",
because that target is unreachable by construction -- a live adaptive system
pays a cold start, a rollout latency, and a continuous canary tax on unproven
candidates, so its ceiling sits well below a pre-trained frozen drafter's. The
comparison measured the gap between "adaptive" and "clairvoyant", not the thing
the experiment is about, which is **which policy recovers faster**.

So recovery is measured against a target both policies can actually reach: the
lower of the two policies' own post-shift plateaus. That keeps the target
policy-independent (neither policy sets its own bar) while remaining
achievable, and it leaves the comparison between them intact.

Two numbers are reported, because one of them can be undefined:

  requests_to_recover  how long until rolling throughput reaches the shared
                       target and stays there. Directly answers the question,
                       but is None if a policy never gets there.
  deficit_per_request  mean shortfall below the target across the whole
                       post-shift stream. Always defined, and it captures
                       *depth* of the dip as well as its duration -- a policy
                       that recovers quickly but collapses first is not better.

The oracle is still reported, as context for how much of the achievable gain
either policy captured.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any


@dataclass
class PolicyRecovery:
    policy: str
    plateau: float
    requests_to_recover: int | None
    deficit_per_request: float
    fraction_of_oracle: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "post_shift_plateau_tokens_per_s": round(self.plateau, 3),
            "requests_to_recover": self.requests_to_recover,
            "deficit_per_request_tokens_per_s": round(self.deficit_per_request, 4),
            "plateau_fraction_of_oracle": (
                round(self.fraction_of_oracle, 4) if self.fraction_of_oracle else None
            ),
        }


def plateau_of(rolling: list[float], *, tail_frac: float = 0.25) -> float:
    """A policy's settled post-shift throughput: the median of its final slice.

    Median rather than mean so one slow request does not move the bar.
    """
    if not rolling:
        return 0.0
    n = max(1, int(len(rolling) * tail_frac))
    return statistics.median(rolling[-n:])


def first_sustained(rolling: list[float], *, start: int, target: float, sustain: int) -> int | None:
    n = len(rolling)
    for i in range(max(0, start), n):
        if i + sustain > n:
            break
        if all(rolling[j] >= target for j in range(i, i + sustain)):
            return i
    return None


def mean_deficit(rolling: list[float], *, start: int, target: float) -> float:
    tail = rolling[start:]
    if not tail:
        return 0.0
    return sum(max(0.0, target - v) for v in tail) / len(tail)


def analyse(
    policies: dict[str, dict[str, Any]],
    *,
    shift_index: int,
    window: int,
    recovery_frac: float,
    sustain: int,
    oracle_tokens_per_s: float | None = None,
) -> dict[str, Any]:
    """Recompute recovery for every policy against a shared, reachable target.

    `policies` maps a policy name to a dict carrying `rolling_tokens_per_s`.
    Works on a saved report, so changing the metric does not require re-running
    the GPU work that produced the traces.
    """
    traces = {
        name: list(p.get("rolling_tokens_per_s") or [])
        for name, p in policies.items()
        if p.get("rolling_tokens_per_s")
    }
    if not traces:
        return {"error": "no rolling throughput traces in the report"}

    plateaus = {name: plateau_of(t) for name, t in traces.items()}
    # The lower plateau: policy-independent, and reachable by both.
    shared_target = recovery_frac * min(plateaus.values())

    # Only start looking once the rolling window has cleared the shift, or the
    # window still contains pre-shift requests and reports instant recovery.
    search_start = shift_index + window

    results = {}
    for name, trace in traces.items():
        results[name] = PolicyRecovery(
            policy=name,
            plateau=plateaus[name],
            requests_to_recover=(
                (idx - shift_index)
                if (
                    idx := first_sustained(
                        trace, start=search_start, target=shared_target, sustain=sustain
                    )
                )
                is not None
                else None
            ),
            deficit_per_request=mean_deficit(trace, start=search_start, target=shared_target),
            fraction_of_oracle=(
                plateaus[name] / oracle_tokens_per_s if oracle_tokens_per_s else None
            ),
        ).as_dict()

    periodic = results.get("periodic", {})
    drift = results.get("drift_triggered", {})
    pr, dr = periodic.get("requests_to_recover"), drift.get("requests_to_recover")
    pd_, dd = (
        periodic.get("deficit_per_request_tokens_per_s"),
        drift.get("deficit_per_request_tokens_per_s"),
    )

    return {
        "method": (
            "recovery measured against the lower of the two policies' own "
            "post-shift plateaus, scaled by recovery_frac -- a target that is "
            "policy-independent and reachable, unlike the frozen pre-trained oracle"
        ),
        "shared_target_tokens_per_s": round(shared_target, 3),
        "recovery_frac": recovery_frac,
        "sustain_requests": sustain,
        "search_started_at_request": search_start,
        "oracle_tokens_per_s": oracle_tokens_per_s,
        "policies": results,
        "comparison": {
            "periodic_requests_to_recover": pr,
            "drift_requests_to_recover": dr,
            "drift_faster_by_fraction": (
                round((pr - dr) / pr, 4) if pr and dr and pr > 0 else None
            ),
            "periodic_deficit_per_request": pd_,
            "drift_deficit_per_request": dd,
            "drift_deficit_lower_by_fraction": (
                round((pd_ - dd) / pd_, 4) if pd_ and dd is not None and pd_ > 0 else None
            ),
        },
    }
