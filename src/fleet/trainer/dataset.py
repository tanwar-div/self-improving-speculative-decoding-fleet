"""Harvesting training data from live inference traffic, and scoring a drafter
offline before it is allowed anywhere near production traffic.

`traffic_samples` stores, per served request, the prompt tokens and the
*target model's own greedy continuation*. That makes the distillation label
free: it is already the thing the drafter has to predict, captured from real
production decoding rather than from a static corpus.

It also makes offline evaluation exact and free. Acceptance rate is a function
of the drafter's proposals and the target's greedy path, and both are already in
the table -- so `simulate_acceptance` replays the engine's accept/reject logic
with zero model forward passes. A candidate that is worse than the incumbent can
be discarded before it costs any production traffic at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from fleet import db
from fleet.config import settings
from fleet.specdec.drafters.base import Drafter

log = logging.getLogger(__name__)


@dataclass
class TrainingSample:
    event_id: int
    domain: str
    prompt_ids: list[int]
    target_ids: list[int]

    @property
    def full_sequence(self) -> list[int]:
        return self.prompt_ids + self.target_ids


def harvest(*, limit: int | None = None, since_event_id: int | None = None) -> list[TrainingSample]:
    """Most recent samples first, so a retrain after a workload shift is
    dominated by the traffic that caused the shift."""
    limit = limit or settings.train_max_samples
    sql = """
        SELECT s.event_id, s.domain, s.prompt_token_ids, s.target_token_ids
          FROM traffic_samples s
         WHERE (%s::bigint IS NULL OR s.event_id > %s::bigint)
         ORDER BY s.event_id DESC
         LIMIT %s
    """
    rows = db.fetchall(sql, (since_event_id, since_event_id, limit))
    return [
        TrainingSample(
            event_id=r["event_id"],
            domain=r["domain"],
            prompt_ids=list(r["prompt_token_ids"]),
            target_ids=list(r["target_token_ids"]),
        )
        for r in rows
    ]


def split(samples: list[TrainingSample], holdout_frac: float = 0.2) -> tuple[list, list]:
    """Chronological split: train on older, evaluate on newer.

    Deliberately not random. A random split would let the drafter be scored on
    traffic it trained on, which is exactly the mistake that makes an offline
    acceptance number useless for predicting production acceptance.
    """
    ordered = sorted(samples, key=lambda s: s.event_id)
    n_holdout = max(1, int(len(ordered) * holdout_frac))
    return ordered[:-n_holdout], ordered[-n_holdout:]


def simulate_acceptance(
    drafter: Drafter, samples: list[TrainingSample], *, gamma: int | None = None
) -> dict[str, float]:
    """Replay the engine's accept/reject logic against recorded target paths.

    Mirrors `SpeculativeEngine.decode`: propose up to gamma tokens, accept the
    longest prefix matching the target's greedy continuation, then commit the
    accepted tokens plus one bonus token (the target's own choice at the point
    of divergence).
    """
    gamma = gamma or settings.gamma
    proposed = accepted = steps = tokens = 0
    per_domain: dict[str, list[int]] = {}

    for s in samples:
        committed = list(s.prompt_ids)
        target_ids = s.target_ids
        if not target_ids:
            continue
        # The engine's first token comes out of the prompt prefill, with nothing
        # drafted for it. Starting the replay at index 0 instead of 1 would add
        # a proposal round the engine never makes and shift the alignment of
        # every subsequent one, so the offline acceptance rate would not match
        # what the engine actually achieves.
        committed.append(target_ids[0])
        i = 1
        d_prop = d_acc = 0
        while i < len(target_ids):
            budget = len(target_ids) - i
            ask = max(0, min(gamma, budget - 1))
            prop = drafter.propose(committed, ask).token_ids[:ask] if ask > 0 else []

            n_acc = 0
            for k, tok in enumerate(prop):
                if tok == target_ids[i + k]:
                    n_acc += 1
                else:
                    break

            proposed += len(prop)
            accepted += n_acc
            d_prop += len(prop)
            d_acc += n_acc
            steps += 1

            take = target_ids[i : i + n_acc + 1]
            committed.extend(take)
            i += len(take)
            tokens += len(take)

        tokens += 1  # the prefill token
        per_domain.setdefault(s.domain, [0, 0])
        per_domain[s.domain][0] += d_prop
        per_domain[s.domain][1] += d_acc

    return {
        "acceptance_rate": (accepted / proposed) if proposed else 0.0,
        "proposed": proposed,
        "accepted": accepted,
        "steps": steps,
        "tokens": tokens,
        # Expected target forwards saved per token: the actual throughput lever.
        "tokens_per_target_forward": (tokens / steps) if steps else 1.0,
        "per_domain_acceptance": {
            d: round(a / p, 4) if p else 0.0 for d, (p, a) in per_domain.items()
        },
    }
