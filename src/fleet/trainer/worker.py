"""Asynchronous draft-model trainer.

Runs as its own process, entirely off the serving path. Serving replicas never
block on training and never share a process with it; the only coupling between
the two is the registry row and the artifact digest. That decoupling is the
point: the fleet keeps serving at full throughput while the next draft version
trains behind it.

The loop is:

    drift check  ->  harvest live traffic  ->  fit  ->  score offline
                 ->  publish artifact      ->  register as 'candidate'

and then it stops. Promotion is the canary controller's decision, not the
trainer's, so a bad version cannot reach full traffic just because training
finished.
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

from fleet import db
from fleet.config import settings
from fleet.drift.detector import (
    DriftDetector,
    TrafficFingerprint,
    fingerprint_from_sequences,
    live_fingerprint,
)
from fleet.registry import store
from fleet.registry.artifacts import ArtifactStore
from fleet.specdec.drafters import load_artifact
from fleet.specdec.drafters.base import DrafterMeta, NullDrafter
from fleet.specdec.drafters.ngram import DEPTH_PENALTIES, POLICIES, NgramDrafter
from fleet.trainer.dataset import harvest, simulate_acceptance, split

log = logging.getLogger(__name__)


def train_candidate(
    *,
    parent_version: int | None,
    reason: str,
    min_samples: int | None = None,
    require_improvement: bool = True,
) -> int | None:
    """Train one draft version from live traffic and register it as a candidate.

    Returns the new version number, or None if training was skipped.
    """
    min_samples = settings.train_min_samples if min_samples is None else min_samples
    samples = harvest()
    if len(samples) < min_samples:
        log.info("skipping train: only %d samples (need %d)", len(samples), min_samples)
        return None

    train_set, holdout = split(samples)
    t0 = time.perf_counter()
    drafter = NgramDrafter.fit(
        (s.full_sequence for s in train_set),
        order=settings.ngram_order,
        min_count=settings.ngram_min_count,
        min_confidence=0.15,
        meta=DrafterMeta(kind="ngram"),
    )
    fit_ms = (time.perf_counter() - t0) * 1000.0

    # Score on traffic the drafter has not seen, using the target's recorded
    # greedy path -- no model forwards needed.
    # Choose how the drafter combines its two proposal strategies, by scoring
    # each option on traffic it has not trained on. Without this a thin table
    # can pre-empt context copy and make a freshly trained version *worse* than
    # the one it replaces.
    best_cfg, config_scores = select_drafter_config(drafter, holdout)
    drafter.policy = best_cfg["policy"]
    drafter.depth_penalty = best_cfg["depth_penalty"]

    scored = simulate_acceptance(drafter, holdout)
    acceptance = scored["acceptance_rate"]
    objective = scored[OBJECTIVE]

    # Compare against the incumbent on the *same* holdout before spending any
    # production traffic on a canary.
    incumbent_acceptance = None
    incumbent_objective = None
    if parent_version is not None:
        parent = store.get_version(parent_version)
        if parent is not None:
            try:
                art = ArtifactStore()
                # Load by the incumbent's own kind: it may be the no-speculation
                # cold-start version, not an n-gram drafter.
                incumbent = load_artifact(
                    parent.drafter_kind,
                    art.fetch(parent.artifact_uri, parent.artifact_sha256),
                )
                incumbent_scored = simulate_acceptance(incumbent, holdout)
                incumbent_acceptance = incumbent_scored["acceptance_rate"]
                incumbent_objective = incumbent_scored[OBJECTIVE]
            except Exception as exc:  # noqa: BLE001 - scoring is best-effort
                log.warning("could not score incumbent v%s: %s", parent_version, exc)

    if (
        require_improvement
        and incumbent_objective is not None
        and objective <= incumbent_objective + 1e-6
    ):
        log.info(
            "discarding candidate: %.4f tokens/forward does not beat incumbent %.4f",
            objective,
            incumbent_objective,
        )
        return None

    fingerprint = fingerprint_from_sequences(
        [(s.domain, s.full_sequence) for s in train_set], mean_acceptance=acceptance
    )
    drafter.meta.training_manifest = fingerprint.to_manifest()
    drafter.meta.offline_acceptance_rate = acceptance
    drafter.meta.hyperparams.update(
        {
            "fit_ms": round(fit_ms, 1),
            "holdout_samples": len(holdout),
            "tokens_per_target_forward": round(scored["tokens_per_target_forward"], 4),
            "per_domain_acceptance": scored["per_domain_acceptance"],
            "incumbent_offline_acceptance": incumbent_acceptance,
            "objective": OBJECTIVE,
            "offline_tokens_per_target_forward": round(objective, 5),
            "incumbent_tokens_per_target_forward": (
                round(incumbent_objective, 5) if incumbent_objective is not None else None
            ),
            "blend_policy": best_cfg["policy"],
            "depth_penalty": best_cfg["depth_penalty"],
            "config_search_scores": config_scores,
        }
    )

    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "drafter.json"
        drafter.save(local)
        stored = ArtifactStore().put(local, name="ngram")

    event_ids = [s.event_id for s in train_set]
    version = store.register_version(
        drafter_kind=drafter.kind,
        target_model=settings.target_model,
        artifact_uri=stored.uri,
        artifact_sha256=stored.sha256,
        artifact_bytes=stored.size_bytes,
        training_manifest=fingerprint.to_manifest(),
        hyperparams=drafter.meta.hyperparams,
        offline_acceptance_rate=acceptance,
        train_sample_count=len(train_set),
        trained_from_event_id=min(event_ids) if event_ids else None,
        trained_to_event_id=max(event_ids) if event_ids else None,
        parent_version=parent_version,
        notes=reason,
    )
    log.info(
        "trained v%s in %.0fms: %.4f tokens/forward (incumbent %s), "
        "acceptance %.3f, policy %s/d%s, %d samples",
        version,
        fit_ms,
        objective,
        f"{incumbent_objective:.4f}" if incumbent_objective is not None else "n/a",
        acceptance,
        best_cfg["policy"],
        best_cfg["depth_penalty"],
        len(train_set),
    )
    return version


# The objective. Acceptance *rate* is the intuitive metric and the wrong one:
# a drafter that proposes one token and lands it scores 1.0 acceptance but only
# 2.0 tokens per target forward, while one that proposes four and lands two
# scores 0.5 acceptance and 3.0 tokens per forward -- and throughput tracks
# tokens per forward, since the forward pass is what costs. Selecting on
# acceptance rate measurably picked the slower drafter.
OBJECTIVE = "tokens_per_target_forward"


def select_drafter_config(
    drafter: NgramDrafter, holdout: list, *, gamma: int | None = None
) -> tuple[dict, dict[str, float]]:
    """Search blend policy x depth penalty on held-out live traffic.

    Scoring is free: `simulate_acceptance` replays the engine's accept/reject
    logic against the target's already-recorded greedy path, so the whole search
    costs zero model forward passes.
    """
    original = (drafter.policy, drafter.depth_penalty)
    scores: dict[str, float] = {}
    best_key, best_score, best_cfg = None, -1.0, {}
    try:
        for policy in POLICIES:
            for penalty in DEPTH_PENALTIES:
                drafter.policy = policy
                drafter.depth_penalty = penalty
                scored = simulate_acceptance(drafter, holdout, gamma=gamma)
                key = f"{policy}/d{penalty}"
                scores[key] = round(scored[OBJECTIVE], 5)
                if scored[OBJECTIVE] > best_score:
                    best_key, best_score = key, scored[OBJECTIVE]
                    best_cfg = {"policy": policy, "depth_penalty": penalty}
    finally:
        drafter.policy, drafter.depth_penalty = original
    log.info(
        "drafter config search (%s): best %s = %.4f of %d candidates",
        OBJECTIVE,
        best_key,
        best_score,
        len(scores),
    )
    return best_cfg, scores


def select_blend_policy(
    drafter: NgramDrafter, holdout: list, *, gamma: int | None = None
) -> tuple[str, dict[str, float]]:
    """Backwards-compatible wrapper returning only the policy name."""
    cfg, scores = select_drafter_config(drafter, holdout, gamma=gamma)
    return cfg["policy"], scores


def bootstrap() -> int:
    """Cold start: register a version that does not speculate at all.

    The obvious cold start is an untrained drafter proposing by context copy.
    Measured, that is a net *loss*: on a mixed prompt pool it reached 82.9 tok/s
    against 91.0 tok/s for plain greedy decoding, because it widens every
    verification forward without earning enough accepted tokens back.

    So the fleet starts with no speculation, and the first speculative version
    has to beat that through the canary like any other candidate. It is the same
    rule as everywhere else here: nothing serves production traffic until it has
    been measured against what it replaces.
    """
    dep = store.get_deployment()
    if dep.active_version is not None:
        log.info("already bootstrapped: active v%s", dep.active_version)
        return dep.active_version

    drafter = NullDrafter()
    drafter.meta.training_manifest = TrafficFingerprint().to_manifest()
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "drafter.json"
        drafter.save(local)
        stored = ArtifactStore().put(local, name="none-coldstart")

    return store.register_version(
        drafter_kind=drafter.kind,
        target_model=settings.target_model,
        artifact_uri=stored.uri,
        artifact_sha256=stored.sha256,
        artifact_bytes=stored.size_bytes,
        training_manifest=drafter.meta.training_manifest,
        hyperparams={"cold_start": True, "speculation": False},
        offline_acceptance_rate=None,
        train_sample_count=0,
        notes="cold start: no speculation until a candidate proves it helps",
        activate_immediately=True,
    )


def new_samples_since(event_id: int | None) -> int:
    """How much training data has arrived since a version was trained."""
    row = db.fetchone(
        "SELECT COUNT(*) AS n FROM traffic_samples WHERE %s::bigint IS NULL "
        "OR event_id > %s::bigint",
        (event_id, event_id),
    )
    return int(row["n"]) if row else 0


def run_forever() -> None:
    """The trainer process entrypoint."""
    db.wait_for_db()
    db.apply_schema()
    bootstrap()
    detector = DriftDetector()
    log.info("trainer up: polling every %.1fs", settings.trainer_poll_s)

    while True:
        try:
            tick(detector)
        except Exception:  # noqa: BLE001 - a trainer crash must not take the fleet down
            log.exception("trainer tick failed")
        time.sleep(settings.trainer_poll_s)


def tick(detector: DriftDetector) -> int | None:
    """One trainer iteration. Returns a new version number if one was trained."""
    return tick_with_policy(detector, require_improvement=True)


def tick_with_policy(detector: DriftDetector, *, require_improvement: bool) -> int | None:
    """One trainer iteration.

    `require_improvement` is exposed because the benchmark's two retraining
    policies have to be compared with it held equal -- otherwise the drift
    policy would also be getting the benefit of the offline-improvement guard
    and the comparison would no longer isolate trigger timing.
    """
    dep = store.get_deployment()
    if dep.canary_version is not None:
        log.debug("canary v%s in flight; deferring training", dep.canary_version)
        return None
    if dep.active_version is None:
        bootstrap()
        return None

    active = store.get_version(dep.active_version)
    if active is None:
        return None

    # Any 'candidate' already waiting is the controller's to deal with.
    pending = db.fetchone(
        "SELECT version FROM draft_versions WHERE status = 'candidate' "
        "ORDER BY version DESC LIMIT 1"
    )
    if pending:
        log.debug("candidate v%s awaiting canary; not training", pending["version"])
        return None

    reference = TrafficFingerprint.from_manifest(active.training_manifest)
    live = live_fingerprint()

    # Cold start has no training distribution to compare against, so the first
    # real version is gated on sample count instead of on drift.
    if reference.is_empty:
        if live.n_events >= settings.train_min_samples:
            return train_candidate(
                parent_version=active.version,
                reason="first trained version after cold start",
                require_improvement=False,
            )
        return None

    result = detector.check(reference, live, reference_acceptance=active.offline_acceptance_rate)
    store.record_drift_check(result.as_row(active.version))

    if not result.triggered:
        # No drift -- but the drafter may simply be undertrained relative to the
        # traffic now available. Without this branch a stationary workload
        # freezes the fleet on whatever version it happened to train first.
        fresh = new_samples_since(active.trained_to_event_id)
        needed = max(
            settings.train_min_samples,
            int(active.train_sample_count * settings.retrain_data_growth),
        )
        if fresh >= needed:
            log.info(
                "no drift, but %d new samples since v%s trained on %d -- retraining "
                "on the larger set",
                fresh,
                active.version,
                active.train_sample_count,
            )
            return train_candidate(
                parent_version=active.version,
                reason=(
                    f"data growth: {fresh} new samples vs {active.train_sample_count} "
                    "the active version trained on"
                ),
                require_improvement=require_improvement,
            )
        log.info(
            "no drift: %s (%d/%d new samples toward a growth retrain)", result.reason, fresh, needed
        )
        return None

    log.warning("drift detected (score %.3f): %s", result.score, result.reason)
    return train_candidate(
        parent_version=active.version,
        reason=f"drift: {result.reason}",
        require_improvement=require_improvement,
    )
