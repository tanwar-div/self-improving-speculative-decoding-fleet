"""Versioned draft-model registry.

Every state change a rollout can make -- register, canary start/scale/abort,
promote, rollback, retire -- goes through this module and lands in
`deployment_events`, so the deployment history of the fleet is queryable rather
than reconstructed from logs.

Two invariants, both enforced in the database rather than in Python:
  * at most one 'active' and one 'canary' version fleet-wide (partial unique
    indexes in sql/001_schema.sql);
  * every traffic_events row references a real draft_versions row, so a served
    response can always be traced to the exact artifact behind it.

`last_known_good` is maintained on every promotion, which is what makes a
rollback target a version that actually passed its SLO gate once, rather than
just the numerically previous one.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from psycopg.types.json import Jsonb

from fleet import db

log = logging.getLogger(__name__)

_REGISTRY_LOCK = 918273645  # advisory lock key for version-number allocation


@dataclass
class Version:
    version: int
    drafter_kind: str
    target_model: str
    status: str
    artifact_uri: str
    artifact_sha256: str
    artifact_bytes: int
    training_manifest: dict[str, Any]
    hyperparams: dict[str, Any]
    offline_acceptance_rate: float | None
    train_sample_count: int
    trained_from_event_id: int | None
    trained_to_event_id: int | None
    parent_version: int | None
    # Why this version exists: "cold start", "drift: ...", "data growth: ...".
    notes: str | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Version:
        return cls(
            version=row["version"],
            drafter_kind=row["drafter_kind"],
            target_model=row["target_model"],
            status=row["status"],
            artifact_uri=row["artifact_uri"],
            artifact_sha256=row["artifact_sha256"],
            artifact_bytes=row["artifact_bytes"],
            training_manifest=row.get("training_manifest") or {},
            hyperparams=row.get("hyperparams") or {},
            offline_acceptance_rate=row.get("offline_acceptance_rate"),
            train_sample_count=row.get("train_sample_count") or 0,
            trained_from_event_id=row.get("trained_from_event_id"),
            trained_to_event_id=row.get("trained_to_event_id"),
            parent_version=row.get("parent_version"),
            notes=row.get("notes"),
        )


@dataclass
class Deployment:
    active_version: int | None
    canary_version: int | None
    canary_pct: int
    last_known_good: int | None
    generation: int
    updated_reason: str | None


# ---------------------------------------------------------------------------
# reads
# ---------------------------------------------------------------------------


def get_deployment() -> Deployment:
    row = db.fetchone("SELECT * FROM active_deployment WHERE id = TRUE")
    assert row is not None, "active_deployment singleton row missing; run apply_schema()"
    return Deployment(
        active_version=row["active_version"],
        canary_version=row["canary_version"],
        canary_pct=row["canary_pct"],
        last_known_good=row["last_known_good"],
        generation=row["generation"],
        updated_reason=row["updated_reason"],
    )


def get_version(version: int) -> Version | None:
    row = db.fetchone("SELECT * FROM draft_versions WHERE version = %s", (version,))
    return Version.from_row(row) if row else None


def get_by_status(status: str) -> Version | None:
    row = db.fetchone(
        "SELECT * FROM draft_versions WHERE status = %s ORDER BY version DESC LIMIT 1",
        (status,),
    )
    return Version.from_row(row) if row else None


def list_versions(limit: int = 50) -> list[Version]:
    rows = db.fetchall("SELECT * FROM draft_versions ORDER BY version DESC LIMIT %s", (limit,))
    return [Version.from_row(r) for r in rows]


# ---------------------------------------------------------------------------
# writes
# ---------------------------------------------------------------------------


def record_event(
    action: str,
    *,
    from_version: int | None = None,
    to_version: int | None = None,
    actor: str = "controller",
    replica_id: str | None = None,
    slo_evaluation_id: int | None = None,
    reason: str | None = None,
    detail: dict[str, Any] | None = None,
    cur: Any = None,
) -> None:
    sql = """
        INSERT INTO deployment_events
            (action, from_version, to_version, actor, replica_id,
             slo_evaluation_id, reason, detail)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    params = (
        action,
        from_version,
        to_version,
        actor,
        replica_id,
        slo_evaluation_id,
        reason,
        Jsonb(detail or {}),
    )
    if cur is not None:
        cur.execute(sql, params)
    else:
        db.execute(sql, params)


def register_version(
    *,
    drafter_kind: str,
    target_model: str,
    artifact_uri: str,
    artifact_sha256: str,
    artifact_bytes: int,
    training_manifest: dict[str, Any] | None = None,
    hyperparams: dict[str, Any] | None = None,
    offline_acceptance_rate: float | None = None,
    train_sample_count: int = 0,
    trained_from_event_id: int | None = None,
    trained_to_event_id: int | None = None,
    parent_version: int | None = None,
    notes: str | None = None,
    activate_immediately: bool = False,
) -> int:
    """Allocate the next version number and insert the row.

    `activate_immediately` exists only for cold start, where there is no
    baseline to canary against yet. Normal versions enter as 'candidate' and
    have to earn promotion through the SLO gate.
    """
    with db.conn() as c, c.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(%s)", (_REGISTRY_LOCK,))
        cur.execute("SELECT COALESCE(MAX(version), 0) + 1 AS v FROM draft_versions")
        version = int(cur.fetchone()["v"])

        status = "active" if activate_immediately else "candidate"
        cur.execute(
            """
            INSERT INTO draft_versions
                (version, drafter_kind, target_model, status, parent_version,
                 artifact_uri, artifact_sha256, artifact_bytes,
                 trained_from_event_id, trained_to_event_id, train_sample_count,
                 training_manifest, hyperparams, offline_acceptance_rate, notes,
                 promoted_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                    CASE WHEN %s THEN now() ELSE NULL END)
            """,
            (
                version,
                drafter_kind,
                target_model,
                status,
                parent_version,
                artifact_uri,
                artifact_sha256,
                artifact_bytes,
                trained_from_event_id,
                trained_to_event_id,
                train_sample_count,
                Jsonb(training_manifest or {}),
                Jsonb(hyperparams or {}),
                offline_acceptance_rate,
                notes,
                activate_immediately,
            ),
        )

        if activate_immediately:
            cur.execute(
                """
                UPDATE active_deployment
                   SET active_version = %s,
                       last_known_good = %s,
                       canary_version = NULL,
                       canary_pct = 0,
                       generation = generation + 1,
                       updated_at = now(),
                       updated_reason = 'cold start'
                 WHERE id = TRUE
                """,
                (version, version),
            )

        record_event(
            "register",
            to_version=version,
            actor="trainer",
            reason=notes,
            detail={
                "drafter_kind": drafter_kind,
                "sha256": artifact_sha256,
                "train_sample_count": train_sample_count,
                "cold_start": activate_immediately,
            },
            cur=cur,
        )
    log.info("registered draft version %s (%s, status=%s)", version, drafter_kind, status)
    return version


def start_canary(version: int, pct: int, *, reason: str = "scheduled canary") -> Deployment:
    with db.conn() as c, c.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(%s)", (_REGISTRY_LOCK,))
        cur.execute("SELECT active_version, canary_version FROM active_deployment WHERE id = TRUE")
        dep = cur.fetchone()
        if dep["canary_version"] is not None:
            raise RuntimeError(f"canary already running: v{dep['canary_version']}")

        cur.execute("UPDATE draft_versions SET status = 'canary' WHERE version = %s", (version,))
        cur.execute(
            """
            UPDATE active_deployment
               SET canary_version = %s, canary_pct = %s,
                   generation = generation + 1, updated_at = now(), updated_reason = %s
             WHERE id = TRUE
            """,
            (version, pct, reason),
        )
        record_event(
            "canary_start",
            from_version=dep["active_version"],
            to_version=version,
            reason=reason,
            detail={"canary_pct": pct},
            cur=cur,
        )
    log.info("canary started: v%s at %s%%", version, pct)
    return get_deployment()


def abort_canary(version: int, *, reason: str, slo_evaluation_id: int | None = None) -> Deployment:
    """Pull a candidate out of traffic without touching the active version."""
    with db.conn() as c, c.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(%s)", (_REGISTRY_LOCK,))
        cur.execute("SELECT active_version FROM active_deployment WHERE id = TRUE")
        active = cur.fetchone()["active_version"]
        cur.execute(
            "UPDATE draft_versions SET status = 'rejected', retired_at = now() WHERE version = %s",
            (version,),
        )
        cur.execute(
            """
            UPDATE active_deployment
               SET canary_version = NULL, canary_pct = 0,
                   generation = generation + 1, updated_at = now(), updated_reason = %s
             WHERE id = TRUE
            """,
            (reason,),
        )
        record_event(
            "canary_abort",
            from_version=version,
            to_version=active,
            slo_evaluation_id=slo_evaluation_id,
            reason=reason,
            cur=cur,
        )
    log.warning("canary v%s aborted: %s", version, reason)
    return get_deployment()


def promote(version: int, *, reason: str, slo_evaluation_id: int | None = None) -> Deployment:
    """Take a candidate to full traffic, demoting the incumbent to 'retired'.

    The outgoing active version becomes `last_known_good`: it served real
    traffic and passed a gate, so it is a safe rollback target.
    """
    with db.conn() as c, c.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(%s)", (_REGISTRY_LOCK,))
        cur.execute("SELECT active_version FROM active_deployment WHERE id = TRUE")
        previous = cur.fetchone()["active_version"]

        if previous is not None and previous != version:
            cur.execute(
                "UPDATE draft_versions SET status='retired', retired_at=now() WHERE version=%s",
                (previous,),
            )
        cur.execute(
            "UPDATE draft_versions SET status='active', promoted_at=now() WHERE version=%s",
            (version,),
        )
        cur.execute(
            """
            UPDATE active_deployment
               SET active_version = %s,
                   last_known_good = COALESCE(%s, %s),
                   canary_version = NULL,
                   canary_pct = 0,
                   generation = generation + 1,
                   updated_at = now(),
                   updated_reason = %s
             WHERE id = TRUE
            """,
            (version, previous, version, reason),
        )
        record_event(
            "promote",
            from_version=previous,
            to_version=version,
            slo_evaluation_id=slo_evaluation_id,
            reason=reason,
            cur=cur,
        )
    log.info("promoted v%s to active (previous=%s)", version, previous)
    return get_deployment()


def rollback(*, reason: str, slo_evaluation_id: int | None = None) -> Deployment:
    """Revert the fleet to the last known-good version."""
    with db.conn() as c, c.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(%s)", (_REGISTRY_LOCK,))
        cur.execute("SELECT active_version, last_known_good FROM active_deployment WHERE id = TRUE")
        dep = cur.fetchone()
        current, target = dep["active_version"], dep["last_known_good"]
        if target is None or target == current:
            raise RuntimeError("no distinct last-known-good version to roll back to")

        if current is not None:
            cur.execute(
                "UPDATE draft_versions SET status='rolled_back', retired_at=now() WHERE version=%s",
                (current,),
            )
        cur.execute(
            "UPDATE draft_versions SET status='active', promoted_at=now() WHERE version=%s",
            (target,),
        )
        cur.execute(
            """
            UPDATE active_deployment
               SET active_version = %s, canary_version = NULL, canary_pct = 0,
                   generation = generation + 1, updated_at = now(), updated_reason = %s
             WHERE id = TRUE
            """,
            (target, reason),
        )
        record_event(
            "rollback",
            from_version=current,
            to_version=target,
            slo_evaluation_id=slo_evaluation_id,
            reason=reason,
            cur=cur,
        )
    log.warning("rolled back v%s -> v%s: %s", current, target, reason)
    return get_deployment()


def ack_hotswap(replica_id: str, version: int, generation: int, swap_ms: float) -> None:
    """A replica reporting that it is now serving `version`."""
    record_event(
        "hotswap_ack",
        to_version=version,
        actor="replica",
        replica_id=replica_id,
        reason=f"generation {generation}",
        detail={"generation": generation, "swap_ms": round(swap_ms, 2)},
    )


def record_slo_evaluation(payload: dict[str, Any]) -> int:
    row = db.fetchone(
        """
        INSERT INTO slo_evaluations
            (candidate_version, baseline_version, window_start, window_end,
             candidate_requests, baseline_requests, candidate_tps, baseline_tps,
             candidate_p95_ms, baseline_p95_ms, candidate_acceptance, baseline_acceptance,
             candidate_greedy_mismatches, tps_ratio, p95_ratio, gate_result,
             failed_checks, thresholds, statistical_tests, decision_note)
        VALUES (%(candidate_version)s, %(baseline_version)s, %(window_start)s, %(window_end)s,
                %(candidate_requests)s, %(baseline_requests)s, %(candidate_tps)s,
                %(baseline_tps)s, %(candidate_p95_ms)s, %(baseline_p95_ms)s,
                %(candidate_acceptance)s, %(baseline_acceptance)s,
                %(candidate_greedy_mismatches)s, %(tps_ratio)s, %(p95_ratio)s,
                %(gate_result)s, %(failed_checks)s, %(thresholds)s,
                %(statistical_tests)s, %(decision_note)s)
        RETURNING id
        """,
        {
            **payload,
            "failed_checks": Jsonb(payload.get("failed_checks", [])),
            "thresholds": Jsonb(payload.get("thresholds", {})),
            "statistical_tests": Jsonb(payload.get("statistical_tests", {})),
            "decision_note": payload.get("decision_note"),
        },
    )
    return int(row["id"])


def record_drift_check(payload: dict[str, Any]) -> int:
    row = db.fetchone(
        """
        INSERT INTO drift_checks
            (reference_version, window_events, domain_js, token_psi, acceptance_drop,
             score, threshold, triggered, per_domain, reason)
        VALUES (%(reference_version)s, %(window_events)s, %(domain_js)s, %(token_psi)s,
                %(acceptance_drop)s, %(score)s, %(threshold)s, %(triggered)s,
                %(per_domain)s, %(reason)s)
        RETURNING id
        """,
        {**payload, "per_domain": Jsonb(payload.get("per_domain", {}))},
    )
    return int(row["id"])


def deployment_history(limit: int = 40) -> list[dict[str, Any]]:
    return db.fetchall(
        """
        SELECT ts, action, from_version, to_version, actor, replica_id,
               slo_evaluation_id, reason, detail
          FROM deployment_events
         ORDER BY id DESC
         LIMIT %s
        """,
        (limit,),
    )


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, default=str, indent=2)
