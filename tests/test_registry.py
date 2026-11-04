"""Registry invariants: the database, not application code, is what guarantees
a single active version and a reversible rollout."""

from __future__ import annotations

import psycopg
import pytest

from fleet import db
from fleet.registry import store


def _register(fake_artifact, **kwargs) -> int:
    stored = fake_artifact()
    params = dict(
        drafter_kind="ngram",
        target_model="test/model",
        artifact_uri=stored.uri,
        artifact_sha256=stored.sha256,
        artifact_bytes=stored.size_bytes,
    )
    params.update(kwargs)
    return store.register_version(**params)


def test_version_numbers_are_monotonic(clean_db, fake_artifact):
    versions = [_register(fake_artifact) for _ in range(4)]
    assert versions == [1, 2, 3, 4]


def test_cold_start_activates_and_sets_known_good(clean_db, fake_artifact):
    v = _register(fake_artifact, activate_immediately=True)
    dep = store.get_deployment()
    assert dep.active_version == v
    assert dep.last_known_good == v
    assert dep.generation == 1
    assert store.get_version(v).status == "active"


def test_only_one_active_version_is_possible(clean_db, fake_artifact):
    v1 = _register(fake_artifact, activate_immediately=True)
    v2 = _register(fake_artifact)
    # Bypass promote() and try to force a second active row directly.
    with pytest.raises(psycopg.errors.UniqueViolation):
        db.execute("UPDATE draft_versions SET status = 'active' WHERE version = %s", (v2,))
    assert store.get_version(v1).status == "active"


def test_only_one_canary_at_a_time(clean_db, fake_artifact):
    _register(fake_artifact, activate_immediately=True)
    v2 = _register(fake_artifact)
    v3 = _register(fake_artifact)
    store.start_canary(v2, 20)
    with pytest.raises(RuntimeError, match="canary already running"):
        store.start_canary(v3, 20)


def test_promotion_retires_incumbent_and_records_known_good(clean_db, fake_artifact):
    v1 = _register(fake_artifact, activate_immediately=True)
    v2 = _register(fake_artifact)
    store.start_canary(v2, 20)
    store.promote(v2, reason="gate passed")

    dep = store.get_deployment()
    assert dep.active_version == v2
    assert dep.last_known_good == v1, "rollback target must be the version that was serving"
    assert dep.canary_version is None and dep.canary_pct == 0
    assert store.get_version(v1).status == "retired"
    assert store.get_version(v2).status == "active"


def test_rollback_returns_to_known_good(clean_db, fake_artifact):
    v1 = _register(fake_artifact, activate_immediately=True)
    v2 = _register(fake_artifact)
    store.start_canary(v2, 20)
    store.promote(v2, reason="gate passed")

    store.rollback(reason="throughput regression")
    dep = store.get_deployment()
    assert dep.active_version == v1
    assert store.get_version(v2).status == "rolled_back"
    assert store.get_version(v1).status == "active"


def test_rollback_without_known_good_is_refused(clean_db, fake_artifact):
    _register(fake_artifact, activate_immediately=True)
    with pytest.raises(RuntimeError, match="no distinct last-known-good"):
        store.rollback(reason="nothing to revert to")


def test_aborted_canary_leaves_active_untouched(clean_db, fake_artifact):
    v1 = _register(fake_artifact, activate_immediately=True)
    v2 = _register(fake_artifact)
    store.start_canary(v2, 25)
    store.abort_canary(v2, reason="p95 regression")

    dep = store.get_deployment()
    assert dep.active_version == v1
    assert dep.canary_version is None
    assert store.get_version(v2).status == "rejected"
    assert store.get_version(v1).status == "active"


def test_every_transition_is_audited(clean_db, fake_artifact):
    v1 = _register(fake_artifact, activate_immediately=True)
    v2 = _register(fake_artifact)
    store.start_canary(v2, 20)
    store.promote(v2, reason="gate passed")
    store.rollback(reason="regression")

    actions = [e["action"] for e in store.deployment_history(50)]
    for expected in ("register", "canary_start", "promote", "rollback"):
        assert expected in actions, f"{expected} missing from the audit trail"

    rollback_event = next(e for e in store.deployment_history(50) if e["action"] == "rollback")
    assert rollback_event["from_version"] == v2
    assert rollback_event["to_version"] == v1
    assert rollback_event["reason"] == "regression"


def test_generation_advances_on_every_rollout_change(clean_db, fake_artifact):
    v1 = _register(fake_artifact, activate_immediately=True)
    gens = [store.get_deployment().generation]
    v2 = _register(fake_artifact)
    store.start_canary(v2, 20)
    gens.append(store.get_deployment().generation)
    store.promote(v2, reason="ok")
    gens.append(store.get_deployment().generation)
    store.rollback(reason="revert")
    gens.append(store.get_deployment().generation)
    assert gens == sorted(set(gens)), f"generation must strictly increase, got {gens}"
    assert v1 == store.get_deployment().active_version


def test_traffic_cannot_reference_an_unknown_version(clean_db, fake_artifact):
    """Provenance is enforced: no event may name a version that is not in the
    registry."""
    _register(fake_artifact, activate_immediately=True)
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        db.execute(
            """
            INSERT INTO traffic_events
                (request_id, replica_id, draft_version, domain, prompt_sha256,
                 prompt_tokens, generated_tokens, total_ms, tokens_per_s)
            VALUES (gen_random_uuid(), 'r1', 999, 'chat', repeat('a',64), 10, 10, 100, 50)
            """
        )
