from __future__ import annotations

import os
import uuid

import pytest

# Point every component at a dedicated test database before fleet.config is
# imported, so a test run can never write into the dev fleet's registry.
os.environ.setdefault("FLEET_DATABASE_URL", "postgresql://fleet:fleet@localhost:5433/fleet_test")
os.environ.setdefault("FLEET_DEVICE", "cpu")
os.environ.setdefault("FLEET_DTYPE", "float32")


def _ensure_test_database() -> None:
    import psycopg

    admin = "postgresql://fleet:fleet@localhost:5433/postgres"
    with psycopg.connect(admin, autocommit=True) as c:
        exists = c.execute("SELECT 1 FROM pg_database WHERE datname = 'fleet_test'").fetchone()
        if not exists:
            c.execute("CREATE DATABASE fleet_test")


@pytest.fixture(scope="session", autouse=True)
def database():
    try:
        _ensure_test_database()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Postgres not reachable for integration tests: {exc}")

    from fleet import db

    db.wait_for_db()
    db.apply_schema()
    yield
    db.close_pool()


@pytest.fixture
def clean_db(database):
    """Truncate the registry between tests so version numbering is predictable."""
    from fleet import db

    db.execute(
        """
        TRUNCATE traffic_samples, traffic_events, slo_evaluations,
                 deployment_events, drift_checks, draft_versions
            RESTART IDENTITY CASCADE
        """
    )
    # active_deployment has FKs into draft_versions, so TRUNCATE ... CASCADE
    # above removes the singleton row too. Put it back.
    db.execute("INSERT INTO active_deployment (id) VALUES (TRUE) ON CONFLICT (id) DO NOTHING")
    db.execute(
        """
        UPDATE active_deployment
           SET active_version = NULL, canary_version = NULL, canary_pct = 0,
               last_known_good = NULL, generation = 0, updated_reason = NULL
         WHERE id = TRUE
        """
    )
    yield db


@pytest.fixture
def fake_artifact(tmp_path):
    """Register-able artifact bytes without training anything."""
    from fleet.specdec.drafters.ngram import NgramDrafter

    def _make(name: str = "test", table: dict | None = None):
        drafter = NgramDrafter(table or {}, order=3, min_count=1)
        path = tmp_path / f"{name}-{uuid.uuid4().hex[:8]}.json"
        drafter.save(path)
        from fleet.registry.artifacts import ArtifactStore

        return ArtifactStore(root=tmp_path / "store").put(path, name=name)

    return _make
