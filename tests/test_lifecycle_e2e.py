"""End-to-end: live traffic -> drift -> retrain -> canary -> SLO gate ->
hot-swap, with a real model doing real speculative decoding.

Uses a tiny model so the control plane is exercised in seconds; the throughput
numbers that matter come from the benchmark, not from here. What is being
tested is that the loop closes with no manual step in it.
"""

from __future__ import annotations

import hashlib
import uuid

import pytest

from fleet import db
from fleet.canary import controller
from fleet.config import settings
from fleet.registry import store
from fleet.serving.hotswap import DeploymentWatcher, load_drafter
from fleet.serving.telemetry import RequestRecord, TelemetryWriter
from fleet.specdec.drafters.base import NullDrafter
from fleet.specdec.engine import SpeculativeEngine
from fleet.specdec.target import TargetModel
from fleet.trainer import worker
from fleet.trainer.dataset import harvest, simulate_acceptance

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"

CODE_PROMPTS = [
    "def reverse(items):\n    return items[::-1]\ndef reverse_again(items):",
    "def total(rows):\n    return sum(rows)\ndef total_again(rows):",
    "class Cache:\n    def get(self, key):\n        return self.store[key]\n    def put(self, key",
    "for i in range(10):\n    print(i)\nfor j in range(10):",
]

LEGAL_PROMPTS = [
    "The appellant submits that the tribunal erred in law. The appellant further submits",
    "Pursuant to section 12 of the Act, the respondent shall. Pursuant to section 13",
    "It is ordered that the application be dismissed. It is further ordered",
    "The court finds that the clause was unenforceable. The court further finds",
]


@pytest.fixture(scope="module")
def engine():
    target = TargetModel.load(TINY, device="cpu", dtype="float32")
    return SpeculativeEngine(target, NullDrafter(), gamma=4)


def serve(
    engine: SpeculativeEngine,
    prompts,
    domain: str,
    *,
    version: int,
    is_canary: bool = False,
    repeats: int = 1,
    audit_every: int = 5,
    drafter=None,
    slow_factor: float = 1.0,
) -> list:
    """Serve requests through the real engine and write real telemetry rows."""
    results = []
    n = 0
    for _ in range(repeats):
        for prompt in prompts:
            ids = engine.target.tokenizer(prompt).input_ids
            use = drafter if drafter is not None else engine.drafter
            res = engine.decode(ids, max_new_tokens=24, drafter=use)
            n += 1

            greedy_match = None
            if n % audit_every == 0:
                ref = engine.greedy_reference(ids, max_new_tokens=24)
                greedy_match = ref.generated_ids == res.generated_ids

            TelemetryWriter._write(
                RequestRecord(
                    request_id=uuid.uuid4(),
                    replica_id="replica-test",
                    draft_version=version,
                    is_canary=is_canary,
                    domain=domain,
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    prompt_tokens=len(ids),
                    generated_tokens=len(res.generated_ids),
                    draft_proposed=res.draft_proposed,
                    draft_accepted=res.draft_accepted,
                    target_forwards=res.target_forwards,
                    ttft_ms=res.ttft_ms,
                    total_ms=res.total_ms * slow_factor,
                    tokens_per_s=res.tokens_per_s / slow_factor,
                    greedy_match=greedy_match,
                    prompt_token_ids=ids,
                    target_token_ids=res.generated_ids,
                )
            )
            results.append(res)
    return results


@pytest.fixture
def tuned(clean_db, monkeypatch):
    """Shrink the thresholds so the loop completes in a test-sized run."""
    monkeypatch.setattr(settings, "train_min_samples", 12)
    monkeypatch.setattr(settings, "canary_min_requests", 8)
    monkeypatch.setattr(settings, "drift_window", 120)
    monkeypatch.setattr(settings, "target_model", TINY)
    monkeypatch.setattr(settings, "replica_urls", "")  # no HTTP push in tests
    return clean_db


def test_cold_start_serves_before_anything_is_trained(tuned, engine):
    version = worker.bootstrap()
    assert version == 1
    assert store.get_deployment().active_version == 1

    drafter = load_drafter(1)
    assert drafter.meta.version == 1

    results = serve(engine, CODE_PROMPTS, "code", version=1, drafter=drafter, audit_every=1)
    # The point of cold start: the fleet serves, and is correct, before any
    # training has happened. Whether the untrained drafter manages to propose
    # anything is model-dependent -- this target has random weights, so its
    # continuations are noise and nothing recurs for context copy to latch
    # onto. That behaviour is covered on controlled sequences in
    # tests/test_drafters.py instead.
    assert all(len(r.generated_ids) > 0 for r in results)
    mismatches = db.fetchone(
        "SELECT COUNT(*) AS n FROM traffic_events WHERE greedy_match IS FALSE"
    )["n"]
    assert mismatches == 0
    assert db.fetchone("SELECT COUNT(*) AS n FROM traffic_samples")["n"] == len(results)


def test_full_self_improvement_loop(tuned, engine):
    # --- cold start -------------------------------------------------------
    worker.bootstrap()
    v1_drafter = load_drafter(1)
    serve(engine, CODE_PROMPTS, "code", version=1, drafter=v1_drafter, repeats=5)

    # --- first trained version comes from live traffic --------------------
    v2 = worker.tick(__import__("fleet.drift.detector", fromlist=["DriftDetector"]).DriftDetector())
    assert v2 == 2, "trainer should produce a version from the cold-start traffic"
    assert store.get_version(v2).status == "candidate"
    assert store.get_version(v2).train_sample_count > 0
    assert store.get_version(v2).training_manifest["domain_counts"] == {"code": 16}

    # --- controller puts it into canary -----------------------------------
    controller.tick()
    dep = store.get_deployment()
    assert dep.canary_version == v2
    assert dep.canary_pct == settings.canary_pct
    assert store.get_version(v2).status == "canary"

    # --- both arms take traffic concurrently ------------------------------
    v2_drafter = load_drafter(v2)
    serve(engine, CODE_PROMPTS, "code", version=1, drafter=v1_drafter, repeats=3)
    serve(engine, CODE_PROMPTS, "code", version=v2, drafter=v2_drafter, is_canary=True, repeats=3)

    # --- the gate decides, and records why --------------------------------
    decision = controller.evaluate_canary()
    assert decision is not None
    assert decision.result in {"pass", "fail"}
    evals = db.fetchall("SELECT * FROM slo_evaluations ORDER BY id DESC LIMIT 1")
    assert evals and evals[0]["candidate_version"] == v2

    dep = store.get_deployment()
    if decision.passed:
        assert dep.active_version == v2
        assert dep.last_known_good == 1
    else:
        assert dep.active_version == 1
        assert store.get_version(v2).status == "rejected"

    # --- no audited request ever diverged from greedy ---------------------
    mismatches = db.fetchone(
        "SELECT COUNT(*) AS n FROM traffic_events WHERE greedy_match IS FALSE"
    )["n"]
    assert mismatches == 0, "speculative decoding changed the output"

    # --- every served response is traceable to an artifact ----------------
    untraceable = db.fetchone(
        "SELECT COUNT(*) AS n FROM response_provenance WHERE artifact_sha256 IS NULL"
    )["n"]
    assert untraceable == 0


def test_drift_gates_retraining(tuned, engine):
    """Stationary traffic must not retrain; a domain shift must."""
    from fleet.drift.detector import DriftDetector

    worker.bootstrap()
    v1_drafter = load_drafter(1)
    serve(engine, CODE_PROMPTS, "code", version=1, drafter=v1_drafter, repeats=5)

    v2 = worker.tick(DriftDetector())
    assert v2 == 2
    controller.tick()  # canary
    controller.evaluate_canary()  # may pass or fail; force v2 active
    if store.get_deployment().active_version != v2:
        store.promote(v2, reason="forced active for drift test")
    v2_drafter = load_drafter(v2)

    # More of the same workload -> no retrain.
    serve(engine, CODE_PROMPTS, "code", version=v2, drafter=v2_drafter, repeats=5)
    detector = DriftDetector()
    assert worker.tick(detector) is None
    check = db.fetchall("SELECT * FROM drift_checks ORDER BY id DESC LIMIT 1")[0]
    assert check["triggered"] is False
    assert "stationary" in check["reason"]

    # Workload moves to an entirely different domain -> retrain.
    serve(engine, LEGAL_PROMPTS, "legal", version=v2, drafter=v2_drafter, repeats=12)
    v3 = worker.tick(DriftDetector())
    check = db.fetchall("SELECT * FROM drift_checks ORDER BY id DESC LIMIT 1")[0]
    assert check["triggered"] is True, check["reason"]
    assert v3 is not None, "a domain shift should produce a new candidate"
    assert "legal" in store.get_version(v3).training_manifest["domain_counts"]


def test_hot_swap_is_picked_up_without_restart(tuned, engine):
    worker.bootstrap()
    watcher = DeploymentWatcher(engine, interval=60)
    watcher.sync(force=True)
    assert watcher.state.active_version == 1
    assert watcher.state.generation == store.get_deployment().generation

    serve(engine, CODE_PROMPTS, "code", version=1, drafter=load_drafter(1), repeats=5)
    from fleet.drift.detector import DriftDetector

    v2 = worker.tick(DriftDetector())
    store.start_canary(v2, 50)

    # The replica has not been restarted; one poll should load the canary too.
    assert watcher.sync() is True
    assert sorted(watcher.state.drafters) == [1, v2]
    assert watcher.state.canary_version == v2

    # Both variants are now resolvable on the same replica.
    seen = {watcher.resolve(force_variant="active")[0], watcher.resolve(force_variant="canary")[0]}
    assert seen == {1, v2}

    store.promote(v2, reason="test promotion")
    assert watcher.sync() is True
    assert watcher.state.active_version == v2
    assert watcher.state.canary_version is None
    assert sorted(watcher.state.drafters) == [v2], "retired version should be evicted"
    assert engine.drafter.meta.version == v2

    acks = [e for e in store.deployment_history(40) if e["action"] == "hotswap_ack"]
    assert acks, "a replica swap should be recorded in the audit trail"


def test_offline_acceptance_matches_what_the_engine_actually_does(tuned, engine):
    """The offline score is only useful if it predicts real acceptance.

    `simulate_acceptance` replays accept/reject against the recorded target
    path with no model forwards; the engine computes it against live logits. On
    the same drafter and the same prompts the two must agree exactly.
    """
    worker.bootstrap()
    v1_drafter = load_drafter(1)
    serve(engine, CODE_PROMPTS, "code", version=1, drafter=v1_drafter, repeats=6)

    from fleet.drift.detector import DriftDetector

    v2 = worker.tick(DriftDetector())
    drafter = load_drafter(v2)

    samples = harvest(limit=24)
    offline = simulate_acceptance(drafter, samples, gamma=settings.gamma)

    proposed = accepted = 0
    for s in samples:
        res = engine.decode(s.prompt_ids, max_new_tokens=len(s.target_ids), drafter=drafter)
        assert res.generated_ids == s.target_ids, "greedy path drifted between runs"
        proposed += res.draft_proposed
        accepted += res.draft_accepted

    assert proposed == offline["proposed"]
    assert accepted == offline["accepted"]


def test_cold_start_does_not_speculate(tuned, engine):
    """Measured on a mixed prompt pool, an untrained context-copy drafter ran
    at 82.9 tok/s against 91.0 for plain greedy decoding -- speculation that has
    never been measured was a net loss. So the fleet starts with none, and the
    first speculative version has to earn its place through the canary.
    """
    version = worker.bootstrap()
    meta = store.get_version(version)
    assert meta.drafter_kind == "none"
    assert meta.status == "active"
    assert meta.hyperparams.get("speculation") is False

    drafter = load_drafter(version)
    results = serve(engine, CODE_PROMPTS, "code", version=version, drafter=drafter)
    assert all(r.draft_proposed == 0 for r in results), "cold start must not speculate"
    assert all(len(r.generated_ids) > 0 for r in results)


def test_data_growth_retrains_on_a_stationary_workload(tuned, engine, monkeypatch):
    """A stationary workload must still improve the drafter.

    Drift detection answers "has the workload moved?", not "is the drafter as
    good as my data now allows?". Without a growth trigger an 800-request
    stationary run produced exactly one retrain and the fleet froze on its
    first trained version -- the opposite of self-improving.
    """
    from fleet.drift.detector import DriftDetector

    monkeypatch.setattr(settings, "retrain_data_growth", 1.0)
    worker.bootstrap()

    # Enough traffic for the first trained version.
    serve(engine, CODE_PROMPTS, "code", version=1, drafter=load_drafter(1), repeats=5)
    v2 = worker.tick(DriftDetector())
    assert v2 == 2
    store.start_canary(v2, 50)
    store.promote(v2, reason="forced active for the growth test")
    trained_on = store.get_version(v2).train_sample_count
    assert trained_on > 0

    # Same workload, no drift. Immediately after promotion there is nothing new
    # to learn from, so nothing should happen.
    detector = DriftDetector()
    assert worker.tick(detector) is None
    check = db.fetchall("SELECT * FROM drift_checks ORDER BY id DESC LIMIT 1")[0]
    assert check["triggered"] is False

    # Now accumulate substantially more traffic of the *same* kind.
    serve(engine, CODE_PROMPTS, "code", version=v2, drafter=load_drafter(v2), repeats=8)
    fresh = worker.new_samples_since(store.get_version(v2).trained_to_event_id)
    assert fresh >= trained_on, f"test needs {trained_on} new samples, got {fresh}"

    # require_improvement=False isolates the *trigger* from the separate
    # improvement guard: a growth-triggered candidate that fails to beat the
    # incumbent on the holdout is still correctly discarded, which would make a
    # plain tick() indistinguishable from "the trigger never fired".
    v3 = worker.tick_with_policy(DriftDetector(), require_improvement=False)
    assert v3 is not None, "more data on a stationary workload should still retrain"
    assert "data growth" in (store.get_version(v3).notes or "")

    # And the drift check still correctly reports no drift -- the retrain came
    # from data growth, not from a false drift positive.
    check = db.fetchall("SELECT * FROM drift_checks ORDER BY id DESC LIMIT 1")[0]
    assert check["triggered"] is False


def test_growth_trigger_respects_the_threshold(tuned, engine, monkeypatch):
    """The growth trigger must not fire on a trickle of new traffic, or it
    degenerates into retraining on every tick."""
    from fleet.drift.detector import DriftDetector

    monkeypatch.setattr(settings, "retrain_data_growth", 10.0)  # demand 10x
    worker.bootstrap()
    serve(engine, CODE_PROMPTS, "code", version=1, drafter=load_drafter(1), repeats=5)
    v2 = worker.tick(DriftDetector())
    store.start_canary(v2, 50)
    store.promote(v2, reason="forced active")

    serve(engine, CODE_PROMPTS, "code", version=v2, drafter=load_drafter(v2), repeats=2)
    assert worker.tick_with_policy(DriftDetector(), require_improvement=False) is None, (
        "a small amount of new traffic must not trigger a 10x growth retrain"
    )
