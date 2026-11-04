"""Drafter behaviour on controlled sequences.

Kept separate from the end-to-end tests on purpose: whether a drafter manages
to propose anything depends on the target model's output being structured, and
the tiny random-weight model used in the e2e tests emits noise. These tests use
sequences with known structure so proposal behaviour is actually observable.
"""

from __future__ import annotations

import pytest

from fleet.registry.artifacts import ArtifactStore
from fleet.specdec.drafters.ngram import NgramDrafter


def test_untrained_drafter_proposes_via_context_copy():
    """An empty table still speculates when the context repeats itself --
    which is what lets a cold-start replica benefit from speculation on day
    one, before any version has been trained."""
    drafter = NgramDrafter({}, order=3, min_count=1)
    seq = [10, 11, 12, 13, 14, 10, 11, 12]  # suffix 10,11,12 occurred at index 0
    proposal = drafter.propose(seq, 3)
    assert proposal.source == "copy"
    # What followed that earlier occurrence, continued for gamma tokens.
    assert proposal.token_ids == [13, 14, 10]
    assert drafter.propose(seq, 2).token_ids == [13, 14]


def test_context_copy_prefers_the_longest_match():
    drafter = NgramDrafter({}, order=4, min_count=1)
    #                  the longer match (7,8,9) should win over (9,)
    seq = [7, 8, 9, 99, 1, 2, 3, 7, 8, 9]
    proposal = drafter.propose(seq, 2)
    assert proposal.token_ids == [99, 1]


def test_no_repetition_means_no_proposal():
    drafter = NgramDrafter({}, order=3, min_count=1)
    proposal = drafter.propose([1, 2, 3, 4, 5, 6, 7], 4)
    assert proposal.token_ids == []
    assert proposal.source == "none"


def test_trained_table_beats_context_copy():
    """A trained table is tried before context copy, since it generalises
    across requests rather than only within one."""
    # 'a b c' is always followed by 'd' in training.
    sequences = [[1, 2, 3, 4, 50 + i] for i in range(8)]
    drafter = NgramDrafter.fit(sequences, order=3, min_count=2, min_confidence=0.2)
    proposal = drafter.propose([9, 9, 1, 2, 3], 2)
    assert proposal.source == "ngram"
    assert proposal.token_ids[0] == 4


def test_fit_prunes_low_confidence_and_rare_keys():
    # Key (1,2) is followed by a different token every time: no confidence.
    noisy = [[1, 2, 10 + i] for i in range(10)]
    drafter = NgramDrafter.fit(noisy, order=2, min_count=2, min_confidence=0.5)
    assert (1, 2) not in drafter.table

    # Key (3,4) is consistently followed by 7.
    consistent = [[3, 4, 7] for _ in range(10)]
    drafter2 = NgramDrafter.fit(consistent, order=2, min_count=2, min_confidence=0.5)
    assert drafter2.table[(3, 4)][0] == 7


def test_deeper_speculation_requires_more_confidence():
    """An early wrong token truncates everything behind it, so confidence
    required should rise with depth rather than stay flat."""
    # 60% confident: enough at depth 0, not at depth 3.
    sequences = [[1, 2, 3]] * 6 + [[1, 2, 9]] * 4
    drafter = NgramDrafter.fit(sequences, order=2, min_count=2, min_confidence=0.5)
    assert drafter._lookup([1, 2], depth=0) == 3
    assert drafter._lookup([1, 2], depth=3) is None


def test_save_load_round_trip_preserves_behaviour(tmp_path):
    sequences = [[1, 2, 3, 4, 5]] * 5
    original = NgramDrafter.fit(sequences, order=3, min_count=2, min_confidence=0.2)
    path = tmp_path / "d.json"
    original.save(path)
    restored = NgramDrafter.load(path)

    assert restored.table == original.table
    assert restored.order == original.order
    assert restored.min_confidence == original.min_confidence
    ctx = [1, 2, 3]
    assert restored.propose(ctx, 2).token_ids == original.propose(ctx, 2).token_ids


def test_artifact_store_verifies_digest_before_loading(tmp_path):
    """A replica must never load weights whose bytes it has not checked."""
    drafter = NgramDrafter.fit([[1, 2, 3, 4]] * 4, order=2, min_count=2)
    local = tmp_path / "d.json"
    drafter.save(local)

    store = ArtifactStore(root=tmp_path / "store", cache=tmp_path / "cache")
    stored = store.put(local, name="t")
    assert store.fetch(stored.uri, stored.sha256).exists()

    with pytest.raises(ValueError, match="digest mismatch"):
        store.fetch(stored.uri, "0" * 64)


def test_artifact_cache_avoids_refetching_unchanged_weights(tmp_path):
    """The weight-sync path should move bytes on a version change, not on
    every poll."""
    drafter = NgramDrafter.fit([[1, 2, 3, 4]] * 4, order=2, min_count=2)
    local = tmp_path / "d.json"
    drafter.save(local)
    store = ArtifactStore(root=tmp_path / "store", cache=tmp_path / "cache")
    stored = store.put(local, name="t")

    for _ in range(5):
        store.fetch(stored.uri, stored.sha256)

    stats = store.sync_stats()
    assert stats["fetches"] == 1, "artifact should be downloaded once"
    assert stats["cache_hits"] == 4


def test_identical_artifacts_deduplicate_in_the_store(tmp_path):
    """Content addressing: retraining that happens to produce identical weights
    must not duplicate storage."""
    drafter = NgramDrafter.fit([[1, 2, 3, 4]] * 4, order=2, min_count=2)
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    drafter.save(a)
    drafter.save(b)
    store = ArtifactStore(root=tmp_path / "store", cache=tmp_path / "cache")
    first, second = store.put(a, name="t"), store.put(b, name="t")
    assert first.sha256 == second.sha256
    assert first.uri == second.uri


def test_config_search_prefers_the_better_strategy():
    """Regression guard for a real bug: a thinly-trained n-gram table used to
    pre-empt context copy and produce a version *worse* than the untrained
    cold-start drafter (offline acceptance 0.033 vs 0.250).

    The blend policy and depth penalty are now chosen by scoring each option on
    held-out traffic, so a table that does not help must not take precedence.
    """
    from fleet.specdec.drafters.ngram import DEPTH_PENALTIES, POLICIES
    from fleet.trainer.dataset import TrainingSample, simulate_acceptance
    from fleet.trainer.worker import OBJECTIVE, select_drafter_config

    # Traffic whose continuations repeat within each request -- context copy
    # nails it. The table, fitted on a different pattern, does not.
    holdout = [
        TrainingSample(
            event_id=i,
            domain="code",
            prompt_ids=[500, 501, 502, 503, 504],
            target_ids=[500, 501, 502, 503, 504, 500, 501, 502, 503, 504],
        )
        for i in range(6)
    ]
    misleading = NgramDrafter.fit(
        [[900, 901, 902, 903]] * 6, order=3, min_count=2, min_confidence=0.1
    )
    # Seed confident-but-wrong unigram continuations for every token the replay
    # will actually encounter, so the table always wins the back-off and always
    # proposes a token that gets rejected.
    for tok in (500, 501, 502, 503, 504):
        misleading.table[(tok,)] = (999, 10, 10)

    best, scores = select_drafter_config(misleading, holdout)
    assert len(scores) == len(POLICIES) * len(DEPTH_PENALTIES)
    assert best["policy"] in POLICIES
    assert best["depth_penalty"] in DEPTH_PENALTIES

    # The search must not leave the drafter mutated.
    assert misleading.policy == "ngram_first"

    copy_best = max(v for k, v in scores.items() if k.startswith("copy_first"))
    ngram_best = max(v for k, v in scores.items() if k.startswith("ngram_first"))
    assert copy_best > ngram_best, scores
    assert scores[f"{best['policy']}/d{best['depth_penalty']}"] == max(scores.values())

    misleading.policy = best["policy"]
    misleading.depth_penalty = best["depth_penalty"]
    chosen = simulate_acceptance(misleading, holdout)[OBJECTIVE]
    misleading.policy, misleading.depth_penalty = "ngram_first", 0.25
    naive = simulate_acceptance(misleading, holdout)[OBJECTIVE]
    assert chosen > naive


def test_objective_is_tokens_per_forward_not_acceptance_rate():
    """Guards the metric choice itself.

    A drafter proposing one token and landing it has perfect acceptance but
    poor throughput; one proposing several and landing some is faster. The
    trainer must optimise the second.
    """
    from fleet.trainer.dataset import TrainingSample, simulate_acceptance
    from fleet.trainer.worker import OBJECTIVE

    assert OBJECTIVE == "tokens_per_target_forward"

    samples = [
        TrainingSample(
            event_id=i,
            domain="code",
            prompt_ids=[1, 2, 3, 4, 5, 6],
            target_ids=[1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
        )
        for i in range(4)
    ]
    timid = NgramDrafter({}, order=3, min_count=1)
    timid.copy_min_match = 2
    bold = NgramDrafter({}, order=8, min_count=1)
    bold.copy_min_match = 2

    timid_scored = simulate_acceptance(timid, samples, gamma=1)
    bold_scored = simulate_acceptance(bold, samples, gamma=6)

    # Proposing one token at a time can match or beat the deeper drafter on
    # acceptance rate while producing strictly fewer tokens per forward pass.
    assert bold_scored[OBJECTIVE] > timid_scored[OBJECTIVE]
    assert timid_scored["acceptance_rate"] >= bold_scored["acceptance_rate"] - 1e-9


def test_blend_policy_survives_save_load(tmp_path):
    drafter = NgramDrafter.fit([[1, 2, 3, 4]] * 5, order=3, min_count=2)
    drafter.policy = "longest"
    path = tmp_path / "d.json"
    drafter.save(path)
    assert NgramDrafter.load(path).policy == "longest"


def test_unknown_policy_falls_back_safely():
    drafter = NgramDrafter({}, order=3, min_count=1, policy="nonsense")
    assert drafter.policy == "ngram_first"


def test_artifact_uri_is_portable_across_filesystem_layouts(tmp_path):
    """Regression guard: the registry stored absolute host paths, so a replica
    in a container could not load a version the host had trained.

    `artifact_uri` must be a logical key that each process resolves against its
    own configured root -- host path, container mount, or S3 prefix.
    """
    from fleet.registry.artifacts import SCHEME, ArtifactStore

    drafter = NgramDrafter.fit([[1, 2, 3, 4]] * 4, order=2, min_count=2)
    local = tmp_path / "d.json"
    drafter.save(local)

    # "Host" writes the artifact.
    host_root = tmp_path / "host-store"
    stored = ArtifactStore(root=host_root, cache=tmp_path / "host-cache").put(local, name="t")

    assert stored.uri.startswith(SCHEME)
    assert "/" not in stored.uri[len(SCHEME) :], "uri must not embed a path"
    assert str(host_root) not in stored.uri

    # A different process with the same store mounted at a different path
    # resolves the identical bytes from the same registry row.
    container_root = tmp_path / "container-mount"
    container_root.mkdir()
    for f in host_root.glob("*.json"):
        (container_root / f.name).write_bytes(f.read_bytes())

    other = ArtifactStore(root=container_root, cache=tmp_path / "container-cache")
    fetched = other.fetch(stored.uri, stored.sha256)
    assert fetched.exists()
    assert NgramDrafter.load(fetched).table == drafter.table


def test_legacy_absolute_uris_still_resolve(tmp_path):
    """Rows written before the fix carry absolute paths; they must not become
    unloadable just because the store moved."""
    from fleet.registry.artifacts import ArtifactStore

    drafter = NgramDrafter.fit([[1, 2, 3, 4]] * 4, order=2, min_count=2)
    local = tmp_path / "d.json"
    drafter.save(local)
    root = tmp_path / "store"
    store = ArtifactStore(root=root, cache=tmp_path / "cache")
    stored = store.put(local, name="legacy")

    filename = stored.uri.split("://")[1]
    legacy_uri = f"/some/old/host/path/{filename}"
    fresh = ArtifactStore(root=root, cache=tmp_path / "cache2")
    assert fresh.fetch(legacy_uri, stored.sha256).exists()


def test_missing_artifact_error_names_what_it_tried(tmp_path):
    from fleet.registry.artifacts import ArtifactStore

    store = ArtifactStore(root=tmp_path / "empty", cache=tmp_path / "c")
    with pytest.raises(FileNotFoundError, match="store root is"):
        store.fetch("artifact://nope-000000000000.json", "0" * 64)
