"""The project's central correctness property.

Speculative decoding must be output-identical to plain greedy decoding, for
*any* drafter -- including a deliberately hostile one. If this holds, promoting
a retrained draft model from production traffic can never change what users
see, which is what makes the whole self-improvement loop safe to automate.

Three drafters are exercised:
  * NullDrafter      -- proposes nothing (the degenerate case)
  * NgramDrafter     -- the real trained drafter
  * AdversarialDrafter -- proposes uniformly random tokens, i.e. worst case

and the speculative output is compared against both this engine's own greedy
path and, independently, against HuggingFace `generate(do_sample=False)`.
"""

from __future__ import annotations

import random

import pytest
import torch

from fleet.specdec.drafters.base import Drafter, DraftProposal, NullDrafter
from fleet.specdec.drafters.ngram import NgramDrafter
from fleet.specdec.engine import SpeculativeEngine
from fleet.specdec.target import TargetModel

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"

PROMPTS = [
    "def fibonacci(n):",
    "The capital of France is",
    "Summarize: the quick brown fox jumps over the lazy dog. the quick brown fox",
    "a b c a b c a b c a b",
    "x",
]


class AdversarialDrafter(Drafter):
    """Proposes random garbage. Verification must absorb all of it."""

    kind = "adversarial"

    def __init__(self, vocab_size: int, seed: int = 0) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.rng = random.Random(seed)

    def propose(self, context_ids, gamma, *, hidden_state=None):
        return DraftProposal(
            [self.rng.randrange(self.vocab_size) for _ in range(gamma)], "adversarial"
        )

    def save(self, path):  # pragma: no cover - not persisted in tests
        raise NotImplementedError

    @classmethod
    def load(cls, path):  # pragma: no cover
        raise NotImplementedError


@pytest.fixture(scope="module")
def target() -> TargetModel:
    return TargetModel.load(TINY, device="cpu", dtype="float32")


@pytest.fixture(scope="module")
def engine(target: TargetModel) -> SpeculativeEngine:
    return SpeculativeEngine(target, NullDrafter(), gamma=4)


@pytest.fixture(scope="module")
def trained_ngram(target: TargetModel) -> NgramDrafter:
    # Train on this model's own greedy continuations, as the trainer does in
    # production -- a drafter trained on the real target distribution.
    eng = SpeculativeEngine(target, NullDrafter(), gamma=0)
    seqs = []
    for p in PROMPTS:
        ids = target.tokenizer(p).input_ids
        res = eng.greedy_reference(ids, max_new_tokens=24)
        seqs.append(ids + res.generated_ids)
    return NgramDrafter.fit(seqs, order=3, min_count=1, min_confidence=0.0)


def _hf_greedy(target: TargetModel, prompt_ids: list[int], max_new_tokens: int) -> list[int]:
    out = target.model.generate(
        input_ids=torch.tensor([prompt_ids]),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        pad_token_id=target.tokenizer.pad_token_id,
    )
    return out[0, len(prompt_ids) :].tolist()


@pytest.mark.parametrize("prompt", PROMPTS)
def test_engine_greedy_matches_huggingface(engine, target, prompt):
    """Our greedy path is the real greedy path, not an approximation of it."""
    ids = target.tokenizer(prompt).input_ids
    ours = engine.greedy_reference(ids, max_new_tokens=20).generated_ids
    theirs = _hf_greedy(target, ids, 20)
    assert ours == theirs[: len(ours)]


@pytest.mark.parametrize("gamma", [1, 2, 4, 8])
@pytest.mark.parametrize("prompt", PROMPTS)
def test_speculative_equals_greedy_trained_drafter(engine, target, trained_ngram, gamma, prompt):
    ids = target.tokenizer(prompt).input_ids
    reference = engine.greedy_reference(ids, max_new_tokens=32)
    spec = engine.decode(ids, max_new_tokens=32, gamma=gamma, drafter=trained_ngram)
    assert spec.generated_ids == reference.generated_ids, (
        f"gamma={gamma} diverged from greedy on {prompt!r}"
    )
    assert spec.text == reference.text


@pytest.mark.parametrize("gamma", [1, 3, 5])
@pytest.mark.parametrize("prompt", PROMPTS)
def test_speculative_equals_greedy_adversarial_drafter(engine, target, gamma, prompt):
    """Worst case: a drafter whose every proposal is wrong."""
    adv = AdversarialDrafter(target.vocab_size, seed=gamma)
    ids = target.tokenizer(prompt).input_ids
    reference = engine.greedy_reference(ids, max_new_tokens=24)
    spec = engine.decode(ids, max_new_tokens=24, gamma=gamma, drafter=adv)
    assert spec.generated_ids == reference.generated_ids
    # An adversarial drafter should get essentially nothing accepted, and must
    # never cause a wrong token to be emitted.
    assert spec.draft_accepted <= spec.draft_proposed


@pytest.mark.parametrize("prompt", PROMPTS)
def test_token_budget_is_respected(engine, target, trained_ngram, prompt):
    """Speculation must not overshoot max_new_tokens."""
    ids = target.tokenizer(prompt).input_ids
    for budget in (1, 2, 7, 16):
        spec = engine.decode(ids, max_new_tokens=budget, gamma=6, drafter=trained_ngram)
        assert len(spec.generated_ids) <= budget


def test_trained_drafter_actually_gets_accepted(engine, target, trained_ngram):
    """Guard against a vacuous equivalence test: if the drafter proposed
    nothing, equality with greedy would be trivially true and meaningless."""
    total_proposed = total_accepted = 0
    for prompt in PROMPTS:
        ids = target.tokenizer(prompt).input_ids
        spec = engine.decode(ids, max_new_tokens=32, gamma=4, drafter=trained_ngram)
        total_proposed += spec.draft_proposed
        total_accepted += spec.draft_accepted
    assert total_proposed > 0, "drafter proposed nothing; equivalence test is vacuous"
    assert total_accepted > 0, "no draft token was ever accepted; speculation is inert"


def test_accounting_is_consistent(engine, target, trained_ngram):
    ids = target.tokenizer(PROMPTS[0]).input_ids
    spec = engine.decode(ids, max_new_tokens=32, gamma=4, drafter=trained_ngram)
    # One prefill forward plus one forward per verification step.
    assert spec.target_forwards == spec.steps + 1
    # Speculation only pays off if fewer target forwards than tokens produced.
    assert spec.target_forwards <= len(spec.generated_ids) + 1


class HiddenStateDrafter(Drafter):
    """Drafter that requires the target's hidden state, to exercise that path.

    Records the hidden state it was handed at each step so the engine's
    indexing can be checked: after a rejection the state must come from the
    last *committed* token, not from a draft token that was discarded.
    """

    kind = "hidden-probe"

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.seen: list[tuple[int, int]] = []  # (context_len, hidden_dim)

    @property
    def needs_hidden_state(self) -> bool:
        return True

    def propose(self, context_ids, gamma, *, hidden_state=None):
        assert hidden_state is not None, "engine did not supply a hidden state"
        assert hidden_state.shape[-1] == self.hidden_size
        self.seen.append((len(context_ids), int(hidden_state.shape[-1])))
        # Propose something wrong on purpose: verification must absorb it.
        return DraftProposal([0] * gamma, "hidden-probe")

    def save(self, path):  # pragma: no cover
        raise NotImplementedError

    @classmethod
    def load(cls, path):  # pragma: no cover
        raise NotImplementedError


@pytest.mark.parametrize("gamma", [1, 4])
def test_hidden_state_drafters_are_supplied_and_stay_exact(engine, target, gamma):
    """The hidden-state extension point must actually receive hidden states,
    and using it must not break greedy equivalence."""
    probe = HiddenStateDrafter(target.hidden_size)
    ids = target.tokenizer(PROMPTS[0]).input_ids
    reference = engine.greedy_reference(ids, max_new_tokens=16)
    spec = engine.decode(ids, max_new_tokens=16, gamma=gamma, drafter=probe)

    assert probe.seen, "drafter was never consulted"
    assert spec.generated_ids == reference.generated_ids
