"""Speculative decoding with exact greedy verification.

The invariant this module exists to hold:

    decode(prompt, drafter=ANY)  ==  greedy_decode(prompt)   token for token

Every speculative token is accepted only if it equals the token the target
model would itself have chosen greedily at that position. A mismatch truncates
the proposal and the target's own choice is committed instead. So the drafter
can be stale, badly trained, or adversarial and the served text is unchanged --
only the token/second changes. That is why retraining the drafter from
production traffic can be fully automated: the blast radius of a bad draft
version is throughput, not correctness.

`tests/test_greedy_equivalence.py` asserts the invariant; the serving path also
audits a sampled fraction of live requests against it and records the result in
`traffic_events.greedy_match`.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field

import torch
from transformers import DynamicCache

from fleet.specdec.drafters.base import Drafter, NullDrafter
from fleet.specdec.target import TargetModel

log = logging.getLogger(__name__)


@dataclass
class DecodeResult:
    prompt_ids: list[int]
    generated_ids: list[int]
    text: str

    draft_proposed: int = 0
    draft_accepted: int = 0
    target_forwards: int = 0
    steps: int = 0

    ttft_ms: float = 0.0
    total_ms: float = 0.0
    proposal_sources: Counter = field(default_factory=Counter)
    stopped_on_eos: bool = False

    @property
    def tokens_per_s(self) -> float:
        if self.total_ms <= 0:
            return 0.0
        return len(self.generated_ids) / (self.total_ms / 1000.0)

    @property
    def acceptance_rate(self) -> float | None:
        if self.draft_proposed == 0:
            return None
        return self.draft_accepted / self.draft_proposed

    @property
    def mean_accepted_per_step(self) -> float:
        return self.draft_accepted / self.steps if self.steps else 0.0


def _cache_length(cache: DynamicCache) -> int:
    return int(cache.get_seq_length())


def _new_cache() -> DynamicCache:
    cache = DynamicCache()
    # Sliding-window layers refuse to be cropped unless they were tracking past
    # states. Speculative decoding always needs to roll the cache back, so opt
    # in up front where the API exists (no-op on plain dynamic layers).
    recorder = getattr(cache, "activate_past_recording", None)
    if callable(recorder):
        recorder()
    return cache


def _crop_cache(cache: DynamicCache, target_length: int) -> None:
    """Roll the KV cache back to `target_length`, dropping entries written for
    draft tokens that verification rejected.

    `crop` is passed a *negative* count of tokens to remove, which is the
    spelling both transformers 4.x (negative == relative) and 5.x (negative is
    the only accepted form) agree on.
    """
    current = _cache_length(cache)
    if current <= target_length:
        return
    cache.crop(target_length - current)


class SpeculativeEngine:
    """Owns the target model and the currently-deployed drafter.

    The drafter is swapped by reference under a lock, so a hot-swap costs one
    pointer write and never interrupts an in-flight request: a request that
    started on version N finishes on version N.
    """

    def __init__(
        self,
        target: TargetModel,
        drafter: Drafter | None = None,
        *,
        gamma: int = 4,
    ) -> None:
        self.target = target
        self._drafter: Drafter = drafter or NullDrafter()
        self.gamma = gamma
        self._eos = target.eos_ids

    @property
    def drafter(self) -> Drafter:
        return self._drafter

    def swap_drafter(self, drafter: Drafter) -> Drafter:
        previous, self._drafter = self._drafter, drafter
        log.info(
            "drafter swapped: %s v%s -> %s v%s",
            previous.kind,
            previous.meta.version,
            drafter.kind,
            drafter.meta.version,
        )
        return previous

    # -----------------------------------------------------------------------

    @torch.inference_mode()
    def decode(
        self,
        prompt_ids: list[int],
        *,
        max_new_tokens: int,
        gamma: int | None = None,
        drafter: Drafter | None = None,
    ) -> DecodeResult:
        gamma = self.gamma if gamma is None else gamma
        drafter = drafter or self._drafter
        drafter.reset()

        model = self.target.model
        device = self.target.device
        started = time.perf_counter()

        committed = list(prompt_ids)
        generated: list[int] = []
        result = DecodeResult(prompt_ids=list(prompt_ids), generated_ids=generated, text="")

        cache = _new_cache()
        # A drafter conditioned on the target's hidden state (Medusa-style
        # heads, for instance) needs them produced; they are off by default
        # because materialising every layer's activations is not free.
        want_hidden = drafter.needs_hidden_state
        last_hidden: torch.Tensor | None = None

        # --- prefill: one forward over the whole prompt ---------------------
        inp = torch.tensor([committed], dtype=torch.long, device=device)
        out = model(
            input_ids=inp,
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.arange(len(committed), device=device),
            output_hidden_states=want_hidden,
        )
        cache = out.past_key_values
        result.target_forwards += 1
        if want_hidden and out.hidden_states is not None:
            last_hidden = out.hidden_states[-1][0, -1]

        first = int(out.logits[0, -1].argmax())
        committed.append(first)
        generated.append(first)
        result.ttft_ms = (time.perf_counter() - started) * 1000.0
        if first in self._eos:
            result.stopped_on_eos = True
            return self._finalize(result, started)

        # --- speculative loop ----------------------------------------------
        while len(generated) < max_new_tokens:
            budget = max_new_tokens - len(generated)
            # One bonus token is always committed per step, so only ask the
            # drafter for what the budget can still absorb.
            ask = max(0, min(gamma, budget - 1))

            proposal = drafter.propose(committed, ask, hidden_state=last_hidden)
            draft = proposal.token_ids[:ask]
            result.proposal_sources[proposal.source] += 1
            result.draft_proposed += len(draft)

            # Feed the last committed token plus the draft. Position j of the
            # output predicts the token that follows position j of the input,
            # so logits[0] is the target's own choice for the slot draft[0]
            # occupies -- which is exactly what verification compares against.
            step_input = [committed[-1], *draft]
            cache_offset = len(committed) - 1
            if _cache_length(cache) != cache_offset:
                _crop_cache(cache, cache_offset)

            out = model(
                input_ids=torch.tensor([step_input], dtype=torch.long, device=device),
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.arange(
                    cache_offset, cache_offset + len(step_input), device=device
                ),
                output_hidden_states=want_hidden,
            )
            cache = out.past_key_values
            result.target_forwards += 1
            result.steps += 1

            target_choice = out.logits[0].argmax(dim=-1).tolist()  # len(draft) + 1

            n_accept = 0
            for i, tok in enumerate(draft):
                if tok == target_choice[i]:
                    n_accept += 1
                else:
                    break
            result.draft_accepted += n_accept

            # The hidden state that conditions the *next* proposal belongs to
            # the position of the last committed token, which is `n_accept` --
            # not the last position of this step's input. Those differ whenever
            # a draft token was rejected, so indexing at -1 would feed the
            # drafter the state of a token that was thrown away.
            if want_hidden and out.hidden_states is not None:
                last_hidden = out.hidden_states[-1][0, n_accept]

            # Accepted drafts, then the target's own token at the first
            # divergence. The bonus token is free: its logits came from the
            # same forward pass.
            new_tokens = [*draft[:n_accept], target_choice[n_accept]]

            for tok in new_tokens:
                committed.append(tok)
                generated.append(tok)
                if tok in self._eos:
                    result.stopped_on_eos = True
                    break
                if len(generated) >= max_new_tokens:
                    break
            if result.stopped_on_eos:
                break

            # Rejected draft tokens left stale entries in the cache.
            _crop_cache(cache, len(committed) - 1)

        return self._finalize(result, started)

    def _finalize(self, result: DecodeResult, started: float) -> DecodeResult:
        result.total_ms = (time.perf_counter() - started) * 1000.0
        result.text = self.target.tokenizer.decode(result.generated_ids, skip_special_tokens=True)
        return result

    # -----------------------------------------------------------------------

    @torch.inference_mode()
    def greedy_reference(self, prompt_ids: list[int], *, max_new_tokens: int) -> DecodeResult:
        """Plain target-only greedy decoding: the correctness oracle.

        Implemented through the same code path with a drafter that proposes
        nothing, so the comparison isolates speculation rather than two
        separately-written decode loops.
        """
        return self.decode(
            prompt_ids, max_new_tokens=max_new_tokens, gamma=0, drafter=NullDrafter()
        )
