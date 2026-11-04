"""Drafter interface.

A drafter proposes continuation tokens cheaply. It is the *only* component the
self-improvement loop retrains, and it is swapped at runtime without touching
the target model or restarting the replica.

Correctness does not depend on drafter quality at all: a bad drafter costs
throughput, never output fidelity, because every proposed token is verified
against the target's own greedy choice. That property is what makes
retraining-from-production safe to automate.
"""

from __future__ import annotations

import abc
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class DraftProposal:
    token_ids: list[int]
    source: str = "none"  # which strategy produced it, for diagnostics


@dataclass
class DrafterMeta:
    kind: str
    version: int | None = None
    train_sample_count: int = 0
    hyperparams: dict[str, Any] = field(default_factory=dict)
    training_manifest: dict[str, Any] = field(default_factory=dict)
    offline_acceptance_rate: float | None = None


class Drafter(abc.ABC):
    """Base class for all draft models."""

    kind: str = "abstract"

    def __init__(self, meta: DrafterMeta | None = None) -> None:
        self.meta = meta or DrafterMeta(kind=self.kind)

    @abc.abstractmethod
    def propose(
        self,
        context_ids: list[int],
        gamma: int,
        *,
        hidden_state: torch.Tensor | None = None,
    ) -> DraftProposal:
        """Return up to `gamma` speculative tokens continuing `context_ids`.

        Returning fewer tokens (or none) is always legal; the engine then falls
        back to a plain target step for that position.
        """

    @property
    def needs_hidden_state(self) -> bool:
        """True if propose() requires the target's last hidden state."""
        return False

    @abc.abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abc.abstractmethod
    def load(cls, path: Path) -> Drafter: ...

    def reset(self) -> None:  # noqa: B027 - optional hook, not abstract
        """Clear any per-request state. Called at the start of each request.

        Intentionally concrete and empty: most drafters are stateless across
        requests, and forcing every implementation to declare that would be
        noise.
        """


class NullDrafter(Drafter):
    """Proposes nothing. The engine degenerates to plain greedy decoding.

    Used as the correctness oracle and as the no-speculation baseline.
    """

    kind = "none"

    def propose(
        self, context_ids: list[int], gamma: int, *, hidden_state: torch.Tensor | None = None
    ) -> DraftProposal:
        return DraftProposal(token_ids=[], source="none")

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"kind": self.kind}))

    @classmethod
    def load(cls, path: Path) -> NullDrafter:
        return cls()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
