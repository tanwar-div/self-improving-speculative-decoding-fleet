"""Drafter implementations and the kind -> loader registry.

Both the serving replica (loading the version it should serve) and the trainer
(loading the incumbent to score a candidate against it) need to turn a registry
`drafter_kind` into a loaded drafter, so the mapping lives here rather than in
either of them.
"""

from __future__ import annotations

from pathlib import Path

from fleet.specdec.drafters.base import Drafter, DrafterMeta, DraftProposal, NullDrafter
from fleet.specdec.drafters.ngram import NgramDrafter

# "none" is a real, deployable version: the cold start serves it, and a rollback
# can legitimately land on it.
LOADERS: dict[str, type[Drafter]] = {
    "ngram": NgramDrafter,
    "none": NullDrafter,
}

__all__ = [
    "LOADERS",
    "DraftProposal",
    "Drafter",
    "DrafterMeta",
    "NgramDrafter",
    "NullDrafter",
    "load_artifact",
]


def load_artifact(kind: str, path: Path) -> Drafter:
    loader = LOADERS.get(kind)
    if loader is None:
        raise LookupError(f"no loader registered for drafter kind {kind!r}")
    return loader.load(path)
