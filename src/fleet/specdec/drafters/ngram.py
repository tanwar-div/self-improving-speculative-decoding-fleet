"""Trained n-gram drafter.

Two proposal strategies, tried in order:

1. A back-off n-gram table *learned from live production traffic* (prompt plus
   the target model's own greedy continuation). This is the part that adapts:
   when the serving workload shifts domain, the table trained on the old
   distribution stops hitting, acceptance rate falls, and the drift detector
   fires a retrain.
2. Context copy (prompt-lookup): the longest suffix of the current context that
   already occurred earlier in the same request, continued with whatever
   followed it. Domain-agnostic and strong on code and summarization, where
   output repeats the input.

Retraining cost is seconds, which is what makes the self-improvement loop a
loop rather than an overnight batch job.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

import torch

from fleet.specdec.drafters.base import Drafter, DrafterMeta, DraftProposal

log = logging.getLogger(__name__)

# Candidate blend policies, all scored on held-out traffic at fit time.
# The first three treat the two strategies as mutually exclusive; the last two
# compose them, letting one fill out a proposal the other left short. That
# matters because the two fail in different places: context copy needs the
# current request to repeat itself, and the table needs the pattern to have been
# seen in earlier traffic. Choosing one throws away the other's coverage.
POLICIES = (
    "ngram_first",
    "copy_first",
    "longest",
    "copy_then_extend",
    "ngram_then_extend",
)

# How much more confident the table must be to keep chaining a proposal deeper.
# 0.0 means depth is free. Worth searching rather than fixing: for small gamma
# the verification forward pass is memory-bound, so carrying more draft tokens
# costs almost nothing and over-caution directly costs throughput -- but a wrong
# token truncates everything behind it, so it is not free either.
DEPTH_PENALTIES = (0.0, 0.25, 0.5)


class NgramDrafter(Drafter):
    kind = "ngram"

    def __init__(
        self,
        table: dict[tuple[int, ...], tuple[int, int, int]] | None = None,
        *,
        order: int = 3,
        min_count: int = 2,
        min_confidence: float = 0.25,
        copy_min_match: int = 2,
        policy: str = "ngram_first",
        depth_penalty: float = 0.25,
        meta: DrafterMeta | None = None,
    ) -> None:
        super().__init__(meta or DrafterMeta(kind=self.kind))
        # key (token suffix) -> (best_next_token, hits, total)
        self.table = table or {}
        self.order = order
        self.min_count = min_count
        self.min_confidence = min_confidence
        self.copy_min_match = copy_min_match
        # How the two proposal strategies are combined. Chosen from held-out
        # live traffic at training time (see POLICIES / trainer.worker) rather
        # than fixed by hand: a thinly-trained table that always pre-empts
        # context copy can be *worse* than no table at all, which is exactly
        # what happened before this was made a learned choice.
        self.policy = policy if policy in POLICIES else "ngram_first"
        self.depth_penalty = depth_penalty
        self._stat_ngram = 0
        self._stat_copy = 0
        self._stat_empty = 0

    # -- inference ----------------------------------------------------------

    def propose(
        self,
        context_ids: list[int],
        gamma: int,
        *,
        hidden_state: torch.Tensor | None = None,
    ) -> DraftProposal:
        if gamma <= 0:
            return DraftProposal([], "none")

        if self.policy == "copy_then_extend":
            out = self._context_copy(context_ids, gamma)
            if out:
                self._stat_copy += 1
                if len(out) < gamma:
                    # Continue from the extended context, so the table picks up
                    # where the copy ran out rather than re-proposing position 0.
                    out = out + self._propose_ngram(context_ids + out, gamma - len(out))
                return DraftProposal(out, "copy+ngram")
            table = self._propose_ngram(context_ids, gamma)
            if table:
                self._stat_ngram += 1
                return DraftProposal(table, "ngram")

        elif self.policy == "ngram_then_extend":
            out = self._propose_ngram(context_ids, gamma)
            if out:
                self._stat_ngram += 1
                if len(out) < gamma:
                    out = out + self._context_copy(context_ids + out, gamma - len(out))
                return DraftProposal(out, "ngram+copy")
            copied = self._context_copy(context_ids, gamma)
            if copied:
                self._stat_copy += 1
                return DraftProposal(copied, "copy")

        elif self.policy == "copy_first":
            copied = self._context_copy(context_ids, gamma)
            if copied:
                self._stat_copy += 1
                return DraftProposal(copied, "copy")
            table = self._propose_ngram(context_ids, gamma)
            if table:
                self._stat_ngram += 1
                return DraftProposal(table, "ngram")

        elif self.policy == "longest":
            # Picking the longer proposal is close to free: for small gamma the
            # verification forward pass is memory-bound, so its cost barely
            # moves with how many draft tokens ride along. Only accepted length
            # matters, and both proposals are verified either way.
            table = self._propose_ngram(context_ids, gamma)
            copied = self._context_copy(context_ids, gamma)
            if len(table) >= len(copied) and table:
                self._stat_ngram += 1
                return DraftProposal(table, "ngram")
            if copied:
                self._stat_copy += 1
                return DraftProposal(copied, "copy")

        else:  # "ngram_first"
            table = self._propose_ngram(context_ids, gamma)
            if table:
                self._stat_ngram += 1
                return DraftProposal(table, "ngram")
            copied = self._context_copy(context_ids, gamma)
            if copied:
                self._stat_copy += 1
                return DraftProposal(copied, "copy")

        self._stat_empty += 1
        return DraftProposal([], "none")

    def _propose_ngram(self, context_ids: list[int], gamma: int) -> list[int]:
        if not self.table:
            return []
        out: list[int] = []
        seq = context_ids
        for _ in range(gamma):
            nxt = self._lookup(seq, len(out))
            if nxt is None:
                break
            out.append(nxt)
            seq = context_ids + out
        return out

    def _lookup(self, seq: Sequence[int], depth: int) -> int | None:
        """Back-off lookup: highest available order first."""
        for order in range(min(self.order, len(seq)), 0, -1):
            entry = self.table.get(tuple(seq[-order:]))
            if entry is None:
                continue
            tok, hits, total = entry
            if total < self.min_count:
                continue
            confidence = hits / total
            # Require more confidence the deeper we chain: an early mistake
            # truncates every token behind it. How much more is a learned
            # hyperparameter (see DEPTH_PENALTIES), because being too cautious
            # here caps tokens-per-forward and therefore throughput.
            if confidence >= self.min_confidence * (1.0 + self.depth_penalty * depth):
                return tok
        return None

    def _context_copy(self, seq: Sequence[int], gamma: int) -> list[int]:
        n = len(seq)
        if n < self.copy_min_match + 1:
            return []
        max_match = min(self.order + 2, n - 1)
        for match in range(max_match, self.copy_min_match - 1, -1):
            pattern = tuple(seq[-match:])
            # Scan backwards for the most recent earlier occurrence.
            for start in range(n - match - 1, -1, -1):
                if tuple(seq[start : start + match]) == pattern:
                    nxt = list(seq[start + match : start + match + gamma])
                    if nxt:
                        return nxt
                    break
        return []

    def stats(self) -> dict[str, int]:
        return {
            "proposals_ngram": self._stat_ngram,
            "proposals_copy": self._stat_copy,
            "proposals_empty": self._stat_empty,
            "table_entries": len(self.table),
            "policy": self.policy,
            "depth_penalty": self.depth_penalty,
        }

    def reset(self) -> None:
        pass  # table is request-independent; context copy reads the live context

    # -- training -----------------------------------------------------------

    @classmethod
    def fit(
        cls,
        sequences: Iterable[Sequence[int]],
        *,
        order: int = 3,
        min_count: int = 2,
        min_confidence: float = 0.25,
        policy: str = "ngram_first",
        meta: DrafterMeta | None = None,
    ) -> NgramDrafter:
        counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
        n_seq = 0
        n_tok = 0
        for seq in sequences:
            seq = list(seq)
            n_seq += 1
            n_tok += len(seq)
            for i in range(1, len(seq)):
                nxt = seq[i]
                lo = max(0, i - order)
                for start in range(lo, i):
                    counts[tuple(seq[start:i])][nxt] += 1

        table: dict[tuple[int, ...], tuple[int, int, int]] = {}
        for key, counter in counts.items():
            total = sum(counter.values())
            if total < min_count:
                continue
            tok, hits = counter.most_common(1)[0]
            if hits / total < min_confidence:
                continue
            table[key] = (int(tok), int(hits), int(total))

        drafter = cls(
            table,
            order=order,
            min_count=min_count,
            min_confidence=min_confidence,
            policy=policy,
            meta=meta or DrafterMeta(kind=cls.kind),
        )
        drafter.meta.train_sample_count = n_seq
        drafter.meta.hyperparams = {
            "order": order,
            "min_count": min_count,
            "min_confidence": min_confidence,
            "train_tokens": n_tok,
            "table_entries": len(table),
        }
        log.info(
            "fit ngram drafter: %d sequences, %d tokens, %d table entries",
            n_seq,
            n_tok,
            len(table),
        )
        return drafter

    # -- persistence --------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": self.kind,
            "order": self.order,
            "min_count": self.min_count,
            "min_confidence": self.min_confidence,
            "copy_min_match": self.copy_min_match,
            "policy": self.policy,
            "depth_penalty": self.depth_penalty,
            "meta": {
                "train_sample_count": self.meta.train_sample_count,
                "hyperparams": self.meta.hyperparams,
                "training_manifest": self.meta.training_manifest,
                "offline_acceptance_rate": self.meta.offline_acceptance_rate,
            },
            # Compact: "k1,k2,k3": [next, hits, total]
            "table": {
                ",".join(map(str, key)): [tok, hits, total]
                for key, (tok, hits, total) in self.table.items()
            },
        }
        path.write_text(json.dumps(payload, separators=(",", ":")))

    @classmethod
    def load(cls, path: Path) -> NgramDrafter:
        payload = json.loads(Path(path).read_text())
        table = {
            tuple(int(x) for x in key.split(",")): (int(v[0]), int(v[1]), int(v[2]))
            for key, v in payload["table"].items()
        }
        m = payload.get("meta", {})
        meta = DrafterMeta(
            kind=cls.kind,
            train_sample_count=m.get("train_sample_count", 0),
            hyperparams=m.get("hyperparams", {}),
            training_manifest=m.get("training_manifest", {}),
            offline_acceptance_rate=m.get("offline_acceptance_rate"),
        )
        return cls(
            table,
            order=payload["order"],
            min_count=payload["min_count"],
            min_confidence=payload["min_confidence"],
            copy_min_match=payload.get("copy_min_match", 2),
            policy=payload.get("policy", "ngram_first"),
            depth_penalty=payload.get("depth_penalty", 0.25),
            meta=meta,
        )
