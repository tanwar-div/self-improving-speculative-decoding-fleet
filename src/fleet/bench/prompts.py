"""Multi-domain prompt pool.

Six domains with genuinely different token distributions, which is what the
drift detector and the workload-shift experiment need: a drafter trained on
Python code should visibly stop hitting when the fleet starts serving clinical
questions.

Datasets are read in *streaming* mode and cached to a local JSONL, so building
a 40k-prompt pool costs a few MB of network rather than downloading several GB
of corpora. Any domain whose source is unreachable falls back to a deterministic
template generator, and the manifest records which domains were synthesised so a
benchmark run is never silently built on different data than it claims.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import asdict, dataclass

from fleet.config import REPO_ROOT

log = logging.getLogger(__name__)

DATA_DIR = REPO_ROOT / "data"
POOL_PATH = DATA_DIR / "prompt_pool.jsonl"
MANIFEST_PATH = DATA_DIR / "prompt_pool_manifest.json"

DOMAINS = ("code", "math", "chat", "summarization", "medical", "legal")

# (hf_dataset, config, split, text_field, instruction_prefix)
# (hf_dataset, config, split, text_field, instruction_prefix)
#
# Several sources per domain, because the obvious single source is usually too
# small: MBPP and HumanEval together hold only 538 prompts, and PubMedQA's
# `pqa_labeled` config holds exactly 1,000 -- far short of a 40k pool's
# per-domain share. Sources are drawn in order until the quota is met, so the
# smaller, higher-quality sets are used first and the large ones top up.
DOMAIN_SOURCES: dict[str, list[tuple[str, str | None, str, str, str]]] = {
    "code": [
        ("google-research-datasets/mbpp", "full", "train", "text", "Write a Python function. "),
        ("openai/openai_humaneval", None, "test", "prompt", "Complete this Python function:\n"),
        ("sahil2801/CodeAlpaca-20k", None, "train", "instruction", ""),
    ],
    "math": [
        ("openai/gsm8k", "main", "train", "question", "Solve step by step. "),
        # NOT gsm8k's "socratic" config: it carries the same questions as
        # "main" with different answers, so the cross-source dedup rejects all
        # of it. aqua_rat is genuinely different problems.
        ("deepmind/aqua_rat", "raw", "train", "question", "Solve step by step. "),
    ],
    "chat": [
        ("tatsu-lab/alpaca", None, "train", "instruction", ""),
    ],
    "summarization": [
        (
            "abisee/cnn_dailymail",
            "3.0.0",
            "train",
            "article",
            "Summarize the following news article in two sentences:\n",
        ),
    ],
    "medical": [
        (
            "qiaojin/PubMedQA",
            "pqa_labeled",
            "train",
            "question",
            "Answer this clinical research question: ",
        ),
        (
            "qiaojin/PubMedQA",
            "pqa_artificial",
            "train",
            "question",
            "Answer this clinical research question: ",
        ),
        ("openlifescienceai/medmcqa", None, "train", "question", "Clinical question: "),
    ],
    "legal": [
        (
            "coastalcph/lex_glue",
            "ecthr_a",
            "train",
            "text",
            "Identify the legal issue raised by this case excerpt:\n",
        ),
        (
            "coastalcph/lex_glue",
            "scotus",
            "train",
            "text",
            "Summarise the question presented in this opinion:\n",
        ),
    ],
}

# Fallback templates, used only when a dataset cannot be reached.
#
# Each template's unique output count is the product of the placeholder
# vocabularies it actually references. An earlier version had three code
# templates referencing only {verb}/{thing}/{struct}, which capped that domain
# at exactly 200 distinct prompts and left a 4,862-prompt hole in the pool
# without failing. Templates here all carry at least one numeric placeholder so
# the combinatorial space can actually fill a quota.
FALLBACK_TEMPLATES: dict[str, list[str]] = {
    "code": [
        "Write a Python function that {verb}s a {thing} with at most {n} elements.",
        "Implement {thing} in Python using a {struct}, sized for {n} entries.",
        "Fix the bug in this Python function that {verb}s a {thing} of length {n}.",
        "Write a Python function that {verb}s {n} {thing}s and returns a {struct}.",
        "Refactor a Python {struct} that {verb}s a {thing} in under {n} lines.",
        "Add tests for a Python function that {verb}s a {thing} given {n} inputs.",
    ],
    "math": [
        "A shop sells {n} {thing}s for ${m} each. How much do {k} cost?",
        "If a train travels {n} km in {m} hours, what is its average speed?",
        "Compute the sum of the first {n} odd numbers, then subtract {m}.",
        "A {struct} holds {n} items. After removing {m}, how many remain?",
        "What is {n} percent of {m}, rounded to {k} decimal places?",
    ],
    "chat": [
        "Explain {thing} in {n} sentences, assuming {m} years of experience.",
        "What are the trade-offs of using a {struct} for {n} concurrent readers?",
        "Give me {n} tips for {verb}ing a {thing} at {m}x scale.",
        "Compare a {struct} and a {thing} in about {n} words for {m} readers.",
        "Why choose a {struct} over a {thing} when handling {n} requests per second?",
    ],
    "summarization": [
        "Summarize the following in {n} sentences:\nThe committee reported that {thing} "
        "had grown by {m} percent, citing {struct} as the principal driver.",
        "Summarize in {n} sentences:\nAnalysts noted {thing} fell {m} percent after the "
        "{struct} review, with {k} further changes expected.",
        "Condense to {n} lines:\nThe {struct} audit of {thing} covered {m} sites and "
        "recorded {k} exceptions.",
    ],
    "medical": [
        "Does {thing} reduce complications in {n} patients receiving {struct} therapy?",
        "What is the first-line treatment for {thing} in adults over {n} with {m} comorbidities?",
        "In a cohort of {n} patients, is {thing} associated with {struct} outcomes at {m} months?",
        "Summarise the evidence on {thing} across {n} randomised trials of {m} patients.",
    ],
    "legal": [
        "Identify the legal issue: the appellant contends the {thing} clause was "
        "unenforceable because {struct} was not disclosed within {n} days.",
        "The respondent argues section {n} bars the claim over {thing} after {m} years. Assess.",
        "Does a {n}-day delay in disclosing {thing} void the {struct} provision agreed "
        "{m} months earlier?",
    ],
}

_WORDS = {
    "verb": [
        "reverse",
        "sort",
        "validate",
        "serialise",
        "flatten",
        "deduplicate",
        "parse",
        "normalise",
        "compress",
        "index",
        "merge",
        "partition",
    ],
    "thing": [
        "linked list",
        "binary tree",
        "hash table",
        "JSON payload",
        "date string",
        "matrix",
        "queue",
        "cache key",
        "token stream",
        "config file",
        "audit log",
        "session record",
        "price feed",
        "message batch",
    ],
    "struct": [
        "hash map",
        "heap",
        "trie",
        "ring buffer",
        "B-tree",
        "bloom filter",
        "skip list",
        "union-find",
        "segment tree",
    ],
}

# Small fixed set used by scripts/precision_study.py -- kept here so the
# precision study and the benchmark draw on the same domain mix.
PRECISION_STUDY_PROMPTS = [
    "Write a Python function that reverses a linked list.",
    "Write a Python function to check whether a string is a palindrome.",
    "Write a Python function that merges two sorted lists.",
    "Explain what a hash table is in two sentences.",
    "Summarize: speculative decoding drafts several tokens then verifies them.",
    "What is the time complexity of binary search?",
    "Write a Python function that reverses a string.",
    "Write a Python function to compute the nth Fibonacci number.",
    "Explain what a binary search tree is in two sentences.",
    "Summarize: a draft model proposes tokens that the target model verifies.",
    "What is the time complexity of inserting into a hash table?",
    "Write a Python function that sorts a list of dictionaries by a key.",
    "Does aspirin reduce the risk of cardiovascular events in healthy adults?",
    "Identify the legal issue: the appellant contends the clause was unenforceable.",
    "A shop sells 7 notebooks for $3 each. How much do 12 cost?",
    "Explain the difference between a process and a thread in two sentences.",
]


@dataclass
class Prompt:
    prompt_id: str
    domain: str
    text: str
    source: str

    @staticmethod
    def make(domain: str, text: str, source: str) -> Prompt:
        pid = hashlib.sha256(f"{domain}|{text}".encode()).hexdigest()[:16]
        return Prompt(prompt_id=pid, domain=domain, text=text, source=source)


def _clean(text: str, max_chars: int) -> str:
    text = " ".join(str(text).split())
    return text[:max_chars]


def _stream_domain(domain: str, want: int, max_chars: int) -> list[Prompt]:
    """Pull `want` prompts for one domain by streaming its source datasets."""
    out: list[Prompt] = []
    seen_ids: set[str] = set()
    for name, config, split, field, prefix in DOMAIN_SOURCES.get(domain, []):
        if len(out) >= want:
            break
        need = want - len(out)
        try:
            from datasets import load_dataset

            ds = load_dataset(name, config, split=split, streaming=True)
            taken = 0
            for row in ds:
                raw = row.get(field)
                if isinstance(raw, list):
                    raw = " ".join(map(str, raw))
                if not raw:
                    continue
                text = prefix + _clean(raw, max_chars)
                prompt = Prompt.make(domain, text, f"{name}:{config or 'default'}")
                # Sources overlap (PubMedQA's labeled split is contained in
                # larger collections); the same prompt twice is not two prompts.
                if prompt.prompt_id in seen_ids:
                    continue
                seen_ids.add(prompt.prompt_id)
                out.append(prompt)
                taken += 1
                if taken >= need:
                    break
            log.info("domain %s: streamed %d prompts from %s", domain, taken, name)
        except Exception as exc:  # noqa: BLE001 - any source failure falls back
            log.warning("domain %s: source %s unavailable (%s)", domain, name, exc)
    return out


def _synthesise(domain: str, want: int, seed: int = 0) -> list[Prompt]:
    rng = random.Random(f"{domain}-{seed}")
    templates = FALLBACK_TEMPLATES[domain]
    out: list[Prompt] = []
    seen: set[str] = set()
    attempts = 0
    while len(out) < want and attempts < want * 40:
        attempts += 1
        tpl = rng.choice(templates)
        text = tpl.format(
            verb=rng.choice(_WORDS["verb"]),
            thing=rng.choice(_WORDS["thing"]),
            struct=rng.choice(_WORDS["struct"]),
            n=rng.randint(2, 99),
            m=rng.randint(2, 99),
            k=rng.randint(2, 99),
        )
        if text in seen:
            continue
        seen.add(text)
        out.append(Prompt.make(domain, text, "synthetic"))
    return out


def build_pool(
    total: int = 40_000,
    *,
    weights: dict[str, float] | None = None,
    max_chars: int = 900,
    force: bool = False,
) -> tuple[list[Prompt], dict]:
    """Build (or load from cache) the multi-domain prompt pool."""
    if POOL_PATH.exists() and MANIFEST_PATH.exists() and not force:
        manifest = json.loads(MANIFEST_PATH.read_text())
        if manifest.get("total") == total:
            prompts = [
                Prompt(**json.loads(line)) for line in POOL_PATH.read_text().splitlines() if line
            ]
            log.info("loaded cached prompt pool: %d prompts", len(prompts))
            return prompts, manifest

    weights = weights or {
        "chat": 0.30,
        "math": 0.20,
        "code": 0.14,
        "summarization": 0.16,
        "medical": 0.10,
        "legal": 0.10,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    prompts: list[Prompt] = []
    per_domain: dict[str, dict] = {}
    for domain in DOMAINS:
        want = int(round(total * weights[domain]))
        got = _stream_domain(domain, want, max_chars)
        n_real = len(got)
        if n_real < want:
            got += _synthesise(domain, want - n_real)
        per_domain[domain] = {
            "requested": want,
            "from_dataset": n_real,
            "synthesised": len(got) - n_real,
            "sources": sorted({p.source for p in got}),
        }
        prompts.extend(got)

    random.Random(1234).shuffle(prompts)
    shortfalls = {
        d: v["requested"] - v["from_dataset"] - v["synthesised"]
        for d, v in per_domain.items()
        if v["requested"] - v["from_dataset"] - v["synthesised"] > 0
    }
    manifest = {
        "total": total,
        "actual": len(prompts),
        "weights": weights,
        "max_chars": max_chars,
        "per_domain": per_domain,
        "fully_real": all(v["synthesised"] == 0 for v in per_domain.values()),
        "shortfalls": shortfalls,
    }
    if len(prompts) < total:
        log.warning(
            "prompt pool is SHORT: %d of %d requested. Per-domain shortfalls: %s. "
            "Any figure quoting a pool size must use `actual`, not the request.",
            len(prompts),
            total,
            shortfalls or "(none)",
        )
    if not manifest["fully_real"]:
        synth = {d: v["synthesised"] for d, v in per_domain.items() if v["synthesised"]}
        log.warning("prompt pool contains synthesised prompts: %s", synth)

    with POOL_PATH.open("w") as fh:
        for p in prompts:
            fh.write(json.dumps(asdict(p)) + "\n")
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    log.info("built prompt pool: %d prompts across %d domains", len(prompts), len(DOMAINS))
    return prompts, manifest


def stratified_sample(prompts: list[Prompt], n: int, *, seed: int = 7) -> list[Prompt]:
    """Even-per-domain sample, for the decode-backed subset of the benchmark."""
    by_domain: dict[str, list[Prompt]] = {d: [] for d in DOMAINS}
    for p in prompts:
        by_domain.setdefault(p.domain, []).append(p)
    rng = random.Random(seed)
    per = max(1, n // len(by_domain))
    out: list[Prompt] = []
    for items in by_domain.values():
        rng.shuffle(items)
        out.extend(items[:per])
    rng.shuffle(out)
    return out[:n]
