# Self-Improving Speculative Decoding Fleet

An LLM serving fleet that retrains its own draft model from production traffic.
Live inference becomes training data, a new draft version trains asynchronously,
a versioned registry promotes it through an SLO-gated canary, and serving
replicas hot-swap it with no restart and no human in the loop. Every served
response is traceable to the exact draft-model version behind it, and every
rollout is reversible to a known-good one.

The thing that makes any of this safe to automate is a correctness property, so
that is where this README starts.

---

## 1. The invariant: speculation cannot change the output

Speculative decoding proposes cheap draft tokens and verifies them against the
target model. Under greedy verification, a draft token is committed only if it
equals the token the target itself would have chosen at that position; the first
mismatch truncates the proposal and the target's own token is committed instead.

```
decode(prompt, drafter = ANY)  ==  greedy_decode(prompt)        token for token
```

So the drafter can be stale, badly trained, or actively adversarial, and the
served text does not change — only tokens/second does. **The blast radius of a
bad draft version is throughput, not correctness.** That is the whole licence for
retraining from production traffic automatically.

This is asserted, not assumed. [`tests/test_greedy_equivalence.py`](tests/test_greedy_equivalence.py)
checks it across gamma ∈ {1,2,4,8} and three drafters, including an
`AdversarialDrafter` whose every proposal is uniform random noise, plus a guard
that the trained drafter actually gets tokens accepted — otherwise the
equivalence test would pass vacuously by proposing nothing.

### What happens in fp16, and why it is not this system's fault

The invariant holds exactly in exact arithmetic. In fp16 it can break, and
understanding how matters, because the obvious reading ("speculative decoding
corrupts output in fp16") is wrong.

[`scripts/precision_study.py`](scripts/precision_study.py) measures three things
on Qwen2.5-0.5B-Instruct over **117 held-out prompts from all six domains
(6,738 reference tokens)**. The third is the control that settles it: **plain
greedy decoding where each forward pass carries `gamma` extra copies of the last
token whose logits are then discarded.** No speculation of any kind — just a
different matmul shape for a mathematically identical computation.

| | fp32 | fp16 |
|---|---|---|
| speculative vs greedy | **117/117 (100%)** | 102/117 (87.2%) |
| padded greedy vs greedy — **no speculation** | **117/117 (100%)** | **100/117 (85.5%)** |
| token-level divergence rate | **0.0** | 0.22% |

Two things fall out of that table:

1. **In fp32 the guarantee is exact** — zero divergence in 6,738 tokens, and the
   control confirms the engine is shape-invariant.
2. **In fp16, merely changing the batch shape diverges more often (14.5% of
   sequences) than speculation does (12.8%).** Speculation is not the cause.

Of the 15 fp16 divergences, **14 sat at a top-2 logit gap of ≤ 1 fp16 ULP** —
ties fp16 physically cannot represent — and one at 2 ULPs. Every domain was
affected, so this is a property of the arithmetic, not of any prompt type.

So the honest statement of the guarantee is:

> Greedy-output equivalence is **exact in fp32** (117/117 sequences, 6,738
> tokens). In fp16 it holds up to argmax ties at the limit of fp16 resolution,
> and changing the batch shape of the forward pass perturbs those ties *more*
> than speculation does.

The serving path does not take this on trust either: it re-decodes a sampled
fraction of live requests target-only and records the comparison in
`traffic_events.greedy_match`. Any audited mismatch on a candidate fails its SLO
gate outright, whatever it did for throughput.

Raw data: [`results/precision_float32.json`](results/precision_float32.json),
[`results/precision_float16.json`](results/precision_float16.json).

## 2. Architecture

```
                         ┌──────────────────────────────────────┐
   client traffic ──────▶│ router  (round-robin, stateless)     │
                         └───────────────┬──────────────────────┘
                     ┌───────────────────┴───────────────────┐
                     ▼                                       ▼
        ┌────────────────────────┐              ┌────────────────────────┐
        │ replica-a              │              │ replica-b              │
        │  target model (frozen) │              │  target model (frozen) │
        │  drafter v4  (active)  │              │  drafter v4  (active)  │
        │  drafter v5  (canary)  │              │  drafter v5  (canary)  │
        └───────┬────────────────┘              └────────────┬───────────┘
                │   per-request: prompt tokens, target's greedy path,
                │   accept/reject counts, latency, domain, draft version
                ▼                                            ▼
        ┌────────────────────────────────────────────────────────────────┐
        │  registry + feedback store                                     │
        │  Postgres 16 (local) / Aurora PostgreSQL (deployed)            │
        │                                                                │
        │  draft_versions · active_deployment · traffic_events           │
        │  traffic_samples · slo_evaluations · deployment_events         │
        │  drift_checks   · response_provenance (view)                   │
        └──┬──────────────────────┬─────────────────────┬────────────────┘
           │                      │                     │
           ▼                      ▼                     ▼
   ┌───────────────┐    ┌──────────────────┐   ┌──────────────────────┐
   │ drift         │    │ trainer          │   │ canary controller    │
   │ detector      │───▶│ (async, off the  │──▶│ SLO gate: promote,   │
   │ domain JS,    │    │  serving path)   │   │ reject, or roll back │
   │ token PSI,    │    │ fit → score on   │   │ + post-promotion     │
   │ acceptance    │    │ holdout → publish│   │   regression watch   │
   └───────────────┘    └──────────────────┘   └──────────┬───────────┘
                                                          │ generation++
                                   replicas poll one row ─┘ then hot-swap
```

Six processes, one shared database, no message broker. The registry row *is* the
coordination primitive: replicas watch `active_deployment.generation`, and
everything else is a consequence of that row changing.

### The loop, concretely

0. **Cold start: serve no speculation at all.** The obvious starting point is
   an untrained drafter proposing by context copy — it needs no training data
   and costs nothing to build. Measured on the mixed prompt pool it ran at
   **82.9 tok/s against 91.0 for plain greedy decoding**: it widens every
   verification forward without earning back enough accepted tokens. So the
   fleet starts with a registered `none` version that does not speculate, and
   the first speculative version has to beat *that* through the canary. Same
   rule as everywhere else here — nothing serves production traffic until it has
   been measured against what it replaces.

1. **Capture.** Every served request writes a `traffic_events` row (metrics,
   provenance) and a `traffic_samples` row — the prompt tokens plus *the target
   model's own greedy continuation*. The distillation label is free: it is
   already what the drafter has to predict, harvested from real production
   decoding rather than from a static corpus. Writes go through a bounded queue
   drained by a background thread, because the SLO gate compares p95 latency and
   a synchronous insert would put database latency inside the number being gated.

2. **Decide whether to train.** See §4.

3. **Train, asynchronously.** The trainer is its own process. It holds no
   reference to the engine and never touches it; it writes a version into the
   registry and the serving side discovers it by polling. The fleet keeps
   serving at full throughput while the next version trains.

4. **Score before spending traffic.** Acceptance rate is a function of the
   drafter's proposals and the target's greedy path — and both are already in
   the database. So `simulate_acceptance` replays the engine's exact
   accept/reject logic with **zero model forward passes**, on a *chronological*
   holdout (train on older, score on newer; a random split would score the
   drafter on traffic it trained on). A candidate that does not beat the
   incumbent is discarded before it costs a single production request.

5. **Canary.** The controller moves the candidate to `canary` at N% and bumps
   `generation`. Each replica loads the candidate *alongside* the active
   version — both arms are served by the same processes on the same hardware, so
   the comparison is not confounded by which replica got the traffic.

6. **Gate.** §3.

7. **Hot-swap.** On promotion, `generation` moves again; replicas fetch the
   artifact only if its sha256 differs from what their local cache holds, verify
   the digest, and swap by pointer. A request binds its drafter by reference at
   entry, so a request that started on v4 finishes on v4 even if v5 lands
   mid-flight. The target model is never reloaded.

---

## 3. Rollout safety

**Two invariants enforced by the database, not by application code** — partial
unique indexes in [`sql/001_schema.sql`](sql/001_schema.sql):

```sql
CREATE UNIQUE INDEX ... ON draft_versions((status)) WHERE status = 'active';
CREATE UNIQUE INDEX ... ON draft_versions((status)) WHERE status = 'canary';
```

`tests/test_registry.py::test_only_one_active_version_is_possible` bypasses the
application layer and issues a raw `UPDATE` to confirm Postgres rejects it.

**Provenance is enforced too.** `traffic_events.draft_version` is
`NOT NULL REFERENCES draft_versions(version)`, so no served response can name a
version that is not in the registry, and the `response_provenance` view joins
any request id to the exact `artifact_sha256` behind it.

**`last_known_good`** is maintained on every promotion and is the outgoing
*active* version — one that actually served production traffic and passed a gate
once. Rolling back therefore targets known-good bytes, not merely the previous
version number.

### The SLO gate

A candidate is promoted or rejected by one computation and nothing else:

| check | threshold | why |
|---|---|---|
| throughput ratio | ≥ 0.98× baseline | must not *lose* throughput |
| p95 latency ratio | ≤ 1.05× baseline | a candidate can hold mean throughput while wrecking the tail |
| audited greedy mismatches | **0** | correctness is a gate, not a trade-off |
| requests per arm | ≥ `canary_min_requests` | otherwise the verdict is noise |

Thresholds are asymmetric on purpose: a merely *neutral* candidate is allowed
through, because the drafter is retrained to track a moving workload and holding
position on current traffic is the point.

The baseline is **concurrent** — drawn from the same window, replicas and load
as the candidate — so the gate never compares "the candidate now" against "the
incumbent an hour ago".

#### A threshold alone is not a gate

The first version compared point estimates against those thresholds. Running the
real containerized fleet showed what that actually does: it **rejected three
consecutive candidates at 0.88×, 0.90× and 0.92× throughput, measured on 15
candidate requests against 23–28 baseline requests.** At that sample size and
CPU-serving variance, a 0.92× ratio is statistically indistinguishable from no
difference. The gate was not enforcing an SLO; it was flipping a weighted coin
and writing the outcome to an audit table.

A threshold can only say "this sample looked worse". So failing now requires
**both**:

1. the point estimate violates the threshold, **and**
2. the difference is statistically distinguishable at `slo_alpha` (default 0.05).

- **Throughput** — Welch's t-test on per-request tokens/second. Welch rather
  than Student because the arms genuinely have unequal variance: a better
  drafter changes both the mean and the spread.
- **Tail latency** — percentile bootstrap (2,000 resamples) on the *ratio* of
  p95s, because a p95 has no usable closed-form standard error. "Significant"
  means the interval excludes 1.0.
- **Correctness is exempt.** One audited mismatch is one user who got different
  output; there is no sample size at which that becomes acceptable, so a greedy
  mismatch fails the candidate regardless of significance.

An apparent regression that *cannot* be measured extends the canary instead of
killing the candidate, up to `canary_max_requests` — after which a candidate
still indistinguishable from the incumbent is promoted, since a neutral
candidate passing is the intended behaviour. Both the test statistics and a
human-readable `decision_note` are persisted on every `slo_evaluations` row, so
the gate's reasoning is reconstructable and not just its verdict.

[`tests/test_slo_gate.py`](tests/test_slo_gate.py) encodes the bug as a
regression test: two arms drawn from the *same* distribution, at the sample
sizes and noise level the container run produced, must not be called a
regression.

**The gate is not a one-time admission test.** A candidate can pass on a 30%
slice and regress once it sees the full prompt mix, so
`check_active_regression` keeps comparing the active version against the
throughput distribution it demonstrated during its own canary — with the same
significance guard — and reverts to `last_known_good` on its own. Every verdict
is written to `slo_evaluations`, and the `deployment_events` row for the
promotion or rollback carries the `slo_evaluation_id` that justified it.

## 4. Drift detection: why retrain, not when

Retraining on a timer is wasteful in both directions — it burns compute when the
workload is stationary, and it leaves a stale drafter in production for up to a
full period when the workload moves. Each version's `training_manifest` stores a
fingerprint of what it was trained on, and a check compares the recent traffic
window against it on three axes:

| axis | catches |
|---|---|
| `domain_js` — Jensen–Shannon divergence of the domain mixture | the fleet now serves clinical questions and the drafter learned Python |
| `token_psi` — Population Stability Index over token unigrams | within-domain shift a domain label alone would miss (same "code" label, Rust instead of Python) |
| `acceptance_drop` — realised acceptance vs. training-time | the outcome measure; it is what directly predicts lost throughput |

Token ids are hashed into a fixed 256 buckets, which keeps the fingerprint small
and — more importantly — keeps two fingerprints comparable when they contain
entirely disjoint vocabulary, which is exactly the situation after a shift.

The negative half of the claim is the half worth testing, since a detector that
fires constantly is just a timer with extra steps.
[`tests/test_drift.py`](tests/test_drift.py) covers: stationary traffic does
**not** trigger; a domain shift does; a within-domain token shift is caught with
`domain_js == 0`; an acceptance collapse triggers on its own; and the score rises
**monotonically** as the live mixture moves further from training. Every check is
written to `drift_checks` whether or not it fired, so "why did/didn't it retrain"
is answerable after the fact.

### Drift is not the only reason to retrain

Drift detection answers *"has the workload moved?"*. It cannot answer *"is the
drafter as good as my data now allows?"* — and those come apart badly.

An 800-request **stationary** run produced exactly **one retrain**: the detector
correctly reported no drift every time, so the fleet froze on the first version
it happened to train, which had learned from roughly 100 requests while 800 sat
in the table. A system that stops improving the moment the workload goes stable
has the premise backwards.

So retraining has a second, independent trigger: **data growth**. When the
number of new samples since the active version's training cutoff reaches
`retrain_data_growth ×` the size of the set that version actually trained on,
the trainer produces a candidate regardless of drift. The two triggers answer
different questions and both are recorded, so an operator can tell a
distribution shift from a simple accumulation of evidence.

The improvement guard still applies: a growth-triggered candidate that cannot
beat the incumbent on a chronological holdout is discarded before it costs any
production traffic, and the canary gate still has to admit the ones that
survive. The trigger decides *when to look*, not *what gets promoted*.

### The drafter learns its own blend policy

The drafter has two proposal strategies: a back-off n-gram table learned from
live traffic, and context copy (the longest suffix of the current context that
occurred earlier, continued with whatever followed it).

Early runs produced a trained version *worse* than the untrained cold-start one
(offline acceptance 0.033 vs 0.250). The cause: a thinly-trained table was
pre-empting the better context-copy path. Rather than hand-tune a precedence
rule, the trainer now scores three blend policies — `ngram_first`, `copy_first`,
`longest` — on held-out live traffic and stores the winner in the artifact.
Scoring is free, because `simulate_acceptance` needs no model forwards.

The same selection is applied to the static baselines in the benchmark. A
baseline that was denied this would make the headline ratio measure a handicap
rather than the value of retraining.

---

## 5. Benchmarks

### Which numbers come from where

This host has a 4 GB RTX 3050 and **no `nvidia-container-toolkit`**, so
containers cannot see the GPU. Rather than fake it, the project has two run
modes and the README says which figure comes from which:

| | containerized fleet (`make fleet-up`) | host benchmark (`make bench`) |
|---|---|---|
| inference | CPU, fp32, small target | **GPU, fp16** |
| what it demonstrates | real replicas, hot-swap, canary split, SLO gate, rollback, provenance | throughput, latency, acceptance, correctness |
| used for | the control-plane claims | **every performance number quoted** |

Both run identical code. Installing `nvidia-container-toolkit` would collapse
the two, and nothing in the design prevents it.

### Methodology

- **40,000 prompts across six domains** — code, math, chat, summarization,
  medical, legal — streamed from real datasets and cached locally:

  | domain | n | sources |
  |---|---|---|
  | chat | 12,000 | Alpaca |
  | math | 8,000 | GSM8K, AQuA-RAT |
  | summarization | 6,400 | CNN/DailyMail |
  | code | 5,600 | MBPP, HumanEval, CodeAlpaca-20k |
  | medical | 4,000 | PubMedQA (`pqa_labeled` + `pqa_artificial`) |
  | legal | 4,000 | LexGLUE (ECtHR) |

  The manifest records a `fully_real` flag, per-domain provenance and any
  shortfall; a domain whose source is unreachable falls back to a template
  generator and is reported as synthesised, so a run is never silently built on
  different data than it claims. The committed pool is **`fully_real: true`
  with zero shortfall** — reaching that took adding sources, since MBPP and
  HumanEval together hold only 538 prompts and PubMedQA's labeled split holds
  1,000. Cross-source dedup is applied per domain (GSM8K's `socratic` config
  repeats `main`'s questions, so it contributes nothing).
  **All 40k prompts drive the traffic, drift and registry path**; a stratified
  subset gets real decoding, and the exact count is in every report.
- Experiments drive the **real control plane** — real engine, registry, drift
  detector, trainer, canary controller — with training in a background thread,
  exactly as the fleet runs it. Training wall-time is reported separately and
  excluded from serving throughput, matching the decoupled architecture.
- The correctness audit is run *after* the measured decode, so the extra
  target-only pass never lands inside the latency being compared.

### E1 — throughput, on two workloads

Which workload you pick determines the answer, and only one resembles
production, so both are reported.

**E1a non-stationary (the real case).** The serving mix walks through domain
regimes: `code+chat → summarization → medical+legal → math+code`. Four arms:

| arm | what it is |
|---|---|
| `no_speculation` | target-only greedy; the correctness oracle |
| `static_initial` | drafter fitted once on the traffic available at deploy time (first regime), then frozen — what "train a drafter and ship it" gets you a week later |
| `static_broad` | drafter fitted once on a stratified sample of **every** domain, including ones not yet seen in production — a deliberately strong baseline, handed data the self-improving arm has to earn |
| `self_improving` | cold start, retrained from live traffic, SLO-gated, hot-swapped |

Headline: `self_improving / static_broad`.

**E1b stationary (control).** One fixed stratified mix. Here a static drafter
trained in-distribution is near-optimal and the self-improving arm cannot beat
it — it pays a cold-start cost and catches up. Reported to show the loop does
not *regress* throughput when there is nothing to adapt to. Quoting E1b as the
headline would understate the mechanism; quoting E1a without E1b would hide its
cost.

Both full-stream and cold-start-excluded ("steady") throughput are reported. The
full-stream figure is the honest headline.

### E2 — workload-shift recovery

The mix changes abruptly from `code+chat` to `medical+legal`, and two retraining
*policies* race on the identical prompt stream:

- `periodic` — retrain every N requests regardless of the workload
- `drift_triggered` — retrain when the detector says the distribution moved

Same trainer, same promotion machinery, same `require_improvement` setting, so
the comparison isolates **trigger timing** rather than training quality.

Recovery is measured against an **oracle**: a drafter trained directly on
post-shift traffic, serving the post-shift stream. "Recovered" means a policy's
rolling throughput reached 95% of that oracle and *stayed* there. Measuring
against *pre-shift* throughput instead would be meaningless, because the new
domains have their own intrinsic speed.

### Results

Full JSON including per-domain breakdowns, rolling throughput traces and every
registry transition lands in `runs/`:

```bash
make bench          # full run on the 40k pool (GPU, hours)
make bench-smoke    # minutes, tiny counts, same code path
make bench-report   # re-render runs/latest.json
```

**Hardware:** RTX 3050 Laptop (4 GB), Qwen2.5-0.5B-Instruct, fp16, gamma 4,
48 new tokens/request, 800 decoded requests per arm, 76 min total.
Raw: [`results/benchmark.json`](results/benchmark.json),
rendered: [`results/benchmark.txt`](results/benchmark.txt).

#### E1a — non-stationary workload (`code+chat → summarization → medical+legal → math+code`)

| arm | tok/s | tokens/forward | acceptance | p95 | vs no speculation |
|---|---|---|---|---|---|
| no_speculation | 91.3 | 1.000 | — | 550 ms | 1.000× |
| static_initial | 116.3 | 1.442 | 0.211 | 518 ms | **1.273×** |
| static_broad | 121.6 | 1.530 | 0.227 | 516 ms | **1.333×** |
| self_improving | 98.2 | 1.276 | 0.210 | 587 ms | 1.076× |

#### E1b — stationary workload (control)

| arm | tok/s | vs no speculation |
|---|---|---|
| no_speculation | 90.6 | 1.000× |
| static_initial | 119.2 | 1.315× |
| static_broad | 118.9 | 1.313× |
| self_improving | 101.4 | 1.119× |

#### E2 — workload-shift recovery (`code+chat` ×300 → `medical+legal` ×600)

| policy | retrains | plateau | % of oracle | recovered after | mean post-shift deficit |
|---|---|---|---|---|---|
| periodic (every 200 req) | 4 | 98.0 tok/s | 77% | 65 requests | 2.92 tok/s |
| drift_triggered | 6 | 93.2 tok/s | 73% | **30 requests** | **1.36 tok/s** |

**drift-triggered recovered 53.8% faster (30 vs 65 requests), with a 53.3%
lower integrated post-shift throughput deficit.** Two independent measures
agreeing is the reason to believe it: requests-to-recover alone could be gamed
by a policy that dips less deeply, and the deficit alone ignores duration.

Recovery is measured against **95% of the lower of the two policies' own
post-shift plateaus** — a target that is policy-independent and reachable. The
first version of this metric used the frozen pre-trained oracle (127.5 tok/s)
and reported "never recovered" for *both* policies, which measured the gap
between adaptive and clairvoyant rather than the thing the experiment is
about. See [`fleet/bench/recovery.py`](src/fleet/bench/recovery.py).

Note the honest trade: drift-triggered retrained more often (6 vs 4), so it
paid more canary tax and settled at a slightly *lower* plateau (93.2 vs 98.0)
while recovering much faster. Faster recovery, marginally lower steady state.

### What these numbers do and do not support

**Supported.** Speculative decoding with greedy verification is real and worth
it: **1.27–1.33× throughput** over target-only greedy decoding, with *lower* p95
latency, and **byte-identical output in fp32** (117/117 sequences, §1). The
drift-triggered trigger recovers from a workload shift **~54% faster** than
retraining on a fixed schedule.

**Not supported.** The self-improving arm did **not** beat a well-trained static
drafter: 0.81× on the non-stationary workload and 0.85× on the stationary one.
It beat *no speculation* (1.08–1.12×), but a static drafter fitted once on a
stratified sample of all six domains beat it comfortably. Reporting this the
other way round would be false.

**Why, concretely.** The limit is arithmetic, not mysterious. A safe promotion
needs `canary_min_requests` on *both* arms, so at a canary share of `p` it costs
about `canary_min_requests / p` requests — here 30 / 0.20 = **150 requests**.
Each regime in E1a is **200 requests**. The fleet is therefore permanently about
one regime behind, and pays a 20% tax on unproven candidates the whole time.
The ratio that decides whether continuous retraining pays for itself is

```
regime length / safe-rollout latency
```

which is **≈ 1.3** in this benchmark and **enormous** in production, where a
regime lasts hours and a fleet at thousands of requests/second clears 150
requests in well under a second. This benchmark is close to the worst case for
the design, and that is worth saying plainly rather than tuning the regime
length until the number flatters it.

**What would test the claim properly:** longer regimes relative to rollout
latency, a 5% canary share instead of 20%, and a sequential test (SPRT) in the
gate to decide on fewer requests. All three reduce the adaptation lag that
dominates the result here, and none of them required changing the system — only
the experiment.

**Two fixes this benchmark forced**, both of which moved the numbers and both of
which were found by measuring rather than reasoning:

- The stationary arm went **82.9 → 101.4 tok/s** (2 → 5 versions served) once
  retraining could be triggered by *data growth* and not only by drift. Before
  that it was *below* no-speculation.
- Cold start stopped speculating at all. An untrained context-copy drafter
  measured **82.9 tok/s against 91.0 for plain greedy** — speculation that has
  not been measured was a net loss, so the fleet no longer ships it by default.



---

### What the containerized fleet run actually showed

A 300-request run through the Docker fleet, workload shifted from `code+chat` to
`medical+legal` halfway ([`results/fleet_loop.txt`](results/fleet_loop.txt)):

- **300 requests across two replicas (150/150), 30 re-decoded target-only,
  0 greedy mismatches, 0 responses without a traceable artifact digest.**
- Cold start served **no speculation**. The trainer then produced five
  candidates from live traffic; the gate **rejected three and promoted one**,
  which was hot-swapped into both replicas with no restart.

| candidate | tokens/forward | throughput | Welch p | p95 | bootstrap CI | verdict |
|---|---|---|---|---|---|---|
| v2 | 1.401 | 0.990× | 0.78 | 1.160× | 1.054–1.181 | **fail** (p95) |
| v3 | 1.246 | 0.962× | 0.52 | 1.238× | 1.191–1.291 | **fail** (p95) |
| v4 | 1.406 | 1.064× | 0.20 | 1.124× | 1.039–1.188 | **fail** (p95) |
| v5 | 1.481 | 1.024× | 0.61 | 1.049× | 0.961–1.136 | **pass → promoted** |

Every one of those throughput differences is statistically indistinguishable
from zero (p = 0.20–0.78). A threshold-only gate would have failed v4 on its
0.962×-looking neighbours and possibly passed v2 on its 1.064× sibling; this one
ignored the noise and decided on the one signal that *was* significant — the
bootstrap CI on p95, which excluded 1.0 for v2/v3/v4 and included it for v5.

**Why the p95 regressions are real.** The trainer maximises tokens per target
forward, which is the right objective when the forward pass is *memory-bound* —
on a GPU, carrying four extra draft tokens costs almost nothing. On **CPU the
forward is compute-bound**, so a wider speculative batch costs real time. v2 and
v4 were *better* on the metric the trainer optimises (1.40 tokens/forward against
the cold start's 1.00) and still slower at the tail. The objective is
mis-specified for CPU serving, and **the SLO gate caught that and kept three
regressions out of production with nobody watching a dashboard.** A cost-aware
objective is the fix and is listed in §8. On GPU, where the memory-bound
assumption holds, the same class of drafter does improve throughput — §5.

---

## 6. Running it

```bash
make setup          # uv venv on Python 3.12 + dependencies
make db             # Postgres 16 (the registry) on localhost:5433
make test           # 106 tests

make fleet-build    # build the CPU serving image
make fleet-up       # router :8080, replica-a :8081, replica-b :8082,
                    # plus trainer and canary controller
make fleet-status   # registry state vs. what each replica reports serving

# drive traffic and watch the loop close
PYTHONPATH=src .venv/bin/python scripts/drive_traffic.py --shift --requests 600
make fleet-logs
```

`fleet-status` is the interesting one during a rollout: it shows the registry's
view and each replica's view side by side, and the registry moves first.

### Configuration

Everything is env-overridable with a `FLEET_` prefix — see
[`src/fleet/config.py`](src/fleet/config.py). The knobs that matter most:
`FLEET_GAMMA` (draft tokens per verification step), `FLEET_CANARY_PCT`,
`FLEET_SLO_MIN_TPS_RATIO`, `FLEET_SLO_MAX_P95_RATIO`, `FLEET_DRIFT_THRESHOLD`,
`FLEET_AUDIT_SAMPLE_RATE`.

---

## 7. Aurora deployment

[`terraform/`](terraform/) provisions Aurora PostgreSQL-Compatible
(Serverless v2, 0.5 ACU floor) for the registry. The registry workload is a few
small rows per request plus some aggregates per canary evaluation, so it does
not need provisioned capacity, and scaling to the floor between runs keeps the
cost of standing it up in the low single dollars.

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
# set allowed_cidrs to your own address: curl -s https://checkip.amazonaws.com
terraform init && terraform apply

export FLEET_DATABASE_URL="postgresql://fleet:...@<writer-endpoint>:5432/fleet"
PYTHONPATH=src .venv/bin/python -m fleet.cli bootstrap   # applies the same DDL
```

The schema uses **core SQL only** — no extensions — so the identical DDL applies
to the local container and to Aurora, and development cannot drift away from what
Aurora will accept. The security group refuses `0.0.0.0/0` via a Terraform
`validation` block. The RDS Data API is enabled so the registry can also be
reached over HTTPS + SigV4 without a VPC connection, and an optional reader
instance lets the drift detector's window scans and the gate's aggregates run
off the writer that serving telemetry is inserting into.

Credentials go to Secrets Manager; Terraform outputs a command that reads the
password from there rather than printing it.

---

## 8. Limitations, stated plainly

- **The draft model is an n-gram/context-copy hybrid, not a neural drafter.**
  That is a deliberate fit to a 4 GB GPU: it retrains in *milliseconds*, which
  is what makes the self-improvement loop a loop rather than an overnight batch
  job, and what makes drift-triggered retraining observable in a single run. The
  `Drafter` interface ([`drafters/base.py`](src/fleet/specdec/drafters/base.py))
  is where a distilled transformer draft model or Medusa-style heads would plug
  in; nothing above that interface assumes which it is.
- **Target models are small** (Qwen2.5-0.5B / 1.5B). 4 GB of VRAM does not hold
  a 7B target plus a draft model plus KV cache. Speculative speedups generally
  grow with target size, so these figures should be read as conservative.
- **Containers are CPU-only** on this host, for the `nvidia-container-toolkit`
  reason above.
- **fp16 greedy equivalence is exact only up to sub-ULP argmax ties** — measured
  in §1, and not specific to speculation.
- **The training objective assumes a memory-bound target forward.** Maximising
  tokens per target forward is right on GPU and wrong on CPU, where a wider
  speculative batch costs real time — demonstrated in §5. The fix is to weight
  proposal width by measured per-step cost on the serving hardware; until then,
  the SLO gate is what stands between that mismatch and production.
- **The Aurora Terraform has not been applied.** It is written against the AWS
  provider's documented resource schema, but `terraform validate` and
  `terraform apply` have not been run against it — the Terraform CLI is not
  installed on this machine and no AWS account was attached. Treat
  `terraform/` as reviewed-but-unexecuted infrastructure; the schema in
  `sql/001_schema.sql` *is* exercised, against Postgres 16, on every test run.
- **Single-node.** The registry and artifact store are designed for multi-node
  (content-addressed artifacts, digest verification, a generation counter, one
  writer), but it has only been run on one machine, and the artifact store is a
  shared volume rather than S3.
- **The canary gate uses a threshold comparison, not a significance test.** With
  `canary_min_requests` in the dozens-to-hundreds, a sequential test
  (e.g. SPRT) would reject bad candidates sooner and with a stated error rate.
- **Promotion latency is measured in requests, not seconds, and the benchmark is
  single-stream.** A gate needs `canary_min_requests` on *both* arms before it
  can decide, so at a canary share of `p` it takes roughly
  `canary_min_requests / p` requests. A production fleet at thousands of
  requests per second crosses that in well under a second; this benchmark
  issues requests one at a time, so the same threshold costs hundreds of
  requests of adaptation lag. That lag is charged against the self-improving arm
  in every figure below — it is a property of the measurement setup, not of the
  design, and it makes the reported speedup conservative.

---

## 9. Layout

```
src/fleet/
  specdec/        speculative decoding engine, target wrapper, drafters
  registry/       versioned registry, content-addressed artifact store
  serving/        replica, hot-swap watcher, telemetry writer
  router/         round-robin router + fleet status
  trainer/        async trainer, traffic harvesting, offline scoring
  drift/          domain-aware drift detection
  canary/         SLO gate and rollout controller
  bench/          prompt pool, harness, experiments, report
sql/              registry schema (core SQL; Postgres and Aurora)
terraform/        Aurora Serverless v2
scripts/          precision study, traffic driver, specdec smoke test
tests/            106 tests
```
