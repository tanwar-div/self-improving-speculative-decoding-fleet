-- Self-improving speculative decoding fleet: registry + feedback schema.
-- Target: PostgreSQL 14+ / Amazon Aurora PostgreSQL-Compatible.
-- Deliberately core-only SQL (no extensions) so the same DDL applies to a local
-- Postgres container and to Aurora via the RDS Data API.

CREATE SCHEMA IF NOT EXISTS fleet;
SET search_path TO fleet, public;

-- ---------------------------------------------------------------------------
-- Model registry
-- ---------------------------------------------------------------------------

-- Lifecycle of a draft-model version.
--   candidate  : trained, artifact stored, not yet serving
--   canary     : receiving a fraction of production traffic under SLO watch
--   active     : serving all traffic
--   retired    : superseded by a newer active version (still rollback-eligible)
--   rejected   : failed its SLO gate, never took full traffic
--   rolled_back: was active, reverted after regression
CREATE TABLE IF NOT EXISTS draft_versions (
    version          BIGINT PRIMARY KEY,
    drafter_kind     TEXT        NOT NULL,
    target_model     TEXT        NOT NULL,
    status           TEXT        NOT NULL DEFAULT 'candidate'
        CHECK (status IN ('candidate','canary','active','retired','rejected','rolled_back')),
    parent_version   BIGINT      REFERENCES draft_versions(version),

    -- Artifact identity. sha256 is what makes a rollout reversible to a
    -- *known-good* version rather than merely to a version number.
    artifact_uri     TEXT        NOT NULL,
    artifact_sha256  CHAR(64)    NOT NULL,
    artifact_bytes   BIGINT      NOT NULL,

    -- Which slice of live traffic this version learned from.
    trained_from_event_id BIGINT,
    trained_to_event_id   BIGINT,
    train_sample_count    INTEGER NOT NULL DEFAULT 0,

    -- Fingerprint of the training distribution. Drift detection compares the
    -- live request distribution against this, so retraining fires on actual
    -- workload shift instead of on a timer.
    training_manifest JSONB      NOT NULL DEFAULT '{}'::jsonb,
    hyperparams       JSONB      NOT NULL DEFAULT '{}'::jsonb,

    -- Offline acceptance rate measured on a held-out slice at train time.
    offline_acceptance_rate DOUBLE PRECISION,

    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    promoted_at      TIMESTAMPTZ,
    retired_at       TIMESTAMPTZ,
    notes            TEXT
);

CREATE INDEX IF NOT EXISTS idx_draft_versions_status  ON draft_versions(status);
CREATE INDEX IF NOT EXISTS idx_draft_versions_created ON draft_versions(created_at DESC);

-- At most one version may be 'active' and at most one 'canary' fleet-wide.
CREATE UNIQUE INDEX IF NOT EXISTS uq_draft_versions_one_active
    ON draft_versions((status)) WHERE status = 'active';
CREATE UNIQUE INDEX IF NOT EXISTS uq_draft_versions_one_canary
    ON draft_versions((status)) WHERE status = 'canary';

-- ---------------------------------------------------------------------------
-- Deployment pointer: the single row every serving replica polls.
-- Replicas hot-swap when `generation` changes; they never need a restart.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS active_deployment (
    id               BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (id),
    active_version   BIGINT  REFERENCES draft_versions(version),
    canary_version   BIGINT  REFERENCES draft_versions(version),
    canary_pct       INTEGER NOT NULL DEFAULT 0 CHECK (canary_pct BETWEEN 0 AND 100),
    last_known_good  BIGINT  REFERENCES draft_versions(version),
    generation       BIGINT  NOT NULL DEFAULT 0,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_reason   TEXT
);

INSERT INTO active_deployment (id) VALUES (TRUE) ON CONFLICT (id) DO NOTHING;

-- ---------------------------------------------------------------------------
-- Live inference feedback. One row per served request.
-- draft_version is NOT NULL on purpose: every served response is traceable to
-- the exact draft-model version behind it.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS traffic_events (
    event_id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    request_id       UUID        NOT NULL,
    ts               TIMESTAMPTZ NOT NULL DEFAULT now(),
    replica_id       TEXT        NOT NULL,
    draft_version    BIGINT      NOT NULL REFERENCES draft_versions(version),
    is_canary        BOOLEAN     NOT NULL DEFAULT FALSE,

    domain           TEXT        NOT NULL,
    prompt_sha256    CHAR(64)    NOT NULL,
    prompt_tokens    INTEGER     NOT NULL,
    generated_tokens INTEGER     NOT NULL,

    -- Speculative decoding accounting.
    draft_proposed   INTEGER     NOT NULL DEFAULT 0,
    draft_accepted   INTEGER     NOT NULL DEFAULT 0,
    target_forwards  INTEGER     NOT NULL DEFAULT 0,

    -- Latency / throughput as served.
    ttft_ms          DOUBLE PRECISION,
    total_ms         DOUBLE PRECISION NOT NULL,
    tokens_per_s     DOUBLE PRECISION NOT NULL,

    -- Greedy-equivalence check, when this request was audited against plain
    -- target-only greedy decoding. NULL = not audited.
    greedy_match     BOOLEAN,

    CONSTRAINT ck_accepted_le_proposed CHECK (draft_accepted <= draft_proposed)
);

CREATE INDEX IF NOT EXISTS idx_traffic_ts       ON traffic_events(ts DESC);
CREATE INDEX IF NOT EXISTS idx_traffic_version  ON traffic_events(draft_version, ts DESC);
CREATE INDEX IF NOT EXISTS idx_traffic_canary   ON traffic_events(is_canary, ts DESC);
CREATE INDEX IF NOT EXISTS idx_traffic_domain   ON traffic_events(domain, ts DESC);
CREATE INDEX IF NOT EXISTS idx_traffic_request  ON traffic_events(request_id);

-- Token-level training data harvested from live traffic. Split from
-- traffic_events so the hot metrics path stays narrow.
CREATE TABLE IF NOT EXISTS traffic_samples (
    event_id         BIGINT PRIMARY KEY REFERENCES traffic_events(event_id) ON DELETE CASCADE,
    domain           TEXT   NOT NULL,
    prompt_token_ids INTEGER[] NOT NULL,
    -- The target model's own greedy continuation: the distillation label.
    target_token_ids INTEGER[] NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_samples_domain ON traffic_samples(domain, created_at DESC);

-- ---------------------------------------------------------------------------
-- SLO gate: what decides a promotion. No human in the loop.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS slo_evaluations (
    id                 BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    candidate_version  BIGINT NOT NULL REFERENCES draft_versions(version),
    baseline_version   BIGINT NOT NULL REFERENCES draft_versions(version),
    window_start       TIMESTAMPTZ NOT NULL,
    window_end         TIMESTAMPTZ NOT NULL,

    candidate_requests INTEGER NOT NULL,
    baseline_requests  INTEGER NOT NULL,

    candidate_tps      DOUBLE PRECISION NOT NULL,
    baseline_tps       DOUBLE PRECISION NOT NULL,
    candidate_p95_ms   DOUBLE PRECISION NOT NULL,
    baseline_p95_ms    DOUBLE PRECISION NOT NULL,
    candidate_acceptance DOUBLE PRECISION,
    baseline_acceptance  DOUBLE PRECISION,
    candidate_greedy_mismatches INTEGER NOT NULL DEFAULT 0,

    tps_ratio          DOUBLE PRECISION NOT NULL,
    p95_ratio          DOUBLE PRECISION NOT NULL,

    gate_result        TEXT NOT NULL CHECK (gate_result IN ('pass','fail','insufficient_data')),
    failed_checks      JSONB NOT NULL DEFAULT '[]'::jsonb,
    thresholds         JSONB NOT NULL DEFAULT '{}'::jsonb,
    decided_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_slo_candidate ON slo_evaluations(candidate_version, decided_at DESC);

-- ---------------------------------------------------------------------------
-- Full audit trail. Every rollout and every revert is a row here.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS deployment_events (
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ts            TIMESTAMPTZ NOT NULL DEFAULT now(),
    action        TEXT NOT NULL CHECK (action IN (
                      'register','canary_start','canary_scale','canary_abort',
                      'promote','rollback','hotswap_ack','retire')),
    from_version  BIGINT REFERENCES draft_versions(version),
    to_version    BIGINT REFERENCES draft_versions(version),
    actor         TEXT NOT NULL DEFAULT 'controller',
    replica_id    TEXT,
    slo_evaluation_id BIGINT REFERENCES slo_evaluations(id),
    reason        TEXT,
    detail        JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_deploy_ts     ON deployment_events(ts DESC);
CREATE INDEX IF NOT EXISTS idx_deploy_action ON deployment_events(action, ts DESC);

-- ---------------------------------------------------------------------------
-- Drift detection log: why (or why not) retraining fired.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS drift_checks (
    id                 BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ts                 TIMESTAMPTZ NOT NULL DEFAULT now(),
    reference_version  BIGINT NOT NULL REFERENCES draft_versions(version),
    window_events      INTEGER NOT NULL,
    domain_js          DOUBLE PRECISION,   -- Jensen-Shannon divergence, domain mixture
    token_psi          DOUBLE PRECISION,   -- population stability index, token unigrams
    acceptance_drop    DOUBLE PRECISION,   -- live acceptance rate vs. training-time
    score              DOUBLE PRECISION NOT NULL,
    threshold          DOUBLE PRECISION NOT NULL,
    triggered          BOOLEAN NOT NULL,
    per_domain         JSONB NOT NULL DEFAULT '{}'::jsonb,
    reason             TEXT
);

CREATE INDEX IF NOT EXISTS idx_drift_ts ON drift_checks(ts DESC);

-- ---------------------------------------------------------------------------
-- Traceability view: a served request -> the exact artifact behind it.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW response_provenance AS
SELECT e.request_id,
       e.ts,
       e.replica_id,
       e.domain,
       e.is_canary,
       e.tokens_per_s,
       e.total_ms,
       e.greedy_match,
       CASE WHEN e.draft_proposed = 0 THEN NULL
            ELSE e.draft_accepted::double precision / e.draft_proposed END AS acceptance_rate,
       v.version        AS draft_version,
       v.drafter_kind,
       v.target_model,
       v.artifact_uri,
       v.artifact_sha256,
       v.status         AS version_status_now
FROM traffic_events e
JOIN draft_versions v ON v.version = e.draft_version;

-- ---------------------------------------------------------------------------
-- Additive migrations. Kept in this file and written with IF NOT EXISTS so
-- that applying the schema stays a single idempotent operation against both a
-- fresh database and one already carrying traffic.
-- ---------------------------------------------------------------------------

-- The gate originally compared point estimates against fixed thresholds and
-- would fail a candidate whose sample was merely noisy. It now also records
-- whether each apparent regression was statistically distinguishable.
ALTER TABLE slo_evaluations
    ADD COLUMN IF NOT EXISTS statistical_tests JSONB NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE slo_evaluations
    ADD COLUMN IF NOT EXISTS decision_note TEXT;
