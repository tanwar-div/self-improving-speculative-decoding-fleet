"""Central configuration. Every knob is env-overridable so the same code runs
on the host with a GPU and inside a CPU-only serving container."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FLEET_", env_file=".env", extra="ignore")

    # --- storage -----------------------------------------------------------
    database_url: str = "postgresql://fleet:fleet@localhost:5433/fleet"
    # The shared artifact store: a volume locally, an S3 prefix in a real
    # deployment. Written by the trainer, read by every replica.
    artifact_dir: Path = REPO_ROOT / "artifacts"
    # Each replica's own verified download cache. Deliberately NOT inside
    # artifact_dir: it is per-process scratch, not shared state, and nesting it
    # in the shared store meant one process's cache permissions could block
    # every other process's access to the store.
    artifact_cache_dir: Path = REPO_ROOT / ".artifact-cache"

    # --- models ------------------------------------------------------------
    # The frozen target. Speculative decoding never changes its output; only
    # the draft model is retrained.
    target_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = "cpu"  # "cuda" on the host benchmark, "cpu" in containers
    dtype: str = "auto"  # auto | float32 | float16 | bfloat16
    load_in_4bit: bool = False

    # --- speculative decoding ---------------------------------------------
    gamma: int = Field(default=4, description="draft tokens proposed per verification step")
    max_new_tokens: int = 48

    # --- drafter -----------------------------------------------------------
    drafter_kind: str = "ngram"  # ngram | head | none
    ngram_order: int = 3
    ngram_min_count: int = 2

    # --- serving -----------------------------------------------------------
    replica_id: str = "replica-local"
    replica_urls: str = ""  # comma-separated; the controller pushes swaps here
    poll_interval_s: float = 2.0
    audit_sample_rate: float = 0.02  # fraction of requests re-decoded target-only

    # --- canary / SLO gate -------------------------------------------------
    canary_pct: int = 20
    canary_min_requests: int = 60
    # Candidate must not lose throughput and must not inflate tail latency.
    slo_min_tps_ratio: float = 0.98
    slo_max_p95_ratio: float = 1.05
    slo_require_greedy_match: bool = True
    # A candidate only fails when the regression is *measurable*: the point
    # estimate must violate the threshold AND the difference must be
    # significant at this level. Without it the gate rejects on noise.
    slo_alpha: float = 0.05
    # Cap on how long an undecidable canary keeps running. On reaching it, a
    # candidate that still cannot be distinguished from the incumbent is
    # promoted -- a neutral candidate passing is intended behaviour.
    canary_max_requests: int = 400

    # --- drift detection ---------------------------------------------------
    drift_window: int = 400
    drift_threshold: float = 0.15
    drift_min_interval_s: float = 30.0

    # --- trainer -----------------------------------------------------------
    train_min_samples: int = 200
    # Retrain once this many times the active version's training-set size has
    # accumulated as *new* traffic, even with no drift. Drift detection answers
    # "has the workload moved?" -- it cannot answer "is the drafter as good as
    # my data now allows?", and without this the fleet stops improving the
    # moment the workload goes stationary, which is the opposite of the point.
    retrain_data_growth: float = 1.0
    train_max_samples: int = 6000
    trainer_poll_s: float = 10.0

    @property
    def replica_url_list(self) -> list[str]:
        return [u.strip().rstrip("/") for u in self.replica_urls.split(",") if u.strip()]

    @property
    def resolved_artifact_dir(self) -> Path:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        return self.artifact_dir

    @property
    def resolved_cache_dir(self) -> Path:
        path = self.artifact_cache_dir / self.replica_id
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
