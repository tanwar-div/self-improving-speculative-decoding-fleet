"""Serving replica.

One process = one target model + the draft versions it is currently able to
serve. Endpoints are plain `def` (not `async def`) so FastAPI runs them in its
threadpool: decoding is a blocking compute workload, and pretending otherwise
with an async handler would serialise the whole event loop behind one request.

Every response carries the draft version that produced it, both in the body and
in an `X-Draft-Version` header, so provenance is visible at the HTTP layer and
not only in the database.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from fleet import db
from fleet.config import settings
from fleet.serving.hotswap import DeploymentWatcher
from fleet.serving.telemetry import RequestRecord, writer
from fleet.specdec.engine import SpeculativeEngine
from fleet.specdec.target import TargetModel

log = logging.getLogger(__name__)

STATE: dict[str, object] = {}


class GenerateRequest(BaseModel):
    prompt: str
    domain: str = "chat"
    max_new_tokens: int | None = None
    gamma: int | None = None
    # Lets the benchmark harness pin a variant instead of sampling it, so the
    # A/B comparison can be balanced by construction rather than by luck.
    variant: str | None = Field(default=None, pattern="^(active|canary)$")
    audit: bool | None = None
    store_sample: bool = True


class GenerateResponse(BaseModel):
    request_id: str
    text: str
    draft_version: int
    is_canary: bool
    replica_id: str
    prompt_tokens: int
    generated_tokens: int
    draft_proposed: int
    draft_accepted: int
    acceptance_rate: float | None
    target_forwards: int
    tokens_per_s: float
    ttft_ms: float
    total_ms: float
    greedy_match: bool | None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=os.environ.get("FLEET_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    db.wait_for_db()
    db.apply_schema()

    from fleet.trainer.worker import bootstrap

    bootstrap()  # idempotent; guarantees an active version exists to serve

    target = TargetModel.load()
    engine = SpeculativeEngine(target, gamma=settings.gamma)
    watcher = DeploymentWatcher(engine)
    watcher.start()
    writer.start()

    STATE.update(target=target, engine=engine, watcher=watcher)
    log.info("replica %s ready", settings.replica_id)
    try:
        yield
    finally:
        watcher.stop()
        writer.stop()
        db.close_pool()


app = FastAPI(title="fleet serving replica", lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict:
    watcher: DeploymentWatcher = STATE.get("watcher")  # type: ignore[assignment]
    if watcher is None or watcher.state.active_version is None:
        raise HTTPException(status_code=503, detail="no active draft version")
    return {"status": "ok", "replica_id": settings.replica_id}


@app.get("/version")
def version() -> dict:
    watcher: DeploymentWatcher = STATE["watcher"]  # type: ignore[assignment]
    return watcher.snapshot()


@app.get("/stats")
def stats() -> dict:
    watcher: DeploymentWatcher = STATE["watcher"]  # type: ignore[assignment]
    engine: SpeculativeEngine = STATE["engine"]  # type: ignore[assignment]
    drafter_stats = getattr(engine.drafter, "stats", lambda: {})()
    return {
        "deployment": watcher.snapshot(),
        "telemetry": writer.stats(),
        "drafter": drafter_stats,
        "target_model": settings.target_model,
        "device": settings.device,
        "gamma": settings.gamma,
    }


@app.post("/admin/sync")
def admin_sync() -> dict:
    """Force an immediate reconcile instead of waiting for the poll.

    Used by the canary controller and the benchmark harness to make rollouts
    deterministic in tests; the polling loop is what does this in production.
    """
    watcher: DeploymentWatcher = STATE["watcher"]  # type: ignore[assignment]
    swapped = watcher.sync(force=True)
    return {"swapped": swapped, **watcher.snapshot()}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest, response: Response) -> GenerateResponse:
    engine: SpeculativeEngine = STATE["engine"]  # type: ignore[assignment]
    target: TargetModel = STATE["target"]  # type: ignore[assignment]
    watcher: DeploymentWatcher = STATE["watcher"]  # type: ignore[assignment]

    try:
        draft_version, drafter, is_canary = watcher.resolve(force_variant=req.variant)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    max_new = req.max_new_tokens or settings.max_new_tokens
    prompt_ids = target.encode_chat(req.prompt)

    result = engine.decode(prompt_ids, max_new_tokens=max_new, gamma=req.gamma, drafter=drafter)

    # Correctness audit on a sampled fraction. Deliberately run *after* the
    # measured decode so the extra target-only pass never lands inside the
    # latency or throughput numbers the SLO gate reads.
    should_audit = (
        req.audit if req.audit is not None else random.random() < settings.audit_sample_rate
    )
    greedy_match: bool | None = None
    if should_audit:
        reference = engine.greedy_reference(prompt_ids, max_new_tokens=max_new)
        greedy_match = reference.generated_ids == result.generated_ids
        if not greedy_match:
            log.warning(
                "greedy mismatch on v%s (%s): audited request diverged",
                draft_version,
                req.domain,
            )

    request_id = uuid.uuid4()
    writer.submit(
        RequestRecord(
            request_id=request_id,
            replica_id=settings.replica_id,
            draft_version=draft_version,
            is_canary=is_canary,
            domain=req.domain,
            prompt_sha256=hashlib.sha256(req.prompt.encode()).hexdigest(),
            prompt_tokens=len(prompt_ids),
            generated_tokens=len(result.generated_ids),
            draft_proposed=result.draft_proposed,
            draft_accepted=result.draft_accepted,
            target_forwards=result.target_forwards,
            ttft_ms=result.ttft_ms,
            total_ms=result.total_ms,
            tokens_per_s=result.tokens_per_s,
            greedy_match=greedy_match,
            prompt_token_ids=prompt_ids,
            target_token_ids=result.generated_ids,
            store_sample=req.store_sample,
        )
    )

    response.headers["X-Draft-Version"] = str(draft_version)
    response.headers["X-Replica-Id"] = settings.replica_id
    return GenerateResponse(
        request_id=str(request_id),
        text=result.text,
        draft_version=draft_version,
        is_canary=is_canary,
        replica_id=settings.replica_id,
        prompt_tokens=len(prompt_ids),
        generated_tokens=len(result.generated_ids),
        draft_proposed=result.draft_proposed,
        draft_accepted=result.draft_accepted,
        acceptance_rate=result.acceptance_rate,
        target_forwards=result.target_forwards,
        tokens_per_s=result.tokens_per_s,
        ttft_ms=result.ttft_ms,
        total_ms=result.total_ms,
        greedy_match=greedy_match,
    )


@app.post("/admin/flush-telemetry")
def flush_telemetry() -> dict:
    writer.flush()
    return writer.stats()
