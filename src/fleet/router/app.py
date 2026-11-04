"""Fleet router.

Round-robins `/generate` across serving replicas and aggregates fleet state.
Deliberately thin: it does not decide canary assignment. Each replica resolves
its own variant per request from the registry's `canary_pct`, which keeps the
split correct even for traffic that bypasses the router (the benchmark harness
talks to replicas directly) and leaves the router with no state to go stale.
"""

from __future__ import annotations

import itertools
import logging
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request, Response

from fleet import db
from fleet.config import settings
from fleet.registry import store

log = logging.getLogger(__name__)

STATE: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=os.environ.get("FLEET_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    urls = settings.replica_url_list
    if not urls:
        raise RuntimeError("FLEET_REPLICA_URLS is empty; the router has nothing to route to")
    db.wait_for_db()
    STATE["urls"] = urls
    STATE["cycle"] = itertools.cycle(urls)
    STATE["client"] = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
    log.info("router up over %d replicas: %s", len(urls), ", ".join(urls))
    try:
        yield
    finally:
        await STATE["client"].aclose()  # type: ignore[union-attr]
        db.close_pool()


app = FastAPI(title="fleet router", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "replicas": STATE.get("urls", [])}


@app.post("/generate")
async def generate(request: Request, response: Response) -> dict:
    client: httpx.AsyncClient = STATE["client"]  # type: ignore[assignment]
    urls: list[str] = STATE["urls"]  # type: ignore[assignment]
    body = await request.json()

    # One pass over the ring: a replica reloading its drafter should not fail a
    # request that another replica can serve.
    last_error: str | None = None
    for _ in range(len(urls)):
        url = next(STATE["cycle"])  # type: ignore[arg-type]
        try:
            r = await client.post(f"{url}/generate", json=body)
            if r.status_code == 200:
                payload = r.json()
                response.headers["X-Draft-Version"] = str(payload.get("draft_version", ""))
                response.headers["X-Upstream"] = url
                return payload
            last_error = f"{url} -> {r.status_code}: {r.text[:200]}"
        except Exception as exc:  # noqa: BLE001
            last_error = f"{url} -> {exc}"
        log.warning("replica attempt failed: %s", last_error)

    raise HTTPException(status_code=502, detail=f"all replicas failed; last: {last_error}")


@app.get("/fleet/status")
async def fleet_status() -> dict:
    """Registry state plus what each replica reports it is actually serving.

    Divergence between the two is the thing you want to see during a rollout:
    the registry moves first, then each replica acknowledges.
    """
    client: httpx.AsyncClient = STATE["client"]  # type: ignore[assignment]
    dep = store.get_deployment()
    replicas = []
    for url in STATE["urls"]:  # type: ignore[union-attr]
        try:
            r = await client.get(f"{url}/version", timeout=10.0)
            replicas.append({"url": url, **r.json()})
        except Exception as exc:  # noqa: BLE001
            replicas.append({"url": url, "error": str(exc)})

    return {
        "registry": {
            "active_version": dep.active_version,
            "canary_version": dep.canary_version,
            "canary_pct": dep.canary_pct,
            "last_known_good": dep.last_known_good,
            "generation": dep.generation,
            "updated_reason": dep.updated_reason,
        },
        "replicas": replicas,
        "versions": [
            {
                "version": v.version,
                "status": v.status,
                "kind": v.drafter_kind,
                "offline_acceptance": v.offline_acceptance_rate,
                "train_samples": v.train_sample_count,
                "sha256": v.artifact_sha256[:12],
            }
            for v in store.list_versions(12)
        ],
        "history": store.deployment_history(15),
    }
