"""Hot-swapping draft versions underneath a live serving replica.

A replica polls one row (`active_deployment`) and reacts to its `generation`
counter. When the generation moves it resolves the versions it should now be
able to serve, fetches any artifact it does not already hold (digest-verified,
see registry/artifacts.py), and swaps them in. The target model is never
reloaded and the process is never restarted.

Each replica holds *both* the active and the canary drafter simultaneously
rather than dedicating separate replicas to the canary. That matters for the
SLO gate: candidate and baseline requests are then served by the same
processes on the same hardware, so the throughput and latency comparison is not
confounded by which replica happened to get the traffic.

In-flight requests are unaffected by a swap: a request binds its drafter by
reference at entry, so a request that started on v4 finishes on v4 even if v5
lands mid-generation.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

from fleet.config import settings
from fleet.registry import store
from fleet.registry.artifacts import ArtifactStore
from fleet.specdec.drafters import LOADERS, load_artifact
from fleet.specdec.drafters.base import Drafter
from fleet.specdec.engine import SpeculativeEngine

log = logging.getLogger(__name__)


def load_drafter(version: int) -> Drafter:
    """Resolve a version number to a loaded, digest-verified drafter."""
    meta = store.get_version(version)
    if meta is None:
        raise LookupError(f"version {version} not in registry")
    if meta.drafter_kind not in LOADERS:
        raise LookupError(f"no loader for drafter kind {meta.drafter_kind!r}")
    path = ArtifactStore().fetch(meta.artifact_uri, meta.artifact_sha256)
    drafter = load_artifact(meta.drafter_kind, path)
    drafter.meta.version = version
    drafter.meta.offline_acceptance_rate = meta.offline_acceptance_rate
    drafter.meta.training_manifest = meta.training_manifest
    return drafter


@dataclass
class FleetState:
    """What this replica currently believes it should be serving."""

    generation: int = -1
    active_version: int | None = None
    canary_version: int | None = None
    canary_pct: int = 0
    drafters: dict[int, Drafter] = field(default_factory=dict)
    last_swap_ms: float = 0.0
    swaps: int = 0
    errors: int = 0

    def drafter_for(self, version: int | None) -> Drafter | None:
        return self.drafters.get(version) if version is not None else None


class DeploymentWatcher:
    """Background poller that keeps a replica's drafters in step with the
    registry."""

    def __init__(self, engine: SpeculativeEngine, interval: float | None = None) -> None:
        self.engine = engine
        self.interval = interval or settings.poll_interval_s
        self.state = FleetState()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        self.sync(force=True)
        self._thread = threading.Thread(target=self._loop, name="deployment-watcher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.sync()
            except Exception:  # noqa: BLE001 - a replica must keep serving
                self.state.errors += 1
                log.exception(
                    "deployment sync failed; continuing on v%s", self.state.active_version
                )

    # -- the swap -----------------------------------------------------------

    def sync(self, *, force: bool = False) -> bool:
        """Reconcile with the registry. Returns True if a swap happened."""
        dep = store.get_deployment()
        if not force and dep.generation == self.state.generation:
            return False
        if dep.active_version is None:
            log.warning("no active version in registry yet")
            return False

        t0 = time.perf_counter()
        wanted = {v for v in (dep.active_version, dep.canary_version) if v is not None}

        with self._lock:
            drafters = dict(self.state.drafters)
            for version in wanted:
                if version not in drafters:
                    drafters[version] = load_drafter(version)
            # Drop versions no longer addressable, freeing their tables.
            for version in list(drafters):
                if version not in wanted:
                    drafters.pop(version, None)

            previous = self.state.active_version
            self.state.drafters = drafters
            self.state.generation = dep.generation
            self.state.active_version = dep.active_version
            self.state.canary_version = dep.canary_version
            self.state.canary_pct = dep.canary_pct
            self.engine.swap_drafter(drafters[dep.active_version])
            swap_ms = (time.perf_counter() - t0) * 1000.0
            self.state.last_swap_ms = swap_ms
            self.state.swaps += 1

        log.info(
            "generation %s: active=v%s canary=%s (%s%%) in %.1fms",
            dep.generation,
            dep.active_version,
            f"v{dep.canary_version}" if dep.canary_version else "none",
            dep.canary_pct,
            swap_ms,
        )
        if previous != dep.active_version:
            store.ack_hotswap(settings.replica_id, dep.active_version, dep.generation, swap_ms)
        return True

    # -- per-request resolution --------------------------------------------

    def resolve(self, *, force_variant: str | None = None) -> tuple[int, Drafter, bool]:
        """Pick the version for one request.

        Returns (version, drafter, is_canary). Binding the drafter by reference
        here is what makes an in-flight request immune to a concurrent swap.
        """
        import random

        with self._lock:
            state = self.state
            active_v, canary_v = state.active_version, state.canary_version
            pct = state.canary_pct
            drafters = state.drafters

        if active_v is None:
            raise RuntimeError("replica has no active draft version")

        want_canary = False
        if force_variant == "canary":
            want_canary = canary_v is not None
        elif force_variant == "active":
            want_canary = False
        elif canary_v is not None and pct > 0:
            want_canary = random.randint(1, 100) <= pct

        if want_canary and canary_v in drafters:
            return canary_v, drafters[canary_v], True
        return active_v, drafters[active_v], False

    def snapshot(self) -> dict:
        with self._lock:
            s = self.state
            return {
                "replica_id": settings.replica_id,
                "generation": s.generation,
                "active_version": s.active_version,
                "canary_version": s.canary_version,
                "canary_pct": s.canary_pct,
                "loaded_versions": sorted(s.drafters),
                "last_swap_ms": round(s.last_swap_ms, 2),
                "swaps": s.swaps,
                "sync_errors": s.errors,
                "artifact_sync": ArtifactStore().sync_stats(),
            }
