"""Writing live inference feedback to the registry database.

Every served request produces one `traffic_events` row (metrics, provenance)
and one `traffic_samples` row (prompt tokens plus the target's greedy
continuation -- the training label).

Writes go through a bounded queue drained by a background thread. That is not
an optimisation for its own sake: the SLO gate compares candidate and baseline
p95 latency, so a synchronous insert on the serving path would put database
latency inside the number being gated on. The queue is bounded and drops
(loudly) under backpressure rather than growing without limit, because losing a
telemetry row is survivable and stalling a replica is not.
"""

from __future__ import annotations

import logging
import queue
import threading
import uuid
from dataclasses import dataclass
from typing import Any

from fleet import db

log = logging.getLogger(__name__)

QUEUE_MAX = 4096


@dataclass
class RequestRecord:
    request_id: uuid.UUID
    replica_id: str
    draft_version: int
    is_canary: bool
    domain: str
    prompt_sha256: str
    prompt_tokens: int
    generated_tokens: int
    draft_proposed: int
    draft_accepted: int
    target_forwards: int
    ttft_ms: float
    total_ms: float
    tokens_per_s: float
    greedy_match: bool | None
    prompt_token_ids: list[int]
    target_token_ids: list[int]
    store_sample: bool = True


class TelemetryWriter:
    def __init__(self) -> None:
        self._q: queue.Queue[RequestRecord | None] = queue.Queue(maxsize=QUEUE_MAX)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.written = 0
        self.dropped = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._q.put(None)
        if self._thread:
            self._thread.join(timeout=10)

    def submit(self, record: RequestRecord) -> None:
        try:
            self._q.put_nowait(record)
        except queue.Full:
            self.dropped += 1
            if self.dropped % 100 == 1:
                log.warning("telemetry queue full; dropped %d records", self.dropped)

    def flush(self, timeout: float = 10.0) -> None:
        """Block until the queue drains. Used by the benchmark harness, which
        needs every event durable before it computes statistics."""
        import time

        deadline = time.monotonic() + timeout
        while not self._q.empty() and time.monotonic() < deadline:
            time.sleep(0.02)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                record = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if record is None:
                break
            try:
                self._write(record)
                self.written += 1
            except Exception:  # noqa: BLE001 - telemetry loss must not kill serving
                log.exception("failed to write telemetry for %s", record.request_id)

    @staticmethod
    def _write(r: RequestRecord) -> None:
        with db.conn() as c, c.cursor() as cur:
            cur.execute(
                """
                INSERT INTO traffic_events
                    (request_id, replica_id, draft_version, is_canary, domain,
                     prompt_sha256, prompt_tokens, generated_tokens,
                     draft_proposed, draft_accepted, target_forwards,
                     ttft_ms, total_ms, tokens_per_s, greedy_match)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING event_id
                """,
                (
                    str(r.request_id),
                    r.replica_id,
                    r.draft_version,
                    r.is_canary,
                    r.domain,
                    r.prompt_sha256,
                    r.prompt_tokens,
                    r.generated_tokens,
                    r.draft_proposed,
                    r.draft_accepted,
                    r.target_forwards,
                    r.ttft_ms,
                    r.total_ms,
                    r.tokens_per_s,
                    r.greedy_match,
                ),
            )
            event_id = cur.fetchone()["event_id"]
            if r.store_sample and r.target_token_ids:
                cur.execute(
                    """
                    INSERT INTO traffic_samples
                        (event_id, domain, prompt_token_ids, target_token_ids)
                    VALUES (%s,%s,%s,%s)
                    """,
                    (event_id, r.domain, r.prompt_token_ids, r.target_token_ids),
                )

    def stats(self) -> dict[str, Any]:
        return {"written": self.written, "dropped": self.dropped, "queued": self._q.qsize()}


writer = TelemetryWriter()
