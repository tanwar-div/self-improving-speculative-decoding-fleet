"""Content-addressed artifact store with a verified local cache.

This is the weight-synchronisation path between training and serving. Training
writes an artifact exactly once into a store that every replica can read
(a shared volume locally, an S3 prefix in a real deployment). Replicas then:

  * resolve the version pointer from the registry (one small row),
  * compare the registry's sha256 against what is already in their local cache,
  * download only on a genuine change, and
  * verify the digest before the swap.

So a hot-swap across the fleet moves one artifact per version, not one per
replica per poll, and a replica never loads weights whose bytes it has not
checked. The digest is also what makes "reversible to a known-good version"
mean something: rolling back resolves the same bytes, not merely the same
version number.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from fleet.config import settings

log = logging.getLogger(__name__)


@dataclass
class StoredArtifact:
    uri: str
    sha256: str
    size_bytes: int


SCHEME = "artifact://"


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ArtifactStore:
    """Shared store plus a per-replica verified cache."""

    def __init__(self, root: Path | None = None, cache: Path | None = None) -> None:
        self.root = Path(root or settings.resolved_artifact_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache = Path(cache or settings.resolved_cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._fetches = 0
        self._bytes_fetched = 0

    # -- producer side ------------------------------------------------------

    def put(self, local_path: Path, *, name: str) -> StoredArtifact:
        """Publish a trained artifact.

        The returned URI is a store-relative key (`artifact://<file>`), never an
        absolute path. The registry row is read by processes that do not share a
        filesystem layout -- the trainer may write from the host while replicas
        read from a container mount -- so an absolute path in the registry makes
        a version loadable only by whoever wrote it. The key is logical; each
        process resolves it against its own configured root, which is also what
        lets the same row point at an S3 prefix instead of a volume.

        Writes are atomic: a replica polling mid-write can never observe a
        partial artifact.
        """
        sha = _digest(local_path)
        size = local_path.stat().st_size
        filename = f"{name}-{sha[:12]}.json"
        dest = self.root / filename
        if not dest.exists():
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            shutil.copyfile(local_path, tmp)
            os.replace(tmp, dest)
        return StoredArtifact(uri=f"{SCHEME}{filename}", sha256=sha, size_bytes=size)

    # -- consumer side ------------------------------------------------------

    def _resolve(self, uri: str) -> Path:
        """Map a registry URI onto this process's filesystem."""
        if uri.startswith(SCHEME):
            return self.root / uri[len(SCHEME) :]
        # Tolerate absolute paths written by older versions of this code: try
        # them as given, then fall back to the basename under this root.
        candidate = Path(uri)
        if candidate.exists():
            return candidate
        return self.root / candidate.name

    def fetch(self, uri: str, sha256: str) -> Path:
        """Return a local path holding exactly `sha256`, fetching only if the
        cache does not already hold those bytes."""
        cached = self.cache / f"{sha256}.json"
        if cached.exists():
            self._hits += 1
            return cached

        src = self._resolve(uri)
        if not src.exists():
            raise FileNotFoundError(
                f"artifact missing from store: {uri} (resolved to {src}; store root is {self.root})"
            )

        tmp = cached.with_suffix(".partial")
        shutil.copyfile(src, tmp)
        actual = _digest(tmp)
        if actual != sha256:
            tmp.unlink(missing_ok=True)
            raise ValueError(f"artifact digest mismatch for {uri}: expected {sha256}, got {actual}")
        os.replace(tmp, cached)
        self._fetches += 1
        self._bytes_fetched += cached.stat().st_size
        log.info("fetched artifact %s (%s bytes)", sha256[:12], cached.stat().st_size)
        return cached

    def sync_stats(self) -> dict[str, int]:
        return {
            "cache_hits": self._hits,
            "fetches": self._fetches,
            "bytes_fetched": self._bytes_fetched,
        }
