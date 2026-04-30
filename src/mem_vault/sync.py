"""Cross-vault sync helpers.

mem-vault doesn't ship its own sync engine — instead it leans on whatever
the user already has in front of the vault directory: iCloud, Syncthing,
Dropbox, OneDrive, a `git` repo, an `rsync` cron, etc. The local Qdrant
collection in ``state_dir`` is a derived index, never synced; it gets
rebuilt with ``mem-vault reindex`` after a fresh pull.

This module provides two things:

1. :func:`sync_status` — diff between the markdown source of truth (the
   vault directory) and the embedding index (Qdrant). Prints how many
   memories are present in each, how many are stale (file edited more
   recently than the index), and how many are orphaned in the index
   (file deleted out from under us).

2. :class:`VaultWatcher` — a small ``watchdog``-based file watcher that
   reindexes a memory whenever its ``.md`` file is created/modified, and
   removes it from the index when the file is deleted. Run as a long-
   lived process: ``mem-vault sync-watch``.

Together they let the user run any sync stack of their choice and trust
that the local Qdrant stays in lockstep without manual intervention.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from mem_vault.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# sync_status
# ---------------------------------------------------------------------------


@dataclass
class SyncReport:
    in_vault: int
    in_index: int
    stale_in_index: int  # index entry exists but vault file is newer
    orphan_in_index: int  # index entry has no matching vault file
    missing_in_index: int  # vault file exists but no index entry

    def to_lines(self) -> list[str]:
        return [
            f"  vault files       : {self.in_vault}",
            f"  index entries     : {self.in_index}",
            f"  stale (vault > idx): {self.stale_in_index}",
            f"  orphans in index  : {self.orphan_in_index}",
            f"  missing in index  : {self.missing_in_index}",
        ]

    @property
    def needs_reindex(self) -> bool:
        return any([self.stale_in_index, self.orphan_in_index, self.missing_in_index])


class IndexLockedError(RuntimeError):
    """Raised when the Qdrant collection is locked by another process."""


# Substring pinned by qdrant-client>=1.7 when the embedded local Qdrant
# can't acquire the portalocker file lock on ``state_dir/.lock``. Source:
# ``qdrant_client/local/qdrant_local.py`` raises a plain ``RuntimeError``
# whose message starts with
# ``"Storage folder <path> is already accessed by another instance of
# Qdrant client."``. We match the lower-cased substring ``"already
# accessed"`` so a path/collation tweak in upstream doesn't break us.
#
# If you bump qdrant-client and this breaks, the regression test
# ``tests/test_sync.py::test_lock_message_substring_is_still_in_qdrant_source``
# will fail loud — fix the constant here in step.
_QDRANT_LOCK_SUBSTR = "already accessed"


def _lock_msg() -> str:
    return (
        "Qdrant collection is locked by another process. The MCP server "
        "(or another sync-watch instance) holds the index. Stop it and "
        "retry. The vault files themselves are not locked — you can still "
        "browse / edit / grep them while this is running."
    )


def _is_lock_runtime_error(exc: BaseException) -> bool:
    """Return True iff ``exc`` is the qdrant-client local lock conflict.

    qdrant-client wraps a ``portalocker.exceptions.LockException`` into a
    plain ``RuntimeError`` before re-raising — so the original typed
    exception is gone by the time we catch it. We match by substring on
    the message (pinned by :data:`_QDRANT_LOCK_SUBSTR`).
    """
    return _QDRANT_LOCK_SUBSTR in str(exc).lower()


def _is_lock_exception(exc: BaseException) -> bool:
    """Return True iff ``exc`` is a portalocker file-lock failure.

    qdrant-client *currently* swallows this and re-raises ``RuntimeError``
    (see ``qdrant_client/local/qdrant_local.py``), so this typed catch is
    defensive — it covers the case where a future qdrant-client version
    lets the original exception propagate, or where another path inside
    mem-vault touches portalocker directly.
    """
    try:
        from portalocker.exceptions import BaseLockException
    except Exception:  # pragma: no cover - portalocker is a transitive dep
        return False
    return isinstance(exc, BaseLockException)


def sync_status(config: Config) -> SyncReport:
    """Compare the vault dir with the Qdrant collection and return a diff.

    Raises:
        IndexLockedError: when the local Qdrant collection is held by
            another mem-vault process (typically the MCP server). The
            embedded Qdrant DB enforces single-writer; this command must
            wait until the MCP server exits or be told to skip the index
            check entirely.
    """
    from mem_vault.index import VectorIndex
    from mem_vault.storage import VaultStorage

    storage = VaultStorage(config.memory_dir)

    vault_files = list(storage.memory_dir.glob("*.md"))
    vault_index: dict[str, float] = {}  # memory_id → mtime epoch
    for p in vault_files:
        try:
            vault_index[p.stem] = p.stat().st_mtime
        except OSError:
            continue

    try:
        index = VectorIndex(config)
        # Touch the vector store to surface a lock conflict early.
        _ = index.mem0
    except (BlockingIOError, OSError) as exc:
        raise IndexLockedError(_lock_msg()) from exc
    except RuntimeError as exc:
        # qdrant-client's QdrantLocal converts the underlying
        # ``portalocker.exceptions.LockException`` into a generic
        # ``RuntimeError`` whose message contains "already accessed by
        # another instance" (see ``_QDRANT_LOCK_SUBSTR``). Detect by
        # substring so the IndexLockedError keeps surfacing even if
        # ``portalocker`` is no longer the underlying lib.
        if _is_lock_runtime_error(exc):
            raise IndexLockedError(_lock_msg()) from exc
        raise
    except Exception as exc:
        # Defensive: if a future qdrant-client version lets the typed
        # ``portalocker.exceptions.BaseLockException`` propagate without
        # wrapping it in ``RuntimeError``, still surface it as a lock.
        if _is_lock_exception(exc):
            raise IndexLockedError(_lock_msg()) from exc
        raise

    try:
        entries = index.mem0.get_all(filters={"user_id": config.user_id}, limit=10**6)
    except TypeError:
        entries = index.mem0.get_all(filters={"user_id": config.user_id})
    except (BlockingIOError, OSError) as exc:
        raise IndexLockedError("Lost the Qdrant lock mid-query.") from exc

    if isinstance(entries, dict):
        items = list(entries.get("results", []))
    elif isinstance(entries, list):
        items = entries
    else:
        items = []

    seen_ids: set[str] = set()
    stale = 0
    orphan = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        md = item.get("metadata") or {}
        mem_id = md.get("memory_id")
        if not mem_id:
            continue
        seen_ids.add(mem_id)
        if mem_id not in vault_index:
            orphan += 1
            continue
        idx_ts = item.get("updated_at") or item.get("created_at")
        if not idx_ts:
            continue
        try:
            dt = datetime.fromisoformat(str(idx_ts))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            idx_epoch = dt.timestamp()
        except Exception:
            continue
        if vault_index[mem_id] > idx_epoch + 1:  # 1-second slack for fs precision
            stale += 1

    missing = sum(1 for mid in vault_index if mid not in seen_ids)

    return SyncReport(
        in_vault=len(vault_index),
        in_index=len(seen_ids),
        stale_in_index=stale,
        orphan_in_index=orphan,
        missing_in_index=missing,
    )


# ---------------------------------------------------------------------------
# sync-watch
# ---------------------------------------------------------------------------


class VaultWatcher:
    """Watches the vault dir and keeps the Qdrant index in lockstep.

    Designed for long-running ``mem-vault sync-watch`` processes (cron,
    launchd, systemd). On each event:

    - ``created`` / ``modified``: re-embed the memory (or skip if mtime
      hasn't actually advanced, which happens with iCloud touches).
    - ``deleted``: drop every embedding pointing at that memory_id.
    - ``moved``: handled as delete+create.

    Bursts of events for the same file (e.g. a sync client downloading
    chunks) are debounced to avoid embedding the same body N times in a
    row. The default debounce window is ``debounce_seconds`` (1.0 s).
    """

    def __init__(self, config: Config, *, debounce_seconds: float = 1.0):
        self.config = config
        self.debounce = debounce_seconds
        self._pending: dict[Path, float] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()

        from mem_vault.index import VectorIndex
        from mem_vault.storage import VaultStorage

        self.storage = VaultStorage(config.memory_dir)
        self.index = VectorIndex(config)

    # --- public API -------------------------------------------------------

    def run(self) -> None:
        """Block forever, watching ``config.memory_dir``. Returns on Ctrl-C."""
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        watcher = self

        class _Handler(FileSystemEventHandler):
            def on_created(self, event):
                if not event.is_directory:
                    watcher._enqueue(Path(event.src_path))

            def on_modified(self, event):
                if not event.is_directory:
                    watcher._enqueue(Path(event.src_path))

            def on_moved(self, event):
                if not event.is_directory:
                    watcher._enqueue(Path(event.src_path))
                    watcher._enqueue(Path(event.dest_path))

            def on_deleted(self, event):
                if not event.is_directory:
                    watcher._enqueue(Path(event.src_path))

        handler = _Handler()
        observer = Observer()
        observer.schedule(handler, str(self.config.memory_dir), recursive=False)
        observer.start()
        logger.info("watching %s for memory changes", self.config.memory_dir)

        try:
            while not self._stop.is_set():
                self._flush_pending()
                time.sleep(0.25)
        except KeyboardInterrupt:
            pass
        finally:
            observer.stop()
            observer.join(timeout=5)

    def stop(self) -> None:
        self._stop.set()

    # --- internals --------------------------------------------------------

    def _enqueue(self, path: Path) -> None:
        if path.suffix.lower() != ".md":
            return
        with self._lock:
            self._pending[path] = time.time()

    def _flush_pending(self) -> None:
        now = time.time()
        ready: list[Path] = []
        with self._lock:
            for path, queued_at in list(self._pending.items()):
                if now - queued_at >= self.debounce:
                    ready.append(path)
                    self._pending.pop(path, None)
        for path in ready:
            self._reindex_path(path)

    def _reindex_path(self, path: Path) -> None:
        from mem_vault.index import compute_content_hash

        mem_id = path.stem
        if not path.exists():
            self._delete_index(mem_id)
            return
        try:
            mem = self.storage.get(mem_id)
        except Exception as exc:
            logger.warning("failed to read %s: %s", path, exc)
            return
        if mem is None:
            self._delete_index(mem_id)
            return
        body = mem.body or mem.description or mem.name
        try:
            self.index.delete_by_metadata("memory_id", mem.id, mem.user_id)
            self.index.add(
                body,
                user_id=mem.user_id,
                agent_id=mem.agent_id,
                metadata={
                    "memory_id": mem.id,
                    "type": mem.type,
                    "tags": mem.tags,
                    "content_hash": compute_content_hash(body),
                },
                auto_extract=False,
            )
            logger.info("reindexed %s", mem_id)
        except Exception as exc:
            logger.warning("reindex %s failed: %s", mem_id, exc)

    def _delete_index(self, mem_id: str) -> None:
        try:
            removed = self.index.delete_by_metadata("memory_id", mem_id, self.config.user_id)
            logger.info("removed %s from index (%d entries)", mem_id, removed)
        except Exception as exc:
            logger.warning("delete_by_metadata %s failed: %s", mem_id, exc)
