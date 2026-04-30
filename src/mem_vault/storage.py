"""Vault storage backend for memories.

Each memory is a single ``.md`` file inside ``<vault>/<memory_subdir>``. The
schema mirrors the existing 99-AI/memory format already produced by other
agents, so files written by mem-vault are interchangeable with manual notes:

```
---
name: Short title
description: One-line synopsis
type: feedback | preference | decision | fact | note
tags: [foo, bar]
created: 2026-04-28T19:50:00-03:00
updated: 2026-04-28T19:50:00-03:00
agent_id: devin
user_id: default
origin_session_id: abc-123
contradicts: []
---
**Body in markdown**, freeform.
```

The filename is a slug derived from the title (or content). IDs are simply the
slug — they round-trip cleanly through the MCP tools and double as filenames.
"""

from __future__ import annotations

import io
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import frontmatter

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_VALID_TYPES = {"feedback", "preference", "decision", "fact", "note", "bug", "todo"}


VISIBLE_TO_ALL = "*"


@dataclass
class Memory:
    """A single memory entry (one .md file in the vault).

    Visibility model:
    - ``agent_id`` records the agent that created or last touched the memory.
    - ``visible_to`` is a list of agent ids that can read it. ``["*"]`` (the
      default for legacy / unset) means every agent. An empty list ``[]``
      means private to ``agent_id`` (and any caller that explicitly asks
      for that agent's memories).
    """

    id: str
    name: str
    description: str
    body: str = ""
    type: str = "note"
    tags: list[str] = field(default_factory=list)
    created: str = ""
    updated: str = ""
    agent_id: str | None = None
    user_id: str = "default"
    origin_session_id: str | None = None
    contradicts: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    visible_to: list[str] = field(default_factory=lambda: [VISIBLE_TO_ALL])
    extra: dict[str, Any] = field(default_factory=dict)

    def is_visible_to(self, agent_id: str | None) -> bool:
        """Returns True iff ``agent_id`` (or anyone, if None) may read this memory."""
        if not self.visible_to:
            # Empty list = strictly private to the owner agent.
            return agent_id is not None and agent_id == self.agent_id
        if VISIBLE_TO_ALL in self.visible_to:
            return True
        if agent_id is None:
            # No agent context → only see public memories.
            return False
        if agent_id == self.agent_id:
            return True
        return agent_id in self.visible_to

    def to_frontmatter(self) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "tags": list(self.tags),
            "created": self.created,
            "updated": self.updated,
            "user_id": self.user_id,
        }
        if self.agent_id:
            meta["agent_id"] = self.agent_id
        if self.origin_session_id:
            meta["origin_session_id"] = self.origin_session_id
        if self.contradicts:
            meta["contradicts"] = list(self.contradicts)
        if self.related:
            meta["related"] = list(self.related)
        # Only emit ``visible_to`` when it's non-default — files written by
        # other tools (engram, manual notes) should round-trip cleanly
        # without sprouting a noisy field they didn't ask for.
        if self.visible_to != [VISIBLE_TO_ALL]:
            meta["visible_to"] = list(self.visible_to)
        meta.update(self.extra)
        return meta

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "body": self.body,
            "type": self.type,
            "tags": self.tags,
            "created": self.created,
            "updated": self.updated,
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "origin_session_id": self.origin_session_id,
            "contradicts": self.contradicts,
            "related": self.related,
            "visible_to": self.visible_to,
        }


def slugify(text: str, max_len: int = 64) -> str:
    """ASCII-safe filename slug. Idempotent."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = _SLUG_RE.sub("_", text).strip("_")
    return (text or "memory")[:max_len]


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically.

    Strategy: write to a temp file in the same directory, ``fsync`` it, then
    ``os.replace`` it to the final name. ``os.replace`` is atomic on POSIX
    (single ``rename(2)`` syscall) and "as atomic as the OS allows" on
    Windows (``MoveFileExW`` with ``REPLACE_EXISTING``). Because the temp
    file lives in the same directory as the target, the rename never crosses
    filesystem boundaries (which would silently degrade to copy+unlink).

    Why this matters: without it, a process crash mid-write can leave a
    half-written ``.md`` on disk — frontmatter truncated, body cut off mid-
    line. With ``os.replace``, the target either points at the old contents
    or the fully-written new contents, never something in between.

    Notes for syncing filesystems (iCloud, Syncthing, Dropbox):
    - The temp file is named ``.<target>.<random>.tmp`` (leading dot) so
      most sync clients ignore it as a hidden file. Some clients still
      upload it briefly before the rename — that's harmless, just noisy.
    - We don't ``fsync`` the parent directory after the rename. On Linux
      that's recommended for full crash-durability, but it costs a
      noticeable I/O hit on iCloud-mounted dirs and the use case here
      (memory snapshot, recoverable from `mem-vault reindex`) doesn't need
      strict durability.
    """
    path = Path(path)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                # fsync can fail on some network/synced FSes (e.g. SMB,
                # certain iCloud edge cases). Keep going — we still get
                # the rename atomicity, just without strict durability.
                pass
        os.replace(tmp_path, path)
    except Exception:
        # On any failure, try to remove the temp file so we don't leak
        # ``.something.tmp`` files into the user's vault.
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


class VaultStorage:
    """Read/write memory ``.md`` files inside the vault.

    Pure filesystem layer — no embeddings, no LLM. The vector index is a
    separate concern in :mod:`mem_vault.index`.
    """

    def __init__(self, memory_dir: Path):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, memory_id: str) -> Path:
        return self.memory_dir / f"{memory_id}.md"

    def exists(self, memory_id: str) -> bool:
        return self.path_for(memory_id).exists()

    def _unique_id(self, base_slug: str) -> str:
        candidate = base_slug
        i = 2
        while self.exists(candidate):
            candidate = f"{base_slug}_{i}"
            i += 1
        return candidate

    def save(
        self,
        *,
        content: str,
        title: str | None = None,
        description: str | None = None,
        type: str = "note",
        tags: list[str] | None = None,
        agent_id: str | None = None,
        user_id: str = "default",
        origin_session_id: str | None = None,
        memory_id: str | None = None,
        visible_to: list[str] | None = None,
    ) -> Memory:
        """Write a new memory file. Returns the created Memory.

        - ``title`` defaults to the first line of ``content`` (truncated).
        - ``description`` defaults to the first 200 chars of ``content``.
        - ``memory_id``, if provided, overwrites any existing file with that id.
        - ``visible_to`` controls which agents can read this memory:
          ``["*"]`` (default) = visible to everyone,
          ``[]`` = strictly private to ``agent_id``,
          ``["claude-code", "cursor"]`` = visible to those agents + the owner.
        """
        if type not in _VALID_TYPES:
            type = "note"

        first_line = content.strip().splitlines()[0] if content.strip() else "memory"
        if title is None:
            title = first_line[:80].strip()
        if description is None:
            description = content.strip().replace("\n", " ")[:200]

        if memory_id is None:
            memory_id = self._unique_id(slugify(title))

        now = _now_iso()
        mem = Memory(
            id=memory_id,
            name=title,
            description=description,
            body=content.strip(),
            type=type,
            tags=tags or [],
            created=now,
            updated=now,
            agent_id=agent_id,
            user_id=user_id,
            origin_session_id=origin_session_id,
            visible_to=list(visible_to) if visible_to is not None else [VISIBLE_TO_ALL],
        )
        self._write(mem)
        return mem

    def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        title: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        visible_to: list[str] | None = None,
        related: list[str] | None = None,
    ) -> Memory:
        mem = self.get(memory_id)
        if mem is None:
            raise FileNotFoundError(f"Memory not found: {memory_id}")
        if content is not None:
            mem.body = content.strip()
        if title is not None:
            mem.name = title
        if description is not None:
            mem.description = description
        if tags is not None:
            mem.tags = tags
        if visible_to is not None:
            mem.visible_to = list(visible_to)
        if related is not None:
            mem.related = list(related)
        mem.updated = _now_iso()
        self._write(mem)
        return mem

    def delete(self, memory_id: str) -> bool:
        path = self.path_for(memory_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def get(self, memory_id: str) -> Memory | None:
        path = self.path_for(memory_id)
        if not path.exists():
            return None
        return self._read(path)

    def list(
        self,
        *,
        type: str | None = None,
        tags: list[str] | None = None,
        user_id: str | None = None,
        viewer_agent_id: str | None = None,
        limit: int = 50,
    ) -> list[Memory]:
        """List memories sorted by mtime desc, applying optional filters.

        ``viewer_agent_id`` enforces the visibility model: when provided,
        memories with restricted ``visible_to`` are filtered out unless the
        viewer is in the allowlist. Pass ``None`` (default) to skip the
        visibility check entirely — useful for admin / reindex flows.
        """
        results: list[Memory] = []
        files = sorted(self.memory_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in files:
            try:
                mem = self._read(path)
            except Exception:
                continue
            if type and mem.type != type:
                continue
            if tags and not set(tags).issubset(set(mem.tags)):
                continue
            if user_id and mem.user_id != user_id:
                continue
            if viewer_agent_id is not None and not mem.is_visible_to(viewer_agent_id):
                continue
            results.append(mem)
            if len(results) >= limit:
                break
        return results

    def _read(self, path: Path) -> Memory:
        post = frontmatter.load(path)
        meta = dict(post.metadata)
        # Strip well-known keys; everything else lands in `extra`.
        known = {
            "name",
            "description",
            "type",
            "tags",
            "created",
            "updated",
            "agent_id",
            "user_id",
            "origin_session_id",
            "originSessionId",
            "contradicts",
            "related",
            "visible_to",
        }
        # accept legacy "originSessionId" camelCase from existing files
        origin = meta.pop("origin_session_id", None) or meta.pop("originSessionId", None)
        extra = {k: v for k, v in meta.items() if k not in known}
        # Memories without an explicit ``visible_to`` (engram imports, manual
        # files, anything pre-visibility) default to public — the safest
        # backward-compatible behavior.
        visible_to_raw = meta.get("visible_to")
        if visible_to_raw is None:
            visible_to = [VISIBLE_TO_ALL]
        elif isinstance(visible_to_raw, list):
            visible_to = [str(v) for v in visible_to_raw]
        else:
            visible_to = [str(visible_to_raw)]
        return Memory(
            id=path.stem,
            name=str(meta.get("name") or path.stem),
            description=str(meta.get("description") or ""),
            body=post.content.strip(),
            type=str(meta.get("type") or "note"),
            tags=list(meta.get("tags") or []),
            created=str(meta.get("created") or ""),
            updated=str(meta.get("updated") or ""),
            agent_id=meta.get("agent_id"),
            user_id=str(meta.get("user_id") or "default"),
            origin_session_id=origin,
            contradicts=list(meta.get("contradicts") or []),
            related=[str(r) for r in (meta.get("related") or [])],
            visible_to=visible_to,
            extra=extra,
        )

    def _write(self, mem: Memory) -> None:
        post = frontmatter.Post(content=mem.body, **mem.to_frontmatter())
        # Serialize to bytes first so an exception while rendering the
        # frontmatter doesn't leave a partially-written file behind.
        buf = io.BytesIO()
        frontmatter.dump(post, buf)
        atomic_write_bytes(self.path_for(mem.id), buf.getvalue())
