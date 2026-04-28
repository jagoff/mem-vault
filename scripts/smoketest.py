"""End-to-end smoke test against a temporary vault dir.

Runs the full save → search → get → delete cycle with both modes:
  - auto_extract=False (literal save)
  - auto_extract=True  (LLM extraction; requires Ollama)

Usage:
    uv run python scripts/smoketest.py [--skip-llm]
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

from mem_vault.config import Config
from mem_vault.server import MemVaultService


def _print(label: str, payload, took: float | None = None):
    suffix = f"  ({took:.2f}s)" if took is not None else ""
    print(f"\n=== {label}{suffix} ===", flush=True)
    print(json.dumps(payload, ensure_ascii=False, indent=2)[:1200], flush=True)


async def _timed(label: str, coro):
    t0 = time.time()
    res = await coro
    return res, time.time() - t0


async def run(skip_llm: bool = False) -> int:
    tmp_vault = Path(tempfile.mkdtemp(prefix="memvault_test_"))
    tmp_state = Path(tempfile.mkdtemp(prefix="memvault_state_"))
    try:
        config = Config(
            vault_path=tmp_vault,
            memory_subdir="memory",
            state_dir=tmp_state,
            qdrant_collection="memvault_smoke",
            user_id="smoketest",
            agent_id="smoketest-runner",
        )
        config.memory_dir.mkdir(parents=True, exist_ok=True)
        service = MemVaultService(config)

        # 1) literal save (no LLM)
        save1, took = await _timed("save literal", service.save({
            "content": "Fer prefers Spanish rioplatense in agent replies; technical jargon stays English.",
            "title": "Idioma preferido para agents",
            "type": "preference",
            "tags": ["language", "rioplatense"],
            "auto_extract": False,
        }))
        _print("save (literal)", save1, took)
        assert save1.get("ok") and save1.get("indexed"), "literal save+index failed"
        mid1 = save1["memory"]["id"]

        # 2) LLM-assisted save (auto_extract=True). Skipped if --skip-llm passed.
        if not skip_llm:
            save2, took = await _timed("save auto_extract", service.save({
                "content": (
                    "Decision: mem-vault uses Ollama as the only LLM provider and bge-m3 as embedder. "
                    "Vector store is Qdrant in embedded mode at ~/.local/share/mem-vault/qdrant. "
                    "No external API keys are read or required."
                ),
                "title": "Decision: 100% local stack for mem-vault",
                "type": "decision",
                "tags": ["architecture", "local-first"],
                "auto_extract": True,
            }))
            _print("save (auto_extract=True)", save2, took)
            if save2.get("indexed") is False:
                print("WARNING: auto_extract path failed to index — likely Ollama unreachable. Continuing.")

        # 3) search
        srch, took = await _timed("search", service.search(
            {"query": "what language should the agent use?", "k": 5}
        ))
        _print("search", srch, took)
        assert srch.get("ok"), "search failed"
        assert any(r.get("id") == mid1 for r in srch.get("results", [])), \
            f"expected hit on {mid1} not in {[r.get('id') for r in srch.get('results', [])]}"

        # 4) get
        got, took = await _timed("get", service.get({"id": mid1}))
        _print("get", got, took)
        assert got.get("ok") and got["memory"]["id"] == mid1

        # 5) update
        upd, took = await _timed("update", service.update(
            {"id": mid1, "tags": ["language", "rioplatense", "preference"]}
        ))
        _print("update", upd, took)
        assert upd.get("ok") and "preference" in upd["memory"]["tags"]

        # 6) list
        lst, took = await _timed("list", service.list_({"type": "preference", "limit": 10}))
        _print("list (type=preference)", lst, took)
        assert lst.get("ok") and lst.get("count", 0) >= 1

        # 7) delete
        deleted, took = await _timed("delete", service.delete({"id": mid1}))
        _print("delete", deleted, took)
        assert deleted.get("ok") and deleted.get("deleted_file")

        # 8) verify it's gone
        gone, took = await _timed("get-after-delete", service.get({"id": mid1}))
        _print("get (after delete)", gone, took)
        assert not gone.get("ok")

        print("\nALL CHECKS PASSED")
        return 0
    finally:
        shutil.rmtree(tmp_vault, ignore_errors=True)
        shutil.rmtree(tmp_state, ignore_errors=True)


if __name__ == "__main__":
    skip_llm = "--skip-llm" in sys.argv
    sys.exit(asyncio.run(run(skip_llm=skip_llm)))
