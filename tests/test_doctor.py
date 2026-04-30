"""Tests for ``mem-vault doctor`` — the one-shot health diagnostic.

No network / no Ollama: we stub the external touch-points (``ollama.Client``,
``sync_status``) and drive the report builder directly. The goal is the
report structure (name, status, hint) and the exit-code contract, not the
pretty-printing (we only smoke-test that ``_render`` doesn't explode).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from mem_vault.cli.doctor import (
    _check_extras,
    _check_feedback_loop,
    _check_models,
    _check_ollama,
    _check_state_dir,
    _check_vault,
    _model_is_installed,
    _render,
    _Report,
    run,
)
from mem_vault.config import Config

# ---------------------------------------------------------------------------
# Fixtures — a Config pointing at a tmp vault with sane defaults
# ---------------------------------------------------------------------------


@pytest.fixture
def doctor_config(tmp_path: Path) -> Config:
    vault = tmp_path / "vault"
    vault.mkdir()
    memory_dir = vault / "memory"
    memory_dir.mkdir()
    state = tmp_path / "state"
    state.mkdir()

    cfg = Config(
        vault_path=str(vault),
        memory_subdir="memory",
        state_dir=str(state),
        user_id="tester",
        max_content_size=0,
        llm_timeout_s=0,
    )
    cfg.qdrant_collection = "test"
    cfg.qdrant_path.mkdir(parents=True, exist_ok=True)
    return cfg


# ---------------------------------------------------------------------------
# _model_is_installed — prefix match, tag-aware
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target,installed,expected",
    [
        ("bge-m3:latest", {"bge-m3:latest"}, True),
        ("bge-m3:latest", {"bge-m3:q4_k_m"}, True),  # same base, different tag
        ("bge-m3", {"bge-m3:latest"}, True),
        ("qwen2.5:3b", {"qwen2.5:3b-instruct-q4_K_M"}, True),
        ("qwen2.5:3b", {"llama3:8b"}, False),
        ("", {"bge-m3"}, False),
        ("anything", set(), False),
    ],
)
def test_model_is_installed(target, installed, expected):
    assert _model_is_installed(target, installed) is expected


# ---------------------------------------------------------------------------
# _check_vault / _check_state_dir
# ---------------------------------------------------------------------------


def test_check_vault_ok_for_writable_dir(doctor_config):
    report = _Report()
    _check_vault(doctor_config, report)
    entry = next(c for c in report.checks if c.name == "vault")
    assert entry.status == "ok"


def test_check_vault_missing_vault_path_errors(tmp_path, monkeypatch):
    """vault_path not existing → err."""
    cfg = Config(
        vault_path=str(tmp_path / "nope"),
        memory_subdir="memory",
        state_dir=str(tmp_path / "state"),
        user_id="t",
        max_content_size=0,
        llm_timeout_s=0,
    )
    report = _Report()
    _check_vault(cfg, report)
    entry = next(c for c in report.checks if c.name == "vault")
    assert entry.status == "err"
    assert "does not exist" in entry.detail


def test_check_state_dir_ok_for_writable_dir(doctor_config):
    report = _Report()
    _check_state_dir(doctor_config, report)
    entry = next(c for c in report.checks if c.name == "state")
    assert entry.status == "ok"


# ---------------------------------------------------------------------------
# _check_ollama — stub the client
# ---------------------------------------------------------------------------


class _StubOllamaClient:
    def __init__(self, host, resp=None, raise_on_list=None):
        self.host = host
        self._resp = resp or {
            "models": [
                {"name": "bge-m3:latest"},
                {"name": "qwen2.5:3b"},
                {"name": "llama3:8b"},
            ]
        }
        self._raise = raise_on_list

    def list(self):
        if self._raise is not None:
            raise self._raise
        return self._resp


def test_check_ollama_up_returns_model_set(doctor_config, monkeypatch):
    import ollama

    monkeypatch.setattr(ollama, "Client", lambda host: _StubOllamaClient(host))
    report = _Report()
    ok, installed = _check_ollama(doctor_config, report)
    assert ok is True
    assert "bge-m3:latest" in installed
    assert "qwen2.5:3b" in installed
    assert next(c for c in report.checks if c.name == "ollama").status == "ok"


def test_check_ollama_down_reports_error(doctor_config, monkeypatch):
    import ollama

    monkeypatch.setattr(
        ollama,
        "Client",
        lambda host: _StubOllamaClient(host, raise_on_list=ConnectionError("boom")),
    )
    report = _Report()
    ok, installed = _check_ollama(doctor_config, report)
    assert ok is False
    assert installed == set()
    entry = next(c for c in report.checks if c.name == "ollama")
    assert entry.status == "err"
    assert "unreachable" in entry.detail
    assert entry.hint and "ollama serve" in entry.hint


def test_check_ollama_newer_response_shape(doctor_config, monkeypatch):
    """Newer ollama client versions use ``.models`` + ``.model`` attrs."""

    class _Model:
        def __init__(self, name):
            self.model = name
            self.name = name

    class _Resp:
        def __init__(self):
            self.models = [_Model("bge-m3"), _Model("qwen2.5:3b")]

    import ollama

    monkeypatch.setattr(
        ollama,
        "Client",
        lambda host: _StubOllamaClient(host, resp=_Resp()),
    )
    report = _Report()
    ok, installed = _check_ollama(doctor_config, report)
    assert ok is True
    assert "bge-m3" in installed


# ---------------------------------------------------------------------------
# _check_models
# ---------------------------------------------------------------------------


def test_check_models_all_green(doctor_config):
    report = _Report()
    installed = {"bge-m3:latest", "qwen2.5:3b"}
    _check_models(doctor_config, installed, report)
    statuses = {c.name: c.status for c in report.checks}
    assert statuses["model:embedder"] == "ok"
    assert statuses["model:llm"] == "ok"


def test_check_models_missing_embedder_errors(doctor_config):
    report = _Report()
    installed = {"qwen2.5:3b"}  # no bge-m3
    _check_models(doctor_config, installed, report)
    entry = next(c for c in report.checks if c.name == "model:embedder")
    assert entry.status == "err"
    assert entry.hint and "ollama pull" in entry.hint


# ---------------------------------------------------------------------------
# _check_extras — reranker + fastembed interaction
# ---------------------------------------------------------------------------


def test_check_extras_fastembed_present_reranker_off_warns(doctor_config, monkeypatch):
    """Common misconfig: user has the extra installed but forgot the flag."""
    # Pretend fastembed is importable by stubbing the import
    import sys

    monkeypatch.setitem(sys.modules, "fastembed", type(sys)("fastembed"))
    doctor_config.reranker_enabled = False
    report = _Report()
    _check_extras(doctor_config, report)
    entry = next(c for c in report.checks if c.name == "rerank")
    assert entry.status == "warn"
    assert "MEM_VAULT_RERANK" in (entry.hint or "")


def test_check_extras_fastembed_missing_rerank_enabled_errors(doctor_config, monkeypatch):
    """reranker_enabled=True without fastembed → config mismatch."""
    import sys

    monkeypatch.delitem(sys.modules, "fastembed", raising=False)
    # Force ``import fastembed`` to fail inside _check_extras
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "fastembed":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    doctor_config.reranker_enabled = True
    report = _Report()
    _check_extras(doctor_config, report)
    entry = next(c for c in report.checks if c.name == "rerank")
    assert entry.status == "err"


# ---------------------------------------------------------------------------
# _check_feedback_loop — surface usage_boost / tracking flags
# ---------------------------------------------------------------------------


def test_check_feedback_loop_all_on(doctor_config):
    doctor_config.usage_tracking_enabled = True
    doctor_config.usage_boost_enabled = True
    doctor_config.usage_boost = 0.3
    report = _Report()
    _check_feedback_loop(doctor_config, report)
    entry = next(c for c in report.checks if c.name == "feedback")
    assert entry.status == "ok"
    assert "boost=0.30" in entry.detail


def test_check_feedback_loop_tracking_off_warns(doctor_config):
    doctor_config.usage_tracking_enabled = False
    report = _Report()
    _check_feedback_loop(doctor_config, report)
    entry = next(c for c in report.checks if c.name == "feedback")
    assert entry.status == "warn"


def test_check_feedback_loop_boost_off_is_ok(doctor_config):
    """Boost disabled should be OK, not a warning — it's a valid choice."""
    doctor_config.usage_tracking_enabled = True
    doctor_config.usage_boost_enabled = False
    report = _Report()
    _check_feedback_loop(doctor_config, report)
    entry = next(c for c in report.checks if c.name == "feedback")
    assert entry.status == "ok"
    assert "boost disabled" in entry.detail


# ---------------------------------------------------------------------------
# _Report.exit_code — precedence ok > warn > err
# ---------------------------------------------------------------------------


def test_report_exit_code_zero_when_all_ok():
    report = _Report()
    report.add("x", "ok", "")
    assert report.exit_code() == 0


def test_report_exit_code_one_on_warn_only():
    report = _Report()
    report.add("x", "ok", "")
    report.add("y", "warn", "")
    assert report.exit_code() == 1


def test_report_exit_code_two_on_any_err():
    report = _Report()
    report.add("x", "ok", "")
    report.add("y", "warn", "")
    report.add("z", "err", "")
    assert report.exit_code() == 2


# ---------------------------------------------------------------------------
# _render — smoke test (no crash, writes to stdout)
# ---------------------------------------------------------------------------


def test_render_does_not_crash(capsys):
    report = _Report()
    report.add("x", "ok", "fine")
    report.add("y", "warn", "careful", hint="try Z")
    report.add("z", "err", "broken", hint="fix W")
    _render(report)
    out = capsys.readouterr().out
    assert "mem-vault doctor" in out
    assert "fine" in out
    assert "careful" in out
    assert "broken" in out


# ---------------------------------------------------------------------------
# run — end-to-end with every external touch-point stubbed
# ---------------------------------------------------------------------------


def _make_args(**overrides) -> argparse.Namespace:
    defaults = {"skip_ollama": True, "skip_index": True}
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_run_returns_zero_when_all_skipped_and_vault_healthy(doctor_config, monkeypatch, capsys):
    monkeypatch.setattr("mem_vault.config.load_config", lambda: doctor_config)
    # Skip both the expensive checks → only vault+state+extras+feedback run.
    # All of those should be ok for a freshly built config.
    rc = run(_make_args())
    assert rc in (0, 1)  # warnings from --skip flags themselves are OK
    out = capsys.readouterr().out
    assert "mem-vault doctor" in out


def test_run_config_load_failure_reports_err_and_exits_2(monkeypatch, capsys):
    def _boom():
        raise RuntimeError("vault not configured")

    monkeypatch.setattr("mem_vault.config.load_config", _boom)
    rc = run(_make_args())
    assert rc == 2
    out = capsys.readouterr().out
    assert "load_config failed" in out


def test_run_without_ollama_stub_surfaces_ollama_error(doctor_config, monkeypatch, capsys):
    monkeypatch.setattr("mem_vault.config.load_config", lambda: doctor_config)
    # Force the ollama client's list() call to raise — doctor should
    # record err. The constructor itself must stay cheap (matches prod
    # behavior where ``ollama.Client(host=...)`` doesn't roundtrip).
    import ollama

    monkeypatch.setattr(
        ollama,
        "Client",
        lambda host: _StubOllamaClient(host, raise_on_list=ConnectionError("refused")),
    )
    # Also stub sync_status so the index check doesn't try to touch qdrant
    from mem_vault import sync

    def _stub_status(cfg: Any):
        from mem_vault.sync import SyncReport

        return SyncReport(
            in_vault=0, in_index=0, stale_in_index=0, orphan_in_index=0, missing_in_index=0
        )

    monkeypatch.setattr(sync, "sync_status", _stub_status)
    rc = run(_make_args(skip_ollama=False, skip_index=False))
    # Ollama failure = err → exit 2
    assert rc == 2
