"""Contract tests — schema↔implementation symmetry guards.

These tests catch a specific class of regression: drift between a declarative
schema (the ``_TOOLS`` list, the ``ENV_TO_CONFIG_FIELD`` registry, request
Pydantic models) and the runtime implementation that's supposed to honor it.

Each contract is enforced by a single parametric test that fails loud at CI
time, before the broken pair ever reaches a user.

Examples of bugs these tests catch:

1. **TOOLS↔Service drift** — adding a new tool to ``_TOOLS`` (e.g.
   ``memory_briefing``) without adding the matching ``MemVaultService.briefing``
   method. The MCP server crashes at boot in remote mode with
   ``AttributeError: Service RemoteMemVaultService is missing handler``.
   This test catches it before the build ships. (This was the actual bug
   shipped on 2026-04-30 in commit be15062 — fixed in dd26525, this test
   prevents recurrence.)

2. **Env-var typos** — adding ``MEM_VAULT_NUEVO`` to ``ENV_TO_CONFIG_FIELD``
   pointing at a misspelled field like ``new_feature_enable`` (no ``d``).
   The env var would silently no-op until someone tried to use it.

3. **Pydantic model drift** — ``MemoryCreate`` accepting a field
   (``some_new_arg``) that ``MemVaultService.save`` doesn't read; or
   the reverse, ``save`` consuming an arg the schema doesn't allow,
   so remote callers get 422 errors that local callers don't.
"""

from __future__ import annotations

from mem_vault.config import ENV_TO_CONFIG_FIELD, Config
from mem_vault.remote import RemoteMemVaultService
from mem_vault.server import _HANDLER_OVERRIDES, _TOOLS, MemVaultService

# ---------------------------------------------------------------------------
# Contract 1: every _TOOLS entry has a callable on BOTH service classes
# ---------------------------------------------------------------------------
#
# The MCP server enforces this at boot via ``_build_handlers``, but at boot
# time it only sees ONE service (the one ``build_service`` selected based on
# ``MEM_VAULT_REMOTE_URL``). If a method is missing on the OTHER service the
# bug stays latent until someone flips that env var. This test walks both
# sides at once so neither can rot in isolation.


def _expected_attr(tool_name: str) -> str:
    """Mirror of ``server._build_handlers``'s name-resolution rule."""
    return _HANDLER_OVERRIDES.get(tool_name, tool_name.removeprefix("memory_"))


def test_local_service_implements_every_tool():
    """``MemVaultService`` (in-process) must have a callable for each tool."""
    missing = []
    for tool in _TOOLS:
        attr = _expected_attr(tool.name)
        fn = getattr(MemVaultService, attr, None)
        if fn is None or not callable(fn):
            missing.append(f"{tool.name} → MemVaultService.{attr}")
    assert not missing, "Tools without a MemVaultService handler:\n  - " + "\n  - ".join(missing)


def test_remote_service_implements_every_tool():
    """``RemoteMemVaultService`` (HTTP) must have a callable for each tool.

    Same contract as ``test_local_service_implements_every_tool`` but for the
    remote path. This is the test that would have caught commit be15062 →
    dd26525 before it shipped.
    """
    svc = RemoteMemVaultService("http://test")
    missing = []
    for tool in _TOOLS:
        attr = _expected_attr(tool.name)
        fn = getattr(svc, attr, None)
        if fn is None or not callable(fn):
            missing.append(f"{tool.name} → RemoteMemVaultService.{attr}")
    assert not missing, "Tools without a RemoteMemVaultService handler:\n  - " + "\n  - ".join(
        missing
    )


def test_no_orphan_methods_in_local_service():
    """Every public ``async def <verb>`` in ``MemVaultService`` should map to
    a tool in ``_TOOLS`` (or be in an explicit allowlist of helpers).

    This catches the OPPOSITE drift: someone adds a service method but forgets
    to declare its tool — no MCP client ever sees it.
    """
    # Allowlist: methods that are intentionally NOT exposed as MCP tools
    # (helpers, internal computation, lifecycle).
    NOT_A_TOOL = {
        # Lifecycle / private helpers
        "_to_thread",
        "_call_llm_for_synthesis",
        # Synonyms / Python builtins
        "list_",  # exposed as memory_list via _HANDLER_OVERRIDES
    }
    declared_attrs = {_expected_attr(t.name) for t in _TOOLS}
    orphans = []
    for name in dir(MemVaultService):
        if name.startswith("_"):
            continue
        if name in NOT_A_TOOL or name in declared_attrs:
            continue
        attr = getattr(MemVaultService, name)
        # Only flag callables that look like tool handlers (async def).
        if callable(attr) and not isinstance(attr, type):
            # ``async def`` shows as coroutine_function on the class.
            import inspect

            if inspect.iscoroutinefunction(attr):
                orphans.append(name)
    assert not orphans, (
        "MemVaultService methods without a matching _TOOLS entry:\n  - "
        + "\n  - ".join(orphans)
        + "\n\nEither declare them in _TOOLS, prefix with `_`, or add to NOT_A_TOOL."
    )


# ---------------------------------------------------------------------------
# Contract 2: every env-var maps to a real Config field
# ---------------------------------------------------------------------------


def test_every_env_var_maps_to_real_config_field():
    """Every entry in ``ENV_TO_CONFIG_FIELD`` must reference a defined Config field.

    Catches typos: adding ``MEM_VAULT_PROJECT_DEFAULT → project_default_xx``
    would silently no-op until someone tried to set it.
    """
    config_fields = set(Config.model_fields.keys())
    invalid = [
        f"{env} → {field!r}"
        for env, field in ENV_TO_CONFIG_FIELD.items()
        if field not in config_fields
    ]
    assert not invalid, (
        "ENV_TO_CONFIG_FIELD entries pointing at non-existent Config fields:\n  - "
        + "\n  - ".join(invalid)
    )


def test_env_var_naming_is_consistent():
    """Every key in ``ENV_TO_CONFIG_FIELD`` must follow the ``MEM_VAULT_*`` prefix.

    Cheap convention guard. Unprefixed env vars silently shadow shell builtins
    or other apps' config (e.g. ``USER_ID`` would collide with PostgreSQL
    client libs).
    """
    bad = [k for k in ENV_TO_CONFIG_FIELD if not k.startswith("MEM_VAULT_")]
    assert not bad, f"env vars without MEM_VAULT_ prefix: {bad}"


# ---------------------------------------------------------------------------
# Contract 3: Pydantic request models match the service's args.get(...) keys
# ---------------------------------------------------------------------------
#
# Looser than the previous two: we don't grok the AST of every method, but we
# at least verify that the key fields the request model declares (``content``,
# ``id``, ``query``, etc.) are referenced in the service method's source. A
# field that's accepted at the HTTP layer and silently dropped by the service
# is a contract violation worth surfacing.


def _service_method_source(method_name: str) -> str:
    import inspect

    fn = getattr(MemVaultService, method_name)
    return inspect.getsource(fn)


def test_memory_create_fields_are_consumed_by_save():
    """Every field in ``MemoryCreate`` must appear in ``MemVaultService.save``."""
    from mem_vault.ui.server import MemoryCreate

    source = _service_method_source("save")
    missing = [
        name
        for name in MemoryCreate.model_fields
        if f'"{name}"' not in source and f"'{name}'" not in source
    ]
    assert not missing, f"MemoryCreate fields not referenced in MemVaultService.save: {missing}"


def test_memory_update_fields_are_consumed_by_update():
    """Every field in ``MemoryUpdate`` must appear in ``MemVaultService.update``."""
    from mem_vault.ui.server import MemoryUpdate

    source = _service_method_source("update")
    missing = [
        name
        for name in MemoryUpdate.model_fields
        if f'"{name}"' not in source and f"'{name}'" not in source
    ]
    assert not missing, f"MemoryUpdate fields not referenced in MemVaultService.update: {missing}"


def test_synthesize_request_fields_are_consumed():
    """``SynthesizeRequest`` fields must appear in ``MemVaultService.synthesize``."""
    from mem_vault.ui.server import SynthesizeRequest

    source = _service_method_source("synthesize")
    missing = [
        name
        for name in SynthesizeRequest.model_fields
        if f'"{name}"' not in source and f"'{name}'" not in source
    ]
    assert not missing, f"SynthesizeRequest fields not referenced in synthesize: {missing}"


def test_derive_metadata_request_fields_are_consumed():
    from mem_vault.ui.server import DeriveMetadataRequest

    source = _service_method_source("derive_metadata")
    missing = [
        name
        for name in DeriveMetadataRequest.model_fields
        if f'"{name}"' not in source and f"'{name}'" not in source
    ]
    assert not missing, f"DeriveMetadataRequest fields not referenced: {missing}"


def test_feedback_request_fields_are_consumed():
    from mem_vault.ui.server import FeedbackRequest

    source = _service_method_source("feedback")
    missing = [
        name
        for name in FeedbackRequest.model_fields
        if f'"{name}"' not in source and f"'{name}'" not in source
    ]
    assert not missing, f"FeedbackRequest fields not referenced: {missing}"


# ---------------------------------------------------------------------------
# Contract 4: every tool's ``required`` schema arg is referenced in the method
# ---------------------------------------------------------------------------


def test_tool_required_args_are_consumed_by_their_method():
    """Every ``required`` arg in a tool's ``inputSchema`` must show up in the
    matching method's source code (as ``args["X"]`` or ``args.get("X")``)."""
    failures = []
    for tool in _TOOLS:
        required = tool.inputSchema.get("required", [])
        if not required:
            continue
        attr = _expected_attr(tool.name)
        try:
            source = _service_method_source(attr)
        except (OSError, TypeError):
            # Method doesn't exist as a source-introspectable function
            # (e.g. a dynamic attr). The handler-symmetry test catches that
            # case separately.
            continue
        for name in required:
            if f'"{name}"' not in source and f"'{name}'" not in source:
                failures.append(f"{tool.name}.{name} not consumed by {attr}")
    assert not failures, "Required args not consumed:\n  - " + "\n  - ".join(failures)
