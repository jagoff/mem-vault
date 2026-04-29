"""Lifecycle hook entrypoints for mem-vault.

Each module here exposes a single ``run()`` function that:
- reads the hook event JSON from stdin (best-effort, never crashes on bad input),
- does its job (search/log/etc.),
- writes the hook output JSON to stdout (or nothing for no-op hooks),
- never raises — failures are logged to stderr.

These are wired up via the ``mem-vault`` CLI (``mem-vault hook-sessionstart``,
``mem-vault hook-stop``) so they always run inside the right virtualenv.
"""
