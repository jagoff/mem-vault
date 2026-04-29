# Lifecycle hooks

These hooks are **wired through the CLI** (`mem-vault hook-sessionstart`,
`mem-vault hook-stop`) so they always run inside the same virtualenv as the
MCP server, with all dependencies available. The standalone scripts that used
to live here have been moved to `src/mem_vault/hooks/` and exposed as CLI
subcommands.

## Configure (Devin)

Append to `~/.config/devin/config.json`:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "/Users/you/.local/bin/mem-vault hook-sessionstart",
            "timeout": 15
          }
        ]
      }
    ],
    "Stop": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "/Users/you/.local/bin/mem-vault hook-stop",
            "timeout": 5
          }
        ]
      }
    ]
  }
}
```

## Configure (Claude Code)

Same JSON shape, in `~/.claude.json` or `~/.claude/settings.json`. Devin's hook
format is compatible with Claude Code's, see the
[Claude Code hooks docs](https://docs.anthropic.com/en/docs/claude-code/hooks).

## What each hook does

| Hook | Purpose | Side effects |
| --- | --- | --- |
| `mem-vault hook-sessionstart` | Reads the SessionStart event, queries the vault for `type=preference` and `type=feedback` memories, emits an `additionalContext` block so the agent sees them at session start | Stdin consumed; stdout JSON; never blocks |
| `mem-vault hook-stop` | Appends a tab-separated audit line to `~/.local/share/mem-vault/sessions.log` whenever the agent finishes its turn | One log line; never blocks |

## Failure modes (handled)

- mem-vault not installed → `command not found` → Devin logs and continues
- vault path wrong → hook prints to stderr, emits empty stdout, exits 0
- Ollama down → hook prints to stderr, emits empty stdout, exits 0
- vault empty → hook emits empty stdout, exits 0

None of these block the session. The hook is best-effort by design.

## Verify

After configuring, run `/hooks` inside Devin or Claude Code to confirm the
hooks are loaded. Tail `~/.local/share/mem-vault/sessions.log` to see Stop
hook entries land.
