"""mem-vault — Local MCP server with infinite memory backed by an Obsidian vault.

Components:
- `config`: load + validate user config (vault path, models, vector store path).
- `storage`: read/write `.md` memory files with YAML frontmatter inside the vault.
- `extractor`: optional `mem0`-powered fact extraction + dedup against existing memories.
- `index`: vector search via `mem0` (Ollama embedder + Qdrant embedded store).
- `server`: MCP stdio server exposing six memory tools.

Public surface is intentionally tiny — most consumers just spawn the MCP server
via `mem-vault-mcp` and talk to it through stdio.
"""

from mem_vault.config import Config, load_config
from mem_vault.storage import Memory, VaultStorage

__all__ = ["Config", "Memory", "VaultStorage", "load_config"]
__version__ = "0.2.0"
