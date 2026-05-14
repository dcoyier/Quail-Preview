#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
QUAIL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$QUAIL_DIR/.." && pwd)"

if [[ -f "$ROOT_DIR/openrouter.env" ]]; then
	set -a
	source "$ROOT_DIR/openrouter.env"
	set +a
elif [[ -f "$QUAIL_DIR/openrouter.env" ]]; then
	set -a
	source "$QUAIL_DIR/openrouter.env"
	set +a
fi

export QUAIL_EMBEDDING_MODEL="${QUAIL_EMBEDDING_MODEL:-qwen/qwen3-embedding-8b}"
export QUAIL_OPENROUTER_PROVIDER_ONLY="${QUAIL_OPENROUTER_PROVIDER_ONLY:-deepinfra}"
export QUAIL_OPENROUTER_EMBED_PROXY_HOST="${QUAIL_OPENROUTER_EMBED_PROXY_HOST:-127.0.0.1}"
export QUAIL_OPENROUTER_EMBED_PROXY_PORT="${QUAIL_OPENROUTER_EMBED_PROXY_PORT:-11435}"

exec node "$SCRIPT_DIR/openrouter-ollama-embed-proxy.mjs" "$@"
