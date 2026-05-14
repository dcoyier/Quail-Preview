# Quail v0.7 Setup

This README only covers Quail-specific setup. For Pi-inherited CLI and model-provider behavior, see the package docs in `Quail/docs/providers.md` and `Quail/docs/models.md`, or the upstream Pi README: https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent.

## Default Quail Embeddings

Quail v0.7 is configured to use OpenRouter embeddings by default:

- Provider: OpenRouter
- Model: `qwen/qwen3-embedding-8b`
- Default dataset embedding batch size: `256`
- Default OpenRouter embedding concurrency: `20` requests in flight

Set an OpenRouter key before processing datasets or running DSL queries that use `embeddings similarity`:

```sh
export OPENROUTER_API_KEY="sk-or-v1-..."
```

You can also copy the local example env file:

```sh
cp openrouter.env.example openrouter.env
```

Then edit `openrouter.env` and either source it yourself or use the proxy script below, which sources it automatically.
The bundled `./Quail/hatch` launcher also sources `openrouter.env` from this directory automatically. If you use a PATH-installed `hatch`, launch it from this directory and Quail will load this local `openrouter.env` before embedding.

From this directory, run Quail through the bundled hatch launcher:

```sh
./Quail/hatch dataset list
./Quail/hatch dataset process --name "Dataset Name" --input "/absolute/path/to/data.csv"
```

The `--model`, `--batch-size`, and `--embedding-concurrency` flags still override the defaults:

```sh
./Quail/hatch dataset process --name "Dataset Name" --input "/absolute/path/to/data.csv" --batch-size 128 --embedding-concurrency 2
```

For persistent local defaults, set `QUAIL_EMBEDDING_BATCH_SIZE` and `QUAIL_EMBEDDING_CONCURRENCY` in `openrouter.env`.
Quail also retries transient empty, malformed, rate-limited, or server-error embedding responses; tune this with `QUAIL_EMBEDDING_MAX_RETRIES` and `QUAIL_EMBEDDING_RETRY_BASE_MS`.

## Optional Ollama-Compatible Proxy

Quail no longer needs Ollama for embeddings. If a workflow expects an Ollama `/api/embed` URL, start the local OpenRouter-backed proxy:

```sh
./Quail/scripts/start-openrouter-embed-proxy.sh
```

In another shell, point Quail's Ollama-compatible embedding backend at the proxy:

```sh
export QUAIL_EMBEDDING_PROVIDER=ollama
export QUAIL_OLLAMA_EMBED_URL=http://127.0.0.1:11435/api/embed
export QUAIL_EMBEDDING_MODEL=qwen/qwen3-embedding-8b
./Quail/hatch dataset process --name "Dataset Name" --input "/absolute/path/to/data.csv"
```

The proxy defaults to OpenRouter plus `qwen/qwen3-embedding-8b`; `openrouter.env.example` shows the local knobs.

## Existing Processed Datasets

BM25, field retrieval, counts, tags, and contains search keep working on existing datasets. Semantic embedding search must use the same embedding family and dimensionality as the processed dataset. If a dataset was processed with `embeddinggemma:latest`, reprocess it to use `qwen/qwen3-embedding-8b` before relying on `embeddings similarity`.
