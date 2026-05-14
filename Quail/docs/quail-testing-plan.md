# Quail Testing Plan

Quail inherits a large Pi test suite for terminal, session, provider, package, and SDK behavior. That upstream suite is useful as a platform layer, but it should not be the default Quail gate in its raw form: many inherited files assert Pi-specific prompt text, `.pi` project paths, or upstream extension/package UI assumptions that are no longer Quail behavior. Quail's product risk now lives in corpus processing, DSL execution, analysis state, evidence reporting, embedding/BM25 indexes, and long-running executor performance.

## Test Layers

1. **Inherited Pi Compatibility**
   - Keep a curated inherited suite for CLI parsing, sessions, RPC, provider/model routing, package handling, and terminal behavior.
   - Treat failures here as regressions in the host application shell.
   - Do not include inherited tests that assert Pi-only prompt text, `.pi` project resource paths, stale upstream default model names, or upstream extension behavior that Quail intentionally replaces.
   - Settings, package-command, and resource-loader tests that still matter have been rewritten to use `quail` command text and `.quail` project paths, then added back to the inherited gate.

2. **Quail App Boundary**
   - Cover the active app adapter, default tool selection, Quail CLI command registration, Quail-specific slash commands, processing-thread environment, prompt override, and post-assistant DSL follow-up hook.
   - Verify the shared Pi shell asks the app adapter for behavior instead of checking Quail directly.
   - Keep these tests fast and deterministic; they should not load real corpora or call model/embedding providers.
   - The default system prompt test now asserts Quail's DSL prompt and custom-prompt bypass behavior rather than Pi's generic coding-agent prompt.

3. **Quail Dataset Processing**
   - Cover CSV, TSV, JSON, JSONL, and text ingestion.
   - Verify text-field inference, field-type inference/overrides, metadata preservation, stable IDs, root manifests, removal, overwrite behavior, and malformed input errors.
   - Verify BM25 and embedding artifact structure, including binary `embeddings.f32` loading.

4. **Quail DSL Semantics**
   - Maintain golden tests for every public DSL construct: `retrieve`, `get`, `count`, `count_by`, `group`, `temp`, `group_expr`, `tag`, `untag`, loops, conditionals, variables, indexing, slices, field filters, tag filters, BM25, embeddings, and distributions.
   - For every optimization, compare optimized output against a simple reference path on small deterministic corpora.
   - Keep syntax/error tests for parse/runtime feedback because model-facing repairability is part of the product.

5. **Runtime Performance and Cache Correctness**
   - Track score-vector cache hits/misses, threshold-mask reuse, text-filter reuse, field-comparison reuse, and query-embedding reuse.
   - Use synthetic threshold-sweep tests modeled on ASRS-style workloads.
   - Include cold/warm executor tests: first call, repeated call, same query with new threshold, new query, cache clear, and process restart.
   - Assert identical counts/IDs/distributions before and after cache reuse.

6. **Large-Corpus Smoke Benchmarks**
   - Add opt-in tests gated by environment variables, not default CI, for local large datasets such as ASRS or BRIGHT.
   - Measure wall time, rows scored, output equality, memory use, and executor cache stats.
   - Store benchmark fixtures/configs separately from generated run output.

7. **Executor and RPC**
   - Test `/health`, `/status`, `/cache/clear`, request validation, invalid JSON, oversized bodies, queued requests, and warm-cache behavior.
   - Verify `QUAIL_DSL_EXECUTOR_URL` routes execution remotely and `QUAIL_DSL_EXECUTOR_DISABLE=1` prevents recursive executor calls.
   - The executor now returns structured JSON errors with stable `code` fields for bad JSON, invalid payloads, oversized request bodies, wrong methods, and missing routes.
   - Executor close restores the prior `QUAIL_DSL_EXECUTOR_DISABLE` environment value so tests and embedded callers do not inherit stale executor state.
   - Add graceful shutdown/cancellation tests when cancellation support is added.

8. **State, Branching, and Auditability**
   - Verify branch-aware analysis state, saved groups, tags, variables, compaction survival, and fork behavior.
   - Confirm only printed output enters model-visible evidence while state mutations persist.

## Relationship to the Pi Suite

The old Pi suite should become the lower platform layer. Quail should add its own named suite on top:

```bash
npm test
npm run test:quail
npm run test:pi-inherited
```

`npm test` now runs the Quail product suite and the curated Pi-inherited platform suite. `npm run test:quail` is the fast product-specific suite for app/workspace/DSL behavior. `npm run test:pi-inherited` runs the platform compatibility layer selected in `vitest.pi-inherited.config.ts`.

The current default suite intentionally excludes old Pi files that still need Quail-specific rewrites before they should gate releases: broad upstream package-manager internals, upstream extension discovery/runner assumptions, stdout help plumbing that depends on the old tsx-based test harness, and flaky low-level watcher coverage. Quail-native replacements now cover settings, resource loading, package command paths, system prompt behavior, app behavior, workspace resolution, DSL semantics, and executor HTTP behavior.

## Near-Term Test Backlog

- Split the current large `quail-dsl.test.ts` into focused files: dataset processing, DSL semantics, runtime caches, executor, and analysis-state integration.
- Add a small reference evaluator for group expressions so optimized bitset paths can be compared against a straightforward `Set<string>` implementation.
- Expand `quail-app.test.ts` into adapter tests for CLI routing, processing-thread launch arguments, and active-dataset prompt context.
- Add executor integration tests using an ephemeral port.
- Add deterministic embedding fixtures that do not call Ollama.
- Rewrite the remaining broad package-manager internals around `.quail` if Quail keeps package installation as a first-class product surface.
- Decide whether the upstream extension system is supported, deprecated, or hidden in Quail; then add tests for that explicit policy instead of re-enabling old Pi extension assumptions wholesale.
- Add opt-in ASRS/BRIGHT benchmark commands with stable JSON summaries.
- Add memory ceiling tests for cache eviction once cache byte accounting is implemented.
