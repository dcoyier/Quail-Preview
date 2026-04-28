# Quail v0.7 Specification

## Summary

Quail v0.7 is a terminal-first qualitative analysis harness forked from Pi. It keeps Pi's agent runtime, model/provider integrations, session system, TUI, extension hooks, and SDK shape, but repurposes the default experience from a general coding assistant into a corpus analysis assistant for qualitative research.

The main user experience is a conversation with an LLM that can query processed text datasets through a dedicated `quail` tool. Instead of giving the model broad filesystem tools by default, Quail activates a constrained qualitative-analysis tool that can retrieve evidence, count subsets, inspect metadata, create temporary or saved groups, and apply session-scoped tags.

In this workspace, the source package lives in `Quail/`. Runtime research data lives under the root-level `workspace/` directory.

## Product Intent

Quail is designed for people who want to ask interpretive questions of a text corpus while still preserving a trail of evidence. It should make it natural to:

- Load a corpus of survey answers, interviews, speeches, documents, or similar text entries.
- Ask high-level qualitative questions in normal language.
- Have the assistant inspect the dataset using repeatable, code-like operations rather than vague context-only guessing.
- Back claims with quotes, counts, score distributions, and metadata breakdowns.
- Preserve analytic state across a research session, including saved groups, ad hoc tags, and variables.

The intended user does not need to write the Quail DSL directly in ordinary use. The assistant is prompted to translate the user's question into `quail` tool calls and then summarize the result in audience-readable language.

## Non-Goals

Quail v0.7 is not a full graphical QDA application. It does not try to replace tools with visual coding panes, drag-and-drop codebooks, multi-user review workflows, or publication-ready report builders.

Quail is also not a database server. Datasets are local files in a workspace directory, loaded by the CLI/runtime as needed.

Finally, Quail does not make the LLM's interpretation authoritative by itself. The product direction is evidence-centered: the assistant should use retrieval, counts, metadata, and quotes to support any interpretive answer.

## Relationship To Pi

Quail keeps much of Pi's infrastructure:

- Terminal interactive mode, print/text mode, JSON mode, RPC mode, and SDK entry points.
- Provider and model support, including API-key and OAuth-backed providers.
- Session persistence, session branching, tree navigation, compaction, export, and resume flows.
- Extensions, skills, prompt templates, themes, custom commands, and package management.
- The built-in TUI shell, editor, footer, keybindings, and message rendering.

Quail changes the product posture and default tool surface:

- Package name: `quail`.
- Version: `0.7.0`.
- Description: "Qualitative analysis harness forked from Pi".
- Binaries: `quail` and `hatch`, both pointing at the same CLI entrypoint.
- Config directory: `.quail` instead of `.pi`.
- Default active tool in normal Quail sessions: `quail`.
- Default active tools in `/process` sessions: `read,bash,edit,write`, because processing a dataset is a temporary operational task.

Some upstream Pi documentation remains in the tree. The Quail-specific behavior is concentrated in `Quail/src/quail/` plus integration points in the CLI, SDK, session runtime, system prompt builder, and interactive mode.

## Core User Workflow

1. Start Quail with `quail` or `hatch`.
2. Process or inspect datasets through `/process`, or directly through dataset CLI commands.
3. Activate a processed dataset in the main chat by typing `@"Dataset Name"`.
4. Ask research questions in natural language.
5. The assistant calls the `quail` tool with the active dataset name and DSL code.
6. Quail executes the DSL against local processed indexes and appends the result to the session.
7. The assistant answers using human-readable conclusions backed by evidence.

Dataset activation is intentionally textual and visible in the conversation. Quail scans user messages for `@"Dataset Name"` mentions, validates them against `workspace/datasets/manifest.json`, and rebuilds the system prompt with the active dataset list before the next model call. The interactive editor also adds autocomplete for dataset names after `@`.

## Dataset Processing

Datasets are managed by the dataset CLI:

```bash
hatch dataset list
hatch dataset process --name <unique name> --input <file> [--tag field=value]
hatch dataset process --name <unique name> --text "entry one\nentry two"
hatch dataset remove <name> --yes
```

Supported input formats are:

- `txt`
- `csv`
- `tsv`
- `json`
- `jsonl`
- `auto`, based on file extension

For tabular or JSON data, Quail chooses a text field from preferred names such as `text`, `response`, `content`, `answer`, `comment`, and `body`, or from `--text-column` when supplied. Other scalar fields are preserved as metadata tags. Global tags can be added to every entry with repeated `--tag field=value` arguments.

Processing produces:

- Stable entry IDs based on dataset slug and ordinal.
- Normalized text for exact `contains` search.
- A BM25 index for lexical relevance.
- Optional embedding vectors through Ollama.
- A dataset manifest and root dataset registry.

The default embedding model is `embeddinggemma:latest`, called through Ollama at `http://127.0.0.1:11434/api/embed`. The default batch size is `64`. `--skip-embeddings` exists for repair and debug workflows, but semantic retrieval requires embeddings.

Processed datasets live under:

```text
workspace/datasets/<dataset-slug>/
```

Each dataset directory contains:

- `manifest.json`
- `entries.jsonl`
- `bm25.json`
- `embeddings.json`
- `embeddings.f32`

The root dataset registry is:

```text
workspace/datasets/manifest.json
```

The current workspace includes a processed dataset named `US Party Platforms` with 39,809 entries and metadata fields `party`, `words`, and `year`.

## `/process` Thread

The `/process` command opens a temporary Quail processing thread. This side thread uses a specialized system prompt and the normal coding tools so the assistant can stage pasted text, run dataset commands, inspect file paths, and remove datasets after confirmation.

Important behavior:

- The processing thread is launched with `--no-session`, so its conversation is not kept in the main research thread.
- It disables extensions, skills, prompt templates, and context files to stay focused.
- It receives `read,bash,edit,write` tools, unlike the main Quail thread.
- It sets `QUAIL_PROCESS_THREAD=1`, which changes the header and onboarding copy.
- It returns to the main Quail session when closed.

The processing prompt requires the assistant to confirm the dataset source, metadata treatment, unique dataset name, and embedding/BM25 processing plan before running a destructive or expensive command.

## The `quail` Tool

The main Quail session registers a custom tool named `quail`. Its input schema is:

```typescript
{
  datasets: string[];
  code: string;
}
```

The model is instructed to pass dataset names separately and pass only DSL statements in `code`. It should not write raw `$ ... $` call blocks in assistant text during normal tool use.

The tool executes sequentially and returns:

- Full textual output for the model.
- TUI-rendered previews for the user.
- Details including datasets, block count, error status, preview line count, and output line count.

After execution, Quail appends updated analysis state into the session as a custom entry. This is how saved groups, tags, and variables persist across turns and across session branches.

## Quail DSL

The Quail DSL is a small, line-oriented analysis language. It exists so the model's research steps are explicit, debuggable, and repeatable.

Core commands:

- `retrieve(top|middle|bottom N in (BM25: "query") of <group_expression>)`
- `retrieve(top|middle|bottom N in (embeddings: "query") of <group_expression>)`
- `get(<id>)`
- `get(groups)`
- `get(tag_fields)`
- `get(["field"])`
- `get(["field1", "field2"])`
- `get((<filter>) distribution of (<group_expression>))`
- `count(<group_expression>)`
- `count_by(["field1", "field2"] of <group_expression>)`
- `group(<spec>)`
- `temp(<spec>)`
- `group_expr(<group_expression>)`
- `tag(<id> with <field> set to <tag>)`
- `untag(<field> from <id>)`

Group specs can combine:

- `BM25: "text" > threshold`
- `embeddings: "text" > threshold`
- `contains: "text"`
- `contains_word: "word"`
- `tags: ["field": "value"]`
- `include: [id, id]`
- `exclude: [id, id]`

Group expressions support:

- `all`
- saved group IDs such as `G1`
- `temp(...)`
- `group(...)`
- boolean `and`, `or`, and `not`
- parenthesized combinations
- variables created through `group_expr(...)`

The language also supports variables, `for` loops, `if`/`else`, comparisons, `in`/`not in`, `print(...)`, indexing, property access, and basic numeric/string arithmetic. It intentionally rejects overly broad metadata dumps such as `get(tags)` and instead encourages `get(tag_fields)` plus targeted field lookup.

## Analysis State

Quail analysis state is versioned and session-scoped:

```typescript
{
  version: 1,
  nextGroupNumber: number,
  groups: Record<string, QuailGroupState>,
  tagsByEntry: Record<string, Record<string, string>>,
  variables: Record<string, unknown>
}
```

Saved groups include an ID, source dataset names, original spec string, matching entry IDs, and creation timestamp. Tags are layered on top of the immutable processed dataset entries, so research coding can evolve during a session without rewriting the dataset files.

When the user navigates or branches a session, Quail recovers the most recent `quail.analysis_state` custom entry from the active branch. This makes the analytic state branch-aware.

## Answering Contract

The Quail main system prompt tells the assistant to:

- Be thorough.
- Ask clarifying questions when the dataset or task is unclear.
- Use the `quail` tool for activated datasets.
- Avoid exposing internal terms such as "groups" or "entries" in final answers.
- Write for an ordinary audience.
- Support claims with evidence such as quotes, counts, or statistics.
- Avoid overcalling the tool because all tool results enter the context.

This creates a two-layer interaction: the model uses DSL operations internally, then translates those operations into a research answer the user can understand.

## Modes And Interfaces

Quail can be used through:

- Interactive terminal mode.
- Non-interactive print mode with `-p` or `--print`.
- JSON mode.
- RPC mode for process integration.
- SDK APIs inherited from Pi.

The interactive Quail header is branded with Quail-specific ASCII art and onboarding. It tells users to activate a processed dataset with `@"Dataset Name"` or use `/process` to add one.

## Technical Architecture

Important Quail-specific modules:

- `Quail/src/quail/dataset-store.ts` parses, indexes, embeds, writes, lists, loads, and removes datasets.
- `Quail/src/quail/dataset-cli.ts` implements `hatch dataset ...` commands.
- `Quail/src/quail/dsl.ts` parses and executes the Quail DSL.
- `Quail/src/quail/query-tool.ts` registers and renders the `quail` tool.
- `Quail/src/quail/analysis-state.ts` defines branch-aware analysis state.
- `Quail/src/quail/analysis-runner.ts` supports legacy `$` call block execution from assistant text.
- `Quail/src/quail/prompts.ts` builds the main and processing prompts.
- `Quail/src/quail/paths.ts` centralizes workspace paths.

Important integration points:

- `Quail/src/core/sdk.ts` sets the default active Quail tool and registers the custom tool.
- `Quail/src/core/system-prompt.ts` swaps in the Quail main prompt when `APP_NAME === "quail"`.
- `Quail/src/core/agent-session.ts` rebuilds active dataset context before prompts and persists analysis state after tool execution.
- `Quail/src/modes/interactive/interactive-mode.ts` adds Quail header behavior, dataset autocomplete, and `/process`.
- `Quail/src/cli/args.ts` changes CLI help text for Quail.
- `Quail/src/config.ts` reads `package.json` `piConfig` to set app name and config directory.

## Current Limitations And Risks

- Quail depends on a local Ollama embedding endpoint for semantic search and dataset processing unless embeddings are skipped.
- Upstream Pi docs are still present, so users may see Pi-centric language outside Quail-specific surfaces.
- Quail-specific release notes are not bundled yet; the app suppresses Pi's upstream changelog in Quail mode.
- The DSL is custom and powerful but not a general programming language. The model should recover from parse/runtime feedback by issuing corrected calls.
- The local file dataset store is simple and inspectable, but very large corpora may need future performance work around loading, indexing, and incremental updates.
- Tags and groups are session state rather than dataset mutations. This is useful for exploratory analysis, but project-level sharing of codebooks or annotations would need additional design.

## Definition Of Quail v0.7

Quail v0.7 is best understood as a local, terminal-based qualitative research companion: a Pi-derived agent shell whose default behavior has been narrowed around evidence retrieval and corpus analysis. Its distinctive pieces are local dataset processing, BM25 plus embedding search, a constrained analysis DSL, session-persistent analytic state, dataset activation through `@"Dataset Name"`, and a dedicated `quail` tool that forces the assistant to inspect the corpus before answering.

