import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { defaultEmbeddingBatchSize, defaultEmbeddingConcurrency, defaultEmbeddingModel, listDatasets, loadLocalEmbeddingEnv, } from "./dataset-store.js";
import { getQuailDatasetsDir, getQuailStagingDir, getQuailWorkspaceDir } from "./paths.js";
const ACTIVATION_RE = /@"([^"]+)"/g;
function normalizeMentionText(value) {
    return value
        .normalize("NFKC")
        .toLowerCase()
        .replace(/[_-]+/g, " ")
        .replace(/\s+/g, " ")
        .trim();
}
function uniqueAliases(values) {
    const seen = new Set();
    const aliases = [];
    for (const value of values) {
        const normalized = normalizeMentionText(value);
        if (!normalized || seen.has(normalized))
            continue;
        seen.add(normalized);
        aliases.push(normalized);
    }
    return aliases;
}
export function getDatasetMentionAliases(dataset) {
    const baseValues = [dataset.name, dataset.slug];
    const aliases = [...baseValues];
    for (const value of baseValues) {
        const normalized = normalizeMentionText(value);
        aliases.push(normalized);
        const retrievalMatch = normalized.match(/^retrieval\s+(.+)$/);
        if (retrievalMatch) {
            const retrievalSubject = retrievalMatch[1].trim();
            aliases.push(`bright ${retrievalSubject}`);
            aliases.push(`bright ${retrievalSubject.replace(/\bmax\d+k\b/g, "").trim()}`);
        }
    }
    return uniqueAliases(aliases);
}
function escapeRegExp(value) {
    return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
function mentionAliasPattern(alias) {
    const source = normalizeMentionText(alias)
        .split(" ")
        .filter(Boolean)
        .map(escapeRegExp)
        .join("[\\s_-]+");
    return new RegExp(`^\\s*${source}(?=$|[\\s.,;:!?\\)\\]\\}])`, "i");
}
function buildDatasetActivationIndex(datasets) {
    return datasets
        .flatMap((dataset) => getDatasetMentionAliases(dataset).map((alias) => ({
        name: dataset.name,
        alias,
        pattern: mentionAliasPattern(alias),
    })))
        .sort((a, b) => b.alias.length - a.alias.length);
}
function resolveDatasetMentionName(rawMention, datasets) {
    const normalized = normalizeMentionText(rawMention);
    if (!normalized)
        return undefined;
    return buildDatasetActivationIndex(datasets).find((candidate) => candidate.alias === normalized)?.name;
}
function getKnownDatasetMentions(text, datasets) {
    const activated = new Set();
    if (datasets.length === 0)
        return [];
    for (const match of text.matchAll(/@\s*"([^"]+)"/g)) {
        const resolved = resolveDatasetMentionName(match[1], datasets);
        if (resolved)
            activated.add(resolved);
    }
    const candidates = buildDatasetActivationIndex(datasets);
    for (const match of text.matchAll(/@/g)) {
        const atIndex = match.index ?? -1;
        if (atIndex < 0)
            continue;
        const previous = atIndex > 0 ? text[atIndex - 1] : "";
        if (/[A-Za-z0-9._%+-]/.test(previous))
            continue;
        const tail = text.slice(atIndex + 1);
        const candidate = candidates.find((item) => item.pattern.test(tail));
        if (candidate)
            activated.add(candidate.name);
    }
    return [...activated];
}
function getMessageText(message) {
    if (message.role !== "user")
        return "";
    const content = message.content;
    if (typeof content === "string")
        return content;
    return content
        .filter((part) => part.type === "text" && typeof part.text === "string")
        .map((part) => part.text)
        .join("\n");
}
export function getActivatedDatasetNamesFromMessages(messages, datasets = []) {
    const active = new Set();
    for (const message of messages) {
        const text = getMessageText(message);
        if (!text)
            continue;
        if (datasets.length > 0) {
            for (const name of getKnownDatasetMentions(text, datasets))
                active.add(name);
        }
        else {
            for (const match of text.matchAll(ACTIVATION_RE)) {
                active.add(match[1]);
            }
        }
    }
    return [...active];
}
export function getActiveDatasetsForPrompt(cwd, messages) {
    const datasets = listDatasets(cwd);
    const activated = getActivatedDatasetNamesFromMessages(messages, datasets);
    const registry = new Map(datasets.map((dataset) => [dataset.name, dataset]));
    return activated.flatMap((name) => {
        const dataset = registry.get(name);
        return dataset ? [{ name: dataset.name, entries: dataset.entryCount }] : [];
    });
}
function formatActiveDatasets(activeDatasets) {
    if (!activeDatasets || activeDatasets.length === 0) {
        return "- (none)";
    }
    return activeDatasets
        .map((dataset) => `- "${dataset.name}", ${dataset.entries} entries`)
        .join("\n");
}
// The main Quail prompt lives in field-based-system-prompt.md so the shipped prompt matches the editable source.
function loadQuailMainSystemPromptTemplate() {
    return readFileSync(fileURLToPath(new URL("./field-based-system-prompt.md", import.meta.url)), "utf8").trimEnd();
}
export function buildQuailMainSystemPrompt(options) {
    return loadQuailMainSystemPromptTemplate().replace("{{ACTIVE_DATASETS}}", formatActiveDatasets(options.activeDatasets));
}
export function buildQuailProcessingSystemPrompt(cwd) {
    loadLocalEmbeddingEnv(cwd);
    const workspaceDir = getQuailWorkspaceDir(cwd);
    const stagingDir = getQuailStagingDir(cwd);
    const datasetsDir = getQuailDatasetsDir(cwd);
    const embeddingModel = defaultEmbeddingModel();
    const batchSize = defaultEmbeddingBatchSize();
    const embeddingConcurrency = defaultEmbeddingConcurrency();
    return `You are the Quail processing agent. You are a temporary side agent whose conversation is not kept in the main research thread. Your job is to help the user process, add, inspect, or remove field-based qualitative datasets for Quail.

Quail datasets are based on fields and tags. A record may have many fields, and each field has an entry tag value after processing. Later analysis can add or update field/tag values using the same field model.

You are a Pi coding agent with file and shell tools. Be conversational until the user has supplied everything required for a processing or removal run. Once you send the command that performs processing or removal, do not ask for or accept more messages in this processing thread; let the command output provide progress and finish with a concise status.

Workspace rules:
- The Quail repo cwd is: ${cwd}
- The active Quail workspace is: ${workspaceDir}
- You may write staging files under: ${stagingDir}
- Processed datasets live under: ${datasetsDir}
- Do not modify source files unless the user explicitly asks for Quail code changes. Processing normal datasets should use the dataset CLI.

Required before processing a dataset:
1. The dataset itself: either a file path or pasted text. If the user pasted text, write it to the active staging directory before running the CLI.
2. Global fields/tags. Ask whether any global field/tag values should be added to every response, such as source, audience, year, cohort, or project. Use --tag field=value for these global values.
3. Field/type confirmation. Run hatch dataset inspect --input "/absolute/path" before processing, including any --tag field=value global values. Show the inferred field types as a Markdown table with columns Field, Type, Embedded, Non-empty, and Sample, along with the record count. String fields are embedded and prepared for BM25/contains text search; non-string fields are preserved for exact matching and counting. Ask the user to confirm or provide type overrides.
4. A unique dataset name. Check uniqueness with hatch dataset list. If hatch is unavailable, use node dist/cli.js dataset list after npm run build.
5. Confirmation of processing procedure: default embedding backend is OpenRouter, default embedding model is ${embeddingModel}, default batch size is ${batchSize}, default embedding concurrency is ${embeddingConcurrency}, plus BM25, embeddings, and exact contains preparation for string fields.

Preferred commands:
- List datasets: hatch dataset list
- Inspect before processing: hatch dataset inspect --input "/absolute/path" --tag field=value
- Override an inferred field type during inspect or process: --field-type "Field Name=string"
- Process from a file: hatch dataset process --name "Dataset Name" --input "/absolute/path" --model ${embeddingModel} --batch-size ${batchSize} --embedding-concurrency ${embeddingConcurrency} --tag field=value
- Process pasted text: write the paste to the active staging directory, then run the same process command with --input.
- Remove a dataset: hatch dataset remove "Dataset Name" --yes

If hatch has not been installed or points to another command, run npm run build from ${cwd}, then use node dist/cli.js dataset ... from ${cwd}.

The dataset CLI prints clear progress for each processing step. If OpenRouter authentication or embedding generation fails, report the exact error and ask the user to set OPENROUTER_API_KEY or QUAIL_OPENROUTER_API_KEY. If QUAIL_EMBEDDING_PROVIDER=ollama is configured, report the endpoint error exactly.

After processing, report the dataset name, record count, inferred/overridden field types, and embedded fields.

Removal rules:
- Confirm the dataset name and show the current dataset list before removing.
- Use hatch dataset remove "Dataset Name" --yes only after confirmation.
- Report whether the dataset was removed or was not found.
`;
}
//# sourceMappingURL=prompts.js.map