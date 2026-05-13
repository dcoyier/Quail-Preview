import { request } from "undici";
import { bm25ScoreTerms, cosineSimilarity, embedTexts, fieldDocumentId, loadDatasets, } from "./dataset-store.js";
import { cloneAnalysisState } from "./analysis-state.js";
import { normalizeContainsText, tokenize } from "./text.js";
class DslRuntimeError extends Error {
    code;
    line;
    constructor(code, message, line) {
        super(message);
        this.code = code;
        this.line = line;
    }
}
class LruMap {
    maxEntries;
    values = new Map();
    constructor(maxEntries) {
        this.maxEntries = maxEntries;
    }
    get(key) {
        const value = this.values.get(key);
        if (value === undefined)
            return undefined;
        this.values.delete(key);
        this.values.set(key, value);
        return value;
    }
    set(key, value) {
        if (this.maxEntries <= 0)
            return;
        if (this.values.has(key))
            this.values.delete(key);
        this.values.set(key, value);
        while (this.values.size > this.maxEntries) {
            const firstKey = this.values.keys().next().value;
            if (firstKey === undefined)
                break;
            this.values.delete(firstKey);
        }
    }
    clear() {
        this.values.clear();
    }
    get size() {
        return this.values.size;
    }
}
const runtimeCaches = new Map();
function envPositiveInt(name, fallback) {
    const value = process.env[name]?.trim();
    if (!value)
        return fallback;
    const parsed = Number(value);
    return Number.isFinite(parsed) && parsed >= 0 ? Math.floor(parsed) : fallback;
}
function emptyCacheStats() {
    return {
        datasetContexts: 0,
        scoreVectorEntries: 0,
        thresholdIdSetEntries: 0,
        fieldComparisonEntries: 0,
        textFilterEntries: 0,
        queryEmbeddingEntries: 0,
        scoreVectorHits: 0,
        scoreVectorMisses: 0,
        thresholdIdSetHits: 0,
        thresholdIdSetMisses: 0,
        fieldComparisonHits: 0,
        fieldComparisonMisses: 0,
        textFilterHits: 0,
        textFilterMisses: 0,
        queryEmbeddingHits: 0,
        queryEmbeddingMisses: 0,
    };
}
function createRuntimeCache() {
    return {
        scoreVectors: new LruMap(envPositiveInt("QUAIL_DSL_SCORE_VECTOR_CACHE_ENTRIES", 64)),
        thresholdIdSets: new LruMap(envPositiveInt("QUAIL_DSL_THRESHOLD_CACHE_ENTRIES", 256)),
        fieldComparisonIdSets: new LruMap(envPositiveInt("QUAIL_DSL_FIELD_COMPARE_CACHE_ENTRIES", 256)),
        textFilterIdSets: new LruMap(envPositiveInt("QUAIL_DSL_TEXT_FILTER_CACHE_ENTRIES", 256)),
        queryEmbeddings: new LruMap(envPositiveInt("QUAIL_DSL_QUERY_EMBEDDING_CACHE_ENTRIES", 128)),
        stats: emptyCacheStats(),
    };
}
function getRuntimeCache(cwd, datasets) {
    const key = [
        cwd,
        ...datasets.map((dataset) => [
            dataset.manifest.slug,
            dataset.manifest.updatedAt,
            dataset.manifest.entryCount,
            dataset.manifest.embeddingModel,
            dataset.manifest.embeddingDimensions,
            dataset.entries.length,
        ].join(":")),
    ].join("\0");
    let cache = runtimeCaches.get(key);
    if (!cache) {
        cache = createRuntimeCache();
        runtimeCaches.set(key, cache);
    }
    return cache;
}
export function clearQuailDslRuntimeCaches() {
    runtimeCaches.clear();
}
export function getQuailDslRuntimeCacheStats() {
    const stats = emptyCacheStats();
    stats.datasetContexts = runtimeCaches.size;
    for (const cache of runtimeCaches.values()) {
        stats.scoreVectorEntries += cache.scoreVectors.size;
        stats.thresholdIdSetEntries += cache.thresholdIdSets.size;
        stats.fieldComparisonEntries += cache.fieldComparisonIdSets.size;
        stats.textFilterEntries += cache.textFilterIdSets.size;
        stats.queryEmbeddingEntries += cache.queryEmbeddings.size;
        stats.scoreVectorHits += cache.stats.scoreVectorHits;
        stats.scoreVectorMisses += cache.stats.scoreVectorMisses;
        stats.thresholdIdSetHits += cache.stats.thresholdIdSetHits;
        stats.thresholdIdSetMisses += cache.stats.thresholdIdSetMisses;
        stats.fieldComparisonHits += cache.stats.fieldComparisonHits;
        stats.fieldComparisonMisses += cache.stats.fieldComparisonMisses;
        stats.textFilterHits += cache.stats.textFilterHits;
        stats.textFilterMisses += cache.stats.textFilterMisses;
        stats.queryEmbeddingHits += cache.stats.queryEmbeddingHits;
        stats.queryEmbeddingMisses += cache.stats.queryEmbeddingMisses;
    }
    return stats;
}
export function extractQuailCallBlocks(text) {
    const lines = text.split(/\r?\n/);
    const blocks = [];
    let start = -1;
    for (let i = 0; i < lines.length; i++) {
        if (lines[i].trim() !== "$")
            continue;
        if (start < 0) {
            start = i;
            continue;
        }
        const bodyLines = lines.slice(start + 1, i);
        const parsed = parseCallBody(bodyLines.join("\n"));
        if (parsed)
            blocks.push({ ...parsed, raw: lines.slice(start, i + 1).join("\n") });
        start = -1;
    }
    return blocks;
}
function parseCallBody(body) {
    const lines = body.split(/\r?\n/);
    const firstNonEmpty = lines.findIndex((line) => line.trim().length > 0);
    if (firstNonEmpty < 0)
        return undefined;
    const atLine = lines[firstNonEmpty].trim();
    if (!atLine.startsWith("@"))
        return undefined;
    const datasets = [...atLine.matchAll(/"([^"]+)"/g)].map((match) => match[1]);
    const code = lines.slice(firstNonEmpty + 1).join("\n").trim();
    return { datasets, code };
}
export function formatQuailExecutionResult(result) {
    const parts = [];
    if (result.errors.length > 0) {
        parts.push("Parse/runtime feedback:");
        for (const error of result.errors) {
            const line = error.line !== undefined ? ` line ${error.line}:` : ":";
            parts.push(`- ${error.code}${line} ${error.message}`);
        }
    }
    if (result.output.trim()) {
        parts.push("Output:");
        parts.push(result.output.trim());
    }
    parts.push("Use this result to continue. If there were parse/runtime errors, correct the code call before answering.");
    return parts.join("\n");
}
function quailDslExecutorUrl() {
    if (process.env.QUAIL_DSL_EXECUTOR_DISABLE === "1")
        return undefined;
    const value = process.env.QUAIL_DSL_EXECUTOR_URL?.trim();
    return value ? value.replace(/\/+$/, "") : undefined;
}
function quailDslExecutorTimeoutMs() {
    const fallback = 12 * 60 * 60 * 1000;
    const value = process.env.QUAIL_DSL_EXECUTOR_TIMEOUT_MS?.trim();
    if (!value)
        return fallback;
    const parsed = Number(value);
    return Number.isFinite(parsed) && parsed >= 0 ? parsed : fallback;
}
function isQuailExecutionResult(value) {
    if (!value || typeof value !== "object")
        return false;
    const record = value;
    return (typeof record.blocks === "number" &&
        Array.isArray(record.errors) &&
        typeof record.output === "string" &&
        !!record.state &&
        typeof record.state === "object");
}
async function executeQuailCallBlocksRemote(baseUrl, options) {
    const timeoutMs = quailDslExecutorTimeoutMs();
    const body = JSON.stringify({
        version: 1,
        cwd: options.cwd,
        state: options.state,
        blocks: options.blocks,
    });
    const response = await request(`${baseUrl}/quail/execute`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body,
        headersTimeout: timeoutMs,
        bodyTimeout: timeoutMs,
    });
    const responseText = await response.body.text();
    const payload = JSON.parse(responseText || "{}");
    if (response.statusCode < 200 || response.statusCode >= 300 || !payload?.ok) {
        throw new Error(`Quail shared DSL executor failed (${response.statusCode}): ${payload?.error ?? responseText.slice(0, 500)}`);
    }
    if (!isQuailExecutionResult(payload.result)) {
        throw new Error("Quail shared DSL executor returned an invalid execution result");
    }
    return payload.result;
}
export async function executeQuailCallBlocks(options) {
    const remoteUrl = quailDslExecutorUrl();
    if (remoteUrl)
        return executeQuailCallBlocksRemote(remoteUrl, options);
    const state = cloneAnalysisState(options.state);
    const outputs = [];
    const errors = [];
    for (const [index, block] of options.blocks.entries()) {
        if (block.datasets.length === 0) {
            errors.push({ code: "E_MISSING_DATASET_LINE", message: "Every Quail call must include an @ line with at least one dataset name." });
            continue;
        }
        try {
            const datasets = loadDatasets(options.cwd, block.datasets);
            const entries = datasets.flatMap((dataset) => dataset.entries);
            const entriesById = new Map(entries.map((entry) => [entry.id, entry]));
            const datasetByEntryId = new Map();
            const entryIndexById = new Map();
            const entryOrdinalById = new Map();
            for (const [entryIndex, entry] of entries.entries())
                entryIndexById.set(entry.id, entryIndex);
            for (const dataset of datasets) {
                for (const [entryIndex, entry] of dataset.entries.entries()) {
                    datasetByEntryId.set(entry.id, dataset);
                    entryOrdinalById.set(entry.id, getStableOrdinal(entry, entryIndex));
                }
            }
            const ctx = {
                cwd: options.cwd,
                datasets,
                entries,
                entriesById,
                datasetByEntryId,
                entryIndexById,
                entryOrdinalById,
                state,
                outputs,
                errors,
                activeDatasetNames: block.datasets,
                tagMutations: [],
                bm25QueryTerms: new Map(),
                embeddingQueryVectors: new Map(),
                groupExpressionCache: new Map(),
                runtimeCache: getRuntimeCache(options.cwd, datasets),
            };
            await executeProgram(block.code, ctx);
            flushTagMutationSummary(ctx);
        }
        catch (error) {
            if (error instanceof DslRuntimeError)
                errors.push({ code: error.code, message: error.message, line: error.line });
            else
                errors.push({ code: "E_RUNTIME", message: error instanceof Error ? error.message : String(error) });
        }
    }
    return { state, output: outputs.join("\n"), errors, blocks: options.blocks.length };
}
function parseProgram(code) {
    const roots = [];
    const stack = [];
    for (const logicalLine of logicalLines(code)) {
        const rawLine = logicalLine.text;
        if (rawLine.trim().length === 0 || rawLine.trimStart().startsWith("#"))
            continue;
        const indent = rawLine.match(/^[ \t]*/)?.[0].replace(/\t/g, "    ").length ?? 0;
        const node = { line: logicalLine.line, indent, text: rawLine.trim(), children: [] };
        if (hasUnquotedSemicolon(node.text)) {
            throw new DslRuntimeError("E_SEMICOLON", "Write one statement per line; semicolons are not supported.", node.line);
        }
        if (node.text === "else:" || node.text === "else") {
            while (stack.length > 0 && indent < stack[stack.length - 1].indent)
                stack.pop();
            const parent = stack[stack.length - 1] ?? roots[roots.length - 1];
            if (!parent || !parent.text.startsWith("if "))
                throw new DslRuntimeError("E_PARSE_ELSE", "else must follow an if block", node.line);
            parent.elseChildren = [];
            stack.push({ ...node, children: parent.elseChildren });
            continue;
        }
        while (stack.length > 0 && indent <= stack[stack.length - 1].indent)
            stack.pop();
        if (stack.length === 0)
            roots.push(node);
        else
            stack[stack.length - 1].children.push(node);
        stack.push(node);
    }
    return roots;
}
function logicalLines(code) {
    const out = [];
    let buffer = "";
    let startLine = 1;
    let depth = 0;
    for (const [index, rawLine] of code.split(/\r?\n/).entries()) {
        const trimmed = rawLine.trim();
        if (!buffer && (trimmed.length === 0 || trimmed.startsWith("#"))) {
            out.push({ line: index + 1, text: rawLine });
            continue;
        }
        if (!buffer)
            startLine = index + 1;
        buffer = buffer ? `${buffer} ${trimmed}` : rawLine;
        depth += delimiterBalanceDelta(rawLine);
        if (depth <= 0) {
            out.push({ line: startLine, text: buffer });
            buffer = "";
            depth = 0;
        }
    }
    if (buffer)
        out.push({ line: startLine, text: buffer });
    return out;
}
function delimiterBalanceDelta(text) {
    let depth = 0;
    let quoted;
    for (let i = 0; i < text.length; i++) {
        const ch = text[i];
        if (quoted) {
            if (ch === quoted && text[i - 1] !== "\\")
                quoted = undefined;
            continue;
        }
        if (ch === '"' || ch === "'") {
            quoted = ch;
            continue;
        }
        if ("([{".includes(ch))
            depth++;
        else if (")]}".includes(ch))
            depth--;
    }
    return depth;
}
async function executeProgram(code, ctx) {
    const nodes = parseProgram(code);
    for (const node of nodes)
        await executeNode(node, ctx);
}
async function executeNode(node, ctx) {
    const text = stripTrailingColon(node.text);
    try {
        if (text.startsWith("for ")) {
            const match = text.match(/^for\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+in\s+(.+)$/);
            if (!match)
                throw new DslRuntimeError("E_PARSE_FOR", "Expected for <name> in <expression>", node.line);
            const values = await evaluateExpression(match[2], ctx, node.line);
            if (!Array.isArray(values))
                throw new DslRuntimeError("E_FOR_NON_LIST", "for loop expression must return a list", node.line);
            for (const value of values) {
                ctx.state.variables[match[1]] = value;
                clearGroupExpressionCache(ctx);
                for (const child of node.children)
                    await executeNode(child, ctx);
            }
            return;
        }
        if (text.startsWith("if ")) {
            const condition = text.slice(3).trim();
            const value = await evaluateCondition(condition, ctx, node.line);
            const branch = value ? node.children : (node.elseChildren ?? []);
            for (const child of branch)
                await executeNode(child, ctx);
            return;
        }
        const varMatch = text.match(/^var\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)$/);
        if (varMatch) {
            ctx.state.variables[varMatch[1]] = await evaluateExpression(varMatch[2], ctx, node.line);
            clearGroupExpressionCache(ctx);
            return;
        }
        const assignMatch = text.match(/^([a-zA-Z_][a-zA-Z0-9_]*)\s*(\+=|-=|=)\s*(.+)$/);
        if (assignMatch) {
            const current = ctx.state.variables[assignMatch[1]];
            const value = await evaluateExpression(assignMatch[3], ctx, node.line);
            if (assignMatch[2] === "=")
                ctx.state.variables[assignMatch[1]] = value;
            else if (assignMatch[2] === "+=")
                ctx.state.variables[assignMatch[1]] = addValues(current, value);
            else
                ctx.state.variables[assignMatch[1]] = subtractValues(current, value);
            clearGroupExpressionCache(ctx);
            return;
        }
        if (text.startsWith("tag(")) {
            await executeTag(text, ctx, node.line);
            return;
        }
        if (text.startsWith("untag(")) {
            await executeUntag(text, ctx, node.line);
            return;
        }
        await evaluateExpression(text, ctx, node.line);
    }
    catch (error) {
        if (error instanceof DslRuntimeError)
            ctx.errors.push({ code: error.code, message: error.message, line: error.line ?? node.line });
        else
            ctx.errors.push({ code: "E_RUNTIME", message: error instanceof Error ? error.message : String(error), line: node.line });
    }
}
function stripTrailingColon(text) {
    return text.endsWith(":") ? text.slice(0, -1).trimEnd() : text;
}
function hasUnquotedSemicolon(text) {
    let quoted;
    for (let i = 0; i < text.length; i++) {
        const ch = text[i];
        if (quoted) {
            if (ch === quoted && text[i - 1] !== "\\")
                quoted = undefined;
            continue;
        }
        if (ch === '"' || ch === "'") {
            quoted = ch;
            continue;
        }
        if (ch === ";")
            return true;
    }
    return false;
}
function pushOutput(ctx, text) {
    flushTagMutationSummary(ctx);
    ctx.outputs.push(text);
}
function flushTagMutationSummary(ctx) {
    if (ctx.tagMutations.length === 0)
        return;
    const tags = ctx.tagMutations.filter((mutation) => mutation.type === "tag");
    const untags = ctx.tagMutations.filter((mutation) => mutation.type === "untag");
    if (tags.length > 0)
        ctx.outputs.push(formatTagMutationSummary("tag", tags));
    if (untags.length > 0)
        ctx.outputs.push(formatTagMutationSummary("untag", untags));
    ctx.tagMutations = [];
}
function clearGroupExpressionCache(ctx) {
    ctx.groupExpressionCache.clear();
}
function hasSideEffectfulGroupCall(text) {
    return /(^|[^A-Za-z0-9_])group\s*\(/.test(text);
}
function isCacheableGroupExpressionText(text, ctx, seen = new Set()) {
    const trimmed = trimOuterParens(text.trim());
    if (seen.has(trimmed))
        return true;
    seen.add(trimmed);
    const value = ctx.state.variables[trimmed];
    if (isGroupExpressionValue(value)) {
        return isCacheableGroupExpressionText(value.expression, ctx, seen);
    }
    if (typeof value === "string" && value !== trimmed) {
        return isCacheableGroupExpressionText(value, ctx, seen);
    }
    return !hasSideEffectfulGroupCall(trimmed);
}
function formatTagMutationSummary(type, mutations) {
    const ids = new Set(mutations.map((mutation) => mutation.id));
    const fields = [...new Set(mutations.map((mutation) => mutation.field))].sort();
    const valueCount = mutations.reduce((sum, mutation) => sum + mutation.valueCount, 0);
    const entryWord = ids.size === 1 ? "entry" : "entries";
    const fieldWord = fields.length === 1 ? "field" : "fields";
    const fieldList = fields.join(", ");
    if (type === "tag") {
        const valueWord = valueCount === 1 ? "tag value" : "tag values";
        return `Tagged ${ids.size} ${entryWord} with ${valueCount} ${valueWord} across ${fields.length} ${fieldWord}: ${fieldList}.`;
    }
    const wholeFieldCount = mutations.filter((mutation) => mutation.wholeField).length;
    if (wholeFieldCount === mutations.length) {
        const removedWord = wholeFieldCount === 1 ? "tag field" : "tag fields";
        return `Removed ${wholeFieldCount} ${removedWord} from ${ids.size} ${entryWord} across ${fields.length} ${fieldWord}: ${fieldList}.`;
    }
    const valueWord = valueCount === 1 ? "tag value" : "tag values";
    return `Removed ${valueCount} ${valueWord} from ${ids.size} ${entryWord} across ${fields.length} ${fieldWord}: ${fieldList}.`;
}
function addValues(a, b) {
    if (typeof a === "number" && typeof b === "number")
        return a + b;
    if (typeof a === "string" || typeof b === "string")
        return `${a ?? ""}${b ?? ""}`;
    if (Array.isArray(a) && Array.isArray(b))
        return [...a, ...b];
    throw new DslRuntimeError("E_BAD_ADD", "+= supports numbers, strings, and lists");
}
function subtractValues(a, b) {
    if (typeof a === "number" && typeof b === "number")
        return a - b;
    if (Array.isArray(a))
        return a.filter((value) => value !== b);
    throw new DslRuntimeError("E_BAD_SUBTRACT", "-= supports numbers and removing one value from a list");
}
async function executeTag(text, ctx, line) {
    const match = text.match(/^tag\((.+?)\s+with\s+(.+?)\s+(set\s+to|add)\s+(.+)\)$/);
    if (!match)
        throw new DslRuntimeError("E_PARSE_TAG", "Expected tag(<id|group_expression> with <field> set to <tag>) or tag(<id|group_expression> with <field> add <tag>)", line);
    const targetText = match[1].trim();
    const field = coerceString(resolveBareOrString(match[2], ctx));
    const op = match[3].toLowerCase() === "add" ? "add" : "set";
    const values = normalizeTagValues(await evaluateTagValueExpression(match[4], ctx, line));
    if (values.length === 0)
        throw new DslRuntimeError("E_TAG_VALUE", "tag requires at least one non-empty tag value", line);
    const ids = await resolveTagTargetIds(targetText, ctx, line);
    for (const id of ids) {
        const entryTags = ctx.state.tagsByEntry[id] ?? {};
        if (op === "set") {
            entryTags[field] = normalizeStoredTagValue(values);
        }
        else {
            const current = tagValueToList(getEntryTags(ctx.entriesById.get(id), ctx)[field]);
            entryTags[field] = normalizeStoredTagValue(uniqueStrings([...current, ...values]));
        }
        ctx.state.tagsByEntry[id] = entryTags;
        ctx.tagMutations.push({ type: "tag", id, field, valueCount: values.length });
    }
    ctx.tagIndex = undefined;
    clearGroupExpressionCache(ctx);
}
async function resolveTagTargetIds(targetText, ctx, line) {
    const target = resolveBareOrString(targetText, ctx);
    if (typeof target === "string" && ctx.entriesById.has(target))
        return [target];
    const expression = isGroupExpressionValue(target) ? target.expression : String(target);
    return [...await resolveGroupExpression(expression, ctx, line)];
}
async function executeUntag(text, ctx, line) {
    const removeMatch = text.match(/^untag\((.+?)\s+with\s+(.+?)\s+remove\s+(.+)\)$/);
    if (removeMatch) {
        const id = coerceId(resolveBareOrString(removeMatch[1], ctx));
        const field = coerceString(resolveBareOrString(removeMatch[2], ctx));
        const values = normalizeTagValues(await evaluateTagValueExpression(removeMatch[3], ctx, line));
        if (values.length === 0)
            throw new DslRuntimeError("E_TAG_VALUE", "untag remove requires at least one non-empty tag value", line);
        if (!ctx.entriesById.has(id))
            throw new DslRuntimeError("E_UNKNOWN_ID", `Unknown entry id ${id}`, line);
        const current = tagValueToList(getEntryTags(ctx.entriesById.get(id), ctx)[field]);
        const remaining = current.filter((value) => !values.includes(value));
        const entryTags = ctx.state.tagsByEntry[id] ?? {};
        if (remaining.length === 0)
            delete entryTags[field];
        else
            entryTags[field] = normalizeStoredTagValue(remaining);
        ctx.state.tagsByEntry[id] = entryTags;
        ctx.tagIndex = undefined;
        clearGroupExpressionCache(ctx);
        ctx.tagMutations.push({ type: "untag", id, field, valueCount: values.length });
        return;
    }
    const match = text.match(/^untag\((.+?)\s+from\s+(.+)\)$/);
    if (!match)
        throw new DslRuntimeError("E_PARSE_UNTAG", "Expected untag(<field> from <id>) or untag(<id> with <field> remove <tag>)", line);
    const field = coerceString(resolveBareOrString(match[1], ctx));
    const id = coerceId(resolveBareOrString(match[2], ctx));
    if (ctx.state.tagsByEntry[id])
        delete ctx.state.tagsByEntry[id][field];
    ctx.tagIndex = undefined;
    clearGroupExpressionCache(ctx);
    ctx.tagMutations.push({ type: "untag", id, field, valueCount: 1, wholeField: true });
}
async function evaluateCondition(expr, ctx, line) {
    const orParts = splitTopLevelByWord(expr, "or");
    if (orParts.length > 1) {
        for (const part of orParts)
            if (await evaluateCondition(part, ctx, line))
                return true;
        return false;
    }
    const andParts = splitTopLevelByWord(expr, "and");
    if (andParts.length > 1) {
        for (const part of andParts)
            if (!await evaluateCondition(part, ctx, line))
                return false;
        return true;
    }
    const inMatch = splitByTopLevelOperator(expr, " not in ") ?? splitByTopLevelOperator(expr, " in ");
    if (inMatch) {
        const [left, op, right] = inMatch;
        const leftValue = await evaluateExpression(left, ctx, line);
        const rightValue = await evaluateExpression(right, ctx, line);
        const result = Array.isArray(rightValue) ? rightValue.includes(leftValue) : false;
        return op.trim() === "not in" ? !result : result;
    }
    for (const op of [">=", "<=", "!=", "==", ">", "<"]) {
        const parts = splitByTopLevelOperator(expr, op);
        if (!parts)
            continue;
        const left = await evaluateExpression(parts[0], ctx, line);
        const right = await evaluateExpression(parts[2], ctx, line);
        switch (op) {
            case ">=": return Number(left) >= Number(right);
            case "<=": return Number(left) <= Number(right);
            case "!=": return left !== right;
            case "==": return left === right;
            case ">": return Number(left) > Number(right);
            case "<": return Number(left) < Number(right);
        }
    }
    return Boolean(await evaluateExpression(expr, ctx, line));
}
async function evaluateExpression(expr, ctx, line) {
    let trimmed = expr.trim();
    trimmed = trimOuterParens(trimmed);
    if (isListLiteralExpression(trimmed))
        return parseList(trimmed, ctx, line);
    if (ctx.entriesById.has(trimmed))
        return trimmed;
    const arithmetic = splitArithmetic(trimmed, ["+", "-"]) ?? splitArithmetic(trimmed, ["*", "/"]);
    if (arithmetic)
        return evaluateArithmetic(arithmetic, ctx, line);
    if (trimmed.startsWith("+") && trimmed.length > 1)
        return toNumber(await evaluateExpression(trimmed.slice(1), ctx, line), "+", line);
    if (trimmed.startsWith("-") && trimmed.length > 1 && !/^-?\d+(\.\d+)?$/.test(trimmed)) {
        return -toNumber(await evaluateExpression(trimmed.slice(1), ctx, line), "-", line);
    }
    if (trimmed === "true")
        return true;
    if (trimmed === "false")
        return false;
    if (trimmed === "null")
        return null;
    if (/^-?\d+(\.\d+)?$/.test(trimmed))
        return Number(trimmed);
    const index = splitIndex(trimmed);
    if (index) {
        const base = await evaluateExpression(index.base, ctx, line);
        const slice = splitSliceIndex(index.index);
        if (slice)
            return getSlice(base, await evaluateSliceBound(slice.start, ctx, line), await evaluateSliceBound(slice.end, ctx, line), line);
        const key = await evaluateExpression(index.index, ctx, line);
        return getIndex(base, key, line);
    }
    const property = splitProperty(trimmed);
    if (property) {
        const base = await evaluateExpression(property.base, ctx, line);
        return getProperty(base, property.property, ctx, line);
    }
    if (trimmed.startsWith("print(") && trimmed.endsWith(")")) {
        const args = splitTopLevel(innerCall(trimmed), ",").filter((part) => part.length > 0);
        const values = await Promise.all(args.map((arg) => evaluateExpression(arg, ctx, line)));
        pushOutput(ctx, values.length === 1 ? formatValue(values[0]) : values.map(formatInlineValue).join(" "));
        return undefined;
    }
    if (trimmed.startsWith("str(") && trimmed.endsWith(")"))
        return stringifyExpression(innerCall(trimmed), ctx, line);
    if (trimmed.startsWith("len(") && trimmed.endsWith(")"))
        return lengthExpression(innerCall(trimmed), ctx, line);
    if (trimmed.startsWith("type(") && trimmed.endsWith(")"))
        return typeExpression(innerCall(trimmed), ctx, line);
    if (trimmed.startsWith("retrieve(") && trimmed.endsWith(")"))
        return retrieve(innerCall(trimmed), ctx, line);
    if (trimmed.startsWith("get(") && trimmed.endsWith(")"))
        return getValue(innerCall(trimmed), ctx, line);
    if (trimmed.startsWith("count(") && trimmed.endsWith(")"))
        return countGroup(innerCall(trimmed), ctx, line);
    if (trimmed.startsWith("count_by(") && trimmed.endsWith(")"))
        return countBy(innerCall(trimmed), ctx, line);
    if (trimmed.startsWith("group_expr(") && trimmed.endsWith(")"))
        return createGroupExpression(innerCall(trimmed));
    if (trimmed.startsWith("temp(") && trimmed.endsWith(")"))
        return createGroupExpression(trimmed);
    if (trimmed.startsWith("group(") && trimmed.endsWith(")"))
        return createGroup(innerCall(trimmed), ctx, line);
    if (isStringLiteral(trimmed))
        return parseStringLiteral(trimmed, line);
    if (trimmed in ctx.state.variables)
        return ctx.state.variables[trimmed];
    if (/^[A-Za-z][A-Za-z0-9_:-]*$/.test(trimmed))
        return trimmed;
    throw new DslRuntimeError("E_PARSE_EXPR", `Could not parse expression: ${expr}`, line);
}
async function stringifyExpression(expr, ctx, line) {
    const args = splitTopLevel(expr, ",").filter((part) => part.length > 0);
    if (args.length !== 1) {
        throw new DslRuntimeError("E_PARSE_STR", "Expected str(<expression>) with exactly one argument", line);
    }
    return formatInlineValue(await evaluateExpression(args[0], ctx, line));
}
async function lengthExpression(expr, ctx, line) {
    const args = splitTopLevel(expr, ",").filter((part) => part.length > 0);
    if (args.length !== 1) {
        throw new DslRuntimeError("E_PARSE_LEN", "Expected len(<expression>) with exactly one argument", line);
    }
    const value = await evaluateExpression(args[0], ctx, line);
    if (typeof value === "string")
        return Array.from(value).length;
    if (Array.isArray(value))
        return value.length;
    if (value && typeof value === "object")
        return Object.keys(value).length;
    throw new DslRuntimeError("E_LEN", `len() is not available for ${formatValueForError(value)}`, line);
}
async function typeExpression(expr, ctx, line) {
    const args = splitTopLevel(expr, ",").filter((part) => part.length > 0);
    if (args.length !== 1) {
        throw new DslRuntimeError("E_PARSE_TYPE", "Expected type(<expression>) with exactly one argument", line);
    }
    const value = await evaluateExpression(args[0], ctx, line);
    if (value === undefined)
        return "undefined";
    if (value === null)
        return "null";
    if (typeof value === "boolean")
        return "bool";
    if (typeof value === "number")
        return Number.isInteger(value) ? "int" : "float";
    if (typeof value === "string")
        return "string";
    if (Array.isArray(value))
        return "list";
    if (typeof value === "object")
        return "object";
    return typeof value;
}
async function evaluateArithmetic(expr, ctx, line) {
    const leftValue = await evaluateExpression(expr.left, ctx, line);
    const rightValue = await evaluateExpression(expr.right, ctx, line);
    if (expr.operator === "+" && (typeof leftValue === "string" || typeof rightValue === "string")) {
        return `${formatInlineValue(leftValue)}${formatInlineValue(rightValue)}`;
    }
    const left = toNumber(leftValue, expr.operator, line);
    const right = toNumber(rightValue, expr.operator, line);
    switch (expr.operator) {
        case "+": return left + right;
        case "-": return left - right;
        case "*": return left * right;
        case "/":
            if (right === 0)
                throw new DslRuntimeError("E_DIVIDE_BY_ZERO", "Cannot divide by zero", line);
            return left / right;
        default:
            throw new DslRuntimeError("E_ARITHMETIC", `Unknown arithmetic operator ${expr.operator}`, line);
    }
}
function toNumber(value, operator, line) {
    const number = Number(value);
    if (!Number.isFinite(number)) {
        throw new DslRuntimeError("E_ARITHMETIC", `Operator ${operator} requires numeric values, got ${formatValue(value)}`, line);
    }
    return number;
}
function innerCall(text) {
    return text.slice(text.indexOf("(") + 1, -1).trim();
}
function isStringLiteral(text) {
    return (text.startsWith('"') && text.endsWith('"')) || (text.startsWith("'") && text.endsWith("'"));
}
function parseStringLiteral(text, line) {
    try {
        if (text.startsWith('"'))
            return JSON.parse(text);
        return text.slice(1, -1).replace(/\\'/g, "'");
    }
    catch {
        throw new DslRuntimeError("E_STRING", `Invalid string literal ${text}`, line);
    }
}
function isListLiteralExpression(text) {
    if (!text.startsWith("[") || !text.endsWith("]"))
        return false;
    let depth = 0;
    let quoted;
    for (let i = 0; i < text.length; i++) {
        const ch = text[i];
        if (quoted) {
            if (ch === quoted && text[i - 1] !== "\\")
                quoted = undefined;
            continue;
        }
        if (ch === '"' || ch === "'") {
            quoted = ch;
            continue;
        }
        if (ch === "[")
            depth++;
        else if (ch === "]")
            depth--;
        if (depth === 0 && i < text.length - 1)
            return false;
    }
    return depth === 0;
}
async function parseList(text, ctx, line) {
    const inner = text.slice(1, -1).trim();
    if (!inner)
        return [];
    const parts = splitTopLevel(inner, ",");
    return Promise.all(parts.map((part) => evaluateExpression(part, ctx, line)));
}
function splitArithmetic(text, operators) {
    let depth = 0;
    let quoted;
    let candidate;
    for (let i = 0; i < text.length; i++) {
        const ch = text[i];
        if (quoted) {
            if (ch === quoted && text[i - 1] !== "\\")
                quoted = undefined;
            continue;
        }
        if (ch === '"' || ch === "'") {
            quoted = ch;
            continue;
        }
        if ("([{".includes(ch))
            depth++;
        else if (")]}".includes(ch))
            depth--;
        else if (depth === 0 && operators.includes(ch) && !isUnarySign(text, i)) {
            candidate = { left: text.slice(0, i).trim(), operator: ch, right: text.slice(i + 1).trim() };
        }
    }
    if (!candidate || candidate.left.length === 0 || candidate.right.length === 0)
        return undefined;
    return candidate;
}
function isUnarySign(text, index) {
    const ch = text[index];
    if (ch !== "+" && ch !== "-")
        return false;
    let previous = index - 1;
    while (previous >= 0 && /\s/.test(text[previous]))
        previous--;
    if (previous < 0)
        return true;
    return "([{,+-*/".includes(text[previous]);
}
function splitProperty(text) {
    let depth = 0;
    let quoted;
    for (let i = text.length - 1; i >= 0; i--) {
        const ch = text[i];
        if (quoted) {
            if (ch === quoted && text[i - 1] !== "\\")
                quoted = undefined;
            continue;
        }
        if (ch === '"' || ch === "'") {
            quoted = ch;
            continue;
        }
        if (ch === ")" || ch === "]")
            depth++;
        else if (ch === "(" || ch === "[")
            depth--;
        else if (ch === "." && depth === 0)
            return { base: text.slice(0, i), property: text.slice(i + 1) };
    }
    return undefined;
}
function splitIndex(text) {
    if (!text.endsWith("]"))
        return undefined;
    let depth = 0;
    let quoted;
    for (let i = text.length - 1; i >= 0; i--) {
        const ch = text[i];
        if (quoted) {
            if (ch === quoted && text[i - 1] !== "\\")
                quoted = undefined;
            continue;
        }
        if (ch === '"' || ch === "'") {
            quoted = ch;
            continue;
        }
        if (ch === "]")
            depth++;
        else if (ch === "[") {
            depth--;
            if (depth === 0)
                return { base: text.slice(0, i), index: text.slice(i + 1, -1) };
        }
    }
    return undefined;
}
function splitSliceIndex(text) {
    const parts = splitTopLevel(text, ":");
    if (parts.length === 1)
        return undefined;
    if (parts.length !== 2)
        throw new DslRuntimeError("E_PARSE_SLICE", `Expected slice syntax [start:end], got [${text}]`);
    return {
        start: parts[0] ? parts[0] : undefined,
        end: parts[1] ? parts[1] : undefined,
    };
}
async function evaluateSliceBound(expr, ctx, line) {
    if (expr === undefined)
        return undefined;
    const value = await evaluateExpression(expr, ctx, line);
    if (typeof value !== "number" || !Number.isInteger(value)) {
        throw new DslRuntimeError("E_SLICE", `Slice bounds must be integers, got ${formatValue(value)}`, line);
    }
    return value;
}
function getProperty(base, property, ctx, line) {
    if (base && typeof base === "object" && property in base)
        return base[property];
    throw new DslRuntimeError("E_PROPERTY", `Property .${property} is not available on ${formatValueForError(base)}`, line);
}
function getIndex(base, key, line) {
    if (Array.isArray(base) && typeof key === "number")
        return base[normalizeIndex(key, base.length)];
    if (typeof base === "string" && typeof key === "number") {
        const chars = Array.from(base);
        return chars[normalizeIndex(key, chars.length)];
    }
    if (base && typeof base === "object" && (typeof key === "string" || typeof key === "number")) {
        return base[String(key)];
    }
    throw new DslRuntimeError("E_INDEX", `Cannot index ${formatValueForError(base)} with ${formatValue(key)}`, line);
}
function getSlice(base, start, end, line) {
    if (typeof base === "string") {
        const chars = Array.from(base);
        return chars.slice(normalizeSliceBound(start, chars.length, 0), normalizeSliceBound(end, chars.length, chars.length)).join("");
    }
    if (Array.isArray(base)) {
        return base.slice(normalizeSliceBound(start, base.length, 0), normalizeSliceBound(end, base.length, base.length));
    }
    throw new DslRuntimeError("E_SLICE", `Cannot slice ${formatValueForError(base)}`, line);
}
function normalizeIndex(index, length) {
    return index < 0 ? length + index : index;
}
function normalizeSliceBound(value, length, fallback) {
    if (value === undefined)
        return fallback;
    return value < 0 ? length + value : value;
}
function resolveBareOrString(value, ctx) {
    const trimmed = value.trim();
    if (isStringLiteral(trimmed))
        return parseStringLiteral(trimmed, undefined);
    return ctx.state.variables[trimmed] ?? trimmed;
}
function coerceString(value) {
    return String(value);
}
function coerceId(value) {
    return String(value).trim();
}
async function evaluateTagValueExpression(value, ctx, line) {
    const trimmed = value.trim();
    if (isStringLiteral(trimmed) ||
        isListLiteralExpression(trimmed) ||
        trimmed in ctx.state.variables ||
        (trimmed.startsWith("str(") && trimmed.endsWith(")"))) {
        return evaluateExpression(trimmed, ctx, line);
    }
    return resolveBareOrString(trimmed, ctx);
}
function normalizeTagValues(value) {
    const values = Array.isArray(value) ? value : [value];
    return uniqueStrings(values.map((item) => String(item).trim()).filter((item) => item.length > 0));
}
function normalizeStoredTagValue(values) {
    const unique = uniqueStrings(values);
    return unique.length <= 1 ? (unique[0] ?? "") : unique;
}
function tagValueToList(value) {
    if (value === undefined)
        return [];
    return Array.isArray(value) ? value : [value];
}
function uniqueStrings(values) {
    return [...new Set(values)];
}
async function getValue(arg, ctx, line) {
    const trimmed = trimOuterParens(arg.trim());
    if (trimmed === "groups")
        return Object.keys(ctx.state.groups);
    if (trimmed === "fields")
        return getSourceFields(ctx);
    if (trimmed === "text_fields")
        return getTextFields(ctx);
    if (trimmed === "tag_fields")
        return getTagFields(ctx);
    if (trimmed === "tags")
        throw new DslRuntimeError("E_GET_TAGS_DISABLED", 'get(tags) is disabled. Use get(tag_fields) to list fields or get(["field"]) to list values for a field.', line);
    if (trimmed.startsWith("[") && trimmed.endsWith("]")) {
        const fields = (await parseList(trimmed, ctx, line)).map(String);
        return getFieldOrTagValues(fields, ctx);
    }
    const distribution = parseDistribution(trimmed);
    if (distribution)
        return getDistribution(await parseFilter(distribution.filter, ctx, line), distribution.groupExpression, ctx, line);
    const id = coerceId(await evaluateExpression(trimmed, ctx, line));
    const group = ctx.state.groups[id];
    if (group)
        return group;
    const entry = ctx.entriesById.get(id);
    if (!entry)
        throw new DslRuntimeError("E_UNKNOWN_ID", `Unknown entry or group id ${id}`, line);
    return {
        id: entry.id,
        dataset: entry.dataset,
        fields: entry.fields,
        text: entry.text,
        tags: getEntryTags(entry, ctx),
    };
}
function getSourceFields(ctx) {
    return Array.from(new Set(ctx.entries.flatMap((entry) => Object.keys(entry.fields)))).sort();
}
function getTextFields(ctx) {
    return Array.from(new Set(ctx.datasets.flatMap((dataset) => dataset.manifest.embeddedFields ?? dataset.manifest.textFields ?? []))).sort();
}
function getTagFields(ctx) {
    const fields = new Set();
    for (const entry of ctx.entries) {
        for (const field of Object.keys(getEntryTags(entry, ctx)))
            fields.add(field);
    }
    return [...fields].sort();
}
function getFieldOrTagValues(fields, ctx) {
    const valuesFor = (field) => getValuesForFieldOrTag(field, ctx);
    if (fields.length === 1)
        return valuesFor(fields[0]);
    return Object.fromEntries(fields.map((field) => [field, valuesFor(field)]));
}
function getValuesForFieldOrTag(field, ctx) {
    const values = new Map();
    for (const entry of ctx.entries) {
        const hasSourceField = Object.prototype.hasOwnProperty.call(entry.fields, field);
        if (hasSourceField) {
            const value = entry.fields[field];
            values.set(stableValueKey(value), value);
        }
        const tagSource = hasSourceField ? ctx.state.tagsByEntry[entry.id] : getEntryTags(entry, ctx);
        for (const item of tagValueToList(tagSource?.[field])) {
            if (item.length > 0)
                values.set(stableValueKey(item), item);
        }
    }
    return [...values.values()].sort(compareFieldValues);
}
function stableValueKey(value) {
    return JSON.stringify(value);
}
function compareFieldValues(a, b) {
    if (typeof a === "number" && typeof b === "number")
        return a - b;
    return formatInlineValue(a).localeCompare(formatInlineValue(b));
}
function parseDistribution(text) {
    const marker = ") distribution of ";
    const index = text.indexOf(marker);
    if (!text.startsWith("(") || index < 0)
        return undefined;
    const filterText = text.slice(1, index);
    const rawGroupExpression = text.slice(index + marker.length).trim();
    const groupExpression = trimOuterParens(rawGroupExpression);
    return { filter: filterText, groupExpression };
}
function getTagsDictionary(ctx) {
    const tags = {};
    for (const entry of ctx.entries) {
        for (const [field, value] of Object.entries(getEntryTags(entry, ctx))) {
            for (const item of tagValueToList(value)) {
                if (item.length > 0)
                    (tags[field] ??= new Set()).add(item);
            }
        }
    }
    return Object.fromEntries(Object.entries(tags).map(([field, values]) => [field, [...values].sort()]));
}
function getEntryTags(entry, ctx) {
    return { ...entry.tags, ...(ctx.state.tagsByEntry[entry.id] ?? {}) };
}
function getTagIndex(ctx) {
    if (ctx.tagIndex)
        return ctx.tagIndex;
    const index = new Map();
    for (const entry of ctx.entries) {
        for (const [field, value] of Object.entries(getEntryTags(entry, ctx))) {
            for (const item of tagValueToList(value)) {
                if (item.length === 0)
                    continue;
                let values = index.get(field);
                if (!values) {
                    values = new Map();
                    index.set(field, values);
                }
                let ids = values.get(item);
                if (!ids) {
                    ids = new Set();
                    values.set(item, ids);
                }
                ids.add(entry.id);
            }
        }
    }
    ctx.tagIndex = index;
    return index;
}
function getIdsMatchingTags(tags, ctx) {
    const index = getTagIndex(ctx);
    let ids;
    for (const [field, value] of Object.entries(tags)) {
        const matching = index.get(field)?.get(value);
        if (!matching)
            return new Set();
        ids = ids ? intersectSets(ids, matching) : new Set(matching);
        if (ids.size === 0)
            return ids;
    }
    return ids ?? new Set(ctx.entries.map((entry) => entry.id));
}
function intersectSets(a, b) {
    const [small, large] = a.size <= b.size ? [a, b] : [b, a];
    return new Set([...small].filter((id) => large.has(id)));
}
async function countGroup(arg, ctx, line) {
    return (await resolveGroupExpression(arg, ctx, line)).size;
}
async function countBy(arg, ctx, line) {
    const parts = splitByTopLevelOperator(arg, " of ");
    if (!parts)
        throw new DslRuntimeError("E_PARSE_COUNT_BY", 'Expected count_by(["field", ...] of <group_expression>)', line);
    const fieldValues = await evaluateExpression(parts[0], ctx, line);
    if (!Array.isArray(fieldValues) || fieldValues.length === 0) {
        throw new DslRuntimeError("E_PARSE_COUNT_BY", "count_by requires a non-empty list of fields", line);
    }
    const fields = fieldValues.map(String);
    const ids = [...(await resolveGroupExpression(parts[2], ctx, line))];
    const buckets = new Map();
    for (const id of ids) {
        const entry = ctx.entriesById.get(id);
        if (!entry)
            continue;
        for (const values of expandFieldOrTagValues(fields, entry, ctx)) {
            const key = JSON.stringify(values);
            const bucket = buckets.get(key);
            if (bucket)
                bucket.count++;
            else
                buckets.set(key, { values, count: 1 });
        }
    }
    return [...buckets.values()]
        .sort((a, b) => a.values.map(formatInlineValue).join("\u0000").localeCompare(b.values.map(formatInlineValue).join("\u0000")))
        .map(({ values, count }) => {
        const row = {};
        for (const [index, field] of fields.entries())
            row[field] = values[index];
        row.count = count;
        return row;
    });
}
function expandFieldOrTagValues(fields, entry, ctx) {
    const valuesByField = fields.map((field) => {
        const hasSourceField = Object.prototype.hasOwnProperty.call(entry.fields, field);
        const fieldValue = hasSourceField ? [entry.fields[field]] : [];
        const tagSource = hasSourceField ? ctx.state.tagsByEntry[entry.id] : getEntryTags(entry, ctx);
        const tagValues = tagValueToList(tagSource?.[field]).filter((value) => value.length > 0);
        const values = uniqueFieldValues([...fieldValue, ...tagValues]);
        return values.length > 0 ? values : ["(missing)"];
    });
    let rows = [[]];
    for (const values of valuesByField) {
        const nextRows = [];
        for (const row of rows) {
            for (const value of values)
                nextRows.push([...row, value]);
        }
        rows = nextRows;
    }
    return rows;
}
function uniqueFieldValues(values) {
    const seen = new Set();
    const out = [];
    for (const value of values) {
        const key = stableValueKey(value);
        if (seen.has(key))
            continue;
        seen.add(key);
        out.push(value);
    }
    return out;
}
async function resolveGroupSpec(specText, ctx, line) {
    const spec = await parseGroupSpec(specText, ctx, line);
    const hasConstraints = spec.fieldsCompare.length > 0 ||
        Boolean(spec.tags) ||
        spec.contains.length > 0 ||
        spec.containsWord.length > 0 ||
        spec.bm25.length > 0 ||
        spec.embeddings.length > 0;
    let ids = !hasConstraints && spec.include ? new Set() : new Set(ctx.entries.map((entry) => entry.id));
    if (spec.fieldsCompare.length > 0) {
        ids = getIdsComparingFields(spec.fieldsCompare, ctx, line);
    }
    if (spec.tags)
        ids = intersectSets(ids, getIdsMatchingTags(spec.tags, ctx));
    for (const predicate of spec.contains) {
        ids = intersectSets(ids, getIdsContaining(predicate, ctx));
    }
    for (const predicate of spec.containsWord) {
        ids = intersectSets(ids, getIdsContainingWord(predicate, ctx, line));
    }
    for (const predicate of spec.bm25) {
        ids = intersectSets(ids, getIdsAboveBm25Threshold(predicate, predicate.threshold ?? 0, ctx));
    }
    for (const predicate of spec.embeddings) {
        ids = intersectSets(ids, await getIdsAboveEmbeddingThreshold(predicate, predicate.threshold ?? 0, ctx, line));
    }
    for (const id of spec.include ?? [])
        ids.add(id);
    for (const id of spec.exclude ?? [])
        ids.delete(id);
    return ids;
}
async function createGroup(specText, ctx, line) {
    const ids = await resolveGroupSpec(specText, ctx, line);
    const groupId = `G${ctx.state.nextGroupNumber++}`;
    ctx.state.groups[groupId] = {
        id: groupId,
        datasets: ctx.activeDatasetNames,
        spec: specText,
        entryIds: [...ids],
        createdAt: new Date().toISOString(),
    };
    clearGroupExpressionCache(ctx);
    return groupId;
}
function createGroupExpression(expression) {
    return {
        __quailType: "group_expression",
        expression: trimOuterParens(expression.trim()),
    };
}
async function retrieve(arg, ctx, line) {
    const parsed = parseRetrieveArgs(arg, line);
    const amount = parsed.amount;
    const ids = [...(await resolveGroupExpression(parsed.groupExpression, ctx, line))];
    let selected;
    if (!parsed.filter) {
        if (parsed.location === "top")
            selected = ids.slice(0, amount);
        else if (parsed.location === "bottom")
            selected = ids.slice(-amount);
        else {
            const center = Math.floor(ids.length / 2);
            const start = Math.max(0, center - Math.floor(amount / 2));
            selected = ids.slice(start, start + amount);
        }
        return selected;
    }
    const filter = await parseFilter(parsed.filter, ctx, line);
    if (parsed.location === "top" && filter.type !== "direction") {
        return topBySimilarity(ids, filter, ctx, line, amount);
    }
    const sorted = filter.type === "direction"
        ? sortByDirection(ids, filter, ctx, line)
        : await sortBySimilarity(ids, filter, ctx, line);
    if (parsed.location === "top")
        selected = sorted.slice(0, amount);
    else if (parsed.location === "bottom")
        selected = sorted.slice(-amount).reverse();
    else {
        const center = Math.floor(sorted.length / 2);
        const start = Math.max(0, center - Math.floor(amount / 2));
        selected = sorted.slice(start, start + amount);
    }
    return selected;
}
function parseRetrieveArgs(arg, line) {
    const unrankedPrefix = arg.match(/^(top|middle|bottom)\s+(\d+)\s+of\b\s*(.+)$/s);
    if (unrankedPrefix) {
        const groupExpression = unrankedPrefix[3].trim();
        if (!groupExpression)
            throw new DslRuntimeError("E_PARSE_RETRIEVE", "retrieve requires a group expression after of, such as all or temp(...)", line);
        return {
            location: unrankedPrefix[1],
            amount: Number.parseInt(unrankedPrefix[2], 10),
            groupExpression,
        };
    }
    const prefix = arg.match(/^(top|middle|bottom)\s+(\d+)\s+in\s*/s);
    if (!prefix) {
        throw new DslRuntimeError("E_PARSE_RETRIEVE", "Expected retrieve(<top|middle|bottom> <amount> of <group_expression>) or retrieve(<top|middle|bottom> <amount> in (<filter>) of <group_expression>)", line);
    }
    const rest = arg.slice(prefix[0].length);
    const openIndex = rest.search(/\S/);
    if (openIndex < 0 || rest[openIndex] !== "(") {
        throw new DslRuntimeError("E_PARSE_RETRIEVE", "retrieve requires the ranking filter in parentheses, such as in (BM25: \"freedom\")", line);
    }
    const closeIndex = findMatchingClose(rest, openIndex, "(", ")");
    if (closeIndex < 0)
        throw new DslRuntimeError("E_PARSE_RETRIEVE", "Could not find the closing ) for the retrieve ranking filter", line);
    const afterFilter = rest.slice(closeIndex + 1).trim();
    const ofMatch = afterFilter.match(/^of\b\s*(.+)$/s);
    if (!ofMatch) {
        throw new DslRuntimeError("E_PARSE_RETRIEVE", "Expected retrieve(<location> <amount> in (<filter>) of <group_expression>)", line);
    }
    const groupExpression = ofMatch[1].trim();
    if (!groupExpression)
        throw new DslRuntimeError("E_PARSE_RETRIEVE", "retrieve requires a group expression after of, such as all or temp(...)", line);
    return {
        location: prefix[1],
        amount: Number.parseInt(prefix[2], 10),
        filter: rest.slice(openIndex + 1, closeIndex),
        groupExpression,
    };
}
async function sortBySimilarity(ids, filter, ctx, line) {
    const predicate = { field: filter.field, text: filter.text };
    const scoreVector = filter.type === "BM25" ? getBm25ScoreVector(predicate, ctx) : await getEmbeddingScoreVector(predicate, ctx, line);
    return [...ids].sort((a, b) => scoreForId(scoreVector, b, ctx) - scoreForId(scoreVector, a, ctx));
}
async function topBySimilarity(ids, filter, ctx, line, amount) {
    if (amount <= 0)
        return [];
    const predicate = { field: filter.field, text: filter.text };
    const scoreVector = filter.type === "BM25" ? getBm25ScoreVector(predicate, ctx) : await getEmbeddingScoreVector(predicate, ctx, line);
    const top = [];
    for (const id of ids) {
        const score = scoreForId(scoreVector, id, ctx);
        if (top.length === amount && score <= top[top.length - 1].score)
            continue;
        let index = top.length;
        while (index > 0 && score > top[index - 1].score)
            index--;
        top.splice(index, 0, { id, score });
        if (top.length > amount)
            top.pop();
    }
    return top.map((item) => item.id);
}
function sortByDirection(ids, filter, ctx, line) {
    const source = ctx.entriesById.get(filter.fromId);
    if (!source)
        throw new DslRuntimeError("E_UNKNOWN_ID", `Unknown entry id ${filter.fromId}`, line);
    const sourceOrdinal = ctx.entryOrdinalById.get(source.id);
    if (sourceOrdinal === undefined)
        throw new DslRuntimeError("E_DIRECTION", `Could not determine ordering for entry id ${source.id}`, line);
    const allowed = new Set(ids);
    return ctx.entries
        .filter((entry) => entry.dataset === source.dataset && entry.id !== source.id && allowed.has(entry.id))
        .filter((entry) => {
        const ordinal = ctx.entryOrdinalById.get(entry.id);
        return ordinal !== undefined && (filter.direction === "before" ? ordinal < sourceOrdinal : ordinal > sourceOrdinal);
    })
        .sort((a, b) => {
        const aOrdinal = ctx.entryOrdinalById.get(a.id) ?? 0;
        const bOrdinal = ctx.entryOrdinalById.get(b.id) ?? 0;
        return filter.direction === "before" ? bOrdinal - aOrdinal : aOrdinal - bOrdinal;
    })
        .map((entry) => entry.id);
}
function getStableOrdinal(entry, entryIndex) {
    const ordinal = Number(entry.ordinal);
    return Number.isFinite(ordinal) && ordinal > 0 ? ordinal : entryIndex + 1;
}
async function parseFilter(text, ctx, line) {
    const trimmed = trimOuterParens(text.trim());
    const direction = trimmed.match(/^direction\s*:\s*(.+?)\s+from\s+(.+)$/is);
    if (direction) {
        const rawDirection = trimOuterParens(direction[1].trim());
        const rawDirectionKeyword = rawDirection.toLowerCase();
        const directionValue = rawDirectionKeyword === "before" || rawDirectionKeyword === "after"
            ? rawDirectionKeyword
            : coerceString(await evaluateExpression(rawDirection, ctx, line)).toLowerCase();
        if (directionValue !== "before" && directionValue !== "after") {
            throw new DslRuntimeError("E_PARSE_FILTER", `direction must be "before" or "after", got ${formatValue(directionValue)}`, line);
        }
        return {
            type: "direction",
            direction: directionValue,
            fromId: coerceId(await evaluateExpression(direction[2].trim(), ctx, line)),
        };
    }
    const match = trimmed.match(/^(BM25|embeddings)\s*:\s*(.+)$/is);
    if (!match) {
        if (/^contains(?:_word)?\s*:/i.test(trimmed) || /^tags\s*:/i.test(trimmed)) {
            throw new DslRuntimeError("E_PARSE_FILTER", 'retrieve ranks with BM25: "text" or embeddings: "text". Put contains/tags filters after of, for example retrieve(top 20 in (BM25: "freedom") of temp(contains: "freedom")).', line);
        }
        throw new DslRuntimeError("E_PARSE_FILTER", `Expected BM25: "text", embeddings: "text", or direction: before/after from <id>, got ${text}`, line);
    }
    const rawText = match[2].trim();
    const parsed = rawText.startsWith("[")
        ? await parseFieldTextThreshold(rawText, ctx, line)
        : { text: coerceString(await evaluateExpression(rawText, ctx, line)) };
    return { type: match[1] === "BM25" ? "BM25" : "embeddings", field: parsed.field, text: parsed.text };
}
async function parseGroupSpec(text, ctx, line) {
    const spec = { bm25: [], embeddings: [], contains: [], containsWord: [], fieldsCompare: [] };
    for (const part of splitTopLevel(text, ",")) {
        if (!part.trim())
            continue;
        const [rawKey, rawValue] = splitKeyValue(part, line);
        const key = rawKey.trim().toLowerCase();
        const value = rawValue.trim();
        if (key === "bm25") {
            spec.bm25.push(await parseFieldTextThreshold(value, ctx, line));
        }
        else if (key === "embeddings" || key === "embedding") {
            spec.embeddings.push(await parseFieldTextThreshold(value, ctx, line));
        }
        else if (key === "contains") {
            spec.contains.push(await parseFieldTextThreshold(value, ctx, line));
        }
        else if (key === "contains_word" || key === "containsword") {
            spec.containsWord.push(await parseFieldTextThreshold(value, ctx, line));
        }
        else if (key === "include") {
            spec.include = await parseIdListExpression(value, ctx, line, "include");
        }
        else if (key === "exclude") {
            spec.exclude = await parseIdListExpression(value, ctx, line, "exclude");
        }
        else if (key === "fields_compare" || key === "field_compare") {
            spec.fieldsCompare.push(...await parseFieldComparisons(value, ctx, line));
        }
        else if (key === "tags") {
            spec.tags = await parseTagPairs(value, ctx, line);
        }
        else {
            throw new DslRuntimeError("E_GROUP_SPEC_KEY", `Unknown group spec field ${rawKey}`, line);
        }
    }
    return spec;
}
async function parseIdListExpression(text, ctx, line, key) {
    const trimmed = text.trim();
    const value = isListLiteralExpression(trimmed)
        ? await parseList(trimmed, ctx, line)
        : await evaluateExpression(trimmed, ctx, line);
    if (!Array.isArray(value)) {
        throw new DslRuntimeError("E_GROUP_SPEC_VALUE", `${key} requires a list of entry ids, got ${formatValueForError(value)}`, line);
    }
    return value.map((item) => coerceId(item));
}
function splitKeyValue(text, line) {
    const index = text.indexOf(":");
    if (index < 0)
        throw new DslRuntimeError("E_GROUP_SPEC", `Expected key: value in group spec part ${text}`, line);
    return [text.slice(0, index), text.slice(index + 1)];
}
function parseTextThreshold(text, line) {
    const match = text.match(/^((?:"(?:\\.|[^"])*")|(?:'(?:\\.|[^'])*')|.+?)(?:\s*>\s*(.+))?$/s);
    if (!match)
        throw new DslRuntimeError("E_THRESHOLD", `Could not parse text/threshold: ${text}`, line);
    return { text: match[1].trim(), thresholdExpression: match[2]?.trim() };
}
async function parseFieldTextThreshold(text, ctx, line) {
    const trimmed = text.trim();
    if (!trimmed.startsWith("[")) {
        const parsed = parseTextThreshold(trimmed, line);
        return {
            text: coerceString(await evaluateExpression(parsed.text, ctx, line)),
            threshold: await evaluateOptionalThreshold(parsed.thresholdExpression, ctx, line),
        };
    }
    const closeIndex = findMatchingClose(trimmed, 0, "[", "]");
    if (closeIndex < 0)
        throw new DslRuntimeError("E_FIELD_FILTER", `Could not find closing ] in field filter ${text}`, line);
    const thresholdText = trimmed.slice(closeIndex + 1).trim();
    const thresholdMatch = thresholdText.match(/^(?:>\s*(.+))?$/s);
    if (!thresholdMatch)
        throw new DslRuntimeError("E_THRESHOLD", `Could not parse threshold in ${text}`, line);
    const pairs = await parseFieldPairs(trimmed.slice(0, closeIndex + 1), ctx, line);
    const entries = Object.entries(pairs);
    if (entries.length !== 1)
        throw new DslRuntimeError("E_FIELD_FILTER", `Expected exactly one field: value pair in ${text}`, line);
    const [field, value] = entries[0];
    return { field, text: coerceString(value), threshold: await evaluateOptionalThreshold(thresholdMatch[1]?.trim(), ctx, line) };
}
async function evaluateOptionalThreshold(expr, ctx, line) {
    if (!expr)
        return undefined;
    const value = await evaluateExpression(expr, ctx, line);
    if (typeof value !== "number" || !Number.isFinite(value)) {
        throw new DslRuntimeError("E_THRESHOLD", `Threshold must evaluate to a finite number, got ${formatValue(value)}`, line);
    }
    return value;
}
async function parseFieldPairs(text, ctx, line) {
    const inner = text.trim().replace(/^\[/, "").replace(/\]$/, "");
    const fields = {};
    for (const part of splitTopLevel(inner, ",")) {
        if (!part.trim())
            continue;
        const [fieldExpr, valueExpr] = splitKeyValue(part, line);
        const field = await parsePairFieldName(fieldExpr.trim(), ctx, line);
        fields[field] = normalizeFieldValue(await evaluateExpression(valueExpr.trim(), ctx, line));
    }
    return fields;
}
async function parseFieldComparisons(text, ctx, line) {
    const pairs = await parseFieldPairs(text, ctx, line);
    return Object.entries(pairs).map(([field, rawComparison]) => parseFieldComparison(field, rawComparison, line));
}
function parseFieldComparison(field, rawComparison, line) {
    if (!Array.isArray(rawComparison) || rawComparison.length !== 2) {
        throw new DslRuntimeError("E_FIELD_COMPARE", `fields_compare values must be [operator, value], got ${formatValue(rawComparison)} for field ${field}`, line);
    }
    const [rawOperator, value] = rawComparison;
    if (typeof rawOperator !== "string" || !isFieldComparisonOperator(rawOperator)) {
        throw new DslRuntimeError("E_FIELD_COMPARE", `fields_compare operator must be one of ==, !=, >, <, >=, <=, got ${formatValue(rawOperator)} for field ${field}`, line);
    }
    return { field, operator: rawOperator, value: normalizeFieldValue(value) };
}
function isFieldComparisonOperator(value) {
    return value === "==" || value === "!=" || value === ">" || value === "<" || value === ">=" || value === "<=";
}
async function parsePairFieldName(text, ctx, line) {
    if (isStringLiteral(text))
        return parseStringLiteral(text, line);
    return coerceString(await evaluateExpression(text, ctx, line));
}
function normalizeFieldValue(value) {
    if (value === undefined || value === null)
        return null;
    if (typeof value === "string" || typeof value === "number" || typeof value === "boolean")
        return value;
    if (Array.isArray(value))
        return value.map(normalizeFieldValue);
    if (typeof value === "object") {
        return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, normalizeFieldValue(item)]));
    }
    return String(value);
}
async function parseTagPairs(text, ctx, line) {
    const inner = text.trim().replace(/^\[/, "").replace(/\]$/, "");
    const tags = {};
    for (const part of splitTopLevel(inner, ",")) {
        if (!part.trim())
            continue;
        const [field, value] = splitKeyValue(part, line);
        tags[parseStringLiteral(field.trim(), line)] = String(await evaluateExpression(value.trim(), ctx, line));
    }
    return tags;
}
async function resolveGroupExpression(expr, ctx, line) {
    const text = trimOuterParens(expr.trim());
    const cacheable = isCacheableGroupExpressionText(text, ctx);
    if (cacheable) {
        const cached = ctx.groupExpressionCache.get(text);
        if (cached)
            return cached;
    }
    const result = await resolveGroupExpressionUncached(text, ctx, line);
    if (cacheable) {
        ctx.groupExpressionCache.set(text, result);
    }
    return result;
}
async function resolveGroupExpressionUncached(text, ctx, line) {
    if (!text || text === "all")
        return new Set(ctx.entries.map((entry) => entry.id));
    if (text in ctx.state.variables) {
        const value = ctx.state.variables[text];
        if (isGroupExpressionValue(value))
            return resolveGroupExpression(value.expression, ctx, line);
        if (typeof value === "string")
            return resolveGroupExpression(value, ctx, line);
        throw new DslRuntimeError("E_GROUP_EXPR_VARIABLE", `Variable ${text} does not contain a group expression. Use group_expr(<group_expression>) when assigning reusable group expressions.`, line);
    }
    if (ctx.state.groups[text])
        return new Set(ctx.state.groups[text].entryIds.filter((id) => ctx.entriesById.has(id)));
    const orParts = splitTopLevelByWord(text, "or");
    if (orParts.length > 1) {
        const out = new Set();
        for (const part of orParts)
            for (const id of await resolveGroupExpression(part, ctx, line))
                out.add(id);
        return out;
    }
    const andParts = splitTopLevelByWord(text, "and");
    if (andParts.length > 1) {
        let out;
        for (const part of andParts) {
            const partSet = await resolveGroupExpression(part, ctx, line);
            out = out ? new Set([...out].filter((id) => partSet.has(id))) : partSet;
        }
        return out ?? new Set();
    }
    if (text.startsWith("not ")) {
        const excluded = await resolveGroupExpression(text.slice(4), ctx, line);
        return new Set(ctx.entries.map((entry) => entry.id).filter((id) => !excluded.has(id)));
    }
    if (text.startsWith("group(")) {
        const id = await createGroup(innerCall(text), ctx, line);
        return new Set(ctx.state.groups[id].entryIds);
    }
    if (text.startsWith("group_expr(")) {
        return resolveGroupExpression(innerCall(text), ctx, line);
    }
    if (text.startsWith("temp(")) {
        return resolveGroupSpec(innerCall(text), ctx, line);
    }
    if (looksLikeGroupSpec(text)) {
        return resolveGroupSpec(text, ctx, line);
    }
    throw new DslRuntimeError("E_GROUP_EXPR", `Unknown group expression: ${text}. Use a group id like G1, temp(...), all, or a boolean expression.`, line);
}
function looksLikeGroupSpec(text) {
    return /^(BM25|embeddings?|contains|contains_word|containsword|fields_compare|field_compare|include|exclude|tags)\s*:/i.test(text.trim());
}
function isGroupExpressionValue(value) {
    return (typeof value === "object" &&
        value !== null &&
        value.__quailType === "group_expression" &&
        typeof value.expression === "string");
}
function getFieldText(id, field, ctx) {
    const entry = ctx.entriesById.get(id);
    if (!entry)
        return "";
    if (!field)
        return entry.text;
    const dataset = ctx.datasetByEntryId.get(id);
    if (dataset?.manifest.fieldTypes?.[field] !== "string")
        return "";
    const value = entry.fields[field];
    return typeof value === "string" ? value : "";
}
function getFieldContains(id, field, ctx) {
    const entry = ctx.entriesById.get(id);
    if (!entry)
        return "";
    if (!field)
        return entry.contains;
    const dataset = ctx.datasetByEntryId.get(id);
    if (dataset?.manifest.fieldTypes?.[field] !== "string")
        return "";
    return entry.fieldContains[field] ?? normalizeContainsText(getFieldText(id, field, ctx));
}
function getIdsComparingFields(comparisons, ctx, line) {
    const key = comparisons.map((comparison) => [
        comparison.field,
        comparison.operator,
        stableValueKey(comparison.value),
    ].join("\u0001")).join("\u0002");
    const cached = ctx.runtimeCache.fieldComparisonIdSets.get(key);
    if (cached) {
        ctx.runtimeCache.stats.fieldComparisonHits++;
        return new Set(cached);
    }
    ctx.runtimeCache.stats.fieldComparisonMisses++;
    const ids = new Set(ctx.entries
        .filter((entry) => comparisons.every((comparison) => fieldComparisonMatches(entry.fields[comparison.field], comparison, line)))
        .map((entry) => entry.id));
    ctx.runtimeCache.fieldComparisonIdSets.set(key, ids);
    return new Set(ids);
}
function fieldValueEquals(left, right) {
    return stableValueKey(left) === stableValueKey(right);
}
function fieldComparisonMatches(left, comparison, line) {
    if (left === undefined)
        return false;
    if (comparison.operator === "==")
        return fieldValueEquals(left, comparison.value);
    if (comparison.operator === "!=")
        return !fieldValueEquals(left, comparison.value);
    if (typeof left !== "number" || typeof comparison.value !== "number") {
        throw new DslRuntimeError("E_FIELD_COMPARE_TYPE", `Operator ${comparison.operator} requires numeric values for field ${comparison.field}`, line);
    }
    switch (comparison.operator) {
        case ">": return left > comparison.value;
        case "<": return left < comparison.value;
        case ">=": return left >= comparison.value;
        case "<=": return left <= comparison.value;
    }
}
async function getDistribution(filter, groupExpression, ctx, line) {
    if (filter.type === "direction") {
        throw new DslRuntimeError("E_DISTRIBUTION_FILTER", "distribution supports BM25 and embeddings filters, not direction filters", line);
    }
    const ids = [...(await resolveGroupExpression(groupExpression, ctx, line))];
    const predicate = { field: filter.field, text: filter.text };
    const scoreVector = filter.type === "BM25" ? getBm25ScoreVector(predicate, ctx) : await getEmbeddingScoreVector(predicate, ctx, line);
    const scores = ids.map((id) => scoreForId(scoreVector, id, ctx));
    if (scores.length === 0)
        return { min: 0, q1: 0, q2: 0, avg: 0, q3: 0, max: 0 };
    scores.sort((a, b) => a - b);
    const quantile = (q) => scores[Math.min(scores.length - 1, Math.max(0, Math.floor((scores.length - 1) * q)))] ?? 0;
    return {
        min: scores[0],
        q1: quantile(0.25),
        q2: quantile(0.5),
        avg: scores.reduce((sum, value) => sum + value, 0) / scores.length,
        q3: quantile(0.75),
        max: scores[scores.length - 1],
    };
}
function scoreVectorKey(type, predicate, model) {
    return [
        type,
        model ?? "",
        predicate.field ?? "",
        predicate.text,
    ].join("\0");
}
function embeddingRuntimeModelKey(model) {
    return `${model}\0${process.env.QUAIL_OLLAMA_EMBED_URL ?? ""}`;
}
function thresholdKey(scoreKey, threshold) {
    return `${scoreKey}\0>\0${threshold}`;
}
function getBm25ScoreVector(predicate, ctx) {
    const key = scoreVectorKey("BM25", predicate);
    const cached = ctx.runtimeCache.scoreVectors.get(key);
    if (cached) {
        ctx.runtimeCache.stats.scoreVectorHits++;
        return cached;
    }
    ctx.runtimeCache.stats.scoreVectorMisses++;
    const scores = new Float32Array(ctx.entries.length);
    for (const [index, entry] of ctx.entries.entries()) {
        scores[index] = scoreBm25(entry.id, predicate, ctx);
    }
    ctx.runtimeCache.scoreVectors.set(key, scores);
    return scores;
}
async function getQueryEmbedding(model, predicate, ctx) {
    const key = `${embeddingRuntimeModelKey(model)}\0${predicate.field ?? ""}\0${predicate.text}`;
    const local = ctx.embeddingQueryVectors.get(key);
    if (local)
        return local;
    const cached = ctx.runtimeCache.queryEmbeddings.get(key);
    if (cached) {
        ctx.runtimeCache.stats.queryEmbeddingHits++;
        ctx.embeddingQueryVectors.set(key, cached);
        return cached;
    }
    ctx.runtimeCache.stats.queryEmbeddingMisses++;
    const embedding = (await embedTexts([predicate.text], { model, batchSize: 1 })).vectors["0"];
    ctx.embeddingQueryVectors.set(key, embedding);
    ctx.runtimeCache.queryEmbeddings.set(key, embedding);
    return embedding;
}
async function getEmbeddingScoreVector(predicate, ctx, line) {
    const model = ctx.datasets.find((dataset) => dataset.manifest.embeddingModel)?.manifest.embeddingModel;
    if (!model)
        throw new DslRuntimeError("E_EMBEDDING_MODEL", "No embedding model available for active dataset", line);
    const key = scoreVectorKey("embeddings", predicate, embeddingRuntimeModelKey(model));
    const cached = ctx.runtimeCache.scoreVectors.get(key);
    if (cached) {
        ctx.runtimeCache.stats.scoreVectorHits++;
        return cached;
    }
    ctx.runtimeCache.stats.scoreVectorMisses++;
    const embedding = await getQueryEmbedding(model, predicate, ctx);
    const scores = new Float32Array(ctx.entries.length);
    for (const [index, entry] of ctx.entries.entries()) {
        const vectorId = predicate.field ? fieldDocumentId(entry.id, predicate.field) : entry.id;
        const vectors = ctx.datasetByEntryId.get(entry.id)?.embeddings.vectors;
        const vector = vectors?.[vectorId] ?? (predicate.field && getFieldText(entry.id, predicate.field, ctx) === entry.text ? vectors?.[entry.id] : undefined);
        scores[index] = vector ? cosineSimilarity(embedding, vector) : 0;
    }
    ctx.runtimeCache.scoreVectors.set(key, scores);
    return scores;
}
function scoreForId(scores, id, ctx) {
    const index = ctx.entryIndexById.get(id);
    return index === undefined ? 0 : (scores[index] ?? 0);
}
function getIdsAboveScoreThreshold(scoreKeyValue, scores, threshold, ctx) {
    const key = thresholdKey(scoreKeyValue, threshold);
    const cached = ctx.runtimeCache.thresholdIdSets.get(key);
    if (cached) {
        ctx.runtimeCache.stats.thresholdIdSetHits++;
        return cached;
    }
    ctx.runtimeCache.stats.thresholdIdSetMisses++;
    const ids = new Set();
    for (const [index, entry] of ctx.entries.entries()) {
        if ((scores[index] ?? -1) > threshold)
            ids.add(entry.id);
    }
    ctx.runtimeCache.thresholdIdSets.set(key, ids);
    return ids;
}
function getIdsAboveBm25Threshold(predicate, threshold, ctx) {
    const scoreKeyValue = scoreVectorKey("BM25", predicate);
    return getIdsAboveScoreThreshold(scoreKeyValue, getBm25ScoreVector(predicate, ctx), threshold, ctx);
}
async function getIdsAboveEmbeddingThreshold(predicate, threshold, ctx, line) {
    const model = ctx.datasets.find((dataset) => dataset.manifest.embeddingModel)?.manifest.embeddingModel;
    if (!model)
        throw new DslRuntimeError("E_EMBEDDING_MODEL", "No embedding model available for active dataset", line);
    const scoreKeyValue = scoreVectorKey("embeddings", predicate, embeddingRuntimeModelKey(model));
    return getIdsAboveScoreThreshold(scoreKeyValue, await getEmbeddingScoreVector(predicate, ctx, line), threshold, ctx);
}
function scoreBm25(id, predicate, ctx) {
    const dataset = ctx.datasetByEntryId.get(id);
    if (!dataset)
        return 0;
    const cacheKey = `${predicate.field ?? ""}\0${predicate.text}`;
    let terms = ctx.bm25QueryTerms.get(cacheKey);
    if (!terms) {
        terms = tokenize(predicate.text);
        ctx.bm25QueryTerms.set(cacheKey, terms);
    }
    const docId = predicate.field ? fieldDocumentId(id, predicate.field) : id;
    if (!predicate.field || dataset.bm25.termFreq[docId])
        return bm25ScoreTerms(dataset.bm25, docId, terms);
    return scoreBm25FallbackText(dataset.bm25, getFieldText(id, predicate.field, ctx), terms);
}
function scoreBm25FallbackText(index, text, terms) {
    if (!text || terms.length === 0 || index.docCount === 0)
        return 0;
    const tokens = tokenize(text);
    if (tokens.length === 0)
        return 0;
    const tf = {};
    for (const token of tokens)
        tf[token] = (tf[token] ?? 0) + 1;
    const avgdl = index.avgDocLength || tokens.length || 1;
    let score = 0;
    for (const term of terms) {
        const f = tf[term] ?? 0;
        if (f <= 0)
            continue;
        const n = index.docFreq[term] ?? 0;
        const idf = Math.log(1 + (index.docCount - n + 0.5) / (n + 0.5));
        const denom = f + index.k1 * (1 - index.b + index.b * (tokens.length / avgdl));
        score += idf * ((f * (index.k1 + 1)) / denom);
    }
    return score;
}
async function scoreEmbeddings(ids, predicate, ctx, line) {
    const scoreVector = await getEmbeddingScoreVector(predicate, ctx, line);
    const scores = new Map();
    for (const id of ids)
        scores.set(id, scoreForId(scoreVector, id, ctx));
    return scores;
}
function getTextFilterCacheKey(type, predicate) {
    return [type, predicate.field ?? "", predicate.text].join("\0");
}
function getCachedTextFilterIds(key, compute, ctx) {
    const cached = ctx.runtimeCache.textFilterIdSets.get(key);
    if (cached) {
        ctx.runtimeCache.stats.textFilterHits++;
        return cached;
    }
    ctx.runtimeCache.stats.textFilterMisses++;
    const ids = compute();
    ctx.runtimeCache.textFilterIdSets.set(key, ids);
    return ids;
}
function getIdsContaining(predicate, ctx) {
    const needle = normalizeContainsText(predicate.text);
    return getCachedTextFilterIds(getTextFilterCacheKey("contains", predicate), () => new Set(ctx.entries
        .filter((entry) => getFieldContains(entry.id, predicate.field, ctx).includes(needle))
        .map((entry) => entry.id)), ctx);
}
function getIdsContainingWord(predicate, ctx, line) {
    const needle = tokenize(predicate.text);
    if (needle.length === 0)
        throw new DslRuntimeError("E_CONTAINS_WORD", `contains_word requires at least one word token, got ${formatValue(predicate.text)}`, line);
    return getCachedTextFilterIds(getTextFilterCacheKey("contains_word", predicate), () => {
        if (needle.length === 1) {
            const token = needle[0];
            return new Set(ctx.entries.filter((entry) => {
                const docId = predicate.field ? fieldDocumentId(entry.id, predicate.field) : entry.id;
                const termFreq = ctx.datasetByEntryId.get(entry.id)?.bm25.termFreq[docId];
                if (termFreq)
                    return (termFreq[token] ?? 0) > 0;
                return containsWord(getFieldText(entry.id, predicate.field, ctx), predicate.text, line);
            }).map((entry) => entry.id));
        }
        return new Set(ctx.entries
            .filter((entry) => containsWord(getFieldText(entry.id, predicate.field, ctx), predicate.text, line))
            .map((entry) => entry.id));
    }, ctx);
}
function formatValue(value) {
    if (value === undefined)
        return "undefined";
    if (isGroupExpressionValue(value))
        return `group_expr(${value.expression})`;
    if (typeof value === "string")
        return value;
    return JSON.stringify(value, null, 2);
}
function formatValueForError(value) {
    if (!value || typeof value !== "object")
        return formatValue(value);
    if (Array.isArray(value))
        return `list with ${value.length} items`;
    const keys = Object.keys(value);
    return `object with keys: ${keys.slice(0, 8).join(", ")}${keys.length > 8 ? ", ..." : ""}`;
}
function formatInlineValue(value) {
    if (value === undefined)
        return "undefined";
    if (isGroupExpressionValue(value))
        return `group_expr(${value.expression})`;
    if (typeof value === "string")
        return value;
    if (typeof value === "number" || typeof value === "boolean" || value === null)
        return String(value);
    return JSON.stringify(value);
}
function containsWord(text, word, line) {
    const needle = tokenize(word);
    if (needle.length === 0)
        throw new DslRuntimeError("E_CONTAINS_WORD", `contains_word requires at least one word token, got ${formatValue(word)}`, line);
    const haystack = tokenize(text);
    if (needle.length === 1)
        return haystack.includes(needle[0]);
    return haystack.some((_, index) => needle.every((token, offset) => haystack[index + offset] === token));
}
function trimOuterParens(text) {
    let out = text.trim();
    while (out.startsWith("(") && out.endsWith(")") && matchingOuterParens(out)) {
        out = out.slice(1, -1).trim();
    }
    return out;
}
function matchingOuterParens(text) {
    let depth = 0;
    let quoted;
    for (let i = 0; i < text.length; i++) {
        const ch = text[i];
        if (quoted) {
            if (ch === quoted && text[i - 1] !== "\\")
                quoted = undefined;
            continue;
        }
        if (ch === '"' || ch === "'") {
            quoted = ch;
            continue;
        }
        if (ch === "(")
            depth++;
        if (ch === ")")
            depth--;
        if (depth === 0 && i < text.length - 1)
            return false;
    }
    return depth === 0;
}
function findMatchingClose(text, start, open, close) {
    let depth = 0;
    let quoted;
    for (let i = start; i < text.length; i++) {
        const ch = text[i];
        if (quoted) {
            if (ch === quoted && text[i - 1] !== "\\")
                quoted = undefined;
            continue;
        }
        if (ch === '"' || ch === "'") {
            quoted = ch;
            continue;
        }
        if (ch === open)
            depth++;
        else if (ch === close) {
            depth--;
            if (depth === 0)
                return i;
        }
    }
    return -1;
}
function splitTopLevel(text, delimiter) {
    const parts = [];
    let start = 0;
    let depth = 0;
    let quoted;
    for (let i = 0; i < text.length; i++) {
        const ch = text[i];
        if (quoted) {
            if (ch === quoted && text[i - 1] !== "\\")
                quoted = undefined;
            continue;
        }
        if (ch === '"' || ch === "'") {
            quoted = ch;
            continue;
        }
        if ("([{".includes(ch))
            depth++;
        else if (")]}".includes(ch))
            depth--;
        else if (depth === 0 && text.startsWith(delimiter, i)) {
            parts.push(text.slice(start, i).trim());
            start = i + delimiter.length;
            i += delimiter.length - 1;
        }
    }
    parts.push(text.slice(start).trim());
    return parts;
}
function splitTopLevelByWord(text, word) {
    const parts = [];
    let start = 0;
    let depth = 0;
    let quoted;
    const pattern = new RegExp(`\\b${word}\\b`, "g");
    let match;
    while ((match = pattern.exec(text))) {
        for (let i = start; i < match.index; i++) {
            const ch = text[i];
            if (quoted) {
                if (ch === quoted && text[i - 1] !== "\\")
                    quoted = undefined;
                continue;
            }
            if (ch === '"' || ch === "'") {
                quoted = ch;
                continue;
            }
            if ("([{".includes(ch))
                depth++;
            else if (")]}".includes(ch))
                depth--;
        }
        if (depth === 0 && !quoted) {
            parts.push(text.slice(start, match.index).trim());
            start = match.index + word.length;
        }
    }
    parts.push(text.slice(start).trim());
    return parts.filter(Boolean);
}
function splitByTopLevelOperator(expr, operator) {
    let depth = 0;
    let quoted;
    for (let i = 0; i <= expr.length - operator.length; i++) {
        const ch = expr[i];
        if (quoted) {
            if (ch === quoted && expr[i - 1] !== "\\")
                quoted = undefined;
            continue;
        }
        if (ch === '"' || ch === "'") {
            quoted = ch;
            continue;
        }
        if ("([{".includes(ch))
            depth++;
        else if (")]}".includes(ch))
            depth--;
        if (depth === 0 && expr.startsWith(operator, i))
            return [expr.slice(0, i).trim(), operator.trim(), expr.slice(i + operator.length).trim()];
    }
    return undefined;
}
//# sourceMappingURL=dsl.js.map