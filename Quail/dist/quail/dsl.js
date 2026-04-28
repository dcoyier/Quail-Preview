import { bm25Score, cosineSimilarity, embedTexts, loadDatasets } from "./dataset-store.js";
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
export async function executeQuailCallBlocks(options) {
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
            for (const dataset of datasets) {
                for (const entry of dataset.entries)
                    datasetByEntryId.set(entry.id, dataset);
            }
            const ctx = {
                cwd: options.cwd,
                datasets,
                entries,
                entriesById,
                datasetByEntryId,
                state,
                outputs,
                errors,
                activeDatasetNames: block.datasets,
                tagMutations: [],
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
    for (const [index, rawLine] of code.split(/\r?\n/).entries()) {
        if (rawLine.trim().length === 0 || rawLine.trimStart().startsWith("#"))
            continue;
        const indent = rawLine.match(/^[ \t]*/)?.[0].replace(/\t/g, "    ").length ?? 0;
        const node = { line: index + 1, indent, text: rawLine.trim(), children: [] };
        while (stack.length > 0 && indent <= stack[stack.length - 1].indent)
            stack.pop();
        if (node.text === "else:" || node.text === "else") {
            const parent = stack[stack.length - 1] ?? roots[roots.length - 1];
            if (!parent || !parent.text.startsWith("if "))
                throw new DslRuntimeError("E_PARSE_ELSE", "else must follow an if block", node.line);
            parent.elseChildren = [];
            stack.push({ ...node, children: parent.elseChildren });
            continue;
        }
        if (stack.length === 0)
            roots.push(node);
        else
            stack[stack.length - 1].children.push(node);
        stack.push(node);
    }
    return roots;
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
        const value = await evaluateExpression(text, ctx, node.line);
        if (value !== undefined && !isStandaloneRetrieve(text))
            pushOutput(ctx, formatValue(value));
    }
    catch (error) {
        if (error instanceof DslRuntimeError)
            ctx.errors.push({ code: error.code, message: error.message, line: error.line ?? node.line });
        else
            ctx.errors.push({ code: "E_RUNTIME", message: error instanceof Error ? error.message : String(error), line: node.line });
    }
}
function isStandaloneRetrieve(text) {
    return text.startsWith("retrieve(") && text.endsWith(")");
}
function stripTrailingColon(text) {
    return text.endsWith(":") ? text.slice(0, -1).trimEnd() : text;
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
        throw new DslRuntimeError("E_PARSE_TAG", "Expected tag(<id> with <field> set to <tag>) or tag(<id> with <field> add <tag>)", line);
    const id = coerceId(resolveBareOrString(match[1], ctx));
    const field = coerceString(resolveBareOrString(match[2], ctx));
    const op = match[3].toLowerCase() === "add" ? "add" : "set";
    const values = normalizeTagValues(await evaluateTagValueExpression(match[4], ctx, line));
    if (values.length === 0)
        throw new DslRuntimeError("E_TAG_VALUE", "tag requires at least one non-empty tag value", line);
    if (!ctx.entriesById.has(id))
        throw new DslRuntimeError("E_UNKNOWN_ID", `Unknown evidence id ${id}`, line);
    const entryTags = ctx.state.tagsByEntry[id] ?? {};
    if (op === "set") {
        entryTags[field] = normalizeStoredTagValue(values);
    }
    else {
        const current = tagValueToList(getEntryTags(ctx.entriesById.get(id), ctx)[field]);
        entryTags[field] = normalizeStoredTagValue(uniqueStrings([...current, ...values]));
    }
    ctx.state.tagsByEntry[id] = entryTags;
    ctx.tagIndex = undefined;
    ctx.tagMutations.push({ type: "tag", id, field, valueCount: values.length });
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
            throw new DslRuntimeError("E_UNKNOWN_ID", `Unknown evidence id ${id}`, line);
        const current = tagValueToList(getEntryTags(ctx.entriesById.get(id), ctx)[field]);
        const remaining = current.filter((value) => !values.includes(value));
        const entryTags = ctx.state.tagsByEntry[id] ?? {};
        if (remaining.length === 0)
            delete entryTags[field];
        else
            entryTags[field] = normalizeStoredTagValue(remaining);
        ctx.state.tagsByEntry[id] = entryTags;
        ctx.tagIndex = undefined;
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
    ctx.tagMutations.push({ type: "untag", id, field, valueCount: 1, wholeField: true });
}
async function evaluateCondition(expr, ctx, line) {
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
    const arithmetic = splitArithmetic(trimmed, ["+", "-"]) ?? splitArithmetic(trimmed, ["*", "/"]);
    if (arithmetic)
        return evaluateArithmetic(arithmetic, ctx, line);
    if (trimmed.startsWith("+") && trimmed.length > 1)
        return toNumber(await evaluateExpression(trimmed.slice(1), ctx, line), "+", line);
    if (trimmed.startsWith("-") && trimmed.length > 1 && !/^-?\d+(\.\d+)?$/.test(trimmed)) {
        return -toNumber(await evaluateExpression(trimmed.slice(1), ctx, line), "-", line);
    }
    if (/^-?\d+(\.\d+)?$/.test(trimmed))
        return Number(trimmed);
    const index = splitIndex(trimmed);
    if (index) {
        const base = await evaluateExpression(index.base, ctx, line);
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
function getProperty(base, property, ctx, line) {
    if (base && typeof base === "object" && property in base)
        return base[property];
    throw new DslRuntimeError("E_PROPERTY", `Property .${property} is not available on ${formatValueForError(base)}`, line);
}
function getIndex(base, key, line) {
    if (Array.isArray(base) && typeof key === "number")
        return base[key];
    if (base && typeof base === "object" && (typeof key === "string" || typeof key === "number")) {
        return base[String(key)];
    }
    throw new DslRuntimeError("E_INDEX", `Cannot index ${formatValueForError(base)} with ${formatValue(key)}`, line);
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
    if (isStringLiteral(trimmed) || isListLiteralExpression(trimmed) || trimmed in ctx.state.variables) {
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
    if (trimmed === "tag_fields")
        return getTagFields(ctx);
    if (trimmed === "tags")
        throw new DslRuntimeError("E_GET_TAGS_DISABLED", 'get(tags) is disabled. Use get(tag_fields) to list fields or get(["field"]) to list values for a field.', line);
    if (trimmed.startsWith("[") && trimmed.endsWith("]")) {
        const fields = (await parseList(trimmed, ctx, line)).map(String);
        return getTagValues(fields, ctx);
    }
    const distribution = parseDistribution(trimmed);
    if (distribution)
        return getDistribution(distribution.filter, distribution.groupExpression, ctx, line);
    const id = coerceId(await evaluateExpression(trimmed, ctx, line));
    const group = ctx.state.groups[id];
    if (group)
        return group;
    const entry = ctx.entriesById.get(id);
    if (!entry)
        throw new DslRuntimeError("E_UNKNOWN_ID", `Unknown evidence or group id ${id}`, line);
    return {
        id: entry.id,
        dataset: entry.dataset,
        text: entry.text,
        tags: getEntryTags(entry, ctx),
    };
}
function getTagFields(ctx) {
    const fields = new Set();
    for (const entry of ctx.entries) {
        for (const field of Object.keys(getEntryTags(entry, ctx)))
            fields.add(field);
    }
    return [...fields].sort();
}
function getTagValues(fields, ctx) {
    const tags = getTagsDictionary(ctx);
    const valuesFor = (field) => tags[field] ?? [];
    if (fields.length === 1)
        return valuesFor(fields[0]);
    return Object.fromEntries(fields.map((field) => [field, valuesFor(field)]));
}
function parseDistribution(text) {
    const marker = ") distribution of (";
    const index = text.indexOf(marker);
    if (!text.startsWith("(") || index < 0 || !text.endsWith(")"))
        return undefined;
    const filterText = text.slice(1, index + 1);
    const groupExpression = text.slice(index + marker.length, -1);
    return { filter: parseFilter(filterText), groupExpression };
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
        throw new DslRuntimeError("E_PARSE_COUNT_BY", "count_by requires a non-empty list of tag fields", line);
    }
    const fields = fieldValues.map(String);
    const ids = [...(await resolveGroupExpression(parts[2], ctx, line))];
    const buckets = new Map();
    for (const id of ids) {
        const entry = ctx.entriesById.get(id);
        if (!entry)
            continue;
        const tags = getEntryTags(entry, ctx);
        for (const values of expandTagFieldValues(fields, tags)) {
            const key = JSON.stringify(values);
            const bucket = buckets.get(key);
            if (bucket)
                bucket.count++;
            else
                buckets.set(key, { values, count: 1 });
        }
    }
    return [...buckets.values()]
        .sort((a, b) => a.values.join("\u0000").localeCompare(b.values.join("\u0000")))
        .map(({ values, count }) => {
        const row = {};
        for (const [index, field] of fields.entries())
            row[field] = values[index];
        row.count = count;
        return row;
    });
}
function expandTagFieldValues(fields, tags) {
    const valuesByField = fields.map((field) => {
        const values = tagValueToList(tags[field]).filter((value) => value.length > 0);
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
async function resolveGroupSpec(specText, ctx, line) {
    const spec = await parseGroupSpec(specText, ctx, line);
    let ids = spec.tags ? getIdsMatchingTags(spec.tags, ctx) : new Set(ctx.entries.map((entry) => entry.id));
    if (spec.contains !== undefined) {
        const needle = normalizeContainsText(spec.contains);
        ids = new Set([...ids].filter((id) => ctx.entriesById.get(id)?.contains.includes(needle)));
    }
    if (spec.containsWord !== undefined) {
        ids = new Set([...ids].filter((id) => containsWord(ctx.entriesById.get(id)?.text ?? "", spec.containsWord, line)));
    }
    if (spec.bm25) {
        ids = new Set([...ids].filter((id) => scoreBm25(id, spec.bm25.text, ctx) > (spec.bm25.threshold ?? 0)));
    }
    if (spec.embeddings) {
        const scores = await scoreEmbeddings([...ids], spec.embeddings.text, ctx, line);
        ids = new Set([...ids].filter((id) => (scores.get(id) ?? -1) > (spec.embeddings.threshold ?? 0)));
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
    pushOutput(ctx, `${groupId} = ${ids.size}`);
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
    const amount = Math.min(parsed.amount, 20);
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
    const filter = parseFilter(parsed.filter, line);
    const scores = filter.type === "BM25" ? new Map(ids.map((id) => [id, scoreBm25(id, filter.text, ctx)])) : await scoreEmbeddings(ids, filter.text, ctx, line);
    const sorted = ids.sort((a, b) => (scores.get(b) ?? 0) - (scores.get(a) ?? 0));
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
function parseFilter(text, line) {
    const trimmed = trimOuterParens(text.trim());
    const match = trimmed.match(/^(BM25|embeddings)\s*:\s*(.+)$/is);
    if (!match) {
        if (/^contains(?:_word)?\s*:/i.test(trimmed) || /^tags\s*:/i.test(trimmed)) {
            throw new DslRuntimeError("E_PARSE_FILTER", 'retrieve ranks with BM25: "text" or embeddings: "text". Put contains/tags filters after of, for example retrieve(top 20 in (BM25: "freedom") of temp(contains: "freedom")).', line);
        }
        throw new DslRuntimeError("E_PARSE_FILTER", `Expected BM25: "text" or embeddings: "text", got ${text}`, line);
    }
    const rawText = match[2].trim();
    return { type: match[1] === "BM25" ? "BM25" : "embeddings", text: isStringLiteral(rawText) ? parseStringLiteral(rawText, 0) : rawText };
}
async function parseGroupSpec(text, ctx, line) {
    const spec = {};
    for (const part of splitTopLevel(text, ",")) {
        if (!part.trim())
            continue;
        const [rawKey, rawValue] = splitKeyValue(part, line);
        const key = rawKey.trim().toLowerCase();
        const value = rawValue.trim();
        if (key === "bm25") {
            const parsed = parseTextThreshold(value, line);
            spec.bm25 = { text: parsed.text, threshold: parsed.threshold };
        }
        else if (key === "embeddings" || key === "embedding") {
            const parsed = parseTextThreshold(value, line);
            spec.embeddings = { text: parsed.text, threshold: parsed.threshold };
        }
        else if (key === "contains") {
            spec.contains = coerceString(await evaluateExpression(value, ctx, line));
        }
        else if (key === "contains_word" || key === "containsword") {
            spec.containsWord = coerceString(await evaluateExpression(value, ctx, line));
        }
        else if (key === "include") {
            spec.include = (await parseList(value, ctx, line)).map(String);
        }
        else if (key === "exclude") {
            spec.exclude = (await parseList(value, ctx, line)).map(String);
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
function splitKeyValue(text, line) {
    const index = text.indexOf(":");
    if (index < 0)
        throw new DslRuntimeError("E_GROUP_SPEC", `Expected key: value in group spec part ${text}`, line);
    return [text.slice(0, index), text.slice(index + 1)];
}
function parseTextThreshold(text, line) {
    const match = text.match(/^((?:"(?:\\.|[^"])*")|(?:'(?:\\.|[^'])*')|.+?)(?:\s*>\s*(-?\d+(?:\.\d+)?))?$/s);
    if (!match)
        throw new DslRuntimeError("E_THRESHOLD", `Could not parse text/threshold: ${text}`, line);
    return { text: isStringLiteral(match[1].trim()) ? parseStringLiteral(match[1].trim(), line) : match[1].trim(), threshold: match[2] ? Number(match[2]) : undefined };
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
    if (text.startsWith("temp(")) {
        return resolveGroupSpec(innerCall(text), ctx, line);
    }
    throw new DslRuntimeError("E_GROUP_EXPR", `Unknown group expression: ${expr}. Use a group id like G1, temp(...), all, or a boolean expression.`, line);
}
function isGroupExpressionValue(value) {
    return (typeof value === "object" &&
        value !== null &&
        value.__quailType === "group_expression" &&
        typeof value.expression === "string");
}
async function getDistribution(filter, groupExpression, ctx, line) {
    const ids = [...(await resolveGroupExpression(groupExpression, ctx, line))];
    const scores = filter.type === "BM25" ? ids.map((id) => scoreBm25(id, filter.text, ctx)) : [...(await scoreEmbeddings(ids, filter.text, ctx, line)).values()];
    if (scores.length === 0)
        return { min: 0, q1: 0, q2: 0, avg: 0, q3: 0, max: 0 };
    scores.sort((a, b) => a - b);
    const quantile = (q) => scores[Math.min(scores.length - 1, Math.max(0, Math.floor((scores.length - 1) * q)))] ?? 0;
    const distribution = {
        min: scores[0],
        q1: quantile(0.25),
        q2: quantile(0.5),
        avg: scores.reduce((sum, value) => sum + value, 0) / scores.length,
        q3: quantile(0.75),
        max: scores[scores.length - 1],
    };
    pushOutput(ctx, formatValue(distribution));
    return distribution;
}
function scoreBm25(id, query, ctx) {
    const dataset = ctx.datasetByEntryId.get(id);
    if (!dataset)
        return 0;
    return bm25Score(dataset.bm25, id, query);
}
async function scoreEmbeddings(ids, query, ctx, line) {
    const model = ctx.datasets.find((dataset) => dataset.manifest.embeddingModel)?.manifest.embeddingModel;
    if (!model)
        throw new DslRuntimeError("E_EMBEDDING_MODEL", "No embedding model available for active dataset", line);
    const embedding = (await embedTexts([query], { model, batchSize: 1 })).vectors["0"];
    const scores = new Map();
    for (const id of ids) {
        const vector = ctx.datasetByEntryId.get(id)?.embeddings.vectors[id];
        scores.set(id, vector ? cosineSimilarity(embedding, vector) : 0);
    }
    return scores;
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