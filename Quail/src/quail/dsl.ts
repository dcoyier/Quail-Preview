import { spawn } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { request } from "undici";
import {
	bm25ScoreTerms,
	embeddingBackendCacheKey,
	embedTexts,
	fieldDocumentId,
	fieldValueToText,
	loadLocalEmbeddingEnv,
	loadDatasets,
	scoreEmbeddingVectorValues,
	type EmbeddingVector,
	type FieldValue,
	type LoadedQuailDataset,
	type QuailEntry,
} from "./dataset-store.js";
import { cloneAnalysisState, type QuailAnalysisState, type QuailPythonBindingState } from "./analysis-state.js";
import { normalizeContainsText, tokenize } from "./text.js";

export interface QuailCallBlock {
	datasets: string[];
	code: string;
	raw: string;
}

export interface QuailExecutionResult {
	state: QuailAnalysisState;
	output: string;
	errors: QuailDslError[];
	blocks: number;
}

export interface QuailDslError {
	code: string;
	message: string;
	line?: number;
}

interface RuntimeContext {
	cwd: string;
	datasets: LoadedQuailDataset[];
	entries: QuailEntry[];
	entriesById: Map<string, QuailEntry>;
	datasetByEntryId: Map<string, LoadedQuailDataset>;
	entryIndexById: Map<string, number>;
	entryOrdinalById: Map<string, number>;
	processedFields: string[];
	tagIndex?: TagIndex;
	state: QuailAnalysisState;
	outputs: string[];
	errors: QuailDslError[];
	activeDatasetNames: string[];
	tagMutations: TagMutation[];
	bm25QueryTerms: Map<string, string[]>;
	embeddingQueryVectors: Map<string, ArrayLike<number>>;
	groupExpressionCache: Map<string, IdBitSet>;
	scopedGroupExpressionCache: Map<string, ScopedMembers>;
	similarityTargetsCache: Map<string, Promise<SimilarityTarget[]>>;
	unitValuesCache: Map<string, UnitValue[]>;
	runtimeCache: DslRuntimeCache;
	variables: Record<string, unknown>;
}

type TagIndex = Map<string, Map<string, Set<string>>>;
type TagValue = unknown;
type QuailScope = "entries" | "fields";

interface DslRuntimeCache {
	scoreVectors: LruMap<string, Float32Array>;
	scoreVectorPromises: Map<string, Promise<Float32Array>>;
	thresholdIdSets: LruMap<string, IdBitSet>;
	fieldComparisonIdSets: LruMap<string, IdBitSet>;
	textFilterIdSets: LruMap<string, IdBitSet>;
	queryEmbeddings: LruMap<string, ArrayLike<number>>;
	queryEmbeddingPromises: Map<string, Promise<ArrayLike<number>>>;
	stats: DslRuntimeCacheStats;
}

export interface DslRuntimeCacheStats {
	datasetContexts: number;
	scoreVectorEntries: number;
	thresholdIdSetEntries: number;
	fieldComparisonEntries: number;
	textFilterEntries: number;
	queryEmbeddingEntries: number;
	scoreVectorHits: number;
	scoreVectorMisses: number;
	thresholdIdSetHits: number;
	thresholdIdSetMisses: number;
	fieldComparisonHits: number;
	fieldComparisonMisses: number;
	textFilterHits: number;
	textFilterMisses: number;
	queryEmbeddingHits: number;
	queryEmbeddingMisses: number;
}

interface TagMutation {
	type: "tag" | "untag";
	id: string;
	field: string;
	valueCount: number;
	wholeField?: boolean;
}

interface FilterSpec {
	type: "BM25" | "embeddings";
	field?: string;
	text: string;
}

interface FieldTextPredicate {
	field?: string;
	text: string;
	threshold?: number;
}

type FieldComparisonOperator = "==" | "!=" | ">" | "<" | ">=" | "<=";

interface FieldComparison {
	field: string;
	operator: FieldComparisonOperator;
	value: FieldValue;
}

interface GroupSpec {
	bm25: FieldTextPredicate[];
	embeddings: FieldTextPredicate[];
	contains: FieldTextPredicate[];
	containsWord: FieldTextPredicate[];
	fieldsCompare: FieldComparison[];
	include?: string[];
	exclude?: string[];
	tags?: Record<string, unknown>;
}

interface GroupExpressionValue {
	__quailType: "group_expression";
	expression: string;
}

interface ScopedMembers {
	scope: QuailScope;
	members: string[];
	spec: string;
}

interface UnitSpec {
	scope: QuailScope;
	kind: "entries" | "entryField" | "fields" | "fieldValues";
	field?: string;
	regexOps: RegexOperation[];
	raw: string;
}

interface RegexOperation {
	type: "find" | "remove" | "splice";
	pattern?: string;
	start?: number;
	end?: number;
}

interface UnitValue {
	value: unknown;
	text: string;
	entryId?: string;
	fieldName?: string;
	scoreByEntry?: boolean;
	sourceIndex: number;
}

type SimilarityTarget =
	| { kind: "text"; text: string }
	| { kind: "entryGroup"; members: string[]; spec: string; cacheKey: string };

interface LineNode {
	line: number;
	indent: number;
	text: string;
	children: LineNode[];
	elseChildren?: LineNode[];
}

class DslRuntimeError extends Error {
	constructor(
		readonly code: string,
		message: string,
		readonly line?: number,
	) {
		super(message);
	}
}

class LruMap<K, V> {
	private readonly values = new Map<K, V>();

	constructor(private readonly maxEntries: number) {}

	get(key: K): V | undefined {
		const value = this.values.get(key);
		if (value === undefined) return undefined;
		this.values.delete(key);
		this.values.set(key, value);
		return value;
	}

	set(key: K, value: V): void {
		if (this.maxEntries <= 0) return;
		if (this.values.has(key)) this.values.delete(key);
		this.values.set(key, value);
		while (this.values.size > this.maxEntries) {
			const firstKey = this.values.keys().next().value as K | undefined;
			if (firstKey === undefined) break;
			this.values.delete(firstKey);
		}
	}

	clear(): void {
		this.values.clear();
	}

	get size(): number {
		return this.values.size;
	}
}

class IdBitSet {
	readonly words: Uint32Array;
	readonly size: number;

	private constructor(
		readonly length: number,
		words?: Uint32Array,
		size?: number,
	) {
		this.words = words ?? new Uint32Array(Math.ceil(length / 32));
		this.size = size ?? countWords(this.words);
	}

	static empty(length: number): IdBitSet {
		return new IdBitSet(length, undefined, 0);
	}

	static full(length: number): IdBitSet {
		const words = new Uint32Array(Math.ceil(length / 32));
		words.fill(0xffffffff);
		maskUnusedBits(words, length);
		return new IdBitSet(length, words, length);
	}

	static fromIds(ids: Iterable<string>, ctx: RuntimeContext): IdBitSet {
		const out = IdBitSet.empty(ctx.entries.length);
		for (const id of ids) out.addId(id, ctx);
		return out;
	}

	static fromPredicate(ctx: RuntimeContext, predicate: (entry: QuailEntry, index: number) => boolean): IdBitSet {
		const out = IdBitSet.empty(ctx.entries.length);
		for (const [index, entry] of ctx.entries.entries()) {
			if (predicate(entry, index)) out.addIndex(index);
		}
		return out;
	}

	clone(): IdBitSet {
		return new IdBitSet(this.length, this.words.slice(), this.size);
	}

	addIndex(index: number): void {
		if (index < 0 || index >= this.length) return;
		const wordIndex = index >>> 5;
		const mask = 1 << (index & 31);
		if ((this.words[wordIndex] & mask) !== 0) return;
		this.words[wordIndex] |= mask;
		(this as { size: number }).size++;
	}

	deleteIndex(index: number): void {
		if (index < 0 || index >= this.length) return;
		const wordIndex = index >>> 5;
		const mask = 1 << (index & 31);
		if ((this.words[wordIndex] & mask) === 0) return;
		this.words[wordIndex] &= ~mask;
		(this as { size: number }).size--;
	}

	addId(id: string, ctx: RuntimeContext): void {
		const index = ctx.entryIndexById.get(id);
		if (index !== undefined) this.addIndex(index);
	}

	deleteId(id: string, ctx: RuntimeContext): void {
		const index = ctx.entryIndexById.get(id);
		if (index !== undefined) this.deleteIndex(index);
	}

	hasIndex(index: number): boolean {
		return index >= 0 && index < this.length && (this.words[index >>> 5] & (1 << (index & 31))) !== 0;
	}

	and(other: IdBitSet): IdBitSet {
		const words = new Uint32Array(this.words.length);
		for (let i = 0; i < words.length; i++) words[i] = this.words[i] & other.words[i];
		return new IdBitSet(this.length, words);
	}

	or(other: IdBitSet): IdBitSet {
		const words = new Uint32Array(this.words.length);
		for (let i = 0; i < words.length; i++) words[i] = this.words[i] | other.words[i];
		return new IdBitSet(this.length, words);
	}

	not(): IdBitSet {
		const words = new Uint32Array(this.words.length);
		for (let i = 0; i < words.length; i++) words[i] = ~this.words[i];
		maskUnusedBits(words, this.length);
		return new IdBitSet(this.length, words);
	}

	toIds(ctx: RuntimeContext): string[] {
		const ids: string[] = [];
		for (let index = 0; index < this.length; index++) {
			if (this.hasIndex(index)) ids.push(ctx.entries[index].id);
		}
		return ids;
	}

	toIdSet(ctx: RuntimeContext): Set<string> {
		return new Set(this.toIds(ctx));
	}
}

function maskUnusedBits(words: Uint32Array, length: number): void {
	if (words.length === 0) return;
	const usedBits = length & 31;
	if (usedBits === 0) return;
	words[words.length - 1] &= 0xffffffff >>> (32 - usedBits);
}

function countWords(words: Uint32Array): number {
	let count = 0;
	for (const word of words) count += popcount32(word);
	return count;
}

function popcount32(value: number): number {
	value >>>= 0;
	value -= (value >>> 1) & 0x55555555;
	value = (value & 0x33333333) + ((value >>> 2) & 0x33333333);
	return (((value + (value >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
}

const runtimeCaches = new Map<string, DslRuntimeCache>();

function envPositiveInt(name: string, fallback: number): number {
	const value = process.env[name]?.trim();
	if (!value) return fallback;
	const parsed = Number(value);
	return Number.isFinite(parsed) && parsed >= 0 ? Math.floor(parsed) : fallback;
}

function emptyCacheStats(): DslRuntimeCacheStats {
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

function createRuntimeCache(): DslRuntimeCache {
	return {
		scoreVectors: new LruMap(envPositiveInt("QUAIL_DSL_SCORE_VECTOR_CACHE_ENTRIES", 64)),
		scoreVectorPromises: new Map(),
		thresholdIdSets: new LruMap(envPositiveInt("QUAIL_DSL_THRESHOLD_CACHE_ENTRIES", 256)),
		fieldComparisonIdSets: new LruMap(envPositiveInt("QUAIL_DSL_FIELD_COMPARE_CACHE_ENTRIES", 256)),
		textFilterIdSets: new LruMap(envPositiveInt("QUAIL_DSL_TEXT_FILTER_CACHE_ENTRIES", 256)),
		queryEmbeddings: new LruMap(envPositiveInt("QUAIL_DSL_QUERY_EMBEDDING_CACHE_ENTRIES", 128)),
		queryEmbeddingPromises: new Map(),
		stats: emptyCacheStats(),
	};
}

function getRuntimeCache(cwd: string, datasets: LoadedQuailDataset[]): DslRuntimeCache {
	const key = [
		cwd,
			...datasets.map((dataset) => [
				dataset.datasetDir,
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

export function clearQuailDslRuntimeCaches(): void {
	runtimeCaches.clear();
}

export function getQuailDslRuntimeCacheStats(): DslRuntimeCacheStats {
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

export function extractQuailCallBlocks(text: string): QuailCallBlock[] {
	const lines = text.split(/\r?\n/);
	const blocks: QuailCallBlock[] = [];
	let start = -1;
	for (let i = 0; i < lines.length; i++) {
		if (lines[i].trim() !== "$") continue;
		if (start < 0) {
			start = i;
			continue;
		}
		const bodyLines = lines.slice(start + 1, i);
		const parsed = parseCallBody(bodyLines.join("\n"));
		if (parsed) blocks.push({ ...parsed, raw: lines.slice(start, i + 1).join("\n") });
		start = -1;
	}
	return blocks;
}

function parseCallBody(body: string): Omit<QuailCallBlock, "raw"> | undefined {
	const lines = body.split(/\r?\n/);
	const firstNonEmpty = lines.findIndex((line) => line.trim().length > 0);
	if (firstNonEmpty < 0) return undefined;
	const atLine = lines[firstNonEmpty].trim();
	if (!atLine.startsWith("@")) return undefined;
	const datasets = [...atLine.matchAll(/"([^"]+)"/g)].map((match) => match[1]);
	const code = lines.slice(firstNonEmpty + 1).join("\n").trim();
	return { datasets, code };
}

export function formatQuailExecutionResult(result: QuailExecutionResult): string {
	const parts: string[] = [];
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

function quailDslExecutorUrl(): string | undefined {
	if (process.env.QUAIL_DSL_EXECUTOR_DISABLE === "1") return undefined;
	const value = process.env.QUAIL_DSL_EXECUTOR_URL?.trim();
	return value ? value.replace(/\/+$/, "") : undefined;
}

function quailDslExecutorTimeoutMs(): number {
	const fallback = 12 * 60 * 60 * 1000;
	const value = process.env.QUAIL_DSL_EXECUTOR_TIMEOUT_MS?.trim();
	if (!value) return fallback;
	const parsed = Number(value);
	return Number.isFinite(parsed) && parsed >= 0 ? parsed : fallback;
}

function isQuailExecutionResult(value: unknown): value is QuailExecutionResult {
	if (!value || typeof value !== "object") return false;
	const record = value as Partial<QuailExecutionResult>;
	return (
		typeof record.blocks === "number" &&
		Array.isArray(record.errors) &&
		typeof record.output === "string" &&
		!!record.state &&
		typeof record.state === "object"
	);
}

async function executeQuailCallBlocksRemote(
	baseUrl: string,
	options: {
		cwd: string;
		state: QuailAnalysisState;
		blocks: QuailCallBlock[];
	},
): Promise<QuailExecutionResult> {
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
	const payload = (JSON.parse(responseText || "{}") as
		| { ok?: boolean; result?: unknown; error?: string }
		| undefined);
	if (response.statusCode < 200 || response.statusCode >= 300 || !payload?.ok) {
		throw new Error(
			`Quail shared DSL executor failed (${response.statusCode}): ${payload?.error ?? responseText.slice(0, 500)}`,
		);
	}
	if (!isQuailExecutionResult(payload.result)) {
		throw new Error("Quail shared DSL executor returned an invalid execution result");
	}
	return payload.result;
}

export async function executeQuailCallBlocks(options: {
	cwd: string;
	state: QuailAnalysisState;
	blocks: QuailCallBlock[];
}): Promise<QuailExecutionResult> {
	loadLocalEmbeddingEnv(options.cwd);
	const remoteUrl = quailDslExecutorUrl();
	if (remoteUrl) return executeQuailCallBlocksRemote(remoteUrl, options);

	const state = cloneAnalysisState(options.state);
	state.variables = cloneJsonableRecord(state.variables ?? {});
	const outputs: string[] = [];
	const errors: QuailDslError[] = [];
	for (const [index, block] of options.blocks.entries()) {
		if (block.datasets.length === 0) {
			errors.push({ code: "E_MISSING_DATASET_LINE", message: "Every Quail call must include an @ line with at least one dataset name." });
			continue;
		}
		try {
			const datasets = loadDatasets(options.cwd, block.datasets);
			const entries = datasets.flatMap((dataset) => dataset.entries);
			const entriesById = new Map(entries.map((entry) => [entry.id, entry]));
			const datasetByEntryId = new Map<string, LoadedQuailDataset>();
			const entryIndexById = new Map<string, number>();
			const entryOrdinalById = new Map<string, number>();
			for (const [entryIndex, entry] of entries.entries()) entryIndexById.set(entry.id, entryIndex);
			for (const dataset of datasets) {
				for (const [entryIndex, entry] of dataset.entries.entries()) {
					datasetByEntryId.set(entry.id, dataset);
					entryOrdinalById.set(entry.id, getStableOrdinal(entry, entryIndex));
				}
			}
			const ctx: RuntimeContext = {
				cwd: options.cwd,
				datasets,
				entries,
				entriesById,
				datasetByEntryId,
				entryIndexById,
				entryOrdinalById,
				processedFields: Array.from(new Set(entries.flatMap((entry) => Object.keys(entry.fields)))).sort(),
				state,
				outputs,
				errors,
				activeDatasetNames: block.datasets,
				tagMutations: [],
				bm25QueryTerms: new Map(),
				embeddingQueryVectors: new Map(),
				groupExpressionCache: new Map(),
				scopedGroupExpressionCache: new Map(),
				similarityTargetsCache: new Map(),
				unitValuesCache: new Map(),
				runtimeCache: getRuntimeCache(options.cwd, datasets),
				variables: cloneJsonableRecord(state.variables),
			};
			await executeProgram(block.code, ctx);
			flushTagMutationSummary(ctx);
		} catch (error) {
			if (error instanceof DslRuntimeError) errors.push({ code: error.code, message: error.message, line: error.line });
			else errors.push({ code: "E_RUNTIME", message: error instanceof Error ? error.message : String(error) });
		}
	}
	return { state, output: outputs.join("\n"), errors, blocks: options.blocks.length };
}

function parseProgram(code: string): LineNode[] {
	const roots: LineNode[] = [];
	const stack: LineNode[] = [];
	for (const logicalLine of logicalLines(code)) {
		const rawLine = logicalLine.text;
		if (rawLine.trim().length === 0 || rawLine.trimStart().startsWith("#")) continue;
		const indent = rawLine.match(/^[ \t]*/)?.[0].replace(/\t/g, "    ").length ?? 0;
		const node: LineNode = { line: logicalLine.line, indent, text: rawLine.trim(), children: [] };
		if (hasUnquotedSemicolon(node.text)) {
			throw new DslRuntimeError("E_SEMICOLON", "Write one statement per line; semicolons are not supported.", node.line);
		}
		if (node.text === "else:" || node.text === "else") {
			while (stack.length > 0 && indent < stack[stack.length - 1].indent) stack.pop();
			const parent = stack[stack.length - 1] ?? roots[roots.length - 1];
			if (!parent || !parent.text.startsWith("if ")) throw new DslRuntimeError("E_PARSE_ELSE", "else must follow an if block", node.line);
			parent.elseChildren = [];
			stack.push({ ...node, children: parent.elseChildren });
			continue;
		}
		while (stack.length > 0 && indent <= stack[stack.length - 1].indent) stack.pop();
		if (stack.length === 0) roots.push(node);
		else stack[stack.length - 1].children.push(node);
		stack.push(node);
	}
	return roots;
}

function logicalLines(code: string): Array<{ line: number; text: string }> {
	const out: Array<{ line: number; text: string }> = [];
	let buffer = "";
	let startLine = 1;
	let depth = 0;
	for (const [index, rawLine] of code.split(/\r?\n/).entries()) {
		const trimmed = rawLine.trim();
		if (!buffer && (trimmed.length === 0 || trimmed.startsWith("#"))) {
			out.push({ line: index + 1, text: rawLine });
			continue;
		}
		if (!buffer) startLine = index + 1;
		buffer = buffer ? `${buffer} ${trimmed}` : rawLine;
		depth += delimiterBalanceDelta(rawLine);
		if (depth <= 0) {
			out.push({ line: startLine, text: buffer });
			buffer = "";
			depth = 0;
		}
	}
	if (buffer) out.push({ line: startLine, text: buffer });
	return out;
}

function validateBalancedDelimiters(code: string): void {
	const pairs: Record<string, string> = { "(": ")", "[": "]", "{": "}" };
	const closers = new Set(Object.values(pairs));
	const stack: Array<{ char: string; line: number }> = [];
	let quoted: string | undefined;
	let line = 1;
	for (let i = 0; i < code.length; i++) {
		const ch = code[i];
		if (ch === "\n") {
			line++;
			continue;
		}
		if (quoted) {
			if (ch === quoted && !isEscaped(code, i)) quoted = undefined;
			continue;
		}
		if (ch === "#") {
			while (i < code.length && code[i] !== "\n") i++;
			if (i < code.length) line++;
			continue;
		}
		if (ch === "\"" || ch === "'") {
			quoted = ch;
			continue;
		}
		if (pairs[ch]) {
			stack.push({ char: ch, line });
			continue;
		}
		if (!closers.has(ch)) continue;
		const opener = stack.pop();
		if (!opener || pairs[opener.char] !== ch) {
			const expected = opener ? pairs[opener.char] : "an opening delimiter";
			throw new DslRuntimeError(
				"E_UNBALANCED_DELIMITER",
				`Unexpected "${ch}" on line ${line}; expected ${expected}. Check parentheses/brackets in the DSL call.`,
				line,
			);
		}
	}
	const opener = stack.at(-1);
	if (opener) {
		throw new DslRuntimeError(
			"E_UNBALANCED_DELIMITER",
			`Unclosed "${opener.char}" opened on line ${opener.line}; expected "${pairs[opener.char]}". Check whether a surrounding print(...), retrieve(...), or group expression is missing a final closing delimiter.`,
			opener.line,
		);
	}
	if (quoted) {
		throw new DslRuntimeError("E_UNBALANCED_DELIMITER", `Unclosed string literal; expected closing ${quoted}.`, line);
	}
}

function delimiterBalanceDelta(text: string): number {
	let depth = 0;
	let quoted: string | undefined;
	for (let i = 0; i < text.length; i++) {
		const ch = text[i];
		if (quoted) {
			if (ch === quoted && text[i - 1] !== "\\") quoted = undefined;
			continue;
		}
		if (ch === '"' || ch === "'") {
			quoted = ch;
			continue;
		}
		if ("([{".includes(ch)) depth++;
		else if (")]}".includes(ch)) depth--;
	}
	return depth;
}

const PYTHON_REQUEST_PREFIX = "@@QUAIL_REQUEST@@";
const PYTHON_DONE_PREFIX = "@@QUAIL_DONE@@";
const PYTHON_ERROR_PREFIX = "@@QUAIL_ERROR@@";
const DEFAULT_MAX_SAVED_VARIABLE_BYTES = 256 * 1024;

function shouldUsePythonRuntime(code: string, ctx: RuntimeContext): boolean {
	const codeForPythonSignals = codeWithoutQuailRawCallBodies(code);
	return /(^|\n)\s*(def|while|try|except|finally|with|class|return|break|continue|pass|raise|assert|import|from|global|nonlocal)\b/.test(code) ||
		/(^|\n)\s*elif\b/.test(code) ||
		/(^|\n)\s*for\s+[A-Za-z_][A-Za-z0-9_]*\s+in\b/.test(code) ||
		/(^|\n)\s*if\s+.+:/.test(code) ||
		/\b(True|False|None|range|enumerate|zip|lambda|getattr|hasattr|isinstance)\b/.test(code) ||
		/\[[^\]\n]+\bfor\b[^\]\n]+\]/.test(code) ||
		/(^|\n)\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*\[.*\bfor\b.*\]/s.test(codeForPythonSignals) ||
		/\{[^{}\n]*:/.test(code) ||
		/\{[^{}\n]+\bfor\b[^{}\n]+\}/.test(code) ||
		/\([^()\n]+\bfor\b[^()\n]+\)/.test(code) ||
		/(^|\n)\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*\([^()\n]*,[^()\n]*\)/.test(code) ||
		/(^|\n)\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*\{[^:{}\n]+(?:,[^:{}\n]*)?\}/.test(code) ||
		/(^|\n)\s*[A-Za-z_][A-Za-z0-9_]*\s*,\s*[A-Za-z_][A-Za-z0-9_]*(?:\s*,[^=\n]+)?\s*=/.test(code) ||
		/(^|\n)\s*for\s+[A-Za-z_][A-Za-z0-9_]*\s*,\s*[A-Za-z_][A-Za-z0-9_]*(?:\s*,[^:\n]+)?\s+in\b/.test(code) ||
			/(^|\n)\s*[A-Za-z_][A-Za-z0-9_]*(?:\[[^\n]+\]|\.[A-Za-z_][A-Za-z0-9_]*)\s*(?:=|\+=|-=)/.test(code) ||
			/\.[A-Za-z_][A-Za-z0-9_]*\s*\(/.test(codeForPythonSignals) ||
			containsScopedGroupLiteral(codeForPythonSignals) ||
			usesPersistedPythonBinding(codeForPythonSignals, ctx.state.pythonBindings ?? []);
}

function containsScopedGroupLiteral(code: string): boolean {
	let quoted: string | undefined;
	for (let i = 0; i < code.length; i++) {
		const ch = code[i];
		if (quoted) {
			if (ch === quoted && code[i - 1] !== "\\") quoted = undefined;
			continue;
		}
		if (ch === '"' || ch === "'") {
			quoted = ch;
			continue;
		}
		if (ch === "#" && (i === 0 || code[i - 1] === "\n")) {
			const next = code.indexOf("\n", i);
			if (next < 0) return false;
			i = next;
			continue;
		}
		if (ch !== "(") continue;
		let next = i + 1;
		while (next < code.length && /\s/.test(code[next])) next++;
		if (/^scope\s*:/i.test(code.slice(next))) return true;
	}
	return false;
}

function usesPersistedPythonBinding(code: string, bindings: readonly QuailPythonBindingState[]): boolean {
	if (bindings.length === 0) return false;
	const names = new Set(bindings.map((binding) => binding.name));
	let quoted: string | undefined;
	let token = "";
	const flush = (): boolean => {
		if (!token) return false;
		const found = names.has(token);
		token = "";
		return found;
	};
	for (let i = 0; i < code.length; i++) {
		const ch = code[i];
		if (quoted) {
			if (ch === quoted && code[i - 1] !== "\\") quoted = undefined;
			continue;
		}
		if (ch === '"' || ch === "'") {
			if (flush()) return true;
			quoted = ch;
			continue;
		}
		if (/[A-Za-z0-9_]/.test(ch)) {
			token += ch;
			continue;
		}
		if (flush()) return true;
	}
	return flush();
}

function codeWithoutQuailRawCallBodies(code: string): string {
	const callNames = ["count", "retrieve", "g_save", "tag", "untag", "create_field", "group_expr", "count_by", "get", "save"];
	let out = "";
	let i = 0;
	let quoted: string | undefined;
	while (i < code.length) {
		const ch = code[i];
		if (quoted) {
			out += ch;
			if (ch === quoted && code[i - 1] !== "\\") quoted = undefined;
			i++;
			continue;
		}
		if (ch === '"' || ch === "'") {
			quoted = ch;
			out += ch;
			i++;
			continue;
		}
		const name = callNames.find((candidate) =>
			isBareQuailCallAt(code, i, candidate),
		);
		if (!name) {
			out += ch;
			i++;
			continue;
		}
		const afterName = i + name.length;
		const openIndex = afterName + code.slice(afterName).search(/\S/);
		const closeIndex = findMatchingClose(code, openIndex, "(", ")");
		if (closeIndex < 0) {
			out += ch;
			i++;
			continue;
		}
		out += code.slice(i, openIndex + 1);
		out += " ".repeat(Math.max(0, closeIndex - openIndex - 1));
		out += ")";
		i = closeIndex + 1;
	}
	return out;
}

function isBareQuailCallAt(text: string, index: number, name: string): boolean {
	if (!text.startsWith(name, index)) return false;
	const previous = text[index - 1] ?? "";
	if (/[A-Za-z0-9_]/.test(previous)) return false;
	for (let i = index - 1; i >= 0; i--) {
		if (/\s/.test(text[i])) continue;
		if (text[i] === ".") return false;
		break;
	}
	return (
		!/[A-Za-z0-9_]/.test(text[index + name.length] ?? "") &&
		text.slice(index + name.length).trimStart().startsWith("(")
	);
}

function transpileQuailPython(code: string, ctx: RuntimeContext): string {
	const knownGroupNames = getKnownGroupVariableNames(ctx);
	return logicalLines(code)
		.map(({ text }) => {
			if (text.trim().length === 0 || text.trimStart().startsWith("#")) return text;
			const indent = text.match(/^[ \t]*/)?.[0] ?? "";
			const trimmed = text.trim();
			const trailingColon = trimmed.endsWith(":") ? ":" : "";
			const body = trailingColon ? trimmed.slice(0, -1).trimEnd() : trimmed;
			const varMatch = body.match(/^var\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)$/);
			const source = varMatch ? `${varMatch[1]} = ${varMatch[2]}` : body;
			const statement = transformQuailStatement(source, knownGroupNames, ctx);
			return `${indent}${transformPythonLiterals(statement)}${trailingColon}`;
		})
		.join("\n");
}

function extractPythonSourceBindings(code: string): QuailPythonBindingState[] {
	const bindings: QuailPythonBindingState[] = [];
	const lines = code.split(/\r?\n/);
	let pendingDecoratorStart: number | undefined;
	for (let index = 0; index < lines.length;) {
		const line = lines[index];
		const indent = line.match(/^[ \t]*/)?.[0].replace(/\t/g, "    ").length ?? 0;
		const trimmed = line.trim();
		if (!trimmed) {
			index++;
			continue;
		}
		if (indent === 0 && trimmed.startsWith("@")) {
			pendingDecoratorStart ??= index;
			index++;
			continue;
		}
		const match = indent === 0
			? trimmed.match(/^(?:(async)\s+)?(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)\b/)
			: undefined;
		if (!match) {
			pendingDecoratorStart = undefined;
			index++;
			continue;
		}
		let end = index + 1;
		while (end < lines.length) {
			const next = lines[end];
			const nextTrimmed = next.trim();
			if (!nextTrimmed) {
				end++;
				continue;
			}
			const nextIndent = next.match(/^[ \t]*/)?.[0].replace(/\t/g, "    ").length ?? 0;
			if (nextIndent <= indent) break;
			end++;
		}
		const start = pendingDecoratorStart ?? index;
		bindings.push({
			name: match[3],
			kind: match[2] === "class" ? "class" : "function",
			source: lines.slice(start, end).join("\n"),
		});
		pendingDecoratorStart = undefined;
		index = end;
	}
	return bindings;
}

function transformQuailStatement(text: string, knownGroupNames: Set<string>, ctx: RuntimeContext): string {
	const assignment = text.match(/^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$/s);
	if (assignment) {
		const [, name, rhs] = assignment;
		const gSaveArg = parseGSaveCallExpression(rhs);
		if (gSaveArg !== undefined) {
			knownGroupNames.add(name);
			return `${name} = __quail_g_save_as(${JSON.stringify(name)}, ${JSON.stringify(gSaveArg)}, locals())`;
		}
		const groupExpressionText = assignableGroupExpressionText(rhs, { knownGroupNames, ctx });
		if (groupExpressionText) {
			knownGroupNames.add(name);
			return `${name} = __quail_group_expr(${JSON.stringify(groupExpressionText)}, locals())`;
		}
		if (assignmentProducesGroupValue(rhs, { knownGroupNames, ctx })) knownGroupNames.add(name);
		else knownGroupNames.delete(name);
	}
	const subtractAssignment = text.match(/^([A-Za-z_][A-Za-z0-9_]*)\s*-=\s*(.+)$/s);
	if (subtractAssignment) {
		const [, name, rhs] = subtractAssignment;
		return quoteBareEntryIds(
			transformQuailGroupLiterals(transformSpecialPythonCalls(`${name} = __quail_subtract(${name}, ${rhs})`)),
			ctx,
		);
	}
	for (const name of ["tag", "untag", "create_field"]) {
		if (!text.startsWith(`${name}(`) || !text.endsWith(")")) continue;
		const closeIndex = findMatchingClose(text, name.length, "(", ")");
		if (closeIndex === text.length - 1) {
			const inner = text.slice(name.length + 1, -1).trim();
			return `__quail_${name}(${JSON.stringify(inner)}, locals())`;
		}
	}
	return quoteBareEntryIds(transformQuailGroupLiterals(transformSpecialPythonCalls(text)), ctx);
}

interface GroupExpressionRecognitionOptions {
	knownGroupNames?: ReadonlySet<string>;
	ctx?: RuntimeContext;
}

function getKnownGroupVariableNames(ctx: RuntimeContext): Set<string> {
	const names = new Set<string>();
	for (const [name, value] of Object.entries(ctx.variables)) {
		if (valueCanResolveToGroupExpression(value, ctx)) names.add(name);
	}
	return names;
}

function valueCanResolveToGroupExpression(value: unknown, ctx: RuntimeContext): boolean {
	if (isGroupExpressionValue(value)) return true;
	if (typeof value === "string") {
		const text = trimOuterParens(value.trim());
		return text === "G0" || text === "G1" || /^G\d+$/.test(text) || Boolean(ctx.state.groups[text]);
	}
	if (!value || typeof value !== "object" || Array.isArray(value)) return false;
	const record = value as Record<string, unknown>;
	return (record.scope === "entries" || record.scope === "fields") &&
		(Array.isArray(record.members) || Array.isArray(record.entryIds) || Array.isArray(record.fieldNames));
}

function assignableGroupExpressionText(
	expr: string,
	options: GroupExpressionRecognitionOptions = {},
): string | undefined {
	const text = normalizeGroupExpressionText(expr);
	if (!text) return undefined;
	if (isRawGroupExpressionText(text)) return text;
	if (isDirectGroupReferenceLiteralText(text, options)) return text;
	if (/^temp\s*\(/i.test(text)) return text;
	if (isComposedGroupExpressionText(text, options)) return text;
	return undefined;
}

function assignmentProducesGroupValue(expr: string, options: GroupExpressionRecognitionOptions): boolean {
	const text = normalizeGroupExpressionText(expr);
	return Boolean(assignableGroupExpressionText(text, options)) ||
		isKnownGroupReferenceText(text, options) ||
		/^temp\s*\(/i.test(text) ||
		/^group_expr\s*\(/i.test(text) ||
		/^g_save\s*\(/i.test(text);
}

function parseGSaveCallExpression(expr: string): string | undefined {
	const trimmed = expr.trim();
	if (!/^g_save\s*\(/i.test(trimmed)) return undefined;
	const openIndex = trimmed.indexOf("(");
	if (openIndex < 0) return undefined;
	const closeIndex = findMatchingClose(trimmed, openIndex, "(", ")");
	if (closeIndex !== trimmed.length - 1) return undefined;
	return trimmed.slice(openIndex + 1, closeIndex).trim();
}

function isComposedGroupExpressionText(text: string, options: GroupExpressionRecognitionOptions): boolean {
	const normalized = normalizeGroupExpressionText(text);
	if (!normalized) return false;
	if (normalized.startsWith("not ")) {
		const rest = normalized.slice(4).trim();
		return hasGroupExpressionAnchor(rest, options) && isPlausibleGroupExpressionPart(rest, options);
	}
	const orParts = splitTopLevelByWord(normalized, "or");
	if (orParts.length > 1) return areGroupBooleanParts(orParts, options);
	const andParts = splitTopLevelByWord(normalized, "and");
	if (andParts.length > 1) return areGroupBooleanParts(andParts, options);
	return false;
}

function areGroupBooleanParts(parts: string[], options: GroupExpressionRecognitionOptions): boolean {
	return parts.some((part) => hasGroupExpressionAnchor(part, options)) &&
		parts.every((part) => isPlausibleGroupExpressionPart(part, options));
}

function isPlausibleGroupExpressionPart(text: string, options: GroupExpressionRecognitionOptions): boolean {
	const normalized = normalizeGroupExpressionText(text);
	if (!normalized || isClearlyScalarExpressionText(normalized)) return false;
	if (hasDirectGroupExpressionAnchor(normalized, options)) return true;
	if (normalized.startsWith("not ")) return isPlausibleGroupExpressionPart(normalized.slice(4), options);
	if (isComposedGroupExpressionText(normalized, options)) return true;
	if (isListLiteralExpression(normalized)) return true;
	return /^[A-Za-z_][A-Za-z0-9_]*$/.test(normalized);
}

function hasGroupExpressionAnchor(text: string, options: GroupExpressionRecognitionOptions): boolean {
	const normalized = normalizeGroupExpressionText(text);
	if (hasDirectGroupExpressionAnchor(normalized, options)) return true;
	if (normalized.startsWith("not ")) return hasGroupExpressionAnchor(normalized.slice(4), options);
	const orParts = splitTopLevelByWord(normalized, "or");
	if (orParts.length > 1) return orParts.some((part) => hasGroupExpressionAnchor(part, options));
	const andParts = splitTopLevelByWord(normalized, "and");
	return andParts.length > 1 && andParts.some((part) => hasGroupExpressionAnchor(part, options));
}

function hasDirectGroupExpressionAnchor(text: string, options: GroupExpressionRecognitionOptions): boolean {
	const normalized = normalizeGroupExpressionText(text);
	if (isRawGroupExpressionText(normalized)) return true;
	if (isDirectGroupReferenceLiteralText(normalized, options)) return true;
	if (isKnownGroupReferenceText(normalized, options)) return true;
	if (/^(temp|group|group_expr)\s*\(/i.test(normalized)) return true;
	if (looksLikeGroupSpec(normalized)) return true;
	const ctx = options.ctx;
	return Boolean(ctx && (ctx.entriesById.has(normalized) || getAllFieldNames(ctx).includes(normalized)));
}

function isDirectGroupReferenceLiteralText(text: string, options: GroupExpressionRecognitionOptions): boolean {
	const normalized = normalizeGroupExpressionText(text);
	if (normalized === "G0" || normalized === "G1" || /^G\d+$/.test(normalized)) return true;
	return Boolean(options.ctx?.state.groups[normalized]);
}

function isKnownGroupReferenceText(text: string, options: GroupExpressionRecognitionOptions): boolean {
	const normalized = normalizeGroupExpressionText(text);
	if (normalized === "G0" || normalized === "G1" || /^G\d+$/.test(normalized)) return true;
	if (options.knownGroupNames?.has(normalized)) return true;
	const ctx = options.ctx;
	if (!ctx) return false;
	if (ctx.state.groups[normalized]) return true;
	return Object.prototype.hasOwnProperty.call(ctx.variables, normalized) &&
		valueCanResolveToGroupExpression(ctx.variables[normalized], ctx);
}

function isClearlyScalarExpressionText(text: string): boolean {
	const normalized = normalizeGroupExpressionText(text);
	return isStringLiteral(normalized) ||
		isRawStringLiteral(normalized) ||
		normalized === "true" ||
		normalized === "false" ||
		normalized === "null" ||
		/^-?\d+(\.\d+)?$/.test(normalized);
}

function transformPythonLiterals(text: string): string {
	return replaceOutsideStrings(
		replaceOutsideStrings(
			replaceOutsideStrings(text, /\btrue\b/g, "True"),
			/\bfalse\b/g,
			"False",
		),
		/\bnull\b/g,
		"None",
	);
}

function replaceOutsideStrings(text: string, pattern: RegExp, replacement: string): string {
	let out = "";
	let start = 0;
	let quoted: string | undefined;
	for (let i = 0; i < text.length; i++) {
		const ch = text[i];
		if (quoted) {
			if (ch === quoted && text[i - 1] !== "\\") quoted = undefined;
			continue;
		}
		if (ch === '"' || ch === "'") {
			out += text.slice(start, i).replace(pattern, replacement);
			start = i;
			quoted = ch;
		}
	}
	out += text.slice(start).replace(pattern, replacement);
	return out;
}

function transformSpecialPythonCalls(text: string): string {
	const callNames: Record<string, string> = {
		count: "__quail_count",
		retrieve: "__quail_retrieve",
		g_save: "__quail_g_save",
		group_expr: "__quail_group_expr",
		count_by: "__quail_count_by",
		get: "__quail_get",
		save: "__quail_save",
	};
	let out = "";
	let i = 0;
	let quoted: string | undefined;
	while (i < text.length) {
		const ch = text[i];
		if (quoted) {
			out += ch;
			if (ch === quoted && text[i - 1] !== "\\") quoted = undefined;
			i++;
			continue;
		}
		if (ch === '"' || ch === "'") {
			quoted = ch;
			out += ch;
			i++;
			continue;
		}
		const name = Object.keys(callNames).find((candidate) =>
			isBareQuailCallAt(text, i, candidate),
		);
		if (!name) {
			out += ch;
			i++;
			continue;
		}
		const afterName = i + name.length;
		const openIndex = afterName + text.slice(afterName).search(/\S/);
		const closeIndex = findMatchingClose(text, openIndex, "(", ")");
		if (closeIndex < 0) {
			out += ch;
			i++;
			continue;
		}
		const rawArg = text.slice(openIndex + 1, closeIndex).trim();
		const accessor = name === "get" ? parseGetFieldAccessor(text, closeIndex + 1) : undefined;
		out += `${callNames[name]}(${JSON.stringify(rawArg)}, locals())${accessor ? `[${JSON.stringify(accessor.field)}]` : ""}`;
		i = accessor?.end ?? closeIndex + 1;
	}
	return out;
}

function parseGetFieldAccessor(text: string, start: number): { field: string; end: number } | undefined {
	let index = start;
	while (index < text.length && /\s/.test(text[index])) index++;
	if (text[index] !== "[") return undefined;
	const closeIndex = findMatchingClose(text, index, "[", "]");
	if (closeIndex < 0) return undefined;
	const inner = text.slice(index + 1, closeIndex).trim();
	if (!inner || isStringLiteral(inner) || /^-?\d+$/u.test(inner) || inner.includes(":")) return undefined;
	if (/[()[\]{},+\-*/%<>=!]/u.test(inner)) return undefined;
	return { field: inner, end: closeIndex + 1 };
}

function quoteBareEntryIds(text: string, ctx: RuntimeContext): string {
	let out = "";
	let i = 0;
	while (i < text.length) {
		const ch = text[i];
		if (ch === '"' || ch === "'") {
			const end = scanStringLiteralEnd(text, i);
			out += text.slice(i, end);
			i = end;
			continue;
		}
		if (ch === "#") {
			out += text.slice(i);
			break;
		}
		if (isBareEntryIdTokenChar(ch)) {
			const start = i;
			while (i < text.length && isBareEntryIdTokenChar(text[i])) i++;
			const token = text.slice(start, i);
			out += token.includes(":") && ctx.entriesById.has(token) ? JSON.stringify(token) : token;
			continue;
		}
		out += ch;
		i++;
	}
	return out;
}

function scanStringLiteralEnd(text: string, start: number): number {
	const quote = text[start];
	let index = start + 1;
	while (index < text.length) {
		const ch = text[index];
		if (ch === "\\" && index + 1 < text.length) {
			index += 2;
			continue;
		}
		index++;
		if (ch === quote) break;
	}
	return index;
}

function isBareEntryIdTokenChar(ch: string | undefined): boolean {
	return Boolean(ch && /[A-Za-z0-9_:-]/u.test(ch));
}

function transformQuailGroupLiterals(text: string): string {
	let out = "";
	let i = 0;
	let quoted: string | undefined;
	while (i < text.length) {
		const ch = text[i];
		if (quoted) {
			out += ch;
			if (ch === quoted && text[i - 1] !== "\\") quoted = undefined;
			i++;
			continue;
		}
		if (ch === '"' || ch === "'") {
			quoted = ch;
			out += ch;
			i++;
			continue;
		}
		if (ch === "#") {
			out += text.slice(i);
			break;
		}
		if (ch === "(") {
			let contentStart = i + 1;
			while (contentStart < text.length && /\s/.test(text[contentStart])) contentStart++;
			if (/^scope\s*:/i.test(text.slice(contentStart))) {
				const closeIndex = findMatchingClose(text, i, "(", ")");
				if (closeIndex >= 0) {
					const inner = text.slice(i + 1, closeIndex).trim();
					if (isRawGroupExpressionText(inner)) {
						out += `__quail_group_expr(${JSON.stringify(normalizeGroupExpressionText(inner))}, locals())`;
						i = closeIndex + 1;
						continue;
					}
				}
			}
		}
		out += ch;
		i++;
	}
	return out;
}

function buildPythonRuntimeScript(
	code: string,
	initialVariables: Record<string, unknown>,
	initialBindings: readonly QuailPythonBindingState[],
	currentBindings: readonly QuailPythonBindingState[],
): string {
	const bindingSources = initialBindings.map((binding) => binding.source).join("\n\n");
	const sourceBindingNames = [...new Set([
		...initialBindings.map((binding) => binding.name),
		...currentBindings.map((binding) => binding.name),
	])];
	return `
import json
import sys
import traceback
import builtins

REQUEST_PREFIX = ${JSON.stringify(PYTHON_REQUEST_PREFIX)}
DONE_PREFIX = ${JSON.stringify(PYTHON_DONE_PREFIX)}
ERROR_PREFIX = ${JSON.stringify(PYTHON_ERROR_PREFIX)}

def _jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return None

def _jsonable_locals(values):
    out = {}
    for key, value in values.items():
        if key.startswith("__") or key in _reserved_names:
            continue
        converted = _jsonable(value)
        if converted is not None or value is None:
            out[key] = converted
    return out

def _request(op, arg, local_values):
    payload = {"op": op, "arg": arg, "locals": _jsonable_locals(local_values)}
    sys.stdout.write(REQUEST_PREFIX + json.dumps(payload, separators=(",", ":")) + "\\n")
    sys.stdout.flush()
    line = sys.stdin.readline()
    if not line:
        raise RuntimeError("Quail runtime closed before responding to " + op)
    response = json.loads(line)
    if not response.get("ok"):
        raise RuntimeError(response.get("error") or "Quail runtime request failed")
    return response.get("value")

def __quail_count(arg, local_values):
    return _request("count", arg, local_values)

def __quail_retrieve(arg, local_values):
    return _request("retrieve", arg, local_values)

def __quail_g_save(arg, local_values):
    return _request("g_save", arg, local_values)

def __quail_g_save_as(name, arg, local_values):
    return _request("g_save", {"name": name, "expr": arg}, local_values)

def __quail_group_expr(arg, local_values=None):
    return {"__quailType": "group_expression", "expression": arg}

class _QuailObject(dict):
    def __getitem__(self, key):
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            if isinstance(key, str):
                tags = dict.get(self, "tags")
                if isinstance(tags, dict) and key in tags:
                    return tags[key]
                fields = dict.get(self, "fields")
                if isinstance(fields, dict) and key in fields:
                    return fields[key]
            raise

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

def _wrap_quail_value(value):
    if isinstance(value, dict):
        return _QuailObject({key: _wrap_quail_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_wrap_quail_value(item) for item in value]
    return value

def __quail_get(arg, local_values):
    return _wrap_quail_value(_request("get", arg, local_values))

def __quail_save(arg, local_values):
    return _wrap_quail_value(_request("save", arg, local_values))

def __quail_count_by(arg, local_values):
    return _request("count_by", arg, local_values)

def __quail_tag(arg, local_values):
    return _request("tag", arg, local_values)

def __quail_untag(arg, local_values):
    return _request("untag", arg, local_values)

def __quail_create_field(arg, local_values):
    return _request("create_field", arg, local_values)

def __quail_subtract(left, right):
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left - right
    if isinstance(left, list):
        return [item for item in left if item != right]
    raise TypeError("-= supports numbers and removing one value from a list")

def _quail_type(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return builtins.type(value).__name__

def type(value):
    return _quail_type(value)

def print(*args, sep=" ", end="\\n"):
    text = sep.join(_format_print_value(arg) for arg in args) + end
    _request("print", text, globals())

def _format_print_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(_jsonable(value), separators=(",", ":"))
    return str(value)

def _blocked_import(*args, **kwargs):
    raise ImportError("Imports are disabled in Quail Python code.")

def _blocked_input(*args, **kwargs):
    raise RuntimeError("input() is unavailable in Quail Python code.")

_actual_dir = builtins.dir
_actual_globals = globals

def dir(obj=None):
    if obj is not None:
        return _actual_dir(obj)
    return sorted(
        key for key in _actual_globals().keys()
        if not key.startswith("_") and key not in _reserved_names
    )

def _classify_source_binding(value):
    if value is _missing:
        return "missing"
    if isinstance(value, builtins.type):
        return "class"
    if callable(value) and hasattr(value, "__code__"):
        return "function"
    return "other"

_python_builtins = builtins.__dict__.copy()
_python_builtins["__import__"] = _blocked_import
_python_builtins["input"] = _blocked_input
_missing = object()

_reserved_names = set([
    "_jsonable", "_jsonable_locals", "_request", "_format_print_value",
    "__quail_count", "__quail_retrieve", "__quail_g_save", "__quail_g_save_as", "__quail_group_expr", "__quail_get", "__quail_save",
    "__quail_count_by", "_QuailObject", "_wrap_quail_value", "__quail_tag",
    "__quail_untag", "__quail_create_field", "__quail_subtract", "_quail_type", "type",
    "print", "json", "sys", "builtins",
    "traceback", "REQUEST_PREFIX", "DONE_PREFIX", "ERROR_PREFIX",
    "_blocked_import", "_blocked_input", "_classify_source_binding", "_python_builtins",
    "_missing", "_reserved_names", "_initial_variables", "_source_binding_names",
    "_source_bindings", "_actual_dir", "_actual_globals", "dir",
])

_initial_variables = json.loads(${JSON.stringify(JSON.stringify(initialVariables))})
_source_binding_names = json.loads(${JSON.stringify(JSON.stringify(sourceBindingNames))})
globals().update(_initial_variables)
globals()["__name__"] = "__quail_python__"
globals()["__builtins__"] = _python_builtins

try:
    exec(${JSON.stringify(bindingSources)}, globals(), globals())
    exec(${JSON.stringify(code)}, globals(), globals())
    variables = _jsonable_locals(globals())
    _source_bindings = {
        name: _classify_source_binding(globals().get(name, _missing))
        for name in _source_binding_names
    }
    sys.stdout.write(DONE_PREFIX + json.dumps({
        "variables": variables,
        "sourceBindings": _source_bindings,
    }, separators=(",", ":")) + "\\n")
    sys.stdout.flush()
except BaseException as error:
    sys.stdout.write(ERROR_PREFIX + json.dumps({
        "error": str(error),
        "traceback": traceback.format_exc(limit=5),
    }, separators=(",", ":")) + "\\n")
    sys.stdout.flush()
`;
}

async function executePythonProgram(code: string, ctx: RuntimeContext): Promise<void> {
	if (/^\s*(import|from)\b/m.test(code)) {
		ctx.errors.push({ code: "E_IMPORT_DISABLED", message: "Imports are disabled in Quail Python code; use core Python logic and Quail DSL commands only." });
		return;
	}
	const transpiled = transpileQuailPython(code, ctx);
	const initialBindings = ctx.state.pythonBindings ?? [];
	const currentBindings = extractPythonSourceBindings(transpiled);
	const sourceByName = new Map<string, QuailPythonBindingState>();
	for (const binding of initialBindings) sourceByName.set(binding.name, binding);
	for (const binding of currentBindings) sourceByName.set(binding.name, binding);
	const script = buildPythonRuntimeScript(transpiled, ctx.variables, initialBindings, currentBindings);
	const scriptDir = mkdtempSync(join(tmpdir(), "quail-python-"));
	const scriptPath = join(scriptDir, "runtime.py");
	writeFileSync(scriptPath, script, "utf8");
	const python = spawn(process.env.QUAIL_PYTHON ?? process.env.PYTHON ?? "python3", [scriptPath], {
		stdio: ["pipe", "pipe", "pipe"],
	});
	let stdoutBuffer = "";
	let stderr = "";
	let completed = false;
	let childError: Error | undefined;
	let finalVariables: Record<string, unknown> | undefined;
	let finalSourceBindings: Record<string, "function" | "class" | "other" | "missing"> | undefined;
	let requestChain = Promise.resolve();

	const writeResponse = (payload: unknown): void => {
		python.stdin.write(`${JSON.stringify(payload)}\n`);
	};

	const handleRequest = async (payload: unknown): Promise<void> => {
		if (!payload || typeof payload !== "object") {
			writeResponse({ ok: false, error: "Invalid Python runtime request" });
			return;
		}
		const requestPayload = payload as { op?: unknown; arg?: unknown; locals?: unknown };
		const op = String(requestPayload.op ?? "");
		const rawArg = requestPayload.arg;
		const arg = String(rawArg ?? "");
		if (requestPayload.locals && typeof requestPayload.locals === "object") {
			for (const [key, value] of Object.entries(requestPayload.locals as Record<string, unknown>)) {
					ctx.variables[key] = value;
			}
			clearGroupExpressionCache(ctx);
		}
		try {
			let value: unknown;
			if (op === "count") value = await countUnits(arg, ctx, 1);
				else if (op === "retrieve") value = await retrieveUnits(arg, ctx, 1);
				else if (op === "g_save") {
					const request = parseGSaveRequest(rawArg, 1);
					value = await gSaveGroup(request.name, request.expr, ctx, 1);
				}
				else if (op === "get") value = await getValue(arg, ctx, 1);
				else if (op === "save") value = await saveVariable(arg, ctx, 1);
				else if (op === "count_by") value = await countBy(arg, ctx, 1);
			else if (op === "tag") {
				await executeTag(`tag(${arg})`, ctx, 1);
				value = null;
			} else if (op === "untag") {
				await executeUntag(`untag(${arg})`, ctx, 1);
				value = null;
			} else if (op === "create_field") {
				await executeCreateField(`create_field(${arg})`, ctx, 1);
				value = null;
			} else if (op === "print") {
				pushOutput(ctx, arg.replace(/\n$/, ""));
				value = null;
			} else {
				throw new DslRuntimeError("E_PYTHON_REQUEST", `Unknown Python runtime request ${op}`, 1);
			}
			writeResponse({ ok: true, value });
		} catch (error) {
			writeResponse({ ok: false, error: error instanceof Error ? error.message : String(error) });
		}
	};

	const processStdoutLine = (line: string): void => {
		if (line.startsWith(PYTHON_REQUEST_PREFIX)) {
			const body = line.slice(PYTHON_REQUEST_PREFIX.length);
			requestChain = requestChain.then(() => handleRequest(JSON.parse(body)));
			return;
		}
		if (line.startsWith(PYTHON_DONE_PREFIX)) {
			completed = true;
			const body = JSON.parse(line.slice(PYTHON_DONE_PREFIX.length)) as {
				variables?: Record<string, unknown>;
				sourceBindings?: Record<string, "function" | "class" | "other" | "missing">;
			};
			finalVariables = body.variables ?? {};
			finalSourceBindings = body.sourceBindings ?? {};
			return;
		}
		if (line.startsWith(PYTHON_ERROR_PREFIX)) {
			completed = true;
			const body = JSON.parse(line.slice(PYTHON_ERROR_PREFIX.length)) as { error?: string; traceback?: string };
			ctx.errors.push({
				code: "E_PYTHON_RUNTIME",
				message: `${body.error ?? "Python runtime error"}${body.traceback ? `\n${body.traceback}` : ""}`,
			});
			return;
		}
		if (line.trim().length > 0) pushOutput(ctx, line);
	};

	python.stdout.on("data", (chunk: Buffer) => {
		stdoutBuffer += chunk.toString("utf8");
		let newlineIndex = stdoutBuffer.indexOf("\n");
		while (newlineIndex >= 0) {
			const line = stdoutBuffer.slice(0, newlineIndex);
			stdoutBuffer = stdoutBuffer.slice(newlineIndex + 1);
			processStdoutLine(line);
			newlineIndex = stdoutBuffer.indexOf("\n");
		}
	});
	python.stderr.on("data", (chunk: Buffer) => {
		stderr += chunk.toString("utf8");
	});
	python.on("error", (error) => {
		childError = error;
	});

		await new Promise<void>((resolve) => {
			python.on("close", () => resolve());
		});
		rmSync(scriptDir, { recursive: true, force: true });
		await requestChain;
	if (stdoutBuffer.trim()) processStdoutLine(stdoutBuffer.trim());
	if (finalVariables) {
			for (const [key, value] of Object.entries(finalVariables)) ctx.variables[key] = value;
	}
	if (finalSourceBindings) {
		const nextBindings: QuailPythonBindingState[] = [];
		for (const [name, kind] of Object.entries(finalSourceBindings)) {
			if (kind !== "function" && kind !== "class") continue;
			const binding = sourceByName.get(name);
			if (binding) nextBindings.push({ ...binding, kind });
		}
		ctx.state.pythonBindings = nextBindings;
	}
	if (childError) {
		ctx.errors.push({ code: "E_PYTHON_UNAVAILABLE", message: childError.message });
		return;
	}
	if (!completed && stderr.trim()) {
		ctx.errors.push({ code: "E_PYTHON_RUNTIME", message: stderr.trim() });
	} else if (!completed) {
		ctx.errors.push({ code: "E_PYTHON_RUNTIME", message: "Python runtime exited before completing execution." });
	}
}

async function executeProgram(code: string, ctx: RuntimeContext): Promise<void> {
	validateBalancedDelimiters(code);
	if (shouldUsePythonRuntime(code, ctx)) {
		await executePythonProgram(code, ctx);
		return;
	}
	const nodes = parseProgram(code);
	for (const node of nodes) await executeNode(node, ctx);
}

async function executeNode(node: LineNode, ctx: RuntimeContext): Promise<void> {
	const text = stripTrailingColon(node.text);
	try {
		if (text.startsWith("for ")) {
			const match = text.match(/^for\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+in\s+(.+)$/);
			if (!match) throw new DslRuntimeError("E_PARSE_FOR", "Expected for <name> in <expression>", node.line);
			const values = await evaluateExpression(match[2], ctx, node.line);
			if (!Array.isArray(values)) throw new DslRuntimeError("E_FOR_NON_LIST", "for loop expression must return a list", node.line);
			for (const value of values) {
				ctx.variables[match[1]] = value;
				clearGroupExpressionCache(ctx);
				for (const child of node.children) await executeNode(child, ctx);
			}
			return;
		}
		if (text.startsWith("if ")) {
			const condition = text.slice(3).trim();
			const value = await evaluateCondition(condition, ctx, node.line);
			const branch = value ? node.children : (node.elseChildren ?? []);
			for (const child of branch) await executeNode(child, ctx);
			return;
		}
		const varMatch = text.match(/^var\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)$/);
		if (varMatch) {
			ctx.variables[varMatch[1]] = await evaluateAssignmentExpression(varMatch[2], ctx, node.line, varMatch[1]);
			clearGroupExpressionCache(ctx);
			return;
		}
		const assignMatch = text.match(/^([a-zA-Z_][a-zA-Z0-9_]*)\s*(\+=|-=|=)\s*(.+)$/);
		if (assignMatch) {
			const current = ctx.variables[assignMatch[1]];
			const value = await evaluateAssignmentExpression(assignMatch[3], ctx, node.line, assignMatch[2] === "=" ? assignMatch[1] : undefined);
			if (assignMatch[2] === "=") ctx.variables[assignMatch[1]] = value;
			else if (assignMatch[2] === "+=") ctx.variables[assignMatch[1]] = addValues(current, value);
			else ctx.variables[assignMatch[1]] = subtractValues(current, value);
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
		if (text.startsWith("create_field(")) {
			await executeCreateField(text, ctx, node.line);
			return;
		}
		await evaluateExpression(text, ctx, node.line);
	} catch (error) {
		if (error instanceof DslRuntimeError) ctx.errors.push({ code: error.code, message: error.message, line: error.line ?? node.line });
		else ctx.errors.push({ code: "E_RUNTIME", message: error instanceof Error ? error.message : String(error), line: node.line });
	}
}

async function evaluateAssignmentExpression(expr: string, ctx: RuntimeContext, line: number, targetName?: string): Promise<unknown> {
	const gSaveArg = parseGSaveCallExpression(expr);
	if (gSaveArg !== undefined) {
		if (!targetName) throw gSaveAssignmentError(line);
		return gSaveGroup(targetName, gSaveArg, ctx, line);
	}
	const groupExpressionText = assignableGroupExpressionText(expr, { ctx });
	if (groupExpressionText) return createGroupExpression(groupExpressionText);
	return evaluateExpression(expr, ctx, line);
}

function isRawGroupExpressionText(expr: string): boolean {
	return /^scope\s*:/i.test(normalizeGroupExpressionText(expr));
}

function normalizeGroupExpressionText(expr: string): string {
	return trimOuterParens(expr.trim());
}

function stripTrailingColon(text: string): string {
	return text.endsWith(":") ? text.slice(0, -1).trimEnd() : text;
}

function hasUnquotedSemicolon(text: string): boolean {
	let quoted: string | undefined;
	for (let i = 0; i < text.length; i++) {
		const ch = text[i];
		if (quoted) {
			if (ch === quoted && text[i - 1] !== "\\") quoted = undefined;
			continue;
		}
		if (ch === '"' || ch === "'") {
			quoted = ch;
			continue;
		}
		if (ch === ";") return true;
	}
	return false;
}

function pushOutput(ctx: RuntimeContext, text: string): void {
	flushTagMutationSummary(ctx);
	ctx.outputs.push(text);
}

function flushTagMutationSummary(ctx: RuntimeContext): void {
	if (ctx.tagMutations.length === 0) return;
	const tags = ctx.tagMutations.filter((mutation) => mutation.type === "tag");
	const untags = ctx.tagMutations.filter((mutation) => mutation.type === "untag");
	if (tags.length > 0) ctx.outputs.push(formatTagMutationSummary("tag", tags));
	if (untags.length > 0) ctx.outputs.push(formatTagMutationSummary("untag", untags));
	ctx.tagMutations = [];
}

function clearGroupExpressionCache(ctx: RuntimeContext): void {
	ctx.groupExpressionCache.clear();
	ctx.scopedGroupExpressionCache.clear();
	ctx.similarityTargetsCache.clear();
	ctx.unitValuesCache.clear();
}

function cloneJsonableRecord(value: Record<string, unknown>): Record<string, unknown> {
	return JSON.parse(JSON.stringify(value)) as Record<string, unknown>;
}

function maxSavedVariableBytes(): number {
	const value = process.env.QUAIL_MAX_SAVED_VARIABLE_BYTES?.trim();
	if (!value) return DEFAULT_MAX_SAVED_VARIABLE_BYTES;
	const parsed = Number.parseInt(value, 10);
	return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_MAX_SAVED_VARIABLE_BYTES;
}

function parseSaveVariableName(expr: string, line: number): string {
	const trimmed = expr.trim();
	if (!trimmed) {
		throw new DslRuntimeError("E_PARSE_SAVE", "Expected save(<variable>) with a variable name", line);
	}
	const name = isStringLiteral(trimmed) ? parseStringLiteral(trimmed, line) : trimmed;
	if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) {
		throw new DslRuntimeError("E_PARSE_SAVE", `Expected save(<variable>) with a variable name, got ${expr}`, line);
	}
	return name;
}

function normalizeSavedVariableValue(value: unknown, line: number, path: string): unknown {
	if (value === undefined) {
		throw new DslRuntimeError("E_SAVE_VALUE", `Cannot save ${path} because its value is undefined`, line);
	}
	if (value === null || typeof value === "string" || typeof value === "boolean") return value;
	if (typeof value === "number") {
		if (!Number.isFinite(value)) throw new DslRuntimeError("E_SAVE_VALUE", `Cannot save ${path} because numbers must be finite`, line);
		return value;
	}
	if (Array.isArray(value)) {
		return value.map((item, index) => normalizeSavedVariableValue(item, line, `${path}[${index}]`));
	}
	if (typeof value === "object") {
		return Object.fromEntries(
			Object.entries(value as Record<string, unknown>).map(([key, item]) => [
				key,
				normalizeSavedVariableValue(item, line, `${path}.${key}`),
			]),
		);
	}
	throw new DslRuntimeError("E_SAVE_VALUE", `Cannot save ${path}; save() supports JSON-like values only`, line);
}

async function saveVariable(expr: string, ctx: RuntimeContext, line: number): Promise<unknown> {
	const name = parseSaveVariableName(expr, line);
	if (!(name in ctx.variables)) {
		throw new DslRuntimeError("E_SAVE_VARIABLE", `Cannot save ${name}; no Python/DSL variable named ${name} exists in this execution`, line);
	}
	const value = normalizeSavedVariableValue(ctx.variables[name], line, name);
	const serialized = JSON.stringify(value);
	const size = Buffer.byteLength(serialized, "utf8");
	const maxSize = maxSavedVariableBytes();
	if (size > maxSize) {
		throw new DslRuntimeError(
			"E_SAVE_VALUE",
			`Cannot save ${name}; serialized value is ${size} bytes, above the ${maxSize} byte save() limit. Save compact JSON-like summaries only; use g_save(GROUP-EXPR) for groups or recompute large row/id lists within one execution.`,
			line,
		);
	}
	ctx.variables[name] = value;
	ctx.state.variables[name] = JSON.parse(serialized) as unknown;
	return value;
}

function hasSideEffectfulGroupCall(text: string): boolean {
	return /(^|[^A-Za-z0-9_])(group|g_save|save)\s*\(/.test(text);
}

function isCacheableGroupExpressionText(text: string, ctx: RuntimeContext, seen = new Set<string>()): boolean {
	const trimmed = trimOuterParens(text.trim());
	if (seen.has(trimmed)) return true;
	seen.add(trimmed);

	const value = ctx.variables[trimmed];
	if (isGroupExpressionValue(value)) {
		return isCacheableGroupExpressionText(value.expression, ctx, seen);
	}
	if (typeof value === "string" && value !== trimmed) {
		return isCacheableGroupExpressionText(value, ctx, seen);
	}

	return !hasSideEffectfulGroupCall(trimmed);
}

function formatTagMutationSummary(type: TagMutation["type"], mutations: TagMutation[]): string {
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

function addValues(a: unknown, b: unknown): unknown {
	if (typeof a === "number" && typeof b === "number") return a + b;
	if (typeof a === "string" || typeof b === "string") return `${a ?? ""}${b ?? ""}`;
	if (Array.isArray(a) && Array.isArray(b)) return [...a, ...b];
	throw new DslRuntimeError("E_BAD_ADD", "+= supports numbers, strings, and lists");
}

function subtractValues(a: unknown, b: unknown): unknown {
	if (typeof a === "number" && typeof b === "number") return a - b;
	if (Array.isArray(a)) return a.filter((value) => value !== b);
	throw new DslRuntimeError("E_BAD_SUBTRACT", "-= supports numbers and removing one value from a list");
}

async function executeCreateField(text: string, ctx: RuntimeContext, line: number): Promise<void> {
	const match = text.match(/^create_field\((.+)\)$/);
	if (!match) throw new DslRuntimeError("E_PARSE_CREATE_FIELD", "Expected create_field(FIELD)", line);
	const field = await evaluateFieldName(match[1], ctx, line);
	if (!field) throw new DslRuntimeError("E_CREATE_FIELD", "create_field requires a non-empty field name", line);
	const created = ctx.state.createdFields ?? [];
	if (!created.includes(field)) created.push(field);
	ctx.state.createdFields = created;
}

async function executeTag(text: string, ctx: RuntimeContext, line: number): Promise<void> {
	const match = text.match(/^tag\((.+?)\s+with\s+(.+?)\s+(set\s+to|add)\s+(.+)\)$/);
	if (!match) throw new DslRuntimeError("E_PARSE_TAG", "Expected tag(GROUP-EXPR with FIELD set to TAG)", line);
	const targetText = match[1].trim();
	const field = await evaluateFieldName(match[2], ctx, line);
	const op = match[3].toLowerCase() === "add" ? "add" : "set";
	const values = normalizeTagValues(await evaluateTagValueExpression(match[4], ctx, line));
	if (values.length === 0) throw new DslRuntimeError("E_TAG_VALUE", "tag requires at least one tag value", line);
	const ids = await resolveTagTargetIds(targetText, ctx, line);
	for (const id of ids) {
		const entryTags = ctx.state.tagsByEntry[id] ?? {};
		if (op === "set") {
			entryTags[field] = normalizeStoredTagValue(values);
		} else {
			const current = tagValueToList(getEntryTags(ctx.entriesById.get(id)!, ctx)[field]);
			entryTags[field] = normalizeStoredTagValue(uniqueValues([...current, ...values]));
		}
		ctx.state.tagsByEntry[id] = entryTags;
		ctx.tagMutations.push({ type: "tag", id, field, valueCount: values.length });
	}
	if (!ctx.state.createdFields?.includes(field)) {
		ctx.state.createdFields = [...(ctx.state.createdFields ?? []), field];
	}
	ctx.tagIndex = undefined;
	clearGroupExpressionCache(ctx);
}

async function resolveTagTargetIds(targetText: string, ctx: RuntimeContext, line: number): Promise<string[]> {
	const target = resolveBareOrString(targetText, ctx);
	if (typeof target === "string" && ctx.entriesById.has(target)) return [target];
	if (Array.isArray(target)) {
		const group = inferScopedMembers(target, ctx, targetText, line);
		if (group.scope !== "entries") throw new DslRuntimeError("E_TAG_SCOPE", "tag() requires an entry-scoped group expression", line);
		return group.members;
	}
	const expression = isGroupExpressionValue(target) ? target.expression : String(target);
	const group = await resolveScopedGroupExpression(expression, ctx, line);
	if (group.scope !== "entries") throw new DslRuntimeError("E_TAG_SCOPE", "tag() requires an entry-scoped group expression", line);
	return group.members;
}

async function executeUntag(text: string, ctx: RuntimeContext, line: number): Promise<void> {
	const removeMatch = text.match(/^untag\((.+?)\s+with\s+(.+?)\s+remove\s+(.+)\)$/);
	if (removeMatch) {
		const id = coerceId(resolveBareOrString(removeMatch[1], ctx));
		const field = await evaluateFieldName(removeMatch[2], ctx, line);
		const values = normalizeTagValues(await evaluateTagValueExpression(removeMatch[3], ctx, line));
		if (values.length === 0) throw new DslRuntimeError("E_TAG_VALUE", "untag remove requires at least one tag value", line);
		if (!ctx.entriesById.has(id)) throw new DslRuntimeError("E_UNKNOWN_ID", `Unknown entry id ${id}`, line);
		const current = tagValueToList(getEntryTags(ctx.entriesById.get(id)!, ctx)[field]);
		const remaining = current.filter((value) => !values.some((candidate) => stableValueKey(candidate) === stableValueKey(value)));
		const entryTags = ctx.state.tagsByEntry[id] ?? {};
		if (remaining.length === 0) delete entryTags[field];
		else entryTags[field] = normalizeStoredTagValue(remaining);
		ctx.state.tagsByEntry[id] = entryTags;
		ctx.tagIndex = undefined;
		clearGroupExpressionCache(ctx);
		ctx.tagMutations.push({ type: "untag", id, field, valueCount: values.length });
		return;
	}

	const match = text.match(/^untag\((.+?)\s+from\s+(.+)\)$/);
	if (!match) throw new DslRuntimeError("E_PARSE_UNTAG", "Expected untag(FIELD from GROUP-EXPR)", line);
	const field = await evaluateFieldName(match[1], ctx, line);
	const group = await resolveScopedGroupExpression(match[2], ctx, line);
	if (group.scope !== "entries") throw new DslRuntimeError("E_UNTAG_SCOPE", "untag() requires an entry-scoped group expression", line);
	for (const id of group.members) {
		if (ctx.state.tagsByEntry[id]) delete ctx.state.tagsByEntry[id][field];
		ctx.tagMutations.push({ type: "untag", id, field, valueCount: 1, wholeField: true });
	}
	ctx.tagIndex = undefined;
	clearGroupExpressionCache(ctx);
}

async function evaluateCondition(expr: string, ctx: RuntimeContext, line: number): Promise<boolean> {
	const orParts = splitTopLevelByWord(expr, "or");
	if (orParts.length > 1) {
		for (const part of orParts) if (await evaluateCondition(part, ctx, line)) return true;
		return false;
	}
	const andParts = splitTopLevelByWord(expr, "and");
	if (andParts.length > 1) {
		for (const part of andParts) if (!await evaluateCondition(part, ctx, line)) return false;
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
		if (!parts) continue;
		const left = await evaluateExpression(parts[0], ctx, line);
		const right = await evaluateExpression(parts[2], ctx, line);
		return compareClauseValue(left, op, right);
	}
	return Boolean(await evaluateExpression(expr, ctx, line));
}

async function evaluateExpression(expr: string, ctx: RuntimeContext, line: number): Promise<unknown> {
	let trimmed = expr.trim();
	trimmed = trimOuterParens(trimmed);
	if (isListLiteralExpression(trimmed)) return parseList(trimmed, ctx, line);
	if (ctx.entriesById.has(trimmed)) return trimmed;
	const arithmetic = splitArithmetic(trimmed, ["+", "-"]) ?? splitArithmetic(trimmed, ["*", "/"]);
	if (arithmetic) return evaluateArithmetic(arithmetic, ctx, line);
	if (trimmed.startsWith("+") && trimmed.length > 1) return toNumber(await evaluateExpression(trimmed.slice(1), ctx, line), "+", line);
	if (trimmed.startsWith("-") && trimmed.length > 1 && !/^-?\d+(\.\d+)?$/.test(trimmed)) {
		return -toNumber(await evaluateExpression(trimmed.slice(1), ctx, line), "-", line);
	}
	if (trimmed === "true" || trimmed === "True") return true;
	if (trimmed === "false" || trimmed === "False") return false;
	if (trimmed === "null" || trimmed === "None") return null;
	if (/^-?\d+(\.\d+)?$/.test(trimmed)) return Number(trimmed);
	if (parseGSaveCallExpression(trimmed) !== undefined) throw gSaveAssignmentError(line);
	if (/^(?:entries|fields)\s*\[[^\]]+\]/i.test(trimmed)) throw parseExpressionError(expr, line);
	const index = splitIndex(trimmed);
	if (index) {
		const base = await evaluateExpression(index.base, ctx, line);
		const slice = splitSliceIndex(index.index);
		if (slice) return getSlice(base, await evaluateSliceBound(slice.start, ctx, line), await evaluateSliceBound(slice.end, ctx, line), line);
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
	if (trimmed.startsWith("str(") && trimmed.endsWith(")")) return stringifyExpression(innerCall(trimmed), ctx, line);
	if (trimmed.startsWith("len(") && trimmed.endsWith(")")) return lengthExpression(innerCall(trimmed), ctx, line);
		if (trimmed.startsWith("type(") && trimmed.endsWith(")")) return typeExpression(innerCall(trimmed), ctx, line);
		if (trimmed.startsWith("save(") && trimmed.endsWith(")")) return saveVariable(innerCall(trimmed), ctx, line);
		if (trimmed.startsWith("retrieve(") && trimmed.endsWith(")")) return retrieveUnits(innerCall(trimmed), ctx, line);
	if (trimmed.startsWith("get(") && trimmed.endsWith(")")) return getValue(innerCall(trimmed), ctx, line);
	if (trimmed.startsWith("count(") && trimmed.endsWith(")")) return countUnits(innerCall(trimmed), ctx, line);
	if (trimmed.startsWith("count_by(") && trimmed.endsWith(")")) return countBy(innerCall(trimmed), ctx, line);
	if (trimmed.startsWith("group_expr(") && trimmed.endsWith(")")) return createGroupExpression(innerCall(trimmed));
	if (trimmed.startsWith("temp(") && trimmed.endsWith(")")) return createGroupExpression(trimmed);
	if (trimmed.startsWith("group(") && trimmed.endsWith(")")) return createGroup(innerCall(trimmed), ctx, line);
	if (isStringLiteral(trimmed)) return parseStringLiteral(trimmed, line);
	if (trimmed in ctx.variables) return ctx.variables[trimmed];
	if (/^[A-Za-z][A-Za-z0-9_:-]*$/.test(trimmed)) return trimmed;
	throw parseExpressionError(expr, line);
}

function parseExpressionError(expr: string, line: number): DslRuntimeError {
	const trimmed = trimOuterParens(expr.trim());
	const listMatch = trimmed.match(/^list\s*\((.*)\)$/is);
	if (listMatch) {
		return new DslRuntimeError(
			"E_PARSE_EXPR",
			`Could not parse expression: ${expr}. list(...) is not a Quail helper. retrieve(...) and get([...]) already return lists; to inspect fields use get(fields) or retrieve(top N fields of G1).`,
			line,
		);
	}
	if (/\b(?:entries|fields)\s*\[[^\]]+\]/i.test(trimmed)) {
		return new DslRuntimeError(
			"E_PARSE_EXPR",
			`Could not parse expression: ${expr}. entries[FIELD] and fields[FIELD] are Quail units, so use them inside count(UNIT of GROUP-EXPR) or retrieve(DIRECTION AMOUNT UNIT of GROUP-EXPR), not as standalone Python values.`,
			line,
		);
	}
	const callMatch = trimmed.match(/^([A-Za-z_][A-Za-z0-9_]*)\s*\(/);
	if (callMatch) {
		return new DslRuntimeError(
			"E_PARSE_EXPR",
			`Unknown expression function ${callMatch[1]}(...). Available Quail calls include print, get, retrieve, count, count_by, save, group, group_expr, temp, and g_save when assigned to a variable.`,
			line,
		);
	}
	return new DslRuntimeError("E_PARSE_EXPR", `Could not parse expression: ${expr}. Check parentheses/brackets and use print(...), get(...), retrieve(...), count(...), or a Python literal/variable.`, line);
}

async function stringifyExpression(expr: string, ctx: RuntimeContext, line: number): Promise<string> {
	const args = splitTopLevel(expr, ",").filter((part) => part.length > 0);
	if (args.length !== 1) {
		throw new DslRuntimeError("E_PARSE_STR", "Expected str(<expression>) with exactly one argument", line);
	}
	return formatInlineValue(await evaluateExpression(args[0], ctx, line));
}

async function lengthExpression(expr: string, ctx: RuntimeContext, line: number): Promise<number> {
	const args = splitTopLevel(expr, ",").filter((part) => part.length > 0);
	if (args.length !== 1) {
		throw new DslRuntimeError("E_PARSE_LEN", "Expected len(<expression>) with exactly one argument", line);
	}
	const value = await evaluateExpression(args[0], ctx, line);
	if (typeof value === "string") return Array.from(value).length;
	if (Array.isArray(value)) return value.length;
	if (value && typeof value === "object") return Object.keys(value as Record<string, unknown>).length;
	throw new DslRuntimeError("E_LEN", `len() is not available for ${formatValueForError(value)}`, line);
}

async function typeExpression(expr: string, ctx: RuntimeContext, line: number): Promise<string> {
	const args = splitTopLevel(expr, ",").filter((part) => part.length > 0);
	if (args.length !== 1) {
		throw new DslRuntimeError("E_PARSE_TYPE", "Expected type(<expression>) with exactly one argument", line);
	}
	const value = await evaluateExpression(args[0], ctx, line);
	if (value === undefined) return "undefined";
	if (value === null) return "null";
	if (typeof value === "boolean") return "bool";
	if (typeof value === "number") return Number.isInteger(value) ? "int" : "float";
	if (typeof value === "string") return "string";
	if (Array.isArray(value)) return "list";
	if (typeof value === "object") return "object";
	return typeof value;
}

async function evaluateArithmetic(
	expr: { left: string; operator: string; right: string },
	ctx: RuntimeContext,
	line: number,
): Promise<unknown> {
	const leftValue = await evaluateExpression(expr.left, ctx, line);
	const rightValue = await evaluateExpression(expr.right, ctx, line);
	if (expr.operator === "+" && Array.isArray(leftValue) && Array.isArray(rightValue)) {
		return [...leftValue, ...rightValue];
	}
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
			if (right === 0) throw new DslRuntimeError("E_DIVIDE_BY_ZERO", "Cannot divide by zero", line);
			return left / right;
		default:
			throw new DslRuntimeError("E_ARITHMETIC", `Unknown arithmetic operator ${expr.operator}`, line);
	}
}

function toNumber(value: unknown, operator: string, line: number): number {
	const number = Number(value);
	if (!Number.isFinite(number)) {
		throw new DslRuntimeError("E_ARITHMETIC", `Operator ${operator} requires numeric values, got ${formatValue(value)}`, line);
	}
	return number;
}

function innerCall(text: string): string {
	return text.slice(text.indexOf("(") + 1, -1).trim();
}

function isStringLiteral(text: string): boolean {
	const quote = text[0];
	if ((quote !== '"' && quote !== "'") || text.length < 2) return false;
	for (let i = 1; i < text.length; i++) {
		if (text[i] !== quote || isEscaped(text, i)) continue;
		return i === text.length - 1;
	}
	return false;
}

function parseStringLiteral(text: string, line: number): string {
	try {
		if (text.startsWith('"')) return JSON.parse(text) as string;
		return text.slice(1, -1).replace(/\\'/g, "'");
	} catch {
		throw new DslRuntimeError("E_STRING", `Invalid string literal ${text}`, line);
	}
}

function isRawStringLiteral(text: string): boolean {
	return /^[rR]/.test(text) && isStringLiteral(text.slice(1));
}

function isEscaped(text: string, index: number): boolean {
	let backslashes = 0;
	for (let i = index - 1; i >= 0 && text[i] === "\\"; i--) backslashes++;
	return backslashes % 2 === 1;
}

function isListLiteralExpression(text: string): boolean {
	if (!text.startsWith("[") || !text.endsWith("]")) return false;
	let depth = 0;
	let quoted: string | undefined;
	for (let i = 0; i < text.length; i++) {
		const ch = text[i];
		if (quoted) {
			if (ch === quoted && text[i - 1] !== "\\") quoted = undefined;
			continue;
		}
		if (ch === '"' || ch === "'") { quoted = ch; continue; }
		if (ch === "[") depth++;
		else if (ch === "]") depth--;
		if (depth === 0 && i < text.length - 1) return false;
	}
	return depth === 0;
}

async function parseList(text: string, ctx: RuntimeContext, line: number): Promise<unknown[]> {
	const inner = text.slice(1, -1).trim();
	if (!inner) return [];
	const parts = splitTopLevel(inner, ",");
	return Promise.all(parts.map((part) => evaluateExpression(part, ctx, line)));
}

function splitArithmetic(text: string, operators: string[]): { left: string; operator: string; right: string } | undefined {
	let depth = 0;
	let quoted: string | undefined;
	let candidate: { left: string; operator: string; right: string } | undefined;
	for (let i = 0; i < text.length; i++) {
		const ch = text[i];
		if (quoted) {
			if (ch === quoted && text[i - 1] !== "\\") quoted = undefined;
			continue;
		}
		if (ch === '"' || ch === "'") { quoted = ch; continue; }
		if ("([{".includes(ch)) depth++;
		else if (")]}".includes(ch)) depth--;
		else if (depth === 0 && operators.includes(ch) && !isUnarySign(text, i)) {
			candidate = { left: text.slice(0, i).trim(), operator: ch, right: text.slice(i + 1).trim() };
		}
	}
	if (!candidate || candidate.left.length === 0 || candidate.right.length === 0) return undefined;
	return candidate;
}

function isUnarySign(text: string, index: number): boolean {
	const ch = text[index];
	if (ch !== "+" && ch !== "-") return false;
	let previous = index - 1;
	while (previous >= 0 && /\s/.test(text[previous])) previous--;
	if (previous < 0) return true;
	return "([{,+-*/".includes(text[previous]);
}

function splitProperty(text: string): { base: string; property: string } | undefined {
	let depth = 0;
	let quoted: string | undefined;
	for (let i = text.length - 1; i >= 0; i--) {
		const ch = text[i];
		if (quoted) {
			if (ch === quoted && text[i - 1] !== "\\") quoted = undefined;
			continue;
		}
		if (ch === '"' || ch === "'") { quoted = ch; continue; }
		if (ch === ")" || ch === "]") depth++;
		else if (ch === "(" || ch === "[") depth--;
		else if (ch === "." && depth === 0) return { base: text.slice(0, i), property: text.slice(i + 1) };
	}
	return undefined;
}

function splitIndex(text: string): { base: string; index: string } | undefined {
	if (!text.endsWith("]")) return undefined;
	let depth = 0;
	let quoted: string | undefined;
	for (let i = text.length - 1; i >= 0; i--) {
		const ch = text[i];
		if (quoted) {
			if (ch === quoted && text[i - 1] !== "\\") quoted = undefined;
			continue;
		}
		if (ch === '"' || ch === "'") { quoted = ch; continue; }
		if (ch === "]") depth++;
		else if (ch === "[") {
			depth--;
			if (depth === 0) return { base: text.slice(0, i), index: text.slice(i + 1, -1) };
		}
	}
	return undefined;
}

function splitSliceIndex(text: string): { start?: string; end?: string } | undefined {
	const parts = splitTopLevel(text, ":");
	if (parts.length === 1) return undefined;
	if (parts.length !== 2) throw new DslRuntimeError("E_PARSE_SLICE", `Expected slice syntax [start:end], got [${text}]`);
	return {
		start: parts[0] ? parts[0] : undefined,
		end: parts[1] ? parts[1] : undefined,
	};
}

async function evaluateSliceBound(expr: string | undefined, ctx: RuntimeContext, line: number): Promise<number | undefined> {
	if (expr === undefined) return undefined;
	const value = await evaluateExpression(expr, ctx, line);
	if (typeof value !== "number" || !Number.isInteger(value)) {
		throw new DslRuntimeError("E_SLICE", `Slice bounds must be integers, got ${formatValue(value)}`, line);
	}
	return value;
}

function getProperty(base: unknown, property: string, ctx: RuntimeContext, line: number): unknown {
	if (base && typeof base === "object" && property in base) return (base as Record<string, unknown>)[property];
	throw new DslRuntimeError("E_PROPERTY", `Property .${property} is not available on ${formatValueForError(base)}`, line);
}

function getIndex(base: unknown, key: unknown, line: number): unknown {
	if (Array.isArray(base) && typeof key === "number") return base[normalizeIndex(key, base.length)];
	if (typeof base === "string" && typeof key === "number") {
		const chars = Array.from(base);
		return chars[normalizeIndex(key, chars.length)];
	}
	if (base && typeof base === "object" && (typeof key === "string" || typeof key === "number")) {
		const record = base as Record<string, unknown>;
		const stringKey = String(key);
		if (Object.prototype.hasOwnProperty.call(record, stringKey)) return record[stringKey];
		if (typeof key === "string") {
			const tags = record.tags;
			if (tags && typeof tags === "object" && Object.prototype.hasOwnProperty.call(tags, key)) {
				return (tags as Record<string, unknown>)[key];
			}
			const fields = record.fields;
			if (fields && typeof fields === "object" && Object.prototype.hasOwnProperty.call(fields, key)) {
				return (fields as Record<string, unknown>)[key];
			}
		}
		return undefined;
	}
	throw new DslRuntimeError("E_INDEX", `Cannot index ${formatValueForError(base)} with ${formatValue(key)}`, line);
}

function getSlice(base: unknown, start: number | undefined, end: number | undefined, line: number): unknown {
	if (typeof base === "string") {
		const chars = Array.from(base);
		return chars.slice(normalizeSliceBound(start, chars.length, 0), normalizeSliceBound(end, chars.length, chars.length)).join("");
	}
	if (Array.isArray(base)) {
		return base.slice(normalizeSliceBound(start, base.length, 0), normalizeSliceBound(end, base.length, base.length));
	}
	throw new DslRuntimeError("E_SLICE", `Cannot slice ${formatValueForError(base)}`, line);
}

function normalizeIndex(index: number, length: number): number {
	return index < 0 ? length + index : index;
}

function normalizeSliceBound(value: number | undefined, length: number, fallback: number): number {
	if (value === undefined) return fallback;
	return value < 0 ? length + value : value;
}

function resolveBareOrString(value: string, ctx: RuntimeContext): unknown {
	const trimmed = value.trim();
	if (isStringLiteral(trimmed)) return parseStringLiteral(trimmed, undefined as unknown as number);
	return ctx.variables[trimmed] ?? trimmed;
}

function coerceString(value: unknown): string {
	return String(value);
}

function coerceId(value: unknown): string {
	return String(value).trim();
}

function coerceEntryId(value: unknown, ctx: RuntimeContext): string | undefined {
	const text = coerceId(value);
	if (ctx.entriesById.has(text)) return text;
	const ordinal = typeof value === "number" ? value : (/^\d+$/.test(text) ? Number(text) : NaN);
	if (!Number.isInteger(ordinal) || ordinal <= 0) return undefined;
	for (const [id, entryOrdinal] of ctx.entryOrdinalById) {
		if (entryOrdinal === ordinal) return id;
	}
	return undefined;
}

function resolveCommandTextArgument(arg: string, ctx: RuntimeContext, line: number): string {
	const trimmed = arg.trim();
	if (trimmed in ctx.variables) {
		const value = ctx.variables[trimmed];
		if (typeof value === "string") return value;
	}
	if (isStringLiteral(trimmed)) return parseStringLiteral(trimmed, line);
	return arg;
}

async function evaluateTagValueExpression(value: string, ctx: RuntimeContext, line: number): Promise<unknown> {
	const trimmed = value.trim();
	if (
		isStringLiteral(trimmed) ||
		isListLiteralExpression(trimmed) ||
		trimmed in ctx.variables ||
		(trimmed.startsWith("str(") && trimmed.endsWith(")")) ||
			trimmed === "true" ||
			trimmed === "false" ||
			trimmed === "null" ||
			trimmed === "True" ||
			trimmed === "False" ||
			trimmed === "None" ||
			/^-?\d+(\.\d+)?$/.test(trimmed)
	) {
		return evaluateExpression(trimmed, ctx, line);
	}
	return resolveBareOrString(trimmed, ctx);
}

function normalizeTagValues(value: unknown): unknown[] {
	const values = Array.isArray(value) ? value : [value];
	return uniqueValues(values
		.map((item) => typeof item === "string" ? item.trim() : item)
		.filter((item) => item !== undefined && !(typeof item === "string" && item.length === 0)));
}

function normalizeStoredTagValue(values: unknown[]): TagValue {
	const unique = uniqueValues(values);
	return unique.length <= 1 ? (unique[0] ?? "") : unique;
}

function tagValueToList(value: TagValue | undefined): unknown[] {
	if (value === undefined) return [];
	return Array.isArray(value) ? value : [value];
}

function uniqueStrings(values: string[]): string[] {
	return [...new Set(values)];
}

function uniqueValues(values: unknown[]): unknown[] {
	const seen = new Set<string>();
	const out: unknown[] = [];
	for (const value of values) {
		const key = stableValueKey(value);
		if (seen.has(key)) continue;
		seen.add(key);
		out.push(value);
	}
	return out;
}

async function getValue(arg: string, ctx: RuntimeContext, line: number): Promise<unknown> {
	const trimmed = trimOuterParens(arg.trim());
	if (trimmed === "groups") return Object.keys(ctx.state.groups);
	if (trimmed === "fields") return getAvailableFields(ctx);
	if (trimmed === "text_fields") return getTextFields(ctx);
	if (trimmed === "tag_fields") return getTagFields(ctx);
	if (trimmed === "tags") throw new DslRuntimeError("E_GET_TAGS_DISABLED", 'get(tags) is disabled. Use get(tag_fields) to list fields or get(["field"]) to list values for a field.', line);
	if (trimmed.startsWith("[") && trimmed.endsWith("]")) {
		return getEntriesOrFieldValues(await parseList(trimmed, ctx, line), ctx, line);
	}
	const distribution = parseDistribution(trimmed);
	if (distribution) return getDistribution(await parseFilter(distribution.filter, ctx, line), distribution.groupExpression, ctx, line);
	const evaluated = await evaluateExpression(trimmed, ctx, line);
	if (Array.isArray(evaluated)) return getEntriesOrFieldValues(evaluated, ctx, line);
	const id = coerceEntryId(evaluated, ctx) ?? coerceId(evaluated);
	const group = ctx.state.groups[id];
	if (group) return group;
	const entry = ctx.entriesById.get(id);
	if (!entry) {
		throw new DslRuntimeError(
			"E_UNKNOWN_ID",
			`Unknown entry or group id ${id}. get(...) accepts an entry ID returned by retrieve(... entries of ...), a saved group variable, a list of entry IDs, or a list of field names. Values returned by entries[FIELD] are field/tag values, not entry IDs; retrieve entries of the same group and inspect get(entry)[FIELD] instead.`,
			line,
		);
	}
	return formatEntryForGet(entry, ctx);
}

function getAvailableFields(ctx: RuntimeContext): string[] {
	return getAllFieldNames(ctx);
}

function getTextFields(ctx: RuntimeContext): string[] {
	return Array.from(new Set(ctx.datasets.flatMap((dataset) => dataset.manifest.embeddedFields ?? dataset.manifest.textFields ?? []))).sort();
}

function getTagFields(ctx: RuntimeContext): string[] {
	const fields = new Set<string>();
	for (const field of ctx.state.createdFields ?? []) fields.add(field);
	for (const entry of ctx.entries) {
		for (const field of Object.keys(getEntryTags(entry, ctx))) fields.add(field);
	}
	return [...fields].sort();
}

function getFieldOrTagValues(fields: string[], ctx: RuntimeContext): FieldValue[] | Record<string, FieldValue[]> {
	const valuesFor = (field: string) => getValuesForFieldOrTag(field, ctx);
	if (fields.length === 1) return valuesFor(fields[0]);
	return Object.fromEntries(fields.map((field) => [field, valuesFor(field)]));
}

function getEntriesOrFieldValues(values: unknown[], ctx: RuntimeContext, line: number): unknown {
	if (values.length === 0) return [];
	const strings = values.map((value) => coerceEntryId(value, ctx) ?? String(value));
	if (strings.every((value) => ctx.entriesById.has(value))) {
		return strings.map((id) => formatEntryForGet(ctx.entriesById.get(id)!, ctx));
	}
	const fields = new Set(getAllFieldNames(ctx));
	if (strings.every((value) => fields.has(value))) return getFieldOrTagValues(strings, ctx);
	throw new DslRuntimeError("E_GET_LIST_SCOPE", "get([...]) requires only entry ids or only field names", line);
}

function formatEntryForGet(entry: QuailEntry, ctx: RuntimeContext): Record<string, unknown> {
	return {
		id: entry.id,
		dataset: entry.dataset,
		fields: materializeEntryFields(entry, ctx),
		text: entry.text,
		tags: getEntryTags(entry, ctx),
	};
}

function materializeEntryFields(entry: QuailEntry, ctx: RuntimeContext): Record<string, unknown> {
	const fields: Record<string, unknown> = { ...entry.fields };
	const dataset = ctx.datasetByEntryId.get(entry.id);
	for (const field of ctx.processedFields) {
		if (!Object.prototype.hasOwnProperty.call(fields, field)) fields[field] = defaultMissingSourceFieldValue(field, dataset);
	}
	return fields;
}

function defaultMissingSourceFieldValue(field: string, dataset: LoadedQuailDataset | undefined): unknown {
	const fieldType = dataset?.manifest.fieldTypes?.[field];
	if (fieldType === "string") return "";
	if (dataset?.manifest.embeddedFields?.includes(field) || dataset?.manifest.textFields?.includes(field)) return "";
	return null;
}

function getValuesForFieldOrTag(field: string, ctx: RuntimeContext): FieldValue[] {
	const values = new Map<string, FieldValue>();
	for (const entry of ctx.entries) {
		for (const item of tagValueToList(getEntryTags(entry, ctx)[field])) {
			if (fieldValueToText(item).trim().length > 0) {
				const normalized = normalizeFieldValue(item);
				values.set(stableValueKey(normalized), normalized);
			}
		}
	}
	return [...values.values()].sort(compareFieldValues);
}

function stableValueKey(value: unknown): string {
	return JSON.stringify(value);
}

function compareFieldValues(a: unknown, b: unknown): number {
	if (typeof a === "number" && typeof b === "number") return a - b;
	return formatInlineValue(a).localeCompare(formatInlineValue(b));
}

function parseDistribution(text: string): { filter: string; groupExpression: string } | undefined {
	const marker = ") distribution of ";
	const index = text.indexOf(marker);
	if (!text.startsWith("(") || index < 0) return undefined;
	const filterText = text.slice(1, index);
	const rawGroupExpression = text.slice(index + marker.length).trim();
	const groupExpression = trimOuterParens(rawGroupExpression);
	return { filter: filterText, groupExpression };
}

function getTagsDictionary(ctx: RuntimeContext): Record<string, string[]> {
	const tags: Record<string, Set<string>> = {};
	for (const entry of ctx.entries) {
			for (const [field, value] of Object.entries(getEntryTags(entry, ctx))) {
				for (const item of tagValueToList(value)) {
					const text = formatInlineValue(item);
					if (text.length > 0) (tags[field] ??= new Set()).add(text);
				}
			}
	}
	return Object.fromEntries(Object.entries(tags).map(([field, values]) => [field, [...values].sort()]));
}

function getEntryTags(entry: QuailEntry, ctx: RuntimeContext): Record<string, TagValue> {
	return {
		...(entry.fields as Record<string, TagValue>),
		...entry.tags,
		...(ctx.state.tagsByEntry[entry.id] ?? {}),
	};
}

function getTagIndex(ctx: RuntimeContext): TagIndex {
	if (ctx.tagIndex) return ctx.tagIndex;
	const index: TagIndex = new Map();
	for (const entry of ctx.entries) {
			for (const [field, value] of Object.entries(getEntryTags(entry, ctx))) {
				for (const item of tagValueToList(value)) {
					if (fieldValueToText(item).trim().length === 0) continue;
					let values = index.get(field);
					if (!values) {
						values = new Map();
						index.set(field, values);
					}
					const key = tagIndexKey(item);
					let ids = values.get(key);
					if (!ids) {
						ids = new Set();
						values.set(key, ids);
					}
					ids.add(entry.id);
				}
			}
	}
	ctx.tagIndex = index;
	return index;
}

function tagIndexKey(value: unknown): string {
	return stableValueKey(normalizeFieldValue(value));
}

function getIdsMatchingTags(tags: Record<string, unknown>, ctx: RuntimeContext): Set<string> {
	return getIdsMatchingTagsBits(tags, ctx).toIdSet(ctx);
}

function getIdsMatchingTagsBits(tags: Record<string, unknown>, ctx: RuntimeContext): IdBitSet {
	const index = getTagIndex(ctx);
	let ids: IdBitSet | undefined;
	for (const [field, value] of Object.entries(tags)) {
		const matching = index.get(field)?.get(tagIndexKey(value));
		if (!matching) return IdBitSet.empty(ctx.entries.length);
		const matchingBits = IdBitSet.fromIds(matching, ctx);
		ids = ids ? ids.and(matchingBits) : matchingBits;
		if (ids.size === 0) return ids;
	}
	return ids ?? IdBitSet.full(ctx.entries.length);
}

function intersectSets(a: Set<string>, b: Set<string>): Set<string> {
	const [small, large] = a.size <= b.size ? [a, b] : [b, a];
	const out = new Set<string>();
	for (const id of small) {
		if (large.has(id)) out.add(id);
	}
	return out;
}

function allEntryIds(ctx: RuntimeContext): string[] {
	return ctx.entries.map((entry) => entry.id);
}

function getAllFieldNames(ctx: RuntimeContext): string[] {
	return [
		...new Set([
			...ctx.entries.flatMap((entry) => Object.keys(entry.fields)),
			...getTagFields(ctx),
			...(ctx.state.createdFields ?? []),
		]),
	].sort();
}

function scopedMembers(scope: QuailScope, members: string[], spec: string): ScopedMembers {
	return { scope, members: [...new Set(members)], spec };
}

function inferScopedMembers(values: unknown[], ctx: RuntimeContext, spec: string, line: number): ScopedMembers {
	const members = values.map((value) => coerceString(value));
	if (members.every((member) => ctx.entriesById.has(member))) return scopedMembers("entries", members, spec);
	const fields = new Set(getAllFieldNames(ctx));
	if (members.every((member) => fields.has(member))) return scopedMembers("fields", members, spec);
	throw new DslRuntimeError("E_GROUP_LIST_SCOPE", "List group expressions must contain only entry ids or only field names", line);
}

function cloneScopedMembers(group: ScopedMembers): ScopedMembers {
	return { ...group, members: [...group.members] };
}

async function resolveScopedGroupExpression(expr: string, ctx: RuntimeContext, line: number): Promise<ScopedMembers> {
	const text = trimOuterParens(expr.trim());
	if (isCacheableGroupExpressionText(text, ctx)) {
		const cached = ctx.scopedGroupExpressionCache.get(text);
		if (cached) return cloneScopedMembers(cached);
		const group = await resolveScopedGroupExpressionUncached(text, ctx, line);
		ctx.scopedGroupExpressionCache.set(text, cloneScopedMembers(group));
		return group;
	}
	return resolveScopedGroupExpressionUncached(text, ctx, line);
}

async function resolveScopedGroupExpressionUncached(text: string, ctx: RuntimeContext, line: number): Promise<ScopedMembers> {
	if (!text || text === "G0") return scopedMembers("entries", allEntryIds(ctx), text || "G0");
	if (text === "G1") return scopedMembers("fields", getAllFieldNames(ctx), "G1");
	if (isStringLiteral(text)) {
		const value = parseStringLiteral(text, line);
		if (ctx.entriesById.has(value)) return scopedMembers("entries", [value], text);
		if (getAllFieldNames(ctx).includes(value)) return scopedMembers("fields", [value], text);
	}
	if (isListLiteralExpression(text)) return inferScopedMembers(await parseList(text, ctx, line), ctx, text, line);
	const saved = ctx.state.groups[text];
	if (saved) {
		const scope = saved.scope ?? "entries";
		const members = saved.members ?? saved.entryIds ?? saved.fieldNames ?? [];
		return scopedMembers(scope, members, text);
	}
	if (text in ctx.variables) {
		const value = ctx.variables[text];
		if (isGroupExpressionValue(value)) return resolveScopedGroupExpression(value.expression, ctx, line);
		if (Array.isArray(value)) return inferScopedMembers(value, ctx, text, line);
		if (typeof value === "string") return resolveScopedGroupExpression(value, ctx, line);
		throw new DslRuntimeError("E_GROUP_EXPR_VARIABLE", `Variable ${text} does not contain a group expression.`, line);
	}
	if (text.startsWith("g_save(") && text.endsWith(")")) {
		throw gSaveAssignmentError(line);
	}
	if (text.startsWith("group(") && text.endsWith(")")) {
		return resolveScopedGroupExpression(await createGroup(innerCall(text), ctx, line), ctx, line);
	}
	if (text.startsWith("group_expr(") && text.endsWith(")")) {
		return resolveScopedGroupExpression(innerCall(text), ctx, line);
	}
	if (text.startsWith("temp(") && text.endsWith(")")) {
		const ids = await resolveGroupSpec(innerCall(text), ctx, line);
		return scopedMembers("entries", [...ids], text);
	}
	if (text.startsWith("scope:")) return resolveScopedGroup(text, ctx, line);
	const orParts = splitTopLevelByWord(text, "or");
	if (orParts.length > 1) {
		const parts = await Promise.all(orParts.map((part) => resolveScopedGroupExpression(part, ctx, line)));
		return combineScopedMembers(parts, "or", ctx, line, text);
	}
	const andParts = splitTopLevelByWord(text, "and");
	if (andParts.length > 1) {
		const parts = await Promise.all(andParts.map((part) => resolveScopedGroupExpression(part, ctx, line)));
		return combineScopedMembers(parts, "and", ctx, line, text);
	}
	if (text.startsWith("not ")) {
		const excluded = await resolveScopedGroupExpression(text.slice(4), ctx, line);
		return complementScopedMembers(excluded, ctx, text);
	}
	if (looksLikeGroupSpec(text)) {
		const ids = await resolveGroupSpec(text, ctx, line);
		return scopedMembers("entries", [...ids], text);
	}
	if (ctx.entriesById.has(text)) return scopedMembers("entries", [text], text);
	if (getAllFieldNames(ctx).includes(text)) return scopedMembers("fields", [text], text);
	throw unknownScopedGroupExpressionError(text, line);
}

function unknownScopedGroupExpressionError(text: string, line: number): DslRuntimeError {
	const trimmed = trimOuterParens(text.trim());
	if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(trimmed)) {
		return new DslRuntimeError(
			"E_GROUP_EXPR",
			`No group variable named ${trimmed} exists. If ${trimmed} was assigned from g_save(...), check earlier errors: failed g_save assignments do not create groups. Define it with ${trimmed} = g_save((scope: G0, ...)) or use G0, G1, a saved group variable, a list, or a scoped group expression.`,
			line,
		);
	}
	return new DslRuntimeError("E_GROUP_EXPR", `Unknown group expression: ${text}. Use G0, G1, a saved group variable, a list, or a scoped group expression like (scope: G0, ([field] BM25 similarity to "query" > 0)).`, line);
}

function combineScopedMembers(parts: ScopedMembers[], op: "and" | "or", ctx: RuntimeContext, line: number, spec: string): ScopedMembers {
	const [first] = parts;
	if (!first) return scopedMembers("entries", [], spec);
	if (!parts.every((part) => part.scope === first.scope)) {
		throw new DslRuntimeError("E_GROUP_SCOPE", "Cannot combine entry-scoped and field-scoped groups", line);
	}
	if (op === "or") return scopedMembers(first.scope, parts.flatMap((part) => part.members), spec);
	let current = new Set(first.members);
	for (const part of parts.slice(1)) {
		const next = new Set(part.members);
		current = new Set([...current].filter((member) => next.has(member)));
	}
	return scopedMembers(first.scope, [...current], spec);
}

function complementScopedMembers(group: ScopedMembers, ctx: RuntimeContext, spec: string): ScopedMembers {
	const universe = group.scope === "entries" ? allEntryIds(ctx) : getAllFieldNames(ctx);
	const excluded = new Set(group.members);
	return scopedMembers(group.scope, universe.filter((member) => !excluded.has(member)), spec);
}

async function resolveScopedGroup(text: string, ctx: RuntimeContext, line: number): Promise<ScopedMembers> {
	const parts = splitTopLevel(text, ",");
	const scopePart = trimOuterParens(parts.shift()?.trim() ?? "");
	const scopeMatch = scopePart.match(/^scope\s*:\s*(.+)$/is);
	if (!scopeMatch || !scopeMatch[1].trim()) {
		throw new DslRuntimeError("E_GROUP_SCOPE", "Expected scoped group to start with scope: G0, scope: G1, a saved group id, or a group variable", line);
	}
	const base = await resolveScopedGroupExpression(scopeMatch[1].trim(), ctx, line);
	const clauseText = parts.join(",").trim();
	if (!clauseText) return base;
	const group = await resolveScopedClauseExpression(base.scope, clauseText, ctx, line);
	return combineScopedMembers([base, group], "and", ctx, line, text);
}

async function resolveScopedClauseExpression(scope: QuailScope, expr: string, ctx: RuntimeContext, line: number): Promise<ScopedMembers> {
	const text = trimOuterParens(expr.trim());
	const orParts = splitTopLevelByWord(text, "or");
	if (orParts.length > 1) {
		const parts = await Promise.all(orParts.map((part) => resolveScopedClauseExpression(scope, part, ctx, line)));
		return combineScopedMembers(parts, "or", ctx, line, text);
	}
	const andParts = splitTopLevelByWord(text, "and");
	if (andParts.length > 1) {
		const parts = await Promise.all(andParts.map((part) => resolveScopedClauseExpression(scope, part, ctx, line)));
		return combineScopedMembers(parts, "and", ctx, line, text);
	}
	if (text.startsWith("not ")) {
		const excluded = await resolveScopedClauseExpression(scope, text.slice(4), ctx, line);
		return complementScopedMembers(excluded, ctx, text);
	}
	if (text.startsWith("scope:")) {
		const group = await resolveScopedGroup(text, ctx, line);
		if (group.scope !== scope) throw new DslRuntimeError("E_GROUP_SCOPE", "Nested scoped clauses must use the same scope", line);
		return group;
	}
	return evaluateGroupClause(scope, text, ctx, line);
}

async function evaluateGroupClause(scope: QuailScope, text: string, ctx: RuntimeContext, line: number): Promise<ScopedMembers> {
	const parenthesizedLeft = splitParenthesizedClauseWithTrailingCondition(text);
	if (parenthesizedLeft) return evaluateGroupClause(scope, parenthesizedLeft, ctx, line);
	for (const op of [">=", "<=", "!=", "==", ">", "<"]) {
		const parts = splitByTopLevelOperator(text, op);
		if (!parts) continue;
		const right = await evaluateLooseValue(parts[2], ctx, line);
		const optimized = await tryOptimizedScopedGroupClause(scope, parts[0], parts[1], right, ctx, line);
		if (optimized) return scopedMembers(optimized.scope, optimized.members, text);
		const universe = scope === "entries" ? allEntryIds(ctx) : getAllFieldNames(ctx);
		const matches = (await Promise.all(universe.map(async (member) => {
			const left = await evaluateClauseValue(scope, member, parts[0], ctx, line);
			return compareClauseValue(left, parts[1], right) ? member : undefined;
		}))).filter((member): member is string => member !== undefined);
		return scopedMembers(scope, matches, text);
	}
	const implicitSimilarity = await tryImplicitPositiveSimilarityClause(scope, text, ctx, line);
	if (implicitSimilarity) return implicitSimilarity;
	try {
		const group = await resolveScopedGroupExpression(text, ctx, line);
		if (group.scope !== scope) throw new DslRuntimeError("E_GROUP_SCOPE", "Nested scoped clauses must use the same scope", line);
		return group;
	} catch (error) {
		if (!(error instanceof DslRuntimeError) || error.code !== "E_GROUP_EXPR") throw error;
	}
	throw new DslRuntimeError("E_GROUP_CLAUSE", `Expected a group clause with a condition, got ${text}`, line);
}

function splitParenthesizedClauseWithTrailingCondition(text: string): string | undefined {
	const trimmed = text.trim();
	if (!trimmed.startsWith("(")) return undefined;
	const closeIndex = findMatchingClose(trimmed, 0, "(", ")");
	if (closeIndex < 0 || closeIndex === trimmed.length - 1) return undefined;
	const rest = trimmed.slice(closeIndex + 1).trim();
	const match = rest.match(/^(>=|<=|!=|==|>|<)\s*(.+)$/s);
	if (!match) return undefined;
	return `${trimmed.slice(1, closeIndex).trim()} ${match[1]} ${match[2].trim()}`;
}

async function tryImplicitPositiveSimilarityClause(
	scope: QuailScope,
	text: string,
	ctx: RuntimeContext,
	line: number,
): Promise<ScopedMembers | undefined> {
	if (scope !== "entries") return undefined;
	const input = await parseFunctionInput(text, ctx, line);
	if (input.regexOps.length > 0) return undefined;
	const match = matchSimilarityFunction(input.functionText.trim());
	if (!match) return undefined;
	const targets = await evaluateSimilarityTargets(match.targetText, ctx, line);
	if (targets.length !== 1 || targets[0].kind !== "text") return undefined;
	const predicate: FieldTextPredicate = { field: input.field, text: targets[0].text };
	const ids = match.mode === "embed"
		? await getIdsAboveEmbeddingThreshold(predicate, 0, ctx, line)
		: getIdsAboveBm25Threshold(predicate, 0, ctx);
	return scopedMembers("entries", ids.toIds(ctx), text);
}

async function tryOptimizedScopedGroupClause(
	scope: QuailScope,
	leftText: string,
	operator: string,
	right: unknown,
	ctx: RuntimeContext,
	line: number,
): Promise<ScopedMembers | undefined> {
	if (scope !== "entries" || operator !== ">" || typeof right !== "number" || !Number.isFinite(right)) return undefined;
	const input = await parseFunctionInput(leftText, ctx, line);
	if (input.regexOps.length > 0) return undefined;
	const match = matchSimilarityFunction(input.functionText.trim());
	if (!match) return undefined;
	const targets = await evaluateSimilarityTargets(match.targetText, ctx, line);
	if (targets.length !== 1 || targets[0].kind !== "text") return undefined;
	const predicate: FieldTextPredicate = { field: input.field, text: targets[0].text };
	const ids = match.mode === "embed"
		? await getIdsAboveEmbeddingThreshold(predicate, right, ctx, line)
		: getIdsAboveBm25Threshold(predicate, right, ctx);
	return scopedMembers("entries", ids.toIds(ctx), leftText);
}

function matchSimilarityFunction(functionText: string): { mode: "BM25" | "embed"; targetText: string } | undefined {
	const match = functionText.match(/^(?:(BM25|embed|embeddings?)\s+)?similarity\s+to\s+(.+)$/i);
	if (!match) return undefined;
	const mode = match[1]?.toLowerCase().startsWith("embed") ? "embed" : "BM25";
	return { mode, targetText: match[2] };
}

async function evaluateLooseValue(text: string, ctx: RuntimeContext, line: number): Promise<unknown> {
	const trimmed = text.trim();
	if (isRawStringLiteral(trimmed)) return parseRegexPatternArg(trimmed, ctx, line);
	try {
		return await evaluateExpression(trimmed, ctx, line);
	} catch (error) {
		if (error instanceof DslRuntimeError && error.code === "E_PARSE_EXPR") return trimOuterParens(trimmed);
		throw error;
	}
}

function compareClauseValue(left: unknown, operator: string, right: unknown): boolean {
	if (Array.isArray(left)) {
		if (operator === "!=") return left.every((item) => !compareClauseValue(item, "==", right));
		return left.some((item) => compareClauseValue(item, operator, right));
	}
	if (operator === "==") return valuesEqual(left, right);
	if (operator === "!=") return !valuesEqual(left, right);
	const l = Number(left);
	const r = Number(right);
	if (!Number.isFinite(l) || !Number.isFinite(r)) return false;
	switch (operator) {
		case ">": return l > r;
		case "<": return l < r;
		case ">=": return l >= r;
		case "<=": return l <= r;
		default: return false;
	}
}

function valuesEqual(left: unknown, right: unknown): boolean {
	if (stableValueKey(left) === stableValueKey(right)) return true;
	const leftNumber = finiteNumberValue(left);
	const rightNumber = finiteNumberValue(right);
	if (leftNumber !== undefined && rightNumber !== undefined) return leftNumber === rightNumber;
	const leftBoolean = booleanValue(left);
	const rightBoolean = booleanValue(right);
	if (leftBoolean !== undefined && rightBoolean !== undefined) return leftBoolean === rightBoolean;
	return false;
}

function finiteNumberValue(value: unknown): number | undefined {
	if (typeof value === "number") return Number.isFinite(value) ? value : undefined;
	if (typeof value !== "string" || value.trim() === "") return undefined;
	const number = Number(value);
	return Number.isFinite(number) ? number : undefined;
}

function booleanValue(value: unknown): boolean | undefined {
	if (typeof value === "boolean") return value;
	if (typeof value !== "string") return undefined;
	const lower = value.trim().toLowerCase();
	if (lower === "true") return true;
	if (lower === "false") return false;
	return undefined;
}

async function evaluateClauseValue(scope: QuailScope, member: string, expression: string, ctx: RuntimeContext, line: number): Promise<unknown> {
	const input = await parseFunctionInput(expression, ctx, line);
	const values = scope === "entries"
		? getEntryFunctionValues(member, input.field, ctx)
		: [input.field ? fieldValuesForField(input.field, ctx).join(" ") : member];
	const texts = applyRegexOpsToTexts(values.map(fieldValueToText), input.regexOps, line);
	if (!input.functionText) return texts.length === 1 ? texts[0] : texts;
	return evaluateFunctionText(input.functionText, texts, { entryId: scope === "entries" ? member : undefined, field: input.field }, ctx, line);
}

function gSaveAssignmentError(line: number): DslRuntimeError {
	return new DslRuntimeError("E_G_SAVE_ASSIGNMENT", "g_save(GROUP-EXPR) must be assigned to a variable, e.g. saved_group = g_save((scope: G0, ([field] BM25 similarity to \"query\" > 0))).", line);
}

function parseGSaveRequest(rawArg: unknown, line: number): { name: string; expr: string } {
	if (!rawArg || typeof rawArg !== "object" || Array.isArray(rawArg)) throw gSaveAssignmentError(line);
	const record = rawArg as Record<string, unknown>;
	if (typeof record.name !== "string" || typeof record.expr !== "string") throw gSaveAssignmentError(line);
	return { name: record.name, expr: record.expr };
}

function validateSavedGroupVariableName(name: string, line: number): string {
	const trimmed = name.trim();
	if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(trimmed)) {
		throw new DslRuntimeError("E_G_SAVE_NAME", `g_save() variable names must be Python-style identifiers, got ${name}`, line);
	}
	if (trimmed === "G0" || trimmed === "G1" || /^G\d+$/.test(trimmed)) {
		throw new DslRuntimeError("E_G_SAVE_NAME", `${trimmed} is reserved for built-in/legacy group IDs; assign g_save(...) to a descriptive variable name instead.`, line);
	}
	return trimmed;
}

async function gSaveGroup(name: string, arg: string, ctx: RuntimeContext, line: number): Promise<string> {
	const groupId = validateSavedGroupVariableName(name, line);
	arg = resolveCommandTextArgument(arg, ctx, line);
	const group = await resolveScopedGroupExpression(arg, ctx, line);
	ctx.state.groups[groupId] = {
		id: groupId,
		datasets: ctx.activeDatasetNames,
		scope: group.scope,
		spec: group.spec,
		members: group.members,
		entryIds: group.scope === "entries" ? group.members : undefined,
		fieldNames: group.scope === "fields" ? group.members : undefined,
		createdAt: new Date().toISOString(),
	};
	ctx.variables[groupId] = groupId;
	ctx.state.variables[groupId] = groupId;
	clearGroupExpressionCache(ctx);
	return groupId;
}

async function countUnits(arg: string, ctx: RuntimeContext, line: number): Promise<number> {
	arg = resolveCommandTextArgument(arg, ctx, line);
	if (splitByTopLevelOperator(arg, " out of ")) {
		throw new DslRuntimeError("E_PARSE_COUNT", "Expected count(UNIT of GROUP-EXPR). Quail v0.7 uses of, not out of.", line);
	}
	const parts = splitByTopLevelOperator(arg, " of ");
	if (!parts) return countGroup(arg, ctx, line);
	const unit = await parseUnitSpec(parts[0], ctx, line);
	const group = await resolveScopedGroupExpression(parts[2], ctx, line);
	return buildUnitValues(unit, group, ctx, line).length;
}

async function retrieveUnits(arg: string, ctx: RuntimeContext, line: number): Promise<unknown[]> {
	arg = resolveCommandTextArgument(arg, ctx, line);
	if (splitByTopLevelOperator(arg, " out of ")) {
		throw new DslRuntimeError("E_PARSE_RETRIEVE", "Expected retrieve(DIRECTION AMOUNT UNIT of GROUP-EXPR sorted by RANKING). Quail v0.7 uses of, not out of.", line);
	}
	const sortedParts = splitByTopLevelOperator(arg, " sorted by ");
	const retrievalText = sortedParts ? sortedParts[0] : arg;
	const ranking = sortedParts?.[2];
	const parts = splitByTopLevelOperator(retrievalText, " of ");
	if (!parts) {
		throw new DslRuntimeError("E_PARSE_RETRIEVE", "Expected retrieve(DIRECTION AMOUNT UNIT of GROUP-EXPR sorted by RANKING), for example retrieve(top 5 entries of G0) or retrieve(top 20 entries of (scope: G0, ([text] BM25 similarity to \"freedom\" > 0)) sorted by ([text] BM25 similarity to \"freedom\")).", line);
	}
	const { location, amount, unitText } = parseRetrieveUnitPrefix(parts[0], line);
	const unit = await parseUnitSpec(unitText, ctx, line);
	const group = await resolveScopedGroupExpression(parts[2], ctx, line);
	const values = buildUnitValues(unit, group, ctx, line);
	const ordered = ranking
		? (await Promise.all(values.map(async (value) => ({ value, score: await evaluateRankingExpression(ranking, value, ctx, line) }))))
			.sort((a, b) => b.score - a.score)
			.map((item) => item.value)
		: values;
	return selectByLocation(ordered, location, amount).map((value) => value.value);
}

function parseRetrieveUnitPrefix(text: string, line: number): { location: "top" | "middle" | "bottom"; amount: number; unitText: string } {
	let rest = text.trim();
	let location: "top" | "middle" | "bottom" = "top";
	const direction = rest.match(/^(top|middle|bottom)\b\s*/i);
	if (direction) {
		location = direction[1].toLowerCase() as "top" | "middle" | "bottom";
		rest = rest.slice(direction[0].length).trim();
	}
	let amount = 1;
	const amountMatch = rest.match(/^(\d+)\b\s*/);
	if (amountMatch) {
		amount = Number.parseInt(amountMatch[1], 10);
		rest = rest.slice(amountMatch[0].length).trim();
	}
	if (!rest) throw new DslRuntimeError("E_PARSE_RETRIEVE", "retrieve requires a unit before of GROUP-EXPR: entries, entries[FIELD], fields, or fields[FIELD]", line);
	return { location, amount, unitText: rest };
}

function selectByLocation<T>(items: T[], location: "top" | "middle" | "bottom", amount: number): T[] {
	if (amount <= 0) return [];
	if (location === "top") return items.slice(0, amount);
	if (location === "bottom") return items.slice(-amount).reverse();
	const center = Math.floor(items.length / 2);
	const start = Math.max(0, center - Math.floor(amount / 2));
	return items.slice(start, start + amount);
}

async function parseUnitSpec(text: string, ctx: RuntimeContext, line: number): Promise<UnitSpec> {
	const trimmed = text.trim();
	const match = trimmed.match(/^(entries|fields)(?:\[(.*?)\])?(.*)$/is);
	if (!match) throw new DslRuntimeError("E_PARSE_UNIT", `Expected unit entries, entries[FIELD], fields, or fields[FIELD] before of GROUP-EXPR, got ${text}`, line);
	const base = match[1].toLowerCase() as "entries" | "fields";
	const field = match[2] !== undefined ? await evaluateFieldName(match[2], ctx, line) : undefined;
	const regexText = match[3].trim();
	const dotField = regexText.match(/^\.([A-Za-z_][A-Za-z0-9_]*)\s*\./);
	if (!field && dotField) {
		throw new DslRuntimeError(
			"E_PARSE_UNIT",
			`Use ${base}[${dotField[1]}]${regexText.slice(dotField[1].length + 1)} instead of ${base}.${dotField[1]}${regexText.slice(dotField[1].length + 1)}. Field-specific units use bracket syntax.`,
			line,
		);
	}
	const regexOps = await parseRegexOps(regexText, ctx, line);
	if (base === "entries") {
		return { scope: "entries", kind: field ? "entryField" : "entries", field, regexOps, raw: trimmed };
	}
	return { scope: "fields", kind: field ? "fieldValues" : "fields", field, regexOps, raw: trimmed };
}

function buildUnitValues(unit: UnitSpec, group: ScopedMembers, ctx: RuntimeContext, line: number): UnitValue[] {
	if (unit.scope !== group.scope) {
		throw new DslRuntimeError("E_UNIT_SCOPE", `${unit.raw} cannot be retrieved from a ${group.scope}-scoped GROUP-EXPR`, line);
	}
	const cacheKey = `${unit.raw}\0${group.scope}\0${group.members.join("\0")}`;
	const cached = ctx.unitValuesCache.get(cacheKey);
	if (cached) return cloneUnitValues(cached);
	const values = buildUnitValuesUncached(unit, group, ctx, line);
	ctx.unitValuesCache.set(cacheKey, cloneUnitValues(values));
	return values;
}

function cloneUnitValues(values: UnitValue[]): UnitValue[] {
	return values.map((value) => ({ ...value }));
}

function buildUnitValuesUncached(unit: UnitSpec, group: ScopedMembers, ctx: RuntimeContext, line: number): UnitValue[] {
	let values: UnitValue[] = [];
	let sourceIndex = 0;
	if (unit.kind === "entries") {
		values = group.members.map((entryId) => ({ value: entryId, text: entryId, entryId, scoreByEntry: true, sourceIndex: sourceIndex++ }));
	} else if (unit.kind === "entryField") {
		for (const entryId of group.members) {
			for (const value of getEntryTagValues(entryId, unit.field, ctx)) {
				values.push({ value, text: fieldValueToText(value), entryId, fieldName: unit.field, sourceIndex: sourceIndex++ });
			}
		}
	} else if (unit.kind === "fields") {
		values = group.members.map((fieldName) => ({ value: fieldName, text: fieldName, fieldName, sourceIndex: sourceIndex++ }));
	} else {
		if (!unit.field || !group.members.includes(unit.field)) return [];
		values = tagValuesForField(unit.field, ctx).map((value) => ({
			value,
			text: fieldValueToText(value),
			fieldName: unit.field,
			sourceIndex: sourceIndex++,
		}));
	}
	if (unit.regexOps.length === 0) return values;
	return values.flatMap((value) => applyRegexOpsToTexts([value.text], unit.regexOps, line).map((textValue) => ({
		...value,
		value: textValue,
		text: textValue,
		scoreByEntry: false,
	})));
}

function fieldValuesForField(field: string, ctx: RuntimeContext): unknown[] {
	return getValuesForFieldOrTag(field, ctx);
}

function tagValuesForField(field: string, ctx: RuntimeContext): unknown[] {
	const values = new Map<string, unknown>();
	for (const entry of ctx.entries) {
		for (const item of getEntryTagValues(entry.id, field, ctx)) {
			if (fieldValueToText(item).trim().length === 0) continue;
			const normalized = normalizeFieldValue(item);
			values.set(stableValueKey(normalized), normalized);
		}
	}
	return [...values.values()].sort(compareFieldValues);
}

function getEntryTagValues(entryId: string, field: string | undefined, ctx: RuntimeContext): unknown[] {
	const entry = ctx.entriesById.get(entryId);
	if (!entry || !field) return [];
	const tags = getEntryTags(entry, ctx);
	return Object.prototype.hasOwnProperty.call(tags, field)
		? tagValueToList(tags[field]).flatMap((value) => Array.isArray(value) ? value : [value])
		: [];
}

function getEntryFunctionValues(entryId: string, field: string | undefined, ctx: RuntimeContext): unknown[] {
	const entry = ctx.entriesById.get(entryId);
	if (!entry) return [];
	if (!field) return [entry.text || entry.id];
	return getEntryTagValues(entryId, field, ctx);
}

async function evaluateRankingExpression(expr: string, unit: UnitValue, ctx: RuntimeContext, line: number): Promise<number> {
	const trimmed = trimOuterParens(expr.trim());
	const arithmetic = splitArithmetic(trimmed, ["+", "-"]) ?? splitArithmetic(trimmed, ["*", "/"]);
	if (arithmetic) {
		const left = await evaluateRankingExpression(arithmetic.left, unit, ctx, line);
		const right = await evaluateRankingExpression(arithmetic.right, unit, ctx, line);
		switch (arithmetic.operator) {
			case "+": return left + right;
			case "-": return left - right;
			case "*": return left * right;
			case "/":
				if (right === 0) throw new DslRuntimeError("E_DIVIDE_BY_ZERO", "Cannot divide by zero", line);
				return left / right;
		}
	}
	if (/^-?\d+(\.\d+)?$/.test(trimmed)) return Number(trimmed);
	const input = await parseFunctionInput(trimmed, ctx, line);
	const canScoreEntryField = Boolean(input.field && unit.entryId && unit.scoreByEntry);
	const baseValues = canScoreEntryField && unit.entryId && input.field
		? getEntryFunctionValues(unit.entryId, input.field, ctx)
		: [unit.text];
	const texts = applyRegexOpsToTexts(baseValues.map(fieldValueToText), input.regexOps, line);
	return Number(await evaluateFunctionText(input.functionText || "length", texts, {
		entryId: canScoreEntryField ? unit.entryId : undefined,
		field: input.field ?? unit.fieldName,
	}, ctx, line));
}

async function parseFunctionInput(expression: string, ctx: RuntimeContext, line: number): Promise<{ field?: string; regexOps: RegexOperation[]; functionText: string }> {
	let rest = expression.trim();
	let field: string | undefined;
	if (rest.startsWith("[")) {
		const closeIndex = findMatchingClose(rest, 0, "[", "]");
		if (closeIndex < 0) throw new DslRuntimeError("E_FIELD_FILTER", `Could not find closing ] in ${expression}`, line);
		field = await evaluateFieldName(rest.slice(1, closeIndex), ctx, line);
		rest = rest.slice(closeIndex + 1).trim();
	}
	const regexOps = await parseRegexOpsPrefix(rest, ctx, line);
	rest = regexOps.rest.trim();
	return { field, regexOps: regexOps.ops, functionText: rest };
}

async function evaluateFunctionText(
	functionText: string,
	inputTexts: string[],
	context: { entryId?: string; field?: string },
	ctx: RuntimeContext,
	line: number,
): Promise<number> {
	const text = functionText.trim();
	if (!text || text === "length") {
		return inputTexts.length === 1 ? Array.from(inputTexts[0]).length : inputTexts.length;
	}
	const match = text.match(/^(?:(total|avg)\s+)?(?:per\s+(total|avg)\s+)?(?:(BM25|embed|embeddings?)\s+)?similarity\s+to\s+(.+)$/i);
	if (!match) throw functionSyntaxError(functionText, ctx, line);
	const inputAccumulation = (match[1]?.toLowerCase() as "total" | "avg" | undefined) ?? (inputTexts.length > 1 ? "total" : "avg");
	const testAccumulation = (match[2]?.toLowerCase() as "total" | "avg" | undefined) ?? "avg";
	const mode = match[3]?.toLowerCase().startsWith("embed") ? "embed" : "BM25";
	const targets = await evaluateSimilarityTargets(match[4], ctx, line);
	const scores = await Promise.all(inputTexts.map(async (inputText) => {
		const perTarget = await Promise.all(targets.map((target) => scoreSimilarity(mode, inputText, target, testAccumulation, context, ctx, line)));
		return aggregateNumbers(perTarget, testAccumulation);
	}));
	return aggregateNumbers(scores, inputAccumulation);
}

function functionSyntaxError(functionText: string, ctx: RuntimeContext, line: number): DslRuntimeError {
	const text = functionText.trim();
	if (/^\.length\b/i.test(text)) {
		return new DslRuntimeError(
			"E_FUNCTION",
			"Use space-based function syntax after regex operations: .find(r\"pattern\") length, not .find(r\"pattern\").length.",
			line,
		);
	}

	const regexWithoutDot = text.match(/^(find|remove|splice)\s*\(/i);
	if (regexWithoutDot) {
		return new DslRuntimeError(
			"E_FUNCTION",
			`Regex operations must start with a dot when applied to the current unit: use .${regexWithoutDot[1].toLowerCase()}(...) length, or use a field-qualified expression such as [content].${regexWithoutDot[1].toLowerCase()}(...) length.`,
			line,
		);
	}

	const bareFieldRegex = text.match(/^([A-Za-z_][A-Za-z0-9_]*)\.(find|remove|splice)\s*\(/i);
	if (bareFieldRegex) {
		const name = bareFieldRegex[1];
		const op = bareFieldRegex[2].toLowerCase();
		if (name === "entry") {
			return new DslRuntimeError(
				"E_FUNCTION",
				`There is no entry pseudo-field. To search document text, use a real field such as [content].${op}(...) length or [text].${op}(...) length. Omitting [FIELD] operates on raw entry IDs for entries, not on document text.`,
				line,
			);
		}
		if (getAllFieldNames(ctx).includes(name)) {
			return new DslRuntimeError(
				"E_FUNCTION",
				`Field references require brackets: use [${name}].${op}(...) length, not ${name}.${op}(...) length.`,
				line,
			);
		}
		return new DslRuntimeError(
			"E_FUNCTION",
			`Unknown field-style expression ${name}.${op}(...). Field references require brackets, for example [content].${op}(...) length.`,
			line,
		);
	}

	const bareSimilarity = text.match(/^([A-Za-z_][A-Za-z0-9_]*)\s+(?:(?:total|avg)\s+)?(?:(?:BM25|embed|embeddings?)\s+)?similarity\s+to\b/i);
	if (bareSimilarity) {
		const name = bareSimilarity[1];
		if (name === "entry") {
			return new DslRuntimeError(
				"E_FUNCTION",
				"There is no entry pseudo-field for similarity. To search document text, use a real field such as [content] BM25 similarity to \"query\". Omitting [FIELD] scores raw entry IDs for entries, not document text.",
				line,
			);
		}
		if (getAllFieldNames(ctx).includes(name)) {
			return new DslRuntimeError(
				"E_FUNCTION",
				`Field references require brackets: use [${name}] BM25 similarity to "query", not ${name} BM25 similarity to "query".`,
				line,
			);
		}
	}

	return new DslRuntimeError(
		"E_FUNCTION",
		`Unknown function: ${functionText}. Expected length or similarity syntax such as BM25 similarity to "query". Regex operations use dot syntax before the function, for example .find(r"pattern") length or [content].find(r"pattern") length.`,
		line,
	);
}

async function evaluateSimilarityTargets(text: string, ctx: RuntimeContext, line: number): Promise<SimilarityTarget[]> {
	const cacheKey = similarityTargetExpressionCacheKey(text, ctx);
	const cached = ctx.similarityTargetsCache.get(cacheKey);
	if (cached) return cached;
	const promise = evaluateSimilarityTargetsUncached(text, ctx, line);
	ctx.similarityTargetsCache.set(cacheKey, promise);
	try {
		return await promise;
	} catch (error) {
		ctx.similarityTargetsCache.delete(cacheKey);
		throw error;
	}
}

async function evaluateSimilarityTargetsUncached(text: string, ctx: RuntimeContext, line: number): Promise<SimilarityTarget[]> {
	const group = await tryResolveSimilarityGroupTarget(text, ctx, line);
	if (group) return [group];
	const value = await evaluateLooseValue(text, ctx, line);
	const valueGroup = await similarityTargetFromValue(value, text, ctx, line);
	if (valueGroup) return [valueGroup];
	const values = Array.isArray(value) ? value : [value];
	return values.map((item) => ({ kind: "text", text: fieldValueToText(item) }));
}

function similarityTargetExpressionCacheKey(text: string, ctx: RuntimeContext): string {
	const trimmed = trimOuterParens(text.trim());
	return trimmed;
}

async function tryResolveSimilarityGroupTarget(text: string, ctx: RuntimeContext, line: number): Promise<SimilarityTarget | undefined> {
	const trimmed = trimOuterParens(text.trim());
	if (!looksLikeSimilarityGroupTarget(trimmed, ctx)) return undefined;
	return entryGroupSimilarityTarget(await resolveScopedGroupExpression(trimmed, ctx, line), line);
}

function looksLikeSimilarityGroupTarget(text: string, ctx: RuntimeContext): boolean {
	if (!text || isStringLiteral(text) || isRawStringLiteral(text)) return false;
	if (text === "G0" || text === "G1" || /^G\d+$/.test(text)) return true;
	if (ctx.entriesById.has(text) || ctx.state.groups[text]) return true;
	if (text.startsWith("scope:")) return true;
	if (/^(not\s+|temp\(|group\(|group_expr\()/i.test(text)) return true;
	if (looksLikeGroupSpec(text)) return true;
	return splitTopLevelByWord(text, "and").length > 1 || splitTopLevelByWord(text, "or").length > 1;
}

async function similarityTargetFromValue(value: unknown, rawText: string, ctx: RuntimeContext, line: number): Promise<SimilarityTarget | undefined> {
	if (isGroupExpressionValue(value)) {
		return entryGroupSimilarityTarget(await resolveScopedGroupExpression(value.expression, ctx, line), line);
	}
	const scoped = scopedMembersFromSimilarityValue(value, rawText, ctx, line);
	if (scoped) return entryGroupSimilarityTarget(scoped, line);
	const entryId = entryIdFromSimilarityValue(value, ctx);
	if (entryId && !isStringLiteral(trimOuterParens(rawText.trim()))) return entryGroupSimilarityTarget(scopedMembers("entries", [entryId], rawText), line);
	const rawTextNamesVariable = Object.prototype.hasOwnProperty.call(ctx.variables, trimOuterParens(rawText.trim()));
	if (typeof value === "string" && !rawTextNamesVariable && !isStringLiteral(trimOuterParens(rawText.trim())) && looksLikeSimilarityGroupTarget(value, ctx)) {
		return entryGroupSimilarityTarget(await resolveScopedGroupExpression(value, ctx, line), line);
	}
	if (!Array.isArray(value)) return undefined;
	const ids = value.map((item) => entryIdFromSimilarityValue(item, ctx));
	if (ids.every((id): id is string => id !== undefined)) {
		return entryGroupSimilarityTarget(scopedMembers("entries", ids, rawText), line);
	}
	if (ids.some((id) => id !== undefined)) {
		throw new DslRuntimeError("E_SIMILARITY_TARGET", "Similarity target lists must contain either only entry ids or only literal target values.", line);
	}
	return undefined;
}

function scopedMembersFromSimilarityValue(value: unknown, rawText: string, ctx: RuntimeContext, line: number): ScopedMembers | undefined {
	if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
	const record = value as Record<string, unknown>;
	const scope = record.scope === "fields" ? "fields" : record.scope === "entries" ? "entries" : undefined;
	if (!scope) return undefined;
	const rawMembers = Array.isArray(record.members)
		? record.members
		: scope === "entries" && Array.isArray(record.entryIds)
			? record.entryIds
			: scope === "fields" && Array.isArray(record.fieldNames)
				? record.fieldNames
				: undefined;
	if (!rawMembers) return undefined;
	const members = rawMembers.map((member) => coerceString(member));
	if (scope === "entries" && !members.every((member) => ctx.entriesById.has(member))) {
		throw new DslRuntimeError("E_SIMILARITY_TARGET", "Entry-group similarity target contains unknown entry ids.", line);
	}
	return scopedMembers(scope, members, typeof record.id === "string" ? record.id : rawText);
}

function entryIdFromSimilarityValue(value: unknown, ctx: RuntimeContext): string | undefined {
	if (typeof value === "string" && ctx.entriesById.has(value)) return value;
	if (value && typeof value === "object" && !Array.isArray(value)) {
		const id = (value as Record<string, unknown>).id;
		if (typeof id === "string" && ctx.entriesById.has(id)) return id;
	}
	return undefined;
}

function entryGroupSimilarityTarget(group: ScopedMembers, line: number): SimilarityTarget {
	if (group.scope !== "entries") {
		throw new DslRuntimeError("E_SIMILARITY_TARGET", "Similarity targets must be entry-scoped groups or entry-id lists; field-scoped groups such as G1 cannot be used as similarity targets.", line);
	}
	return {
		kind: "entryGroup",
		members: group.members,
		spec: group.spec,
		cacheKey: similarityGroupMembersCacheKey(group.members),
	};
}

function similarityGroupMembersCacheKey(members: string[]): string {
	return [
		members.length,
		hashStrings(members),
		members[0] ?? "",
		members[members.length - 1] ?? "",
	].join(":");
}

function hashStrings(values: readonly string[]): string {
	let hash = 2166136261;
	for (const value of values) {
		for (let index = 0; index < value.length; index++) {
			hash ^= value.charCodeAt(index);
			hash = Math.imul(hash, 16777619);
		}
		hash ^= 0;
		hash = Math.imul(hash, 16777619);
	}
	return (hash >>> 0).toString(36);
}

async function scoreSimilarity(
	mode: "BM25" | "embed",
	inputText: string,
	target: SimilarityTarget,
	targetAccumulation: "total" | "avg",
	context: { entryId?: string; field?: string },
	ctx: RuntimeContext,
	line: number,
): Promise<number> {
	if (target.kind === "entryGroup") {
		if (context.entryId && ctx.entriesById.has(context.entryId)) {
			const scores = mode === "BM25"
				? await getBm25GroupScoreVector(target, context.field, targetAccumulation, ctx)
				: await getEmbeddingGroupScoreVector(target, context.field, targetAccumulation, ctx, line);
			return scoreForId(scores, context.entryId, ctx);
		}
		if (mode === "embed") {
			throw new DslRuntimeError("E_SIMILARITY_TARGET", "Group embedding targets require an entry-backed unit such as entries of G0 sorted by ([field] embed similarity to G0).", line);
		}
		return lexicalSimilarity(inputText, groupTargetTexts(target, context.field, ctx).join(" "));
	}
	if (context.entryId && ctx.entriesById.has(context.entryId)) {
		const predicate: FieldTextPredicate = { field: context.field, text: target.text };
		if (mode === "BM25") return scoreBm25(context.entryId, predicate, ctx);
		const scoreVector = await getEmbeddingScoreVector(predicate, ctx, line);
		return scoreForId(scoreVector, context.entryId, ctx);
	}
	return lexicalSimilarity(inputText, target.text);
}

function lexicalSimilarity(inputText: string, target: string): number {
	const input = tokenize(inputText);
	const query = tokenize(target);
	if (input.length === 0 || query.length === 0) return 0;
	const counts = new Map<string, number>();
	for (const token of input) counts.set(token, (counts.get(token) ?? 0) + 1);
	let score = 0;
	for (const token of query) score += counts.get(token) ?? 0;
	return score / Math.sqrt(input.length * query.length);
}

function aggregateNumbers(values: number[], mode: "total" | "avg"): number {
	if (values.length === 0) return 0;
	const total = values.reduce((sum, value) => sum + value, 0);
	return mode === "avg" ? total / values.length : total;
}

async function evaluateFieldName(text: string, ctx: RuntimeContext, line: number): Promise<string> {
	const trimmed = text.trim();
	if (isStringLiteral(trimmed)) return parseStringLiteral(trimmed, line);
	if (trimmed in ctx.variables) return coerceString(ctx.variables[trimmed]);
	return trimOuterParens(trimmed);
}

async function parseRegexOps(text: string, ctx: RuntimeContext, line: number): Promise<RegexOperation[]> {
	const parsed = await parseRegexOpsPrefix(text, ctx, line);
	if (parsed.rest.trim()) throw regexSyntaxError(text, parsed.rest.trim(), line);
	return parsed.ops;
}

function regexSyntaxError(original: string, rest: string, line: number): DslRuntimeError {
	const trimmedRest = rest.trim();
	if (/^\.length\b/i.test(trimmedRest)) {
		return new DslRuntimeError(
			"E_REGEX",
			"Use space-based function syntax after regex operations: entries[field].find(r\"pattern\") of GROUP-EXPR sorted by (length), not entries[field].find(r\"pattern\").length.",
			line,
		);
	}
	const missingDot = trimmedRest.match(/^(find|remove|splice)\s*\(/i);
	if (missingDot) {
		return new DslRuntimeError(
			"E_REGEX",
			`Regex operations on units must start with a dot: use .${missingDot[1].toLowerCase()}(...) after entries[FIELD] or fields[FIELD].`,
			line,
		);
	}
	if (trimmedRest.startsWith(".")) {
		return new DslRuntimeError(
			"E_REGEX",
			`Unknown regex operation in ${original}. Valid unit regex operations are .find(...), .remove(...), and .splice(i, j), chained before of GROUP-EXPR.`,
			line,
		);
	}
	return new DslRuntimeError("E_REGEX", `Could not parse regex chain: ${original}. Unit regex operations must look like .find(r"pattern"), .remove(r"pattern"), or .splice(i, j).`, line);
}

async function parseRegexOpsPrefix(text: string, ctx: RuntimeContext, line: number): Promise<{ ops: RegexOperation[]; rest: string }> {
	const ops: RegexOperation[] = [];
	let rest = text.trim();
	while (rest.startsWith(".")) {
		const match = rest.match(/^\.(find|remove|splice)\s*\(/i);
		if (!match) break;
		const openIndex = rest.indexOf("(", match[0].indexOf("("));
		const closeIndex = findMatchingClose(rest, openIndex, "(", ")");
		if (closeIndex < 0) throw new DslRuntimeError("E_REGEX", `Could not find closing ) in regex operation ${rest}`, line);
		const arg = rest.slice(openIndex + 1, closeIndex).trim();
		const name = match[1].toLowerCase();
		if (name === "splice") {
			const parts = splitTopLevel(arg, ",");
			if (parts.length !== 2) throw new DslRuntimeError("E_REGEX_SPLICE", "splice() requires start and end indices", line);
			ops.push({ type: "splice", start: Number(parts[0].trim()), end: Number(parts[1].trim()) });
		} else {
			ops.push({ type: name as "find" | "remove", pattern: await parseRegexPatternArg(arg, ctx, line) });
		}
		rest = rest.slice(closeIndex + 1).trim();
	}
	return { ops, rest };
}

async function parseRegexPatternArg(text: string, ctx: RuntimeContext, line: number): Promise<string> {
	const trimmed = text.trim();
	if (isRawStringLiteral(trimmed)) return parseRawStringLiteral(trimmed.slice(1), line);
	if (isStringLiteral(trimmed)) return parseStringLiteral(trimmed, line);
	try {
		const value = await evaluateExpression(trimmed, ctx, line);
		if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") return String(value);
		if (value === null) return "null";
		throw new DslRuntimeError("E_REGEX", `Regex pattern expression must evaluate to a scalar value, got ${formatValueForError(value)}`, line);
	} catch (error) {
		if (error instanceof DslRuntimeError && error.code !== "E_PARSE_EXPR") throw error;
	}
	return trimmed;
}

function parseRawStringLiteral(text: string, line: number): string {
	if (!isStringLiteral(text)) throw new DslRuntimeError("E_STRING", `Invalid raw string literal r${text}`, line);
	const body = text.slice(1, -1);
	let trailingBackslashes = 0;
	for (let i = body.length - 1; i >= 0 && body[i] === "\\"; i--) trailingBackslashes++;
	if (trailingBackslashes % 2 === 1) {
		throw new DslRuntimeError("E_STRING", `Invalid raw string literal r${text}`, line);
	}
	return body;
}

function applyRegexOpsToTexts(texts: string[], ops: RegexOperation[], line: number): string[] {
	let out = texts;
	for (const op of ops) {
		if (op.type === "splice") {
			out = out.map((value) => value.slice(op.start ?? 0, op.end));
			continue;
		}
		const regex = compileRegex(op.pattern ?? "", line);
		if (op.type === "remove") {
			out = out.map((value) => value.replace(regex, ""));
		} else {
			out = out.flatMap((value) => [...value.matchAll(regex)].map((match) => match[0]).filter((match) => match.length > 0));
		}
	}
	return out;
}

function compileRegex(pattern: string, line: number): RegExp {
	if (!pattern) throw new DslRuntimeError("E_REGEX", "Regex pattern cannot be empty", line);
	try {
		const translated = translatePythonInlineRegexFlags(pattern, line);
		return new RegExp(translated.pattern, translated.flags);
	} catch (error) {
		if (error instanceof DslRuntimeError) throw error;
		throw new DslRuntimeError("E_REGEX", `Invalid regex ${pattern}: ${error instanceof Error ? error.message : String(error)}`, line);
	}
}

function translatePythonInlineRegexFlags(pattern: string, line: number): { pattern: string; flags: string } {
	let source = pattern;
	const flags = new Set(["g"]);
	while (true) {
		const match = source.match(/^\(\?([A-Za-z]+)\)/);
		if (!match) break;
		for (const flag of match[1]) {
			if (flag === "i" || flag === "m" || flag === "s") flags.add(flag);
			else if (flag === "u") {
				// Python's unicode flag is the default for Python 3 strings and is harmless here.
				continue;
			} else {
				throw new DslRuntimeError("E_REGEX", `Unsupported Python regex inline flag (?${flag}). Supported flags are i, m, and s.`, line);
			}
		}
		source = source.slice(match[0].length);
	}
	if (/^\(\?[A-Za-z-]+:/.test(source)) {
		throw new DslRuntimeError("E_REGEX", "Scoped inline regex flags like (?i:...) are not supported; put (?i), (?m), or (?s) at the start of the pattern.", line);
	}
	if (/\(\?[A-Za-z-]+\)/.test(source)) {
		throw new DslRuntimeError("E_REGEX", "Inline regex flags must appear at the start of the pattern.", line);
	}
	return { pattern: source, flags: [...flags].join("") };
}

async function countGroup(arg: string, ctx: RuntimeContext, line: number): Promise<number> {
	return (await resolveGroupExpression(arg, ctx, line)).size;
}

async function countBy(arg: string, ctx: RuntimeContext, line: number): Promise<Array<Record<string, FieldValue | string | number>>> {
	arg = resolveCommandTextArgument(arg, ctx, line);
	const parts = splitByTopLevelOperator(arg, " of ");
	if (!parts) throw new DslRuntimeError("E_PARSE_COUNT_BY", 'Expected count_by(["field", ...] of <group_expression>)', line);
	const fieldValues = await evaluateExpression(parts[0], ctx, line);
	if (!Array.isArray(fieldValues) || fieldValues.length === 0) {
		throw new DslRuntimeError("E_PARSE_COUNT_BY", "count_by requires a non-empty list of fields", line);
	}
	const fields = fieldValues.map(String);
	const group = await resolveScopedGroupExpression(parts[2], ctx, line);
	if (group.scope !== "entries") {
		throw new DslRuntimeError("E_COUNT_BY_SCOPE", "count_by() requires an entry-scoped GROUP-EXPR", line);
	}
	const ids = group.members;
	const buckets = new Map<string, { values: Array<FieldValue | string>; count: number }>();
	for (const id of ids) {
		const entry = ctx.entriesById.get(id);
		if (!entry) continue;
		for (const values of expandFieldOrTagValues(fields, entry, ctx)) {
			const key = JSON.stringify(values);
			const bucket = buckets.get(key);
			if (bucket) bucket.count++;
			else buckets.set(key, { values, count: 1 });
		}
	}
	return [...buckets.values()]
		.sort((a, b) => a.values.map(formatInlineValue).join("\u0000").localeCompare(b.values.map(formatInlineValue).join("\u0000")))
		.map(({ values, count }) => {
			const row: Record<string, FieldValue | string | number> = {};
			for (const [index, field] of fields.entries()) row[field] = values[index];
			row.count = count;
			return row;
		});
}

function expandFieldOrTagValues(fields: string[], entry: QuailEntry, ctx: RuntimeContext): Array<Array<FieldValue | string>> {
	const valuesByField = fields.map((field) => {
		const values = tagValueToList(getEntryTags(entry, ctx)[field])
			.filter((value) => fieldValueToText(value).trim().length > 0)
			.map(normalizeFieldValue);
		const unique = uniqueFieldValues(values);
		return unique.length > 0 ? unique : ["(missing)"];
	});
	let rows: Array<Array<FieldValue | string>> = [[]];
	for (const values of valuesByField) {
		const nextRows: Array<Array<FieldValue | string>> = [];
		for (const row of rows) {
			for (const value of values) nextRows.push([...row, value]);
		}
		rows = nextRows;
	}
	return rows;
}

function uniqueFieldValues(values: Array<FieldValue | string>): Array<FieldValue | string> {
	const seen = new Set<string>();
	const out: Array<FieldValue | string> = [];
	for (const value of values) {
		const key = stableValueKey(value);
		if (seen.has(key)) continue;
		seen.add(key);
		out.push(value);
	}
	return out;
}

async function resolveGroupSpec(specText: string, ctx: RuntimeContext, line: number): Promise<Set<string>> {
	return (await resolveGroupSpecBits(specText, ctx, line)).toIdSet(ctx);
}

async function resolveGroupSpecBits(specText: string, ctx: RuntimeContext, line: number): Promise<IdBitSet> {
	const spec = await parseGroupSpec(specText, ctx, line);
	const hasConstraints = spec.fieldsCompare.length > 0 ||
		Boolean(spec.tags) ||
		spec.contains.length > 0 ||
		spec.containsWord.length > 0 ||
		spec.bm25.length > 0 ||
		spec.embeddings.length > 0;
	let ids = !hasConstraints && spec.include ? IdBitSet.empty(ctx.entries.length) : IdBitSet.full(ctx.entries.length);
	if (spec.fieldsCompare.length > 0) {
		ids = getIdsComparingFields(spec.fieldsCompare, ctx, line);
	}
	if (spec.tags) ids = ids.and(getIdsMatchingTagsBits(spec.tags, ctx));
	for (const predicate of spec.contains) {
		ids = ids.and(getIdsContaining(predicate, ctx));
	}
	for (const predicate of spec.containsWord) {
		ids = ids.and(getIdsContainingWord(predicate, ctx, line));
	}
	for (const predicate of spec.bm25) {
		ids = ids.and(getIdsAboveBm25Threshold(predicate, predicate.threshold ?? 0, ctx));
	}
	for (const predicate of spec.embeddings) {
		ids = ids.and(await getIdsAboveEmbeddingThreshold(predicate, predicate.threshold ?? 0, ctx, line));
	}
	if (spec.include || spec.exclude) ids = ids.clone();
	for (const id of spec.include ?? []) ids.addId(id, ctx);
	for (const id of spec.exclude ?? []) ids.deleteId(id, ctx);
	return ids;
}

async function createGroup(specText: string, ctx: RuntimeContext, line: number): Promise<string> {
	const ids = await resolveGroupSpec(specText, ctx, line);
	while (ctx.state.nextGroupNumber < 2 || ctx.state.groups[`G${ctx.state.nextGroupNumber}`]) {
		ctx.state.nextGroupNumber++;
	}
	const groupId = `G${ctx.state.nextGroupNumber++}`;
	ctx.state.groups[groupId] = {
		id: groupId,
		datasets: ctx.activeDatasetNames,
		scope: "entries",
		spec: specText,
		members: [...ids],
		entryIds: [...ids],
		createdAt: new Date().toISOString(),
	};
	clearGroupExpressionCache(ctx);
	return groupId;
}

function createGroupExpression(expression: string): GroupExpressionValue {
	return {
		__quailType: "group_expression",
		expression: trimOuterParens(expression.trim()),
	};
}

function getStableOrdinal(entry: QuailEntry, entryIndex: number): number {
	const ordinal = Number(entry.ordinal);
	return Number.isFinite(ordinal) && ordinal > 0 ? ordinal : entryIndex + 1;
}

async function parseFilter(text: string, ctx: RuntimeContext, line: number): Promise<FilterSpec> {
	const trimmed = trimOuterParens(text.trim());
	const match = trimmed.match(/^(BM25|embed|embeddings)\s*:\s*(.+)$/is);
	if (!match) {
		if (/^contains(?:_word)?\s*:/i.test(trimmed) || /^tags\s*:/i.test(trimmed)) {
			throw new DslRuntimeError("E_PARSE_FILTER", "Distributions use BM25 or embeddings filters. For contains/tags, build a GROUP-EXPR and use count(UNIT of GROUP-EXPR) or retrieve(DIRECTION AMOUNT UNIT of GROUP-EXPR).", line);
		}
		throw new DslRuntimeError("E_PARSE_FILTER", `Expected BM25: "text", embed: "text", or embeddings: "text", got ${text}`, line);
	}
	const rawText = match[2].trim();
	const parsed = rawText.startsWith("[")
		? await parseFieldTextThreshold(rawText, ctx, line)
		: { text: coerceString(await evaluateExpression(rawText, ctx, line)) };
	return { type: match[1] === "BM25" ? "BM25" : "embeddings", field: parsed.field, text: parsed.text };
}

async function parseGroupSpec(text: string, ctx: RuntimeContext, line: number): Promise<GroupSpec> {
	const spec: GroupSpec = { bm25: [], embeddings: [], contains: [], containsWord: [], fieldsCompare: [] };
	for (const part of splitTopLevel(text, ",")) {
		if (!part.trim()) continue;
		const [rawKey, rawValue] = splitKeyValue(part, line);
		const key = rawKey.trim().toLowerCase();
		const value = rawValue.trim();
		if (key === "bm25") {
			spec.bm25.push(await parseFieldTextThreshold(value, ctx, line));
		} else if (key === "embeddings" || key === "embedding") {
			spec.embeddings.push(await parseFieldTextThreshold(value, ctx, line));
		} else if (key === "contains") {
			spec.contains.push(await parseFieldTextThreshold(value, ctx, line));
		} else if (key === "contains_word" || key === "containsword") {
			spec.containsWord.push(await parseFieldTextThreshold(value, ctx, line));
		} else if (key === "include") {
			spec.include = await parseIdListExpression(value, ctx, line, "include");
		} else if (key === "exclude") {
			spec.exclude = await parseIdListExpression(value, ctx, line, "exclude");
		} else if (key === "fields_compare" || key === "field_compare") {
			spec.fieldsCompare.push(...await parseFieldComparisons(value, ctx, line));
		} else if (key === "tags") {
			spec.tags = await parseTagPairs(value, ctx, line);
		} else {
			throw new DslRuntimeError("E_GROUP_SPEC_KEY", `Unknown group spec field ${rawKey}`, line);
		}
	}
	return spec;
}

async function parseIdListExpression(text: string, ctx: RuntimeContext, line: number, key: "include" | "exclude"): Promise<string[]> {
	const trimmed = text.trim();
	const value = isListLiteralExpression(trimmed)
		? await parseList(trimmed, ctx, line)
		: await evaluateExpression(trimmed, ctx, line);
	if (!Array.isArray(value)) {
		throw new DslRuntimeError("E_GROUP_SPEC_VALUE", `${key} requires a list of entry ids, got ${formatValueForError(value)}`, line);
	}
	return value.map((item) => coerceId(item));
}

function splitKeyValue(text: string, line: number): [string, string] {
	const index = text.indexOf(":");
	if (index < 0) throw new DslRuntimeError("E_GROUP_SPEC", `Expected key: value in group spec part ${text}`, line);
	return [text.slice(0, index), text.slice(index + 1)];
}

function parseTextThreshold(text: string, line: number): { text: string; thresholdExpression?: string } {
	const match = text.match(/^((?:"(?:\\.|[^"])*")|(?:'(?:\\.|[^'])*')|.+?)(?:\s*>\s*(.+))?$/s);
	if (!match) throw new DslRuntimeError("E_THRESHOLD", `Could not parse text/threshold: ${text}`, line);
	return { text: match[1].trim(), thresholdExpression: match[2]?.trim() };
}

async function parseFieldTextThreshold(text: string, ctx: RuntimeContext, line: number): Promise<FieldTextPredicate> {
	const trimmed = text.trim();
	if (!trimmed.startsWith("[")) {
		const parsed = parseTextThreshold(trimmed, line);
		return {
			text: coerceString(await evaluateExpression(parsed.text, ctx, line)),
			threshold: await evaluateOptionalThreshold(parsed.thresholdExpression, ctx, line),
		};
	}
	const closeIndex = findMatchingClose(trimmed, 0, "[", "]");
	if (closeIndex < 0) throw new DslRuntimeError("E_FIELD_FILTER", `Could not find closing ] in field filter ${text}`, line);
	const thresholdText = trimmed.slice(closeIndex + 1).trim();
	const thresholdMatch = thresholdText.match(/^(?:>\s*(.+))?$/s);
	if (!thresholdMatch) throw new DslRuntimeError("E_THRESHOLD", `Could not parse threshold in ${text}`, line);
	const pairs = await parseFieldPairs(trimmed.slice(0, closeIndex + 1), ctx, line);
	const entries = Object.entries(pairs);
	if (entries.length !== 1) throw new DslRuntimeError("E_FIELD_FILTER", `Expected exactly one field: value pair in ${text}`, line);
	const [field, value] = entries[0];
	return { field, text: coerceString(value), threshold: await evaluateOptionalThreshold(thresholdMatch[1]?.trim(), ctx, line) };
}

async function evaluateOptionalThreshold(expr: string | undefined, ctx: RuntimeContext, line: number): Promise<number | undefined> {
	if (!expr) return undefined;
	const value = await evaluateExpression(expr, ctx, line);
	if (typeof value !== "number" || !Number.isFinite(value)) {
		throw new DslRuntimeError("E_THRESHOLD", `Threshold must evaluate to a finite number, got ${formatValue(value)}`, line);
	}
	return value;
}

async function parseFieldPairs(text: string, ctx: RuntimeContext, line: number): Promise<Record<string, FieldValue>> {
	const inner = text.trim().replace(/^\[/, "").replace(/\]$/, "");
	const fields: Record<string, FieldValue> = {};
	for (const part of splitTopLevel(inner, ",")) {
		if (!part.trim()) continue;
		const [fieldExpr, valueExpr] = splitKeyValue(part, line);
		const field = await parsePairFieldName(fieldExpr.trim(), ctx, line);
		fields[field] = normalizeFieldValue(await evaluateExpression(valueExpr.trim(), ctx, line));
	}
	return fields;
}

async function parseFieldComparisons(text: string, ctx: RuntimeContext, line: number): Promise<FieldComparison[]> {
	const pairs = await parseFieldPairs(text, ctx, line);
	return Object.entries(pairs).map(([field, rawComparison]) => parseFieldComparison(field, rawComparison, line));
}

function parseFieldComparison(field: string, rawComparison: FieldValue, line: number): FieldComparison {
	if (!Array.isArray(rawComparison) || rawComparison.length !== 2) {
		throw new DslRuntimeError("E_FIELD_COMPARE", `fields_compare values must be [operator, value], got ${formatValue(rawComparison)} for field ${field}`, line);
	}
	const [rawOperator, value] = rawComparison;
	if (typeof rawOperator !== "string" || !isFieldComparisonOperator(rawOperator)) {
		throw new DslRuntimeError("E_FIELD_COMPARE", `fields_compare operator must be one of ==, !=, >, <, >=, <=, got ${formatValue(rawOperator)} for field ${field}`, line);
	}
	return { field, operator: rawOperator, value: normalizeFieldValue(value) };
}

function isFieldComparisonOperator(value: string): value is FieldComparisonOperator {
	return value === "==" || value === "!=" || value === ">" || value === "<" || value === ">=" || value === "<=";
}

async function parsePairFieldName(text: string, ctx: RuntimeContext, line: number): Promise<string> {
	if (isStringLiteral(text)) return parseStringLiteral(text, line);
	return coerceString(await evaluateExpression(text, ctx, line));
}

function normalizeFieldValue(value: unknown): FieldValue {
	if (value === undefined || value === null) return null;
	if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") return value;
	if (Array.isArray(value)) return value.map(normalizeFieldValue);
	if (typeof value === "object") {
		return Object.fromEntries(Object.entries(value as Record<string, unknown>).map(([key, item]) => [key, normalizeFieldValue(item)]));
	}
	return String(value);
}

async function parseTagPairs(text: string, ctx: RuntimeContext, line: number): Promise<Record<string, unknown>> {
	const inner = text.trim().replace(/^\[/, "").replace(/\]$/, "");
	const tags: Record<string, unknown> = {};
	for (const part of splitTopLevel(inner, ",")) {
		if (!part.trim()) continue;
		const [field, value] = splitKeyValue(part, line);
		tags[parseStringLiteral(field.trim(), line)] = normalizeFieldValue(await evaluateExpression(value.trim(), ctx, line));
	}
	return tags;
}

async function resolveGroupExpression(expr: string, ctx: RuntimeContext, line: number): Promise<IdBitSet> {
	const text = trimOuterParens(expr.trim());
	const cacheable = isCacheableGroupExpressionText(text, ctx);

	if (cacheable) {
		const cached = ctx.groupExpressionCache.get(text);
		if (cached) return cached;
	}

	const result = await resolveGroupExpressionUncached(text, ctx, line);

	if (cacheable) {
		ctx.groupExpressionCache.set(text, result);
	}

	return result;
}

async function resolveGroupExpressionUncached(text: string, ctx: RuntimeContext, line: number): Promise<IdBitSet> {
	if (!text || text === "G0") return IdBitSet.full(ctx.entries.length);
	if (ctx.state.groups[text]) {
		const group = ctx.state.groups[text];
		if ((group.scope ?? "entries") !== "entries") {
			throw new DslRuntimeError("E_GROUP_SCOPE", `${text} is field-scoped and cannot be used where entry ids are required`, line);
		}
		return IdBitSet.fromIds(group.entryIds ?? group.members ?? [], ctx);
	}
	if (text in ctx.variables) {
		const value = ctx.variables[text];
		if (isGroupExpressionValue(value)) return resolveGroupExpression(value.expression, ctx, line);
		if (typeof value === "string") return resolveGroupExpression(value, ctx, line);
		throw new DslRuntimeError("E_GROUP_EXPR_VARIABLE", `Variable ${text} does not contain a group expression. Use group_expr(<group_expression>) when assigning reusable group expressions.`, line);
	}
	const orParts = splitTopLevelByWord(text, "or");
	if (orParts.length > 1) {
		let out = IdBitSet.empty(ctx.entries.length);
		for (const part of orParts) out = out.or(await resolveGroupExpression(part, ctx, line));
		return out;
	}
	const andParts = splitTopLevelByWord(text, "and");
	if (andParts.length > 1) {
		let out: IdBitSet | undefined;
		for (const part of andParts) {
			const partSet = await resolveGroupExpression(part, ctx, line);
			out = out ? out.and(partSet) : partSet;
		}
		return out ?? IdBitSet.empty(ctx.entries.length);
	}
	if (text.startsWith("not ")) {
		const excluded = await resolveGroupExpression(text.slice(4), ctx, line);
		return excluded.not();
	}
	if (text.startsWith("group(")) {
		const id = await createGroup(innerCall(text), ctx, line);
		return IdBitSet.fromIds(ctx.state.groups[id].entryIds ?? ctx.state.groups[id].members, ctx);
	}
	if (text.startsWith("group_expr(")) {
		return resolveGroupExpression(innerCall(text), ctx, line);
	}
	if (text.startsWith("temp(")) {
		return resolveGroupSpecBits(innerCall(text), ctx, line);
	}
	if (looksLikeGroupSpec(text)) {
		return resolveGroupSpecBits(text, ctx, line);
	}
	throw unknownEntryGroupExpressionError(text, line);
}

function unknownEntryGroupExpressionError(text: string, line: number): DslRuntimeError {
	const trimmed = trimOuterParens(text.trim());
	if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(trimmed)) {
		return new DslRuntimeError(
			"E_GROUP_EXPR",
			`No entry group variable named ${trimmed} exists. If ${trimmed} was assigned from g_save(...), check earlier errors: failed g_save assignments do not create groups. Use ${trimmed} = g_save((scope: G0, ...)), G0, temp(...), or count(UNIT of GROUP-EXPR) for field-scoped units.`,
			line,
		);
	}
	return new DslRuntimeError("E_GROUP_EXPR", `Unknown entry group expression: ${text}. Use G0, a saved entry group variable, temp(...), or count(UNIT of GROUP-EXPR) for field-scoped units.`, line);
}

function looksLikeGroupSpec(text: string): boolean {
	return /^(BM25|embeddings?|contains|contains_word|containsword|fields_compare|field_compare|include|exclude|tags)\s*:/i.test(text.trim());
}

function isGroupExpressionValue(value: unknown): value is GroupExpressionValue {
	return (
		typeof value === "object" &&
		value !== null &&
		(value as { __quailType?: unknown }).__quailType === "group_expression" &&
		typeof (value as { expression?: unknown }).expression === "string"
	);
}

function getFieldText(id: string, field: string | undefined, ctx: RuntimeContext): string {
	const entry = ctx.entriesById.get(id);
	if (!entry) return "";
	if (!field) return entry.text;
	const dataset = ctx.datasetByEntryId.get(id);
	const fieldType = dataset?.manifest.fieldTypes?.[field];
	if (fieldType && fieldType !== "string") return "";
	return getEntryTagValues(id, field, ctx).map((value) => typeof value === "string" ? value : fieldValueToText(value)).join(" ");
}

function getFieldContains(id: string, field: string | undefined, ctx: RuntimeContext): string {
	const entry = ctx.entriesById.get(id);
	if (!entry) return "";
	if (!field) return entry.contains;
	const dataset = ctx.datasetByEntryId.get(id);
	const fieldType = dataset?.manifest.fieldTypes?.[field];
	if (fieldType && fieldType !== "string") return "";
	if (Object.prototype.hasOwnProperty.call(ctx.state.tagsByEntry[id] ?? {}, field)) return normalizeContainsText(getFieldText(id, field, ctx));
	return entry.fieldContains[field] ?? normalizeContainsText(getFieldText(id, field, ctx));
}

function getIdsComparingFields(comparisons: FieldComparison[], ctx: RuntimeContext, line: number): IdBitSet {
	const key = comparisons.map((comparison) => [
		comparison.field,
		comparison.operator,
		stableValueKey(comparison.value),
	].join("\u0001")).join("\u0002");
	const cached = ctx.runtimeCache.fieldComparisonIdSets.get(key);
	if (cached) {
		ctx.runtimeCache.stats.fieldComparisonHits++;
		return cached;
	}
	ctx.runtimeCache.stats.fieldComparisonMisses++;
	const ids = IdBitSet.fromPredicate(ctx, (entry) => comparisons.every((comparison) =>
		fieldComparisonValuesMatch(getEntryTagValues(entry.id, comparison.field, ctx).map(normalizeFieldValue), comparison, line),
	));
	ctx.runtimeCache.fieldComparisonIdSets.set(key, ids);
	return ids;
}

function fieldValueEquals(left: unknown, right: unknown): boolean {
	return stableValueKey(left) === stableValueKey(right);
}

function fieldComparisonValuesMatch(values: FieldValue[], comparison: FieldComparison, line: number): boolean {
	if (values.length === 0) return false;
	if (comparison.operator === "!=") return values.every((value) => !fieldValueEquals(value, comparison.value));
	return values.some((value) => fieldComparisonMatches(value, comparison, line));
}

function fieldComparisonMatches(left: FieldValue, comparison: FieldComparison, line: number): boolean {
	if (comparison.operator === "==") return fieldValueEquals(left, comparison.value);
	if (comparison.operator === "!=") return !fieldValueEquals(left, comparison.value);
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

async function getDistribution(filter: FilterSpec, groupExpression: string, ctx: RuntimeContext, line: number): Promise<Record<string, number>> {
	const group = await resolveScopedGroupExpression(groupExpression, ctx, line);
	if (group.scope !== "entries") {
		throw new DslRuntimeError("E_DISTRIBUTION_SCOPE", "Score distributions require an entry-scoped GROUP-EXPR", line);
	}
	const ids = group.members;
	const predicate: FieldTextPredicate = { field: filter.field, text: filter.text };
	const scoreVector = filter.type === "BM25" ? getBm25ScoreVector(predicate, ctx) : await getEmbeddingScoreVector(predicate, ctx, line);
	const scores = ids.map((id) => scoreForId(scoreVector, id, ctx));
	if (scores.length === 0) return { min: 0, q1: 0, q2: 0, avg: 0, q3: 0, max: 0 };
	scores.sort((a, b) => a - b);
	const quantile = (q: number) => scores[Math.min(scores.length - 1, Math.max(0, Math.floor((scores.length - 1) * q)))] ?? 0;
	return {
		min: scores[0],
		q1: quantile(0.25),
		q2: quantile(0.5),
		avg: scores.reduce((sum, value) => sum + value, 0) / scores.length,
		q3: quantile(0.75),
		max: scores[scores.length - 1],
	};
}

function scoreVectorKey(type: "BM25" | "embeddings", predicate: FieldTextPredicate, model?: string): string {
	return [
		type,
		model ?? "",
		predicate.field ?? "",
		predicate.text,
	].join("\0");
}

function embeddingRuntimeModelKey(model: string): string {
	return embeddingBackendCacheKey(model);
}

function thresholdKey(scoreKey: string, threshold: number): string {
	return `${scoreKey}\0>\0${threshold}`;
}

async function getCachedAsyncScoreVector(
	key: string,
	ctx: RuntimeContext,
	compute: () => Promise<Float32Array> | Float32Array,
): Promise<Float32Array> {
	const cached = ctx.runtimeCache.scoreVectors.get(key);
	if (cached) {
		ctx.runtimeCache.stats.scoreVectorHits++;
		return cached;
	}
	const pending = ctx.runtimeCache.scoreVectorPromises.get(key);
	if (pending) {
		ctx.runtimeCache.stats.scoreVectorHits++;
		return pending;
	}
	ctx.runtimeCache.stats.scoreVectorMisses++;
	const promise = Promise.resolve()
		.then(compute)
		.then((scores) => {
			ctx.runtimeCache.scoreVectors.set(key, scores);
			return scores;
		})
		.finally(() => {
			ctx.runtimeCache.scoreVectorPromises.delete(key);
		});
	ctx.runtimeCache.scoreVectorPromises.set(key, promise);
	return promise;
}

function getBm25ScoreVector(predicate: FieldTextPredicate, ctx: RuntimeContext): Float32Array {
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

async function getQueryEmbedding(model: string, predicate: FieldTextPredicate, ctx: RuntimeContext): Promise<ArrayLike<number>> {
	const key = `${embeddingRuntimeModelKey(model)}\0${predicate.field ?? ""}\0${predicate.text}`;
	const local = ctx.embeddingQueryVectors.get(key);
	if (local) return local;
	const cached = ctx.runtimeCache.queryEmbeddings.get(key);
	if (cached) {
		ctx.runtimeCache.stats.queryEmbeddingHits++;
		ctx.embeddingQueryVectors.set(key, cached);
		return cached;
	}
	const pending = ctx.runtimeCache.queryEmbeddingPromises.get(key);
	if (pending) {
		ctx.runtimeCache.stats.queryEmbeddingHits++;
		return pending;
	}
	ctx.runtimeCache.stats.queryEmbeddingMisses++;
	const promise = embedTexts([predicate.text], { model, batchSize: 1 })
		.then((index) => {
			const embedding = index.vectors["0"];
			ctx.embeddingQueryVectors.set(key, embedding);
			ctx.runtimeCache.queryEmbeddings.set(key, embedding);
			return embedding;
		})
		.finally(() => {
			ctx.runtimeCache.queryEmbeddingPromises.delete(key);
		});
	ctx.runtimeCache.queryEmbeddingPromises.set(key, promise);
	return promise;
}

async function getEmbeddingScoreVector(predicate: FieldTextPredicate, ctx: RuntimeContext, line: number): Promise<Float32Array> {
	const model = ctx.datasets.find((dataset) => dataset.manifest.embeddingModel)?.manifest.embeddingModel;
	if (!model) throw new DslRuntimeError("E_EMBEDDING_MODEL", "No embedding model available for active dataset", line);
	const key = scoreVectorKey("embeddings", predicate, embeddingRuntimeModelKey(model));
	return getCachedAsyncScoreVector(key, ctx, async () => {
		const embedding = await getQueryEmbedding(model, predicate, ctx);
		return scoreEmbeddingVectorValues(embeddingVectorsForEntries(allEntryIds(ctx), predicate.field, ctx), embedding);
	});
}

async function getEmbeddingGroupScoreVector(
	target: Extract<SimilarityTarget, { kind: "entryGroup" }>,
	field: string | undefined,
	accumulation: "total" | "avg",
	ctx: RuntimeContext,
	line: number,
): Promise<Float32Array> {
	const model = ctx.datasets.find((dataset) => dataset.manifest.embeddingModel)?.manifest.embeddingModel;
	if (!model) throw new DslRuntimeError("E_EMBEDDING_MODEL", "No embedding model available for active dataset", line);
	const key = [
		"embeddings-group",
		embeddingRuntimeModelKey(model),
		field ?? "",
		accumulation,
		target.cacheKey,
	].join("\0");
	return getCachedAsyncScoreVector(key, ctx, () => {
		const centroid = embeddingCentroidForEntries(target.members, field, accumulation, ctx);
		return scoreEmbeddingVectorValues(embeddingVectorsForEntries(allEntryIds(ctx), field, ctx), centroid);
	});
}

async function getBm25GroupScoreVector(
	target: Extract<SimilarityTarget, { kind: "entryGroup" }>,
	field: string | undefined,
	accumulation: "total" | "avg",
	ctx: RuntimeContext,
): Promise<Float32Array> {
	const key = ["BM25-group", field ?? "", accumulation, target.cacheKey].join("\0");
	return getCachedAsyncScoreVector(key, ctx, () => {
		const targetTexts = groupTargetTexts(target, field, ctx);
		const terms = targetTexts.flatMap(tokenize);
		const divisor = accumulation === "avg" ? Math.max(1, targetTexts.length) : 1;
		const scores = new Float32Array(ctx.entries.length);
		if (terms.length === 0) return scores;
		for (const [index, entry] of ctx.entries.entries()) {
			const dataset = ctx.datasetByEntryId.get(entry.id);
			if (!dataset) continue;
			const docId = field ? fieldDocumentId(entry.id, field) : entry.id;
			scores[index] = bm25ScoreTerms(dataset.bm25, docId, terms) / divisor;
		}
		return scores;
	});
}

function embeddingVectorsForEntries(entryIds: readonly string[], field: string | undefined, ctx: RuntimeContext): Array<EmbeddingVector | undefined> {
	return entryIds.map((id) => getEntryEmbeddingVector(id, field, ctx));
}

function getEntryEmbeddingVector(id: string, field: string | undefined, ctx: RuntimeContext): EmbeddingVector | undefined {
	const dataset = ctx.datasetByEntryId.get(id);
	const entry = ctx.entriesById.get(id);
	if (!dataset || !entry) return undefined;
	const vectorId = field ? fieldDocumentId(id, field) : id;
	return dataset.embeddings.vectors[vectorId]
		?? (field && getFieldText(id, field, ctx) === entry.text ? dataset.embeddings.vectors[id] : undefined);
}

function embeddingCentroidForEntries(
	entryIds: readonly string[],
	field: string | undefined,
	accumulation: "total" | "avg",
	ctx: RuntimeContext,
): Float32Array {
	let centroid: Float32Array | undefined;
	let count = 0;
	for (const id of entryIds) {
		const vector = getEntryEmbeddingVector(id, field, ctx);
		if (!vector) continue;
		const materialized = materializeEmbeddingVector(vector);
		if (!centroid) centroid = new Float32Array(materialized.length);
		const n = Math.min(centroid.length, materialized.length);
		for (let index = 0; index < n; index++) centroid[index] += materialized[index] ?? 0;
		count++;
	}
	if (!centroid || count === 0) return new Float32Array(0);
	if (accumulation === "avg") {
		for (let index = 0; index < centroid.length; index++) centroid[index] /= count;
	}
	return centroid;
}

function materializeEmbeddingVector(vector: EmbeddingVector): ArrayLike<number> {
	const maybeFileBacked = vector as EmbeddingVector & { toFloat32Array?: () => Float32Array };
	return typeof maybeFileBacked.toFloat32Array === "function" ? maybeFileBacked.toFloat32Array() : vector;
}

function groupTargetTexts(target: Extract<SimilarityTarget, { kind: "entryGroup" }>, field: string | undefined, ctx: RuntimeContext): string[] {
	const texts: string[] = [];
	for (const id of target.members) {
		const values = getEntryFunctionValues(id, field, ctx).map(fieldValueToText).filter((value) => value.trim().length > 0);
		if (values.length > 0) texts.push(values.join(" "));
	}
	return texts;
}

function scoreForId(scores: Float32Array, id: string, ctx: RuntimeContext): number {
	const index = ctx.entryIndexById.get(id);
	return index === undefined ? 0 : (scores[index] ?? 0);
}

function getIdsAboveScoreThreshold(scoreKeyValue: string, scores: Float32Array, threshold: number, ctx: RuntimeContext): IdBitSet {
	const key = thresholdKey(scoreKeyValue, threshold);
	const cached = ctx.runtimeCache.thresholdIdSets.get(key);
	if (cached) {
		ctx.runtimeCache.stats.thresholdIdSetHits++;
		return cached;
	}
	ctx.runtimeCache.stats.thresholdIdSetMisses++;
	const ids = IdBitSet.fromPredicate(ctx, (_entry, index) => (scores[index] ?? -1) > threshold);
	ctx.runtimeCache.thresholdIdSets.set(key, ids);
	return ids;
}

function getIdsAboveBm25Threshold(predicate: FieldTextPredicate, threshold: number, ctx: RuntimeContext): IdBitSet {
	const scoreKeyValue = scoreVectorKey("BM25", predicate);
	return getIdsAboveScoreThreshold(scoreKeyValue, getBm25ScoreVector(predicate, ctx), threshold, ctx);
}

async function getIdsAboveEmbeddingThreshold(predicate: FieldTextPredicate, threshold: number, ctx: RuntimeContext, line: number): Promise<IdBitSet> {
	const model = ctx.datasets.find((dataset) => dataset.manifest.embeddingModel)?.manifest.embeddingModel;
	if (!model) throw new DslRuntimeError("E_EMBEDDING_MODEL", "No embedding model available for active dataset", line);
	const scoreKeyValue = scoreVectorKey("embeddings", predicate, embeddingRuntimeModelKey(model));
	return getIdsAboveScoreThreshold(scoreKeyValue, await getEmbeddingScoreVector(predicate, ctx, line), threshold, ctx);
}

function scoreBm25(id: string, predicate: FieldTextPredicate, ctx: RuntimeContext): number {
	const dataset = ctx.datasetByEntryId.get(id);
	if (!dataset) return 0;
	const cacheKey = `${predicate.field ?? ""}\0${predicate.text}`;
	let terms = ctx.bm25QueryTerms.get(cacheKey);
	if (!terms) {
		terms = tokenize(predicate.text);
		ctx.bm25QueryTerms.set(cacheKey, terms);
	}
	const docId = predicate.field ? fieldDocumentId(id, predicate.field) : id;
	if (!predicate.field || dataset.bm25.termFreq[docId]) return bm25ScoreTerms(dataset.bm25, docId, terms);
	return scoreBm25FallbackText(dataset.bm25, getFieldText(id, predicate.field, ctx), terms);
}

function scoreBm25FallbackText(index: { k1: number; b: number; avgDocLength: number; docCount: number; docFreq: Record<string, number> }, text: string, terms: readonly string[]): number {
	if (!text || terms.length === 0 || index.docCount === 0) return 0;
	const tokens = tokenize(text);
	if (tokens.length === 0) return 0;
	const tf: Record<string, number> = {};
	for (const token of tokens) tf[token] = (tf[token] ?? 0) + 1;
	const avgdl = index.avgDocLength || tokens.length || 1;
	let score = 0;
	for (const term of terms) {
		const f = tf[term] ?? 0;
		if (f <= 0) continue;
		const n = index.docFreq[term] ?? 0;
		const idf = Math.log(1 + (index.docCount - n + 0.5) / (n + 0.5));
		const denom = f + index.k1 * (1 - index.b + index.b * (tokens.length / avgdl));
		score += idf * ((f * (index.k1 + 1)) / denom);
	}
	return score;
}

async function scoreEmbeddings(ids: string[], predicate: FieldTextPredicate, ctx: RuntimeContext, line: number): Promise<Map<string, number>> {
	const scoreVector = await getEmbeddingScoreVector(predicate, ctx, line);
	const scores = new Map<string, number>();
	for (const id of ids) scores.set(id, scoreForId(scoreVector, id, ctx));
	return scores;
}

function getTextFilterCacheKey(type: "contains" | "contains_word", predicate: FieldTextPredicate): string {
	return [type, predicate.field ?? "", predicate.text].join("\0");
}

function getCachedTextFilterIds(key: string, compute: () => IdBitSet, ctx: RuntimeContext): IdBitSet {
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

function getIdsContaining(predicate: FieldTextPredicate, ctx: RuntimeContext): IdBitSet {
	const needle = normalizeContainsText(predicate.text);
	return getCachedTextFilterIds(
		getTextFilterCacheKey("contains", predicate),
		() => IdBitSet.fromPredicate(ctx, (entry) => getFieldContains(entry.id, predicate.field, ctx).includes(needle)),
		ctx,
	);
}

function getIdsContainingWord(predicate: FieldTextPredicate, ctx: RuntimeContext, line: number): IdBitSet {
	const needle = tokenize(predicate.text);
	if (needle.length === 0) throw new DslRuntimeError("E_CONTAINS_WORD", `contains_word requires at least one word token, got ${formatValue(predicate.text)}`, line);
	return getCachedTextFilterIds(getTextFilterCacheKey("contains_word", predicate), () => {
		if (needle.length === 1) {
			const token = needle[0];
			return IdBitSet.fromPredicate(ctx, (entry) => {
				const docId = predicate.field ? fieldDocumentId(entry.id, predicate.field) : entry.id;
				const termFreq = ctx.datasetByEntryId.get(entry.id)?.bm25.termFreq[docId];
				if (termFreq) return (termFreq[token] ?? 0) > 0;
				return containsWord(getFieldText(entry.id, predicate.field, ctx), predicate.text, line);
			});
		}
		return IdBitSet.fromPredicate(ctx, (entry) => containsWord(getFieldText(entry.id, predicate.field, ctx), predicate.text, line));
	}, ctx);
}

function formatValue(value: unknown): string {
	if (value === undefined) return "undefined";
	if (isGroupExpressionValue(value)) return `group_expr(${value.expression})`;
	if (typeof value === "string") return value;
	return JSON.stringify(value, null, 2);
}

function formatValueForError(value: unknown): string {
	if (!value || typeof value !== "object") return formatValue(value);
	if (Array.isArray(value)) return `list with ${value.length} items`;
	const keys = Object.keys(value as Record<string, unknown>);
	return `object with keys: ${keys.slice(0, 8).join(", ")}${keys.length > 8 ? ", ..." : ""}`;
}

function formatInlineValue(value: unknown): string {
	if (value === undefined) return "undefined";
	if (isGroupExpressionValue(value)) return `group_expr(${value.expression})`;
	if (typeof value === "string") return value;
	if (typeof value === "number" || typeof value === "boolean" || value === null) return String(value);
	return JSON.stringify(value);
}

function containsWord(text: string, word: string, line: number): boolean {
	const needle = tokenize(word);
	if (needle.length === 0) throw new DslRuntimeError("E_CONTAINS_WORD", `contains_word requires at least one word token, got ${formatValue(word)}`, line);
	const haystack = tokenize(text);
	if (needle.length === 1) return haystack.includes(needle[0]);
	return haystack.some((_, index) => needle.every((token, offset) => haystack[index + offset] === token));
}

function trimOuterParens(text: string): string {
	let out = text.trim();
	while (out.startsWith("(") && out.endsWith(")") && matchingOuterParens(out)) {
		out = out.slice(1, -1).trim();
	}
	return out;
}

function matchingOuterParens(text: string): boolean {
	let depth = 0;
	let quoted: string | undefined;
	for (let i = 0; i < text.length; i++) {
		const ch = text[i];
		if (quoted) {
			if (ch === quoted && text[i - 1] !== "\\") quoted = undefined;
			continue;
		}
		if (ch === '"' || ch === "'") { quoted = ch; continue; }
		if (ch === "(") depth++;
		if (ch === ")") depth--;
		if (depth === 0 && i < text.length - 1) return false;
	}
	return depth === 0;
}

function findMatchingClose(text: string, start: number, open: string, close: string): number {
	let depth = 0;
	let quoted: string | undefined;
	for (let i = start; i < text.length; i++) {
		const ch = text[i];
		if (quoted) {
			if (ch === quoted && text[i - 1] !== "\\") quoted = undefined;
			continue;
		}
		if (ch === '"' || ch === "'") { quoted = ch; continue; }
		if (ch === open) depth++;
		else if (ch === close) {
			depth--;
			if (depth === 0) return i;
		}
	}
	return -1;
}

function splitTopLevel(text: string, delimiter: string): string[] {
	const parts: string[] = [];
	let start = 0;
	let depth = 0;
	let quoted: string | undefined;
	for (let i = 0; i < text.length; i++) {
		const ch = text[i];
		if (quoted) {
			if (ch === quoted && text[i - 1] !== "\\") quoted = undefined;
			continue;
		}
		if (ch === '"' || ch === "'") { quoted = ch; continue; }
		if ("([{".includes(ch)) depth++;
		else if (")]}".includes(ch)) depth--;
		else if (depth === 0 && text.startsWith(delimiter, i)) {
			parts.push(text.slice(start, i).trim());
			start = i + delimiter.length;
			i += delimiter.length - 1;
		}
	}
	parts.push(text.slice(start).trim());
	return parts;
}

function splitTopLevelByWord(text: string, word: string): string[] {
	const parts: string[] = [];
	let start = 0;
	let depth = 0;
	let quoted: string | undefined;
	const pattern = new RegExp(`\\b${word}\\b`, "g");
	let match: RegExpExecArray | null;
	while ((match = pattern.exec(text))) {
		for (let i = start; i < match.index; i++) {
			const ch = text[i];
			if (quoted) {
				if (ch === quoted && text[i - 1] !== "\\") quoted = undefined;
				continue;
			}
			if (ch === '"' || ch === "'") { quoted = ch; continue; }
			if ("([{".includes(ch)) depth++;
			else if (")]}".includes(ch)) depth--;
		}
		if (depth === 0 && !quoted) {
			parts.push(text.slice(start, match.index).trim());
			start = match.index + word.length;
		}
	}
	parts.push(text.slice(start).trim());
	return parts.filter(Boolean);
}

function splitByTopLevelOperator(expr: string, operator: string): [string, string, string] | undefined {
	let depth = 0;
	let quoted: string | undefined;
	for (let i = 0; i <= expr.length - operator.length; i++) {
		const ch = expr[i];
		if (quoted) {
			if (ch === quoted && expr[i - 1] !== "\\") quoted = undefined;
			continue;
		}
		if (ch === '"' || ch === "'") { quoted = ch; continue; }
		if ("([{".includes(ch)) depth++;
		else if (")]}".includes(ch)) depth--;
		if (depth === 0 && expr.startsWith(operator, i)) return [expr.slice(0, i).trim(), operator.trim(), expr.slice(i + operator.length).trim()];
	}
	return undefined;
}
