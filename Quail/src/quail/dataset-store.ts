import {
	closeSync,
	existsSync,
	mkdirSync,
	openSync,
	readdirSync,
	readFileSync,
	readSync,
	rmSync,
	statSync,
	writeFileSync,
	writeSync,
} from "node:fs";
import { basename, dirname, extname, join, resolve } from "node:path";
import { ensureQuailWorkspace, getQuailDatasetsDir, getQuailWorkspaceDir } from "./paths.js";
import { normalizeContainsText, slugifyDatasetName, stableEntryId, tokenize } from "./text.js";

export interface CorpusEntryInput {
	text: string;
	fields: Record<string, FieldValue>;
	tags: Record<string, string>;
	textFields?: string[];
}

export type FieldValue = string | number | boolean | null | FieldValue[] | { [key: string]: FieldValue };
export type FieldType = "string" | "int" | "float" | "bool" | "null" | "list" | "object" | "mixed";
export type FieldTypeOverride = FieldType | "boolean";

export interface FieldInspection {
	name: string;
	type: FieldType;
	embedded: boolean;
	nonEmptyCount: number;
	samples: string[];
}

export interface DatasetInspection {
	entryCount: number;
	fieldTypes: Record<string, FieldType>;
	embeddedFields: string[];
	fields: FieldInspection[];
}

export interface QuailEntry {
	id: string;
	dataset: string;
	ordinal: number;
	text: string;
	fields: Record<string, FieldValue>;
	tags: Record<string, string>;
	contains: string;
	fieldContains: Record<string, string>;
}

export interface Bm25Index {
	k1: number;
	b: number;
	avgDocLength: number;
	docCount: number;
	docLengths: Record<string, number>;
	docFreq: Record<string, number>;
	termFreq: Record<string, Record<string, number>>;
}

export type EmbeddingVector = ArrayLike<number>;

interface DotProductEmbeddingVector extends EmbeddingVector {
	dotProduct(other: EmbeddingVector): number;
}

export interface EmbeddingIndex {
	model: string;
	dimensions: number;
	vectors: Record<string, EmbeddingVector>;
}

export interface QuailDatasetManifest {
	name: string;
	slug: string;
	createdAt: string;
	updatedAt: string;
	entryCount: number;
	metadataFields: string[];
	fieldNames?: string[];
	textFields?: string[];
	fieldTypes?: Record<string, FieldType>;
	embeddedFields?: string[];
	embeddingModel: string;
	embeddingDimensions: number;
	batchSize: number;
	source: {
		fileName?: string;
		format: string;
	};
	files: {
		entries: string;
		bm25: string;
		embeddings: string;
	};
}

export interface LoadedQuailDataset {
	manifest: QuailDatasetManifest;
	workspaceDir: string;
	datasetDir: string;
	entries: QuailEntry[];
	bm25: Bm25Index;
	embeddings: EmbeddingIndex;
}

export interface ProcessDatasetOptions {
	cwd: string;
	inputPath: string;
	name: string;
	format?: string;
	textColumn?: string;
	model?: string;
	batchSize?: number;
	embeddingConcurrency?: number;
	globalTags?: Record<string, string>;
	fieldTypes?: Record<string, FieldTypeOverride>;
	skipEmbeddings?: boolean;
	overwrite?: boolean;
	onProgress?: (message: string) => void;
}

export interface DatasetListItem {
	name: string;
	slug: string;
	entryCount: number;
	embeddingModel: string;
	metadataFields: string[];
	createdAt: string;
}

const ENTRIES_FILE = "entries.jsonl";
const BM25_FILE = "bm25.json";
const EMBEDDINGS_FILE = "embeddings.json";
const EMBEDDINGS_VECTOR_FILE = "embeddings.f32";
const MANIFEST_FILE = "manifest.json";
const ROOT_MANIFEST_FILE = "manifest.json";
export const DEFAULT_EMBEDDING_MODEL = "qwen/qwen3-embedding-8b";
export const DEFAULT_EMBEDDING_PROVIDER = "openrouter";
export const DEFAULT_BATCH_SIZE = 256;
export const DEFAULT_OPENROUTER_EMBEDDING_CONCURRENCY = 20;
export const DEFAULT_OLLAMA_EMBEDDING_CONCURRENCY = 1;
export const DEFAULT_EMBEDDING_MAX_RETRIES = 6;
const DEFAULT_OPENROUTER_EMBED_URL = "https://openrouter.ai/api/v1/embeddings";
const DEFAULT_OLLAMA_EMBED_URL = "http://127.0.0.1:11434/api/embed";
const DEFAULT_EMBEDDING_RETRY_BASE_MS = 500;
const DEFAULT_MAX_EAGER_VECTOR_FILE_BYTES = 1.5 * 1024 * 1024 * 1024;
const FIELD_DOC_SEPARATOR = "\u0000";
const LEGACY_TEXT_FIELD = "text";
const LOCAL_EMBEDDING_ENV_FILE = "openrouter.env";
const LOCAL_EMBEDDING_ENV_DISABLE = "QUAIL_DISABLE_LOCAL_EMBEDDING_ENV";

type EmbeddingProvider = "openrouter" | "ollama";

interface EmbeddingBackendConfig {
	provider: EmbeddingProvider;
	model: string;
	url: string;
	providerOnly?: string;
}

interface LoadedDatasetCacheEntry {
	manifestMtimeMs: number;
	entriesMtimeMs: number;
	bm25MtimeMs: number;
	embeddingsMtimeMs: number;
	vectorMtimeMs?: number;
	dataset: LoadedQuailDataset;
}

const loadedDatasetCache = new Map<string, LoadedDatasetCacheEntry>();
const loadedLocalEmbeddingEnvFiles = new Set<string>();

function datasetDir(cwd: string, slug: string): string {
	return join(getQuailDatasetsDir(cwd), slug);
}

function datasetCacheKey(cwd: string, slug: string): string {
	return datasetDir(cwd, slug);
}

function readJson<T>(path: string): T {
	return JSON.parse(readFileSync(path, "utf8")) as T;
}

function writeJson(path: string, value: unknown): void {
	writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function getMtimeMs(path: string): number {
	return statSync(path).mtimeMs;
}

export function fieldDocumentId(entryId: string, field: string): string {
	return `${entryId}${FIELD_DOC_SEPARATOR}${field}`;
}

export function fieldValueToText(value: unknown): string {
	if (value === undefined || value === null) return "";
	if (typeof value === "string") return value;
	if (typeof value === "number" || typeof value === "boolean") return String(value);
	if (Array.isArray(value)) return value.map(fieldValueToText).filter(Boolean).join(" ");
	if (typeof value === "object") return JSON.stringify(value);
	return String(value);
}

interface BinaryEmbeddingIndexFile {
	format: "float32-binary-v1";
	model: string;
	dimensions: number;
	count: number;
	ids: string[];
	vectorsFile: string;
}

function isBinaryEmbeddingIndexFile(value: unknown): value is BinaryEmbeddingIndexFile {
	if (typeof value !== "object" || value === null) return false;
	const record = value as Record<string, unknown>;
	return record.format === "float32-binary-v1" && typeof record.model === "string";
}

function maxEagerVectorFileBytes(): number {
	const value = process.env.QUAIL_EAGER_VECTOR_FILE_BYTES?.trim();
	if (!value) return DEFAULT_MAX_EAGER_VECTOR_FILE_BYTES;
	const parsed = Number(value);
	return Number.isFinite(parsed) && parsed >= 0 ? parsed : DEFAULT_MAX_EAGER_VECTOR_FILE_BYTES;
}

function hasDotProduct(vector: EmbeddingVector): vector is DotProductEmbeddingVector {
	return typeof (vector as Partial<DotProductEmbeddingVector>).dotProduct === "function";
}

function stripEnvValueQuotes(value: string): string {
	if (value.length >= 2 && ((value.startsWith("\"") && value.endsWith("\"")) || (value.startsWith("'") && value.endsWith("'")))) {
		return value.slice(1, -1);
	}
	return value;
}

function loadLocalEmbeddingEnvFile(path: string): void {
	if (loadedLocalEmbeddingEnvFiles.has(path)) return;
	if (!existsSync(path)) return;
	const text = readFileSync(path, "utf8");
	for (const rawLine of text.split(/\r?\n/)) {
		const line = rawLine.trim();
		if (!line || line.startsWith("#")) continue;
		const assignment = line.startsWith("export ") ? line.slice("export ".length).trim() : line;
		const index = assignment.indexOf("=");
		if (index <= 0) continue;
		const key = assignment.slice(0, index).trim();
		if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(key) || process.env[key] !== undefined) continue;
		process.env[key] = stripEnvValueQuotes(assignment.slice(index + 1).trim());
	}
	loadedLocalEmbeddingEnvFiles.add(path);
}

function localEmbeddingEnvCandidatePaths(cwd: string): string[] {
	const dir = resolve(cwd);
	const parent = dirname(dir);
	return Array.from(new Set([
		join(dir, LOCAL_EMBEDDING_ENV_FILE),
		join(dir, "Quail", LOCAL_EMBEDDING_ENV_FILE),
		join(parent, LOCAL_EMBEDDING_ENV_FILE),
		join(parent, "Quail", LOCAL_EMBEDDING_ENV_FILE),
	]));
}

export function loadLocalEmbeddingEnv(cwd = process.cwd()): void {
	if (process.env[LOCAL_EMBEDDING_ENV_DISABLE]?.trim() === "1") return;
	for (const path of localEmbeddingEnvCandidatePaths(cwd)) loadLocalEmbeddingEnvFile(path);
}

function envValue(name: string): string | undefined {
	loadLocalEmbeddingEnv();
	const value = process.env[name]?.trim();
	return value ? value : undefined;
}

export function defaultEmbeddingModel(): string {
	return envValue("QUAIL_EMBEDDING_MODEL") ?? DEFAULT_EMBEDDING_MODEL;
}

export function defaultEmbeddingBatchSize(): number {
	const value = envValue("QUAIL_EMBEDDING_BATCH_SIZE");
	if (!value) return DEFAULT_BATCH_SIZE;
	const parsed = Number.parseInt(value, 10);
	return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_BATCH_SIZE;
}

export function defaultEmbeddingConcurrency(): number {
	const fallback = embeddingProvider() === "openrouter"
		? DEFAULT_OPENROUTER_EMBEDDING_CONCURRENCY
		: DEFAULT_OLLAMA_EMBEDDING_CONCURRENCY;
	const value = envValue("QUAIL_EMBEDDING_CONCURRENCY");
	if (!value) return fallback;
	const parsed = Number.parseInt(value, 10);
	return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function defaultEmbeddingMaxRetries(): number {
	const value = envValue("QUAIL_EMBEDDING_MAX_RETRIES");
	if (!value) return DEFAULT_EMBEDDING_MAX_RETRIES;
	const parsed = Number.parseInt(value, 10);
	return Number.isFinite(parsed) && parsed >= 0 ? parsed : DEFAULT_EMBEDDING_MAX_RETRIES;
}

function defaultEmbeddingRetryBaseMs(): number {
	const value = envValue("QUAIL_EMBEDDING_RETRY_BASE_MS");
	if (!value) return DEFAULT_EMBEDDING_RETRY_BASE_MS;
	const parsed = Number.parseInt(value, 10);
	return Number.isFinite(parsed) && parsed >= 0 ? parsed : DEFAULT_EMBEDDING_RETRY_BASE_MS;
}

function embeddingProvider(): EmbeddingProvider {
	const value = envValue("QUAIL_EMBEDDING_PROVIDER")?.toLowerCase();
	if (value === "ollama") return "ollama";
	if (value && value !== DEFAULT_EMBEDDING_PROVIDER) {
		throw new Error(`Unsupported QUAIL_EMBEDDING_PROVIDER "${value}". Use "openrouter" or "ollama".`);
	}
	return DEFAULT_EMBEDDING_PROVIDER;
}

function openRouterEmbedUrl(): string {
	const explicitUrl = envValue("QUAIL_OPENROUTER_EMBED_URL");
	if (explicitUrl) return explicitUrl;
	const baseUrl = envValue("QUAIL_OPENROUTER_BASE_URL");
	if (!baseUrl) return DEFAULT_OPENROUTER_EMBED_URL;
	return `${baseUrl.replace(/\/+$/, "")}/embeddings`;
}

function getEmbeddingBackendConfig(model: string): EmbeddingBackendConfig {
	const provider = embeddingProvider();
	if (provider === "ollama") {
		return {
			provider,
			model,
			url: envValue("QUAIL_OLLAMA_EMBED_URL") ?? DEFAULT_OLLAMA_EMBED_URL,
		};
	}
	return {
		provider,
		model,
		url: openRouterEmbedUrl(),
		providerOnly: envValue("QUAIL_OPENROUTER_PROVIDER_ONLY"),
	};
}

export function embeddingBackendCacheKey(model: string): string {
	const config = getEmbeddingBackendConfig(model);
	return [config.provider, config.model, config.url, config.providerOnly ?? ""].join("\0");
}

export function embeddingBackendDescription(model = defaultEmbeddingModel()): string {
	const config = getEmbeddingBackendConfig(model);
	if (config.provider === "ollama") return `Ollama-compatible endpoint ${config.url} with model ${config.model}`;
	const providerText = config.providerOnly ? ` via provider ${config.providerOnly}` : "";
	return `OpenRouter model ${config.model}${providerText}`;
}

class FileBackedEmbeddingVector implements DotProductEmbeddingVector {
	readonly length: number;
	readonly [index: number]: number;

	constructor(
		private readonly store: FileBackedEmbeddingStore,
		private readonly index: number,
		dimensions: number,
	) {
		this.length = dimensions;
	}

	dotProduct(other: EmbeddingVector): number {
		if (other instanceof FileBackedEmbeddingVector) return this.store.dotProduct(this.index, other.toFloat32Array());
		return this.store.dotProduct(this.index, other);
	}

	bulkStore(): FileBackedEmbeddingStore {
		return this.store;
	}

	bulkIndex(): number {
		return this.index;
	}

	toFloat32Array(): Float32Array {
		return this.store.readVector(this.index);
	}
}

class FileBackedEmbeddingStore {
	private readonly scratch: Buffer;

	constructor(
		private readonly fd: number,
		private readonly dimensions: number,
	) {
		this.scratch = Buffer.allocUnsafe(dimensions * 4);
	}

	dotProduct(index: number, other: EmbeddingVector): number {
		readSync(this.fd, this.scratch, 0, this.scratch.byteLength, index * this.dimensions * 4);
		const n = Math.min(other.length, this.dimensions);
		let sum = 0;
		for (let i = 0; i < n; i++) sum += (other[i] ?? 0) * this.scratch.readFloatLE(i * 4);
		return sum;
	}

	dotProducts(indices: readonly number[], other: EmbeddingVector): Float32Array {
		const scores = new Float32Array(indices.length);
		if (this.dimensions <= 0 || other.length === 0 || indices.length === 0) return scores;
		const bytesPerVector = this.dimensions * 4;
		const maxChunkBytes = 8 * 1024 * 1024;
		const maxVectorsPerChunk = Math.max(1, Math.floor(maxChunkBytes / bytesPerVector));
		let offset = 0;
		while (offset < indices.length) {
			const startIndex = indices[offset];
			if (startIndex < 0) {
				offset++;
				continue;
			}
			let count = 1;
			while (
				offset + count < indices.length &&
				count < maxVectorsPerChunk &&
				indices[offset + count] === startIndex + count
			) {
				count++;
			}
			const buffer = Buffer.allocUnsafe(count * bytesPerVector);
			readSync(this.fd, buffer, 0, buffer.byteLength, startIndex * bytesPerVector);
			const n = Math.min(other.length, this.dimensions);
			for (let row = 0; row < count; row++) {
				let sum = 0;
				const rowOffset = row * bytesPerVector;
				for (let dimension = 0; dimension < n; dimension++) {
					sum += (other[dimension] ?? 0) * buffer.readFloatLE(rowOffset + dimension * 4);
				}
				scores[offset + row] = sum;
			}
			offset += count;
		}
		return scores;
	}

	readVector(index: number): Float32Array {
		const buffer = Buffer.allocUnsafe(this.dimensions * 4);
		readSync(this.fd, buffer, 0, buffer.byteLength, index * this.dimensions * 4);
		return new Float32Array(buffer.buffer, buffer.byteOffset, this.dimensions);
	}

	vector(index: number): EmbeddingVector {
		return new FileBackedEmbeddingVector(this, index, this.dimensions);
	}
}

function writeEmbeddingIndex(dir: string, fileName: string, index: EmbeddingIndex): void {
	const ids = Object.keys(index.vectors);
	const dimensions = index.dimensions;
	const vectorsFile = EMBEDDINGS_VECTOR_FILE;
	const vectorPath = join(dir, vectorsFile);

	if (ids.length === 0 || dimensions === 0) {
		writeFileSync(vectorPath, "");
	} else {
		const vectorsPerChunk = Math.max(1, Math.min(256, Math.floor((8 * 1024 * 1024) / (dimensions * 4))));
		const buffer = Buffer.allocUnsafe(vectorsPerChunk * dimensions * 4);
		const fd = openSync(vectorPath, "w");
		try {
			for (let start = 0; start < ids.length; start += vectorsPerChunk) {
				const count = Math.min(vectorsPerChunk, ids.length - start);
				let offset = 0;
				for (let row = 0; row < count; row++) {
					const vector = index.vectors[ids[start + row]];
					for (let dimension = 0; dimension < dimensions; dimension++) {
						const value = Number(vector?.[dimension] ?? 0);
						buffer.writeFloatLE(Number.isFinite(value) ? value : 0, offset);
						offset += 4;
					}
				}
				writeSync(fd, buffer, 0, offset);
			}
		} finally {
			closeSync(fd);
		}
	}

	writeJson(join(dir, fileName), {
		format: "float32-binary-v1",
		model: index.model,
		dimensions,
		count: ids.length,
		ids,
		vectorsFile,
	} satisfies BinaryEmbeddingIndexFile);
}

function loadEmbeddingIndex(dir: string, fileName: string): EmbeddingIndex {
	const raw = readJson<unknown>(join(dir, fileName));
	if (!isBinaryEmbeddingIndexFile(raw)) return raw as EmbeddingIndex;

	const dimensions = Math.max(0, Math.floor(Number(raw.dimensions) || 0));
	const ids = Array.isArray(raw.ids) ? raw.ids.map(String) : [];
	const vectorPath = join(dir, raw.vectorsFile || EMBEDDINGS_VECTOR_FILE);
	const vectorSize = statSync(vectorPath).size;
	const vectors: Record<string, EmbeddingVector> = {};
	const bytesPerVector = dimensions * 4;
	if (dimensions > 0 && vectorSize > maxEagerVectorFileBytes()) {
		const fd = openSync(vectorPath, "r");
		const store = new FileBackedEmbeddingStore(fd, dimensions);
		for (let i = 0; i < ids.length; i++) {
			const byteOffset = i * bytesPerVector;
			vectors[ids[i]] = byteOffset + bytesPerVector <= vectorSize ? store.vector(i) : [];
		}
		return { model: raw.model, dimensions, vectors };
	}

	const data = readFileSync(vectorPath);
	const littleEndian = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

	for (let i = 0; i < ids.length; i++) {
		const byteOffset = i * bytesPerVector;
		if (dimensions === 0 || byteOffset + bytesPerVector > data.byteLength) {
			vectors[ids[i]] = [];
			continue;
		}
		const absoluteOffset = data.byteOffset + byteOffset;
		if (littleEndian && absoluteOffset % 4 === 0) {
			vectors[ids[i]] = new Float32Array(data.buffer, absoluteOffset, dimensions);
			continue;
		}
		const vector = new Float32Array(dimensions);
		for (let dimension = 0; dimension < dimensions; dimension++) vector[dimension] = data.readFloatLE(byteOffset + dimension * 4);
		vectors[ids[i]] = vector;
	}

	return { model: raw.model, dimensions, vectors };
}

function splitDelimitedLine(line: string, delimiter: string): string[] {
	const out: string[] = [];
	let current = "";
	let quoted = false;
	for (let i = 0; i < line.length; i++) {
		const ch = line[i];
		if (ch === '"') {
			if (quoted && line[i + 1] === '"') {
				current += '"';
				i++;
			} else {
				quoted = !quoted;
			}
			continue;
		}
		if (!quoted && ch === delimiter) {
			out.push(current);
			current = "";
			continue;
		}
		current += ch;
	}
	out.push(current);
	return out.map((cell) => cell.trim());
}

function splitDelimitedRecords(content: string, delimiter: string): string[][] {
	const rows: string[][] = [];
	let row: string[] = [];
	let current = "";
	let quoted = false;
	const input = content.replace(/^\uFEFF/, "");
	for (let i = 0; i < input.length; i++) {
		const ch = input[i];
		if (ch === '"') {
			if (quoted && input[i + 1] === '"') {
				current += '"';
				i++;
			} else {
				quoted = !quoted;
			}
			continue;
		}
		if (!quoted && ch === delimiter) {
			row.push(current.trim());
			current = "";
			continue;
		}
		if (!quoted && (ch === "\n" || ch === "\r")) {
			row.push(current.trim());
			current = "";
			if (row.some((cell) => cell.length > 0)) rows.push(row);
			row = [];
			if (ch === "\r" && input[i + 1] === "\n") i++;
			continue;
		}
		current += ch;
	}
	row.push(current.trim());
	if (row.some((cell) => cell.length > 0)) rows.push(row);
	return rows;
}

function inferFieldValue(value: string): FieldValue {
	const trimmed = value.trim();
	if (!trimmed) return "";
	if (/^(true|false)$/i.test(trimmed)) return trimmed.toLowerCase() === "true";
	if (/^-?\d+$/.test(trimmed)) {
		const parsed = Number(trimmed);
		if (Number.isSafeInteger(parsed)) return parsed;
	}
	if (/^-?(?:\d+\.\d+|\d+\.|\.\d+)(?:e[+-]?\d+)?$/i.test(trimmed) || /^-?\d+e[+-]?\d+$/i.test(trimmed)) {
		const parsed = Number(trimmed);
		if (Number.isFinite(parsed)) return parsed;
	}
	return value.trim();
}

function normalizeFieldType(type: FieldTypeOverride): FieldType {
	const normalized = String(type).trim().toLowerCase();
	if (normalized === "boolean") return "bool";
	if (
		normalized === "string" ||
		normalized === "int" ||
		normalized === "float" ||
		normalized === "bool" ||
		normalized === "null" ||
		normalized === "list" ||
		normalized === "object" ||
		normalized === "mixed"
	) {
		return normalized;
	}
	throw new Error(`Unknown field type "${type}". Expected string, int, float, bool, null, list, object, or mixed.`);
}

function inferStringType(value: string): FieldType {
	const trimmed = value.trim();
	if (!trimmed) return "null";
	if (/^(true|false)$/i.test(trimmed)) return "bool";
	if (/^-?(?:0|[1-9]\d*)$/.test(trimmed)) return "int";
	if (/^-?(?:\d+\.\d+|\d+\.|\.\d+)(?:e[+-]?\d+)?$/i.test(trimmed) || /^-?(?:0|[1-9]\d*)e[+-]?\d+$/i.test(trimmed)) return "float";
	return "string";
}

function inferValueType(value: FieldValue): FieldType {
	if (value === null) return "null";
	if (typeof value === "string") return inferStringType(value);
	if (typeof value === "boolean") return "bool";
	if (typeof value === "number") return Number.isInteger(value) ? "int" : "float";
	if (Array.isArray(value)) return "list";
	if (typeof value === "object") return "object";
	return "string";
}

function combineFieldTypes(types: Iterable<FieldType>): FieldType {
	const observed = [...new Set([...types].filter((type) => type !== "null"))];
	if (observed.length === 0) return "null";
	if (observed.length === 1) return observed[0];
	if (observed.includes("mixed")) return "mixed";
	if (observed.every((type) => type === "int" || type === "float")) return "float";
	if (observed.includes("string") && observed.every((type) => type === "string" || type === "int" || type === "float" || type === "bool")) return "string";
	return "mixed";
}

function inferFieldTypes(
	entries: readonly CorpusEntryInput[],
	overrides?: Record<string, FieldTypeOverride>,
): Record<string, FieldType> {
	const observed = new Map<string, FieldType[]>();
	for (const entry of entries) {
		for (const [field, value] of Object.entries(entry.fields)) {
			const types = observed.get(field) ?? [];
			types.push(inferValueType(value));
			observed.set(field, types);
		}
	}
	const fields = [...observed.keys()].sort();
	const normalizedOverrides = Object.fromEntries(Object.entries(overrides ?? {}).map(([field, type]) => [field, normalizeFieldType(type)]));
	const fieldTypes: Record<string, FieldType> = {};
	for (const field of fields) fieldTypes[field] = normalizedOverrides[field] ?? combineFieldTypes(observed.get(field) ?? []);
	for (const field of Object.keys(normalizedOverrides)) {
		if (!(field in fieldTypes)) fieldTypes[field] = normalizedOverrides[field];
	}
	return fieldTypes;
}

function coerceFieldValue(field: string, value: FieldValue, type: FieldType): FieldValue {
	if (value === null) return null;
	if (type === "mixed") return value;
	if (type === "string") return typeof value === "string" ? value.trim() : fieldValueToText(value);
	if (type === "list") {
		if (Array.isArray(value)) return value;
		throw new Error(`Field "${field}" was set to list, but value ${JSON.stringify(value)} is not a list.`);
	}
	if (type === "object") {
		if (typeof value === "object" && !Array.isArray(value)) return value;
		throw new Error(`Field "${field}" was set to object, but value ${JSON.stringify(value)} is not an object.`);
	}
	if (type === "bool") {
		if (typeof value === "boolean") return value;
		if (typeof value === "string" && /^(true|false)$/i.test(value.trim())) return value.trim().toLowerCase() === "true";
		throw new Error(`Field "${field}" was set to bool, but value ${JSON.stringify(value)} cannot be converted to bool.`);
	}
	const numericValue = typeof value === "number" ? value : typeof value === "string" && value.trim() ? Number(value.trim()) : Number.NaN;
	if (!Number.isFinite(numericValue)) {
		throw new Error(`Field "${field}" was set to ${type}, but value ${JSON.stringify(value)} cannot be converted to ${type}.`);
	}
	if (type === "int") {
		if (!Number.isSafeInteger(numericValue)) throw new Error(`Field "${field}" was set to int, but value ${JSON.stringify(value)} is not a safe integer.`);
		return numericValue;
	}
	if (type === "float") return numericValue;
	return null;
}

function coerceFields(fields: Record<string, FieldValue>, fieldTypes: Record<string, FieldType>): Record<string, FieldValue> {
	return Object.fromEntries(
		Object.entries(fields).map(([field, value]) => [field, coerceFieldValue(field, value, fieldTypes[field] ?? inferValueType(value))]),
	);
}

function getStringFieldNames(fieldTypes: Record<string, FieldType>): string[] {
	return Object.entries(fieldTypes)
		.filter(([, type]) => type === "string")
		.map(([field]) => field)
		.sort();
}

function withGlobalFields(entries: readonly CorpusEntryInput[], globalFields: Record<string, string>): CorpusEntryInput[] {
	return entries.map((entry) => ({
		...entry,
		fields: {
			...entry.fields,
			...globalFields,
		},
	}));
}

function sampleFieldValues(entries: readonly CorpusEntryInput[], field: string, type: FieldType): string[] {
	const samples: string[] = [];
	const seen = new Set<string>();
	for (const entry of entries) {
		if (!Object.prototype.hasOwnProperty.call(entry.fields, field)) continue;
		const value = coerceFieldValue(field, entry.fields[field], type);
		const text = fieldValueToText(value).trim();
		if (!text || seen.has(text)) continue;
		seen.add(text);
		samples.push(text.slice(0, 160));
		if (samples.length >= 3) break;
	}
	return samples;
}

function inspectParsedCorpus(
	entries: readonly CorpusEntryInput[],
	options?: { globalFields?: Record<string, string>; fieldTypes?: Record<string, FieldTypeOverride> },
): DatasetInspection {
	const prepared = withGlobalFields(entries, options?.globalFields ?? {});
	const fieldTypes = inferFieldTypes(prepared, options?.fieldTypes);
	const embeddedFields = getStringFieldNames(fieldTypes);
	const fields = Object.keys(fieldTypes).sort().map((field) => {
		const type = fieldTypes[field];
		return {
			name: field,
			type,
			embedded: type === "string",
			nonEmptyCount: prepared.filter((entry) => fieldValueToText(entry.fields[field]).trim().length > 0).length,
			samples: sampleFieldValues(prepared, field, type),
		};
	});
	return { entryCount: prepared.length, fieldTypes, embeddedFields, fields };
}

function getPreferredTextField(headers: string[], textColumn?: string): string | undefined {
	const lowerHeaders = headers.map((h) => h.toLowerCase());
	if (textColumn) {
		const index = lowerHeaders.indexOf(textColumn.toLowerCase());
		return index >= 0 ? headers[index] : undefined;
	}
	const preferredNames = ["text", "response", "content", "answer", "comment", "body"];
	const index = preferredNames.map((name) => lowerHeaders.indexOf(name)).find((candidate) => candidate >= 0) ?? -1;
	return index >= 0 ? headers[index] : undefined;
}

function inferTextFields(fields: Record<string, FieldValue>, preferred?: string): string[] {
	if (preferred && fieldValueToText(fields[preferred]).trim()) return [preferred];
	return Object.entries(fields)
		.filter(([, value]) => typeof value === "string" && value.trim().length > 0)
		.map(([field]) => field);
}

function buildDefaultText(fields: Record<string, FieldValue>, textFields: string[]): string {
	return textFields.map((field) => fieldValueToText(fields[field]).trim()).filter(Boolean).join("\n\n");
}

function parseDelimited(content: string, delimiter: string, textColumn?: string): CorpusEntryInput[] {
	const rows = splitDelimitedRecords(content, delimiter);
	if (rows.length === 0) return [];
	const headers = rows[0].map((h) => h.trim());
	const preferredTextField = getPreferredTextField(headers, textColumn);
	return rows.slice(1).flatMap((cells) => {
		const fields: Record<string, FieldValue> = {};
		for (let i = 0; i < headers.length; i++) {
			const value = (cells[i] ?? "").trim();
			if (value) fields[headers[i]] = value;
		}
		if (Object.keys(fields).length === 0) return [];
		const textFields = inferTextFields(fields, preferredTextField);
		return [{ text: buildDefaultText(fields, textFields), fields, tags: {}, textFields }];
	});
}

function parseJsonCorpus(content: string, textColumn?: string): CorpusEntryInput[] {
	const parsed = JSON.parse(content) as unknown;
	const array = Array.isArray(parsed)
		? parsed
		: typeof parsed === "object" && parsed !== null
			? ((parsed as Record<string, unknown>).entries as unknown[]) ||
				((parsed as Record<string, unknown>).responses as unknown[]) ||
				((parsed as Record<string, unknown>).data as unknown[])
			: undefined;
	if (!Array.isArray(array)) {
		throw new Error("JSON corpus must be an array or an object with entries/responses/data array");
	}
	const preferred = [textColumn, "text", "response", "content", "answer", "comment", "body"].filter(
		(value): value is string => typeof value === "string" && value.length > 0,
	);
	const toFieldValue = (value: unknown): FieldValue | undefined => {
		if (value === undefined || value === null) return;
		if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") return value;
		if (Array.isArray(value)) return value.flatMap((item) => {
			const converted = toFieldValue(item);
			return converted === undefined ? [] : [converted];
		});
		if (typeof value === "object") {
			const out: Record<string, FieldValue> = {};
			for (const [key, nestedValue] of Object.entries(value)) {
				const converted = toFieldValue(nestedValue);
				if (converted !== undefined) out[key] = converted;
			}
			return out;
		}
		return String(value);
	};
	const addFieldFromValue = (
		fields: Record<string, FieldValue>,
		key: string,
		value: unknown,
		options?: { flatten?: boolean },
	): void => {
		if (typeof value === "object" && value !== null && !Array.isArray(value)) {
			for (const [nestedKey, nestedValue] of Object.entries(value)) {
				const converted = toFieldValue(nestedValue);
				if (converted !== undefined) fields[options?.flatten ? nestedKey : `${key}.${nestedKey}`] = converted;
			}
			return;
		}
		const converted = toFieldValue(value);
		if (converted !== undefined) fields[key] = converted;
	};
	return array.flatMap((item) => {
		if (typeof item === "string") {
			const text = item.trim();
			return text ? [{ text, fields: { [LEGACY_TEXT_FIELD]: text }, tags: {}, textFields: [LEGACY_TEXT_FIELD] }] : [];
		}
		if (typeof item !== "object" || item === null) return [];
		const record = item as Record<string, unknown>;
		const textKey = preferred.find((key) => typeof record[key] === "string" && String(record[key]).trim());
		const fields: Record<string, FieldValue> = {};
		for (const [key, value] of Object.entries(record)) {
			if (value === undefined || value === null) continue;
			addFieldFromValue(fields, key, value, { flatten: key === "metadata" });
		}
		if (Object.keys(fields).length === 0) return [];
		const textFields = inferTextFields(fields, textKey);
		return [{ text: buildDefaultText(fields, textFields), fields, tags: {}, textFields }];
	});
}

function parseJsonlCorpus(content: string, textColumn?: string): CorpusEntryInput[] {
	return content
		.split(/\r?\n/)
		.map((line) => line.trim())
		.filter(Boolean)
		.flatMap((line) => parseJsonCorpus(`[${line}]`, textColumn));
}

function parseTextCorpus(content: string): CorpusEntryInput[] {
	const normalized = content.replace(/\r\n/g, "\n").trim();
	if (!normalized) return [];
	const paragraphEntries = normalized.split(/\n\s*\n+/).map((part) => part.trim()).filter(Boolean);
	const entries = paragraphEntries.length > 1 ? paragraphEntries : normalized.split(/\n+/).map((part) => part.trim()).filter(Boolean);
	return entries.map((text) => ({ text, fields: { [LEGACY_TEXT_FIELD]: text }, tags: {}, textFields: [LEGACY_TEXT_FIELD] }));
}

export function parseCorpusFile(inputPath: string, options?: { format?: string; textColumn?: string }): CorpusEntryInput[] {
	const content = readFileSync(inputPath, "utf8");
	const ext = extname(inputPath).toLowerCase();
	const format = (options?.format ?? "auto").toLowerCase();
	if (format === "jsonl" || (format === "auto" && ext === ".jsonl")) return parseJsonlCorpus(content, options?.textColumn);
	if (format === "json" || (format === "auto" && ext === ".json")) return parseJsonCorpus(content, options?.textColumn);
	if (format === "csv" || (format === "auto" && ext === ".csv")) return parseDelimited(content, ",", options?.textColumn);
	if (format === "tsv" || (format === "auto" && ext === ".tsv")) return parseDelimited(content, "\t", options?.textColumn);
	return parseTextCorpus(content);
}

export function buildBm25Index(entries: QuailEntry[], fieldTypes?: Record<string, FieldType>): Bm25Index {
	const k1 = 1.5;
	const b = 0.75;
	const docLengths: Record<string, number> = {};
	const docFreq: Record<string, number> = {};
	const termFreq: Record<string, Record<string, number>> = {};
	let totalLength = 0;
	for (const [docId, text] of getBm25Documents(entries, fieldTypes)) {
		const tokens = tokenize(text);
		docLengths[docId] = tokens.length;
		totalLength += tokens.length;
		const tf: Record<string, number> = {};
		for (const token of tokens) tf[token] = (tf[token] ?? 0) + 1;
		termFreq[docId] = tf;
		for (const token of new Set(tokens)) docFreq[token] = (docFreq[token] ?? 0) + 1;
	}
	const docCount = Object.keys(docLengths).length;
	return {
		k1,
		b,
		avgDocLength: docCount > 0 ? totalLength / docCount : 0,
		docCount,
		docLengths,
		docFreq,
		termFreq,
	};
}

function getBm25Documents(entries: QuailEntry[], fieldTypes?: Record<string, FieldType>): Array<[string, string]> {
	return entries.flatMap((entry) => [
		[entry.id, entry.text] as [string, string],
		...getStringFieldDocuments([entry], fieldTypes),
	]);
}

function getStringFieldDocuments(entries: QuailEntry[], fieldTypes?: Record<string, FieldType>): Array<[string, string]> {
	const stringFields = fieldTypes ? new Set(getStringFieldNames(fieldTypes)) : undefined;
	return entries.flatMap((entry) => {
		const docs: Array<[string, string]> = [];
		for (const [field, value] of Object.entries(entry.fields)) {
			if (stringFields && !stringFields.has(field)) continue;
			if (!stringFields && typeof value !== "string") continue;
			const text = typeof value === "string" ? value : fieldValueToText(value);
			if (text.trim()) docs.push([fieldDocumentId(entry.id, field), text]);
		}
		return docs;
	});
}

function getEmbeddingDocuments(entries: QuailEntry[], fieldTypes?: Record<string, FieldType>): Array<[string, string]> {
	return getStringFieldDocuments(entries, fieldTypes).filter(([, text]) => text.trim().length > 0);
}

export function bm25Score(index: Bm25Index, entryId: string, query: string): number {
	return bm25ScoreTerms(index, entryId, tokenize(query));
}

export function bm25ScoreTerms(index: Bm25Index, entryId: string, terms: readonly string[]): number {
	if (terms.length === 0 || index.docCount === 0) return 0;
	const tf = index.termFreq[entryId] ?? {};
	const dl = index.docLengths[entryId] ?? 0;
	const avgdl = index.avgDocLength || 1;
	let score = 0;
	for (const term of terms) {
		const f = tf[term] ?? 0;
		if (f <= 0) continue;
		const n = index.docFreq[term] ?? 0;
		const idf = Math.log(1 + (index.docCount - n + 0.5) / (n + 0.5));
		const denom = f + index.k1 * (1 - index.b + index.b * (dl / avgdl));
		score += idf * ((f * (index.k1 + 1)) / denom);
	}
	return score;
}

function l2Normalize(vector: number[]): Float32Array {
	let sumSquares = 0;
	for (const value of vector) sumSquares += value * value;
	const norm = Math.sqrt(sumSquares);
	const out = new Float32Array(vector.length);
	if (!Number.isFinite(norm) || norm === 0) {
		out.set(vector);
		return out;
	}
	for (let i = 0; i < vector.length; i++) out[i] = vector[i] / norm;
	return out;
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
	return typeof value === "object" && value !== null ? (value as Record<string, unknown>) : undefined;
}

function numberVector(value: unknown): number[] | undefined {
	if (!Array.isArray(value)) return undefined;
	const vector = value.map((item) => Number(item));
	return vector.every(Number.isFinite) ? vector : undefined;
}

function embeddingVectors(value: unknown): number[][] | undefined {
	if (!Array.isArray(value)) return undefined;
	const vectors = value.map(numberVector);
	if (vectors.some((vector) => vector === undefined)) return undefined;
	return vectors as number[][];
}

function parseOllamaEmbeddingResponse(data: unknown): number[][] | undefined {
	const record = asRecord(data);
	if (!record) return undefined;
	const embeddings = embeddingVectors(record.embeddings);
	if (embeddings) return embeddings;
	const embedding = numberVector(record.embedding);
	if (embedding) return [embedding];
	return undefined;
}

function parseOpenRouterEmbeddingResponse(data: unknown): number[][] | undefined {
	const record = asRecord(data);
	if (!record) return undefined;
	const rows = Array.isArray(record.data)
		? record.data.map((item, fallbackIndex) => {
				const row = asRecord(item);
				const embedding = numberVector(row?.embedding);
				if (!embedding) return undefined;
				const index = typeof row?.index === "number" && Number.isInteger(row.index) ? row.index : fallbackIndex;
				return { index, embedding };
			})
		: undefined;
	if (rows && rows.every((row): row is { index: number; embedding: number[] } => row !== undefined)) {
		return [...rows].sort((a, b) => a.index - b.index).map((row) => row.embedding);
	}
	return parseOllamaEmbeddingResponse(data);
}

export function cosineSimilarity(a: EmbeddingVector, b: EmbeddingVector): number {
	if (hasDotProduct(b)) return b.dotProduct(a);
	if (hasDotProduct(a)) return a.dotProduct(b);
	const n = Math.min(a.length, b.length);
	let sum = 0;
	for (let i = 0; i < n; i++) sum += (a[i] ?? 0) * (b[i] ?? 0);
	return sum;
}

export function scoreEmbeddingVectorValues(vectors: ReadonlyArray<EmbeddingVector | undefined>, query: EmbeddingVector): Float32Array {
	const firstFileBacked = vectors.find((vector): vector is FileBackedEmbeddingVector => vector instanceof FileBackedEmbeddingVector);
	if (firstFileBacked) {
		const store = firstFileBacked.bulkStore();
		const indices: number[] = [];
		let canBulkScan = true;
		for (const vector of vectors) {
			if (!vector) {
				indices.push(-1);
				continue;
			}
			if (!(vector instanceof FileBackedEmbeddingVector) || vector.bulkStore() !== store) {
				canBulkScan = false;
				break;
			}
			indices.push(vector.bulkIndex());
		}
		if (canBulkScan) return store.dotProducts(indices, query);
	}
	const scores = new Float32Array(vectors.length);
	for (let index = 0; index < vectors.length; index++) {
		const vector = vectors[index];
		scores[index] = vector ? cosineSimilarity(query, vector) : 0;
	}
	return scores;
}

class EmbeddingBackendError extends Error {
	constructor(
		message: string,
		readonly retryable = true,
	) {
		super(message);
	}
}

function sleep(ms: number): Promise<void> {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

function embeddingRetryDelayMs(attempt: number): number {
	const base = defaultEmbeddingRetryBaseMs();
	if (base <= 0) return 0;
	const capped = Math.min(30_000, base * (2 ** Math.max(0, attempt - 1)));
	return capped + Math.floor(Math.random() * Math.min(250, base));
}

function shouldRetryEmbeddingError(error: unknown): boolean {
	if (error instanceof EmbeddingBackendError) return error.retryable;
	if (!(error instanceof Error)) return true;
	return !/OPENROUTER_API_KEY|QUAIL_OPENROUTER_API_KEY|Unsupported QUAIL_EMBEDDING_PROVIDER/.test(error.message);
}

async function embedOllamaBatch(config: EmbeddingBackendConfig, inputs: string[]): Promise<EmbeddingVector[]> {
	const response = await fetch(config.url, {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify({ model: config.model, input: inputs }),
	});
	if (!response.ok) {
		const text = await response.text().catch(() => "");
		throw new EmbeddingBackendError(`Ollama-compatible embed failed (${response.status}): ${text || response.statusText}`, response.status >= 500);
	}
	const embeddings = parseOllamaEmbeddingResponse(await response.json());
	if (embeddings) return embeddings.map(l2Normalize);
	throw new EmbeddingBackendError("Ollama-compatible embed response did not include embeddings");
}

async function embedOpenRouterBatch(config: EmbeddingBackendConfig, inputs: string[]): Promise<EmbeddingVector[]> {
	const apiKey = envValue("QUAIL_OPENROUTER_API_KEY") ?? envValue("OPENROUTER_API_KEY");
	if (!apiKey) {
		throw new EmbeddingBackendError("OpenRouter embeddings require OPENROUTER_API_KEY or QUAIL_OPENROUTER_API_KEY.", false);
	}
	const body: Record<string, unknown> = { model: config.model, input: inputs };
	if (config.providerOnly) body.provider = { only: [config.providerOnly] };
	const response = await fetch(config.url, {
		method: "POST",
		headers: {
			"authorization": `Bearer ${apiKey}`,
			"content-type": "application/json",
			"http-referer": "https://github.com/quail",
			"x-title": "Quail v0.7",
		},
		body: JSON.stringify(body),
	});
	if (!response.ok) {
		const text = await response.text().catch(() => "");
		throw new EmbeddingBackendError(`OpenRouter embed failed (${response.status}): ${text || response.statusText}`, response.status === 429 || response.status >= 500);
	}
	const embeddings = parseOpenRouterEmbeddingResponse(await response.json());
	if (embeddings) return embeddings.map(l2Normalize);
	throw new EmbeddingBackendError("OpenRouter embed response did not include embeddings");
}

async function embedBatchOnce(model: string, inputs: string[]): Promise<EmbeddingVector[]> {
	const config = getEmbeddingBackendConfig(model);
	if (config.provider === "ollama") return embedOllamaBatch(config, inputs);
	return embedOpenRouterBatch(config, inputs);
}

async function embedBatch(model: string, inputs: string[]): Promise<EmbeddingVector[]> {
	const maxRetries = defaultEmbeddingMaxRetries();
	let lastError: unknown;
	for (let attempt = 0; attempt <= maxRetries; attempt++) {
		try {
			const embeddings = await embedBatchOnce(model, inputs);
			if (embeddings.length !== inputs.length) {
				throw new EmbeddingBackendError(`Embedding backend returned ${embeddings.length} embedding(s) for ${inputs.length} input(s).`);
			}
			return embeddings;
		} catch (error) {
			lastError = error;
			if (attempt >= maxRetries || !shouldRetryEmbeddingError(error)) throw error;
			await sleep(embeddingRetryDelayMs(attempt + 1));
		}
	}
	throw lastError instanceof Error ? lastError : new Error(String(lastError));
}

export async function embedTexts(
	texts: string[],
	options?: { model?: string; batchSize?: number; concurrency?: number; onProgress?: (message: string) => void },
): Promise<EmbeddingIndex> {
	const model = options?.model ?? defaultEmbeddingModel();
	const batchSize = options?.batchSize ?? defaultEmbeddingBatchSize();
	const concurrency = Math.max(1, Math.floor(options?.concurrency ?? defaultEmbeddingConcurrency()));
	const vectors: Record<string, EmbeddingVector> = {};
	let dimensions = 0;
	const batches: Array<{ start: number; inputs: string[] }> = [];
	for (let start = 0; start < texts.length; start += batchSize) {
		batches.push({ start, inputs: texts.slice(start, start + batchSize) });
	}
	let nextBatchIndex = 0;
	const embedNextBatch = async (): Promise<void> => {
		while (true) {
			const batchIndex = nextBatchIndex++;
			const batchSpec = batches[batchIndex];
			if (!batchSpec) return;
			const { start, inputs: batch } = batchSpec;
			options?.onProgress?.(`Embedding entries ${start + 1}-${start + batch.length} of ${texts.length}`);
			const embeddings = await embedBatch(model, batch);
			for (let i = 0; i < embeddings.length; i++) {
				vectors[String(start + i)] = embeddings[i];
				dimensions = Math.max(dimensions, embeddings[i].length);
			}
		}
	};
	await Promise.all(Array.from({ length: Math.min(concurrency, batches.length) }, () => embedNextBatch()));
	return { model, dimensions, vectors };
}

export function datasetExists(cwd: string, name: string): boolean {
	const slug = slugifyDatasetName(name);
	return existsSync(join(datasetDir(cwd, slug), MANIFEST_FILE));
}

export function listDatasets(cwd: string): DatasetListItem[] {
	ensureQuailWorkspace(cwd);
	const rootManifestPath = join(getQuailDatasetsDir(cwd), ROOT_MANIFEST_FILE);
	if (!existsSync(rootManifestPath)) return [];
	const manifest = readJson<{ datasets: DatasetListItem[] }>(rootManifestPath);
	return [...(manifest.datasets ?? [])].sort((a, b) => a.name.localeCompare(b.name));
}

function writeRootManifest(cwd: string): void {
	const datasetsDir = getQuailDatasetsDir(cwd);
	const datasets: DatasetListItem[] = [];
	if (!existsSync(datasetsDir)) mkdirSync(datasetsDir, { recursive: true });
	for (const name of readdirSync(datasetsDir)) {
		const path = join(datasetsDir, name, MANIFEST_FILE);
		if (!existsSync(path)) continue;
		const manifest = readJson<QuailDatasetManifest>(path);
		datasets.push({
			name: manifest.name,
			slug: manifest.slug,
			entryCount: manifest.entryCount,
			embeddingModel: manifest.embeddingModel,
			metadataFields: manifest.metadataFields,
			createdAt: manifest.createdAt,
		});
	}
	writeJson(join(datasetsDir, ROOT_MANIFEST_FILE), { version: 1, datasets });
}

export function removeDataset(cwd: string, name: string): boolean {
	ensureQuailWorkspace(cwd);
	const slug = slugifyDatasetName(name);
	const dir = datasetDir(cwd, slug);
	if (!existsSync(dir)) return false;
	rmSync(dir, { recursive: true, force: true });
	loadedDatasetCache.delete(datasetCacheKey(cwd, slug));
	writeRootManifest(cwd);
	return true;
}

function buildFieldContains(fields: Record<string, FieldValue>, fieldTypes: Record<string, FieldType>): Record<string, string> {
	return Object.fromEntries(
		Object.entries(fields)
			.filter(([field, value]) => fieldTypes[field] === "string" && typeof value === "string")
			.map(([field, value]) => [field, normalizeContainsText(value as string)]),
	);
}

function normalizeLoadedEntry(raw: QuailEntry, manifest?: QuailDatasetManifest): QuailEntry {
	const rawRecord = raw as QuailEntry & { fields?: Record<string, FieldValue>; fieldContains?: Record<string, string> };
	const fields: Record<string, FieldValue> = rawRecord.fields && typeof rawRecord.fields === "object"
		? { ...rawRecord.fields }
		: {
			...(raw.text ? { [LEGACY_TEXT_FIELD]: raw.text } : {}),
			...Object.fromEntries(Object.entries(raw.tags ?? {}).map(([field, value]) => [field, inferFieldValue(String(value))])),
		};
	const fieldTypes = manifest?.fieldTypes ?? inferFieldTypes([{ text: raw.text ?? "", fields, tags: raw.tags ?? {} }]);
	const textFields = getStringFieldNames(fieldTypes);
	const text = raw.text || buildDefaultText(fields, textFields);
	return {
		...raw,
		text,
		fields,
		tags: raw.tags ?? {},
		contains: raw.contains ?? normalizeContainsText(text),
		fieldContains: rawRecord.fieldContains ?? buildFieldContains(fields, fieldTypes),
	};
}

function getFieldNames(entries: QuailEntry[]): string[] {
	return Array.from(new Set(entries.flatMap((entry) => Object.keys(entry.fields)))).sort();
}

function uniqueStrings(values: string[]): string[] {
	return [...new Set(values)];
}

function getTextFieldNames(entries: QuailEntry[], fieldTypes?: Record<string, FieldType>): string[] {
	if (fieldTypes) return getStringFieldNames(fieldTypes);
	return getFieldNames(entries).filter((field) => entries.some((entry) => typeof entry.fields[field] === "string" && String(entry.fields[field]).trim().length > 0));
}

export function loadDataset(cwd: string, name: string): LoadedQuailDataset {
	const slug = slugifyDatasetName(name);
	const dir = datasetDir(cwd, slug);
	const manifestPath = join(dir, MANIFEST_FILE);
	if (!existsSync(manifestPath)) {
		throw new Error(`Dataset "${name}" is not processed in this Quail workspace`);
	}
	const manifest = readJson<QuailDatasetManifest>(manifestPath);
	const entriesPath = join(dir, manifest.files.entries);
	const bm25Path = join(dir, manifest.files.bm25);
	const embeddingsPath = join(dir, manifest.files.embeddings);
	const vectorPath = join(dir, EMBEDDINGS_VECTOR_FILE);
	const cacheKey = datasetCacheKey(cwd, slug);
	const manifestMtimeMs = getMtimeMs(manifestPath);
	const entriesMtimeMs = getMtimeMs(entriesPath);
	const bm25MtimeMs = getMtimeMs(bm25Path);
	const embeddingsMtimeMs = getMtimeMs(embeddingsPath);
	const vectorMtimeMs = existsSync(vectorPath) ? getMtimeMs(vectorPath) : undefined;
	const cached = loadedDatasetCache.get(cacheKey);
	if (
		cached &&
		cached.manifestMtimeMs === manifestMtimeMs &&
		cached.entriesMtimeMs === entriesMtimeMs &&
		cached.bm25MtimeMs === bm25MtimeMs &&
		cached.embeddingsMtimeMs === embeddingsMtimeMs &&
		cached.vectorMtimeMs === vectorMtimeMs
	) {
		return cached.dataset;
	}
	const entries = readFileSync(entriesPath, "utf8")
		.split(/\r?\n/)
		.map((line) => line.trim())
		.filter(Boolean)
		.map((line) => normalizeLoadedEntry(JSON.parse(line) as QuailEntry, manifest));
	manifest.fieldNames ??= getFieldNames(entries);
	manifest.fieldTypes ??= inferFieldTypes(entries.map((entry) => ({ text: entry.text, fields: entry.fields, tags: entry.tags })));
	manifest.embeddedFields ??= getStringFieldNames(manifest.fieldTypes);
	manifest.textFields ??= manifest.embeddedFields;
	for (const entry of entries) {
		const stringFields = manifest.embeddedFields ?? [];
		entry.text = entry.text || buildDefaultText(entry.fields, stringFields);
		entry.contains = normalizeContainsText(entry.text);
		entry.fieldContains = buildFieldContains(entry.fields, manifest.fieldTypes);
	}
	const bm25 = readJson<Bm25Index>(bm25Path);
	const embeddings = loadEmbeddingIndex(dir, manifest.files.embeddings);
	const dataset = { manifest, workspaceDir: getQuailWorkspaceDir(cwd), datasetDir: dir, entries, bm25, embeddings };
	loadedDatasetCache.set(cacheKey, {
		manifestMtimeMs,
		entriesMtimeMs,
		bm25MtimeMs,
		embeddingsMtimeMs,
		vectorMtimeMs,
		dataset,
	});
	return dataset;
}

export function loadDatasets(cwd: string, names: string[]): LoadedQuailDataset[] {
	return names.map((name) => loadDataset(cwd, name));
}

export function inspectDatasetFile(options: {
	inputPath: string;
	format?: string;
	textColumn?: string;
	globalTags?: Record<string, string>;
	fieldTypes?: Record<string, FieldTypeOverride>;
}): DatasetInspection {
	const parsed = parseCorpusFile(options.inputPath, { format: options.format, textColumn: options.textColumn });
	return inspectParsedCorpus(parsed, { globalFields: options.globalTags, fieldTypes: options.fieldTypes });
}

export async function processDataset(options: ProcessDatasetOptions): Promise<QuailDatasetManifest> {
	const cwd = options.cwd;
	loadLocalEmbeddingEnv(cwd);
	const name = options.name.trim();
	if (!name) throw new Error("Dataset name is required");
	ensureQuailWorkspace(cwd);
	const slug = slugifyDatasetName(name);
	const dir = datasetDir(cwd, slug);
	if (existsSync(join(dir, MANIFEST_FILE)) && !options.overwrite) {
		throw new Error(`Dataset name must be unique. "${name}" already exists.`);
	}
	loadedDatasetCache.delete(datasetCacheKey(cwd, slug));
	options.onProgress?.("[1/6] Reading and structuring corpus");
	const parsed = parseCorpusFile(options.inputPath, { format: options.format, textColumn: options.textColumn });
	if (parsed.length === 0) throw new Error("No entries found in corpus");
	const globalTags = options.globalTags ?? {};
	const prepared = withGlobalFields(parsed, globalTags);
	const inspection = inspectParsedCorpus(prepared, { fieldTypes: options.fieldTypes });
	const fieldTypes = inspection.fieldTypes;
	const embeddedFields = inspection.embeddedFields;
	const entries: QuailEntry[] = parsed.map((entry, index) => ({
		...(() => {
			const rawFields: Record<string, FieldValue> = {
				...entry.fields,
				...globalTags,
			};
			const fields = coerceFields(rawFields, fieldTypes);
			const text = buildDefaultText(fields, embeddedFields);
			return {
				id: stableEntryId(slug, index),
				dataset: name,
				ordinal: index + 1,
				text,
				fields,
				tags: { ...entry.tags },
				contains: normalizeContainsText(text),
				fieldContains: buildFieldContains(fields, fieldTypes),
			};
		})(),
	}));
	const metadataFields = Array.from(new Set(entries.flatMap((entry) => Object.keys(entry.fields)))).sort();
	const textFields = getTextFieldNames(entries, fieldTypes);

	options.onProgress?.("[2/6] Building BM25 preprocessing index");
	const bm25 = buildBm25Index(entries, fieldTypes);

	options.onProgress?.("[3/6] Preparing exact contains search text");
	// Exact string search uses the per-entry normalized `contains` field written below.

	const embeddingModel = options.model ?? defaultEmbeddingModel();
	const batchSize = options.batchSize ?? defaultEmbeddingBatchSize();
	const embeddingConcurrency = options.embeddingConcurrency ?? defaultEmbeddingConcurrency();
	let embeddingIndex: EmbeddingIndex;
	if (options.skipEmbeddings) {
		options.onProgress?.("[4/6] Skipping embeddings by request");
		embeddingIndex = { model: embeddingModel, dimensions: 0, vectors: {} };
	} else {
		options.onProgress?.(
			`[4/6] Embedding with ${embeddingBackendDescription(embeddingModel)} at batch size ${batchSize}, concurrency ${embeddingConcurrency}`,
		);
		const embeddingDocs = getEmbeddingDocuments(entries, fieldTypes);
		const byPosition = await embedTexts(
			embeddingDocs.map(([, text]) => text),
			{ model: embeddingModel, batchSize, concurrency: embeddingConcurrency, onProgress: options.onProgress },
		);
		embeddingIndex = {
			model: byPosition.model,
			dimensions: byPosition.dimensions,
			vectors: Object.fromEntries(embeddingDocs.map(([docId], index) => [docId, byPosition.vectors[String(index)] ?? []])),
		};
	}

	options.onProgress?.(`[5/6] Writing dataset files into ${getQuailDatasetsDir(cwd)}`);
	mkdirSync(dir, { recursive: true });
	writeFileSync(join(dir, ENTRIES_FILE), `${entries.map((entry) => JSON.stringify(entry)).join("\n")}\n`, "utf8");
	writeJson(join(dir, BM25_FILE), bm25);
	writeEmbeddingIndex(dir, EMBEDDINGS_FILE, embeddingIndex);
	const now = new Date().toISOString();
	const manifest: QuailDatasetManifest = {
		name,
		slug,
		createdAt: now,
		updatedAt: now,
		entryCount: entries.length,
		metadataFields,
		fieldNames: metadataFields,
		textFields,
		fieldTypes,
		embeddedFields,
		embeddingModel,
		embeddingDimensions: embeddingIndex.dimensions,
		batchSize,
		source: { fileName: basename(options.inputPath), format: options.format ?? "auto" },
		files: { entries: ENTRIES_FILE, bm25: BM25_FILE, embeddings: EMBEDDINGS_FILE },
	};
	writeJson(join(dir, MANIFEST_FILE), manifest);

	options.onProgress?.("[6/6] Updating dataset registry");
	writeRootManifest(cwd);
	return manifest;
}
